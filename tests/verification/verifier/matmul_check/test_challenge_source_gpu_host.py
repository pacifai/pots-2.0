"""The GPU challenge source's Python side (FS41, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.9) on the
Mac: the plan against the host source's offsets, the pinned structs, the params, the CPU
delegation, the request cache through the ``_upload`` seam, the default source and the build's
source. Nothing here loads a kernel."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verification.commitment import gpu_blake3 as gb
from verification.verifier.matmul_check import challenge_source as cs
from verification.verifier.matmul_check import measure_kernel as mk

H = bytes(range(32))


# ---- the plan ----

def test_plan_units_and_offsets_on_mixed_widths():
    ms, widths, k = [5, 2, 9], [1, 17, 32], 3
    p = cs.xof_plan(ms, k, widths)
    nb = [1, 2, 2]
    assert p.items.dtype == cs.XITEM_DTYPE
    assert list(p.items["m"]) == ms and list(p.items["width"]) == widths
    assert list(p.items["unit0"]) == [0, 3, 3 + 6]
    assert p.units == sum(k * b for b in nb) == 15
    assert list(p.items["out_off"]) == [0, 3, 3 + 51]
    assert p.total == 3 + 51 + 96 and p.k == k
    assert list(p.offsets) == [0, 12, 216] and p.offsets.dtype == np.int64
    assert not p.offsets.flags.writeable


def test_plan_offsets_equal_the_host_sources_draw():
    ms, widths, k = [9, 3, 7, 3, 4], [40, 6, 65, 6, 16], 4
    _, offs, _ = cs.HostChallengeSource().draw(H, ms, k, widths, stream=None)
    p = cs.xof_plan(ms, k, widths)
    assert np.array_equal(p.offsets, offs)
    assert p.total == sum(w * k for w in widths)


@pytest.mark.parametrize("ms, k, widths, msg", [
    ([1], 0, [4], "k must be"), ([1], 256, [4], "k must be"),
    ([1, 2], 2, [4], "2 labels but 1 widths"), ([], 2, [], "no products"),
    ([0], 2, [4], "m must be"), ([1 << 32], 2, [4], "m must be"), ([-1], 2, [4], "m must be"),
    ([1 << 70], 2, [4], "does not fit"),
    ([1], 2, [0], "width is below 1"), ([1], 2, [1 << 32], "2\\^32 or more"),
])
def test_plan_validation(ms, k, widths, msg):
    with pytest.raises(ValueError, match=msg):
        cs.xof_plan(ms, k, widths)


def test_plan_accepts_the_range_ends():
    p = cs.xof_plan([1, (1 << 32) - 1], 255, [1, 2])
    assert p.units == 255 * 2 and p.k == 255


# ---- the structs and params ----

def test_struct_sizes_and_field_offsets():
    assert ctypes.sizeof(cs.XParams) == 64 and cs.XITEM_DTYPE.itemsize == 24
    offs = {n: getattr(cs.XParams, n).offset for n, _ in cs.XParams._fields_}
    assert offs == {"key": 0, "k": 32, "n": 36, "units": 40, "items": 48, "out": 56}
    assert {n: cs.XITEM_DTYPE.fields[n][1] for n in cs.XITEM_DTYPE.names} == \
        {"m": 0, "width": 4, "unit0": 8, "out_off": 16}


def test_params_key_is_h_as_little_endian_words():
    p = cs.xof_plan([3, 4], 7, [20, 5])
    prm = cs.xof_params(H, p, 0x1000, 0x2000)
    assert list(prm.key) == [int.from_bytes(H[4 * i:4 * i + 4], "little") for i in range(8)]
    assert (prm.k, prm.n, prm.units, prm.items, prm.out) == (7, 2, p.units, 0x1000, 0x2000)
    with pytest.raises(ValueError, match="32 bytes"):
        cs.xof_params(H[:31], p, 0, 0)


def test_blocks():
    assert cs.xof_blocks(1) == 1 and cs.xof_blocks(256) == 1 and cs.xof_blocks(257) == 2
    assert cs.xof_blocks(10 ** 12) == cs.XOF_MAX_BLOCKS


def test_source_plan_has_no_device():
    plan, prm = cs.GpuChallengeSource().plan(H, [3], 2, [9])
    assert plan.total == 18 and prm.items == 0 and prm.out == 0


# ---- draw on a CPU ----

def test_cpu_draw_equals_the_host_source_byte_for_byte():
    ms, widths, k = [9, 3, 7, 3], [40, 6, 65, 6], 3
    a, ao, ae = cs.GpuChallengeSource().draw(H, ms, k, widths, stream=None)
    b, bo, be = cs.HostChallengeSource().draw(H, ms, k, widths, stream=None)
    assert ae is None and be is None and torch.equal(a.view(torch.int32), b.view(torch.int32))
    assert np.array_equal(ao, bo)


# ---- the cache ----

class FakeStream:
    device = torch.device("cuda", 0)


def _source(monkeypatch):
    uploads = []

    def upload(self, plan, device, stream):
        uploads.append((plan, device))
        return object(), object()

    monkeypatch.setattr(cs.GpuChallengeSource, "_upload", upload)
    return cs.GpuChallengeSource(), uploads


def test_cache_identity_and_equality_hits(monkeypatch):
    src, ups = _source(monkeypatch)
    dev, st = torch.device("cuda", 0), FakeStream()
    ms, widths = (3, 4, 5), (8, 8, 20)
    e = src._lookup(dev, 2, ms, widths, st)
    assert src._lookup(dev, 2, ms, widths, st) is e and len(ups) == 1  # identity
    assert src._lookup(dev, 2, list(ms), list(widths), st) is e and len(ups) == 1  # equal lists
    assert src._lookup(dev, 2, (3, 4, 5), (8, 8, 20), st) is e and len(ups) == 1  # equal tuples
    assert e.plan is ups[0][0] and e.ms == ms and e.widths == widths


def test_cache_misses_on_k_device_and_members(monkeypatch):
    src, ups = _source(monkeypatch)
    st = FakeStream()
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    base = src._lookup(d0, 2, (3, 4), (8, 9), st)
    assert src._lookup(d0, 3, (3, 4), (8, 9), st) is not base
    assert src._lookup(d1, 2, (3, 4), (8, 9), st) is not base
    assert src._lookup(d0, 2, (3, 5), (8, 9), st) is not base
    assert src._lookup(d0, 2, (3, 4), (8, 10), st) is not base
    assert len(ups) == 5
    assert src._lookup(d0, 2, (3, 4), (8, 9), st) is base and len(ups) == 5


def test_cache_evicts_past_the_cap_least_recently_used_first(monkeypatch):
    src, ups = _source(monkeypatch)
    st, dev = FakeStream(), torch.device("cuda", 0)
    first = src._lookup(dev, 2, (1,), (4,), st)
    for m in range(2, cs.LAYOUT_CACHE_MAX + 1):
        src._lookup(dev, 2, (m,), (4,), st)
    assert len(ups) == cs.LAYOUT_CACHE_MAX
    assert src._lookup(dev, 2, (1,), (4,), st) is first  # touched: now the most recent
    src._lookup(dev, 2, (cs.LAYOUT_CACHE_MAX + 1,), (4,), st)  # one past the cap
    assert src._entries_n == cs.LAYOUT_CACHE_MAX
    assert src._lookup(dev, 2, (1,), (4,), st) is first  # kept
    n = len(ups)
    src._lookup(dev, 2, (2,), (4,), st)  # the oldest untouched one was dropped
    assert len(ups) == n + 1


def test_cache_eviction_inside_one_bucket(monkeypatch):
    src, ups = _source(monkeypatch)
    st, dev = FakeStream(), torch.device("cuda", 0)
    # one bucket: same k, length, first and last labels and widths; the middle differs
    for mid in range(cs.LAYOUT_CACHE_MAX + 5):
        src._lookup(dev, 2, (1, 10 + mid, 2), (4, 4, 4), st)
    assert src._entries_n == cs.LAYOUT_CACHE_MAX and len(ups) == cs.LAYOUT_CACHE_MAX + 5


def test_cache_validates_when_an_entry_is_built(monkeypatch):
    src, ups = _source(monkeypatch)
    for ms, widths in [((0,), (4,)), ((1,), (0,)), ((1, 2), (4,)), ((), ())]:
        with pytest.raises(ValueError):
            src._lookup(torch.device("cuda", 0), 2, ms, widths, FakeStream())
    assert ups == []


# ---- the default source ----

@pytest.fixture
def restore_default():
    before = mk.default_source()
    yield
    mk.set_default_source(before)


def test_default_source_is_the_gpu_source_and_set_swaps(restore_default):
    assert isinstance(mk.default_source(), cs.GpuChallengeSource)
    host = cs.HostChallengeSource()
    prev = mk.set_default_source(host)
    assert isinstance(prev, cs.GpuChallengeSource) and mk.default_source() is host
    assert mk.set_default_source(prev) is host and mk.default_source() is prev


def test_prefetch_follows_the_default_source(restore_default):
    assert mk.prefetch(H, [3, 4], 2, [8, 8]) is None  # the GPU source: nothing to prefetch
    mk.set_default_source(cs.HostChallengeSource())
    pf = mk.prefetch(H, [3, 4], 2, [8, 8])
    assert isinstance(pf, cs.Prefetch)
    pf.future.result()


# ---- the build's source ----

def test_xof_source_names_three_files_and_digests_their_text(tmp_path, monkeypatch):
    mm = tmp_path / "mm"
    cm = tmp_path / "cm"
    mm.mkdir()
    cm.mkdir()
    (mm / cs.XOF_NAME).write_text("kernel")
    (mm / cs.XOF_HEADER).write_text("header")
    (cm / cs.XOF_CORE).write_text("core")
    monkeypatch.setattr(cs, "HERE", mm)
    monkeypatch.setattr(cs, "COMMITMENT", cm)
    s = cs.xof_source()
    assert isinstance(s, gb.Source) and s.cu_name == "challenge_xof.cu" and s.prefix == "challenge_xof"
    assert s.cu_text == "kernel" and set(s.headers) == {"challenge_xof.h", "gpu_blake3_core.h"}
    before = s.digest()
    (cm / cs.XOF_CORE).write_text("core2")
    assert cs.xof_source().digest() != before
    (cm / cs.XOF_CORE).write_text("core")
    (mm / cs.XOF_HEADER).write_text("header2")
    assert cs.xof_source().digest() != before


def test_commitment_dir_points_at_the_core_header():
    assert (cs.COMMITMENT / cs.XOF_CORE).is_file()


def test_importing_builds_and_loads_nothing():
    assert cs._xof_built.cache_info().currsize == 0
    assert cs._xof_kernels.cache_info().currsize == 0
