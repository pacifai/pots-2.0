"""FS37's fused measuring kernel, tested without a GPU.

``measure_phases.h`` splits the core's arithmetic into items and steps that the kernel runs a
warp at a time; ``measure_core_host.c`` runs the same items and steps in loops, with the GPU's
shuffle tree on a 32-slot array. These tests hold that run to the host build of FS36 bit for bit,
and the Python half of the kernel (the table, the work array, the challenge source, the build
with its cache, the in-flight list, the routing) to its own rules. ``test_measure_kernel_cuda.py``
holds the real kernel to the same numbers."""

import ctypes
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    H,
    SHAPES,
    bits,
    host_measure,
    hostlib,
    make,
)
from tests.verification.verifier.matmul_check.test_measure_core_spec import (  # noqa: E402
    fs36_cases,
    same_bits,
)
from verification.commitment import gpu_blake3 as gb
from verification.computation.interface import ReplayError
from verification.verifier.matmul_check import challenge_source as cs
from verification.verifier.matmul_check import freivalds as fv
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check import measure_table as mt
from verification.verifier.matmul_check.challenges import challenge_matrices, challenge_matrix
from verification.verifier.matmul_check.freivalds import MeasureGroup
from verification.verifier.matmul_check.measure_table import MeasureInput, pack


@pytest.fixture(scope="module")
def phased(hostlib, host_measure):
    hostlib.pots_host_sizeof_mwork.restype = ctypes.c_size_t
    hostlib.pots_host_scratch_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_uint32]
    hostlib.pots_host_scratch_bytes.restype = ctypes.c_uint64
    hostlib.pots_host_measure_phased.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
                                                 ctypes.c_void_p, ctypes.c_int]
    hostlib.pots_host_measure_phased.restype = ctypes.c_int

    def run(mis, *, reverse=False, raw=False):
        """The rows of ``mis`` by the kernel's decomposition (Python's table and work array)."""
        table = pack(mis)
        prods = np.frombuffer(table.array, dtype=mk.MPROD_DTYPE).copy()
        scratch = np.zeros(mk.scratch_bytes(prods) + 8, dtype=np.uint8)
        base = (scratch.ctypes.data + 7) & ~7
        works = mk.works_array(prods, base)
        rows = mt.empty_rows(table)
        err = hostlib.pots_host_measure_phased(prods.ctypes.data, len(prods), works.ctypes.data,
                                               rows.ctypes.data, int(reverse))
        if raw:
            return err, rows
        assert err == 0
        return rows

    run.reference = lambda mis: host_measure(pack(mis))
    return run


def same(rows, ref):
    assert np.array_equal(bits(rows), bits(ref))


# ---- the decomposition equals the host build ----

@pytest.mark.parametrize("k", [1, 3, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("guard", [True, False])
def test_phased_equals_the_host_build_on_fs36_inputs(phased, k, dtype, biased, guard):
    mis = [make(n, q, w, k, dtype=dtype, biased=biased, guard=guard, seed=s)
           for s, (n, q, w) in enumerate(SHAPES)]
    same(phased(mis), phased.reference(mis))


def test_phased_equals_the_host_build_on_every_fs36_case(phased):
    """The Mac twin of the CUDA file's ``test_kernel_equals_host_on_fs36_inputs``: the same case
    list, each case alone, the kernel's decomposition against the host build."""
    cases = fs36_cases()
    assert len(cases) >= 200
    for label, mi in cases:
        want, got = phased.reference([mi])[0], phased([mi])[0]
        assert same_bits(got, want), (label, got, want)


def test_mixed_shapes_dtypes_flags_in_one_table_in_any_order(phased):
    mis = [make(n, q, w, 4, dtype=(torch.bfloat16 if s % 2 else torch.float32),
                biased=bool(s % 3 == 0), guard=bool(s % 4), seed=10 + s)
           for s, (n, q, w) in enumerate(SHAPES * 2)]
    ref = phased.reference(mis)
    same(phased(mis), ref)
    same(phased(mis, reverse=True), ref)
    rng = np.random.default_rng(3)
    for _ in range(3):
        order = rng.permutation(len(mis))
        rows = phased([mis[i] for i in order])
        for pos, i in enumerate(order):
            same(rows[pos], ref[i])


def test_phased_on_strided_and_transposed_views(phased):
    mi = make(6, 19, 37, 3, biased=True, seed=4)
    g = torch.Generator().manual_seed(9)

    def embed(t):
        big = torch.randn(t.shape[0] + 3, t.shape[1] * 2 + 5, generator=g).to(t.dtype)
        big[1:1 + t.shape[0], 3:3 + 2 * t.shape[1]:2] = t
        return big[1:1 + t.shape[0], 3:3 + 2 * t.shape[1]:2]

    ref = phased.reference([mi])
    for f in (embed, lambda t: t.t().contiguous().t()):
        v = MeasureInput(f(mi.a), f(mi.b), f(mi.p), mi.r, mi.bias, mi.guard)
        assert not v.a.is_contiguous()
        same(phased([v]), ref)


def special(n=4, q=6, w=40, **kw):
    mi = make(n, q, w, 2, seed=7, **kw)
    return MeasureInput(mi.a.clone(), mi.b.clone(), mi.p.clone(), mi.r, mi.bias, mi.guard)


@pytest.mark.parametrize("biased", [False, True])
@pytest.mark.parametrize("guard", [True, False])
def test_phased_on_nan_inf_zero_and_overflow(phased, biased, guard):
    cases = []
    for edit in (lambda m: m.a.__setitem__((0, 0), float("nan")),
                 lambda m: m.b.__setitem__((2, 5), float("nan")),
                 lambda m: m.p.__setitem__((1, 3), float("inf")),
                 lambda m: m.a.__setitem__((3, 1), float("-inf")),
                 lambda m: [t.zero_() for t in (m.a, m.b, m.p)],
                 lambda m: [m.a.fill_(2e19), m.b.fill_(1.0), m.p.fill_(2e19)]):
        mi = special(biased=biased, guard=guard)
        edit(mi)
        cases.append(mi)
    if biased:
        mi = special(biased=True, guard=guard)
        mi.bias[3] = float("nan")
        cases.append(mi)
        mi = special(biased=True, guard=guard)
        mi.bias[0] = float("inf")
        cases.append(mi)
    same(phased(cases), phased.reference(cases))


def test_layouts_and_scratch_size_match_the_c_side(hostlib, phased):
    assert hostlib.pots_host_sizeof_mwork() == mk.WORK_DTYPE.itemsize == 32
    assert hostlib.pots_host_sizeof_mprod() == mk.MPROD_DTYPE.itemsize == 208
    offs = (ctypes.c_size_t * 16)()
    n = hostlib.pots_host_offsets_mprod(offs)
    assert list(offs)[:n] == [mk.MPROD_DTYPE.fields[f][1] for f in mt.MPROD_FIELDS]
    for f in ("a", "b", "p", "bias"):
        assert [mk._MVIEW.fields[x][1] for x in mt.MVIEW_FIELDS] == [
            getattr(mt.MView, x).offset for x in mt.MVIEW_FIELDS]
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, q, w, k = (int(x) for x in rng.integers(1, 200, 4))
        prods = mk.np.zeros(1, dtype=mk.MPROD_DTYPE)
        prods["a"]["rows"], prods["a"]["cols"], prods["k"] = n, q, k
        assert mk._scratch_sizes(prods)[0] == hostlib.pots_host_scratch_bytes(n, q, k)
        assert hostlib.pots_host_scratch_bytes(n, q, k) % 8 == 0


def test_work_array_prefixes_and_scratch_addresses():
    mis = [make(n, q, w, 2, biased=bool(s % 2), seed=s) for s, (n, q, w) in enumerate(SHAPES)]
    prods = np.frombuffer(pack(mis).array, dtype=mk.MPROD_DTYPE)
    works = mk.works_array(prods, 4096)
    n = len(prods)
    assert works.shape == (n + 1,)
    items = np.array([[q + (s % 2), nn, 7] for s, (nn, q, w) in enumerate(SHAPES)])
    assert np.array_equal(works["start"][0], [0, 0, 0])
    assert np.array_equal(works["start"][1:], np.cumsum(items, axis=0))
    at = works["scratch"][:n].astype(np.int64)
    assert at[0] == 4096 and (at % 8 == 0).all()
    sizes = mk._scratch_sizes(prods).astype(np.int64)
    assert np.array_equal(at[1:], at[:-1] + sizes[:-1])


# ---- the vectorized table ----

def member_inputs(grp, t, r, guard=True):
    return MeasureInput(grp.a[t], grp.b[t], grp.p[t], r, grp.bias, guard)


def groups_cases():
    g = torch.Generator().manual_seed(1)

    def rand(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=g).to(dtype)

    big_a, big_b = rand(12, 9, 7), rand(12, 7, 6)
    cases = {
        "fp32 contiguous": MeasureGroup(rand(4, 5, 7), rand(4, 7, 6), rand(4, 5, 6), [1, 2, 3, 4]),
        "bf16": MeasureGroup(rand(3, 5, 7, dtype=torch.bfloat16), rand(3, 7, 6, dtype=torch.bfloat16),
                             rand(3, 5, 6, dtype=torch.bfloat16), [5, 6, 7]),
        "slice of a bigger operand": MeasureGroup(big_a[3:11:2], big_b[::3], rand(4, 9, 6),
                                                  [8, 9, 10, 11]),
        "transposed b": MeasureGroup(rand(3, 5, 7), rand(3, 6, 7).transpose(1, 2), rand(3, 5, 6),
                                     [1, 2, 3]),
        "transposed all": MeasureGroup(rand(2, 7, 5).transpose(1, 2), rand(2, 6, 7).transpose(1, 2),
                                       rand(2, 6, 5).transpose(1, 2), [4, 5]),
        "biased fp32": MeasureGroup(rand(1, 5, 7), rand(1, 7, 6), rand(1, 5, 6), [2], rand(6)),
        "biased bf16 strided bias": MeasureGroup(
            rand(1, 5, 7, dtype=torch.bfloat16), rand(1, 7, 6, dtype=torch.bfloat16),
            rand(1, 5, 6, dtype=torch.bfloat16), [3], rand(12, dtype=torch.bfloat16)[::2]),
        "mixed dtypes": MeasureGroup(rand(2, 5, 7, dtype=torch.bfloat16), rand(2, 7, 6),
                                     rand(2, 5, 6, dtype=torch.bfloat16), [6, 7]),
        "expanded member stride 0": MeasureGroup(rand(1, 5, 7).expand(3, 5, 7), rand(3, 7, 6),
                                                 rand(3, 5, 6), [1, 2, 3]),
    }
    return cases


@pytest.mark.parametrize("guard", [True, False])
def test_vectorized_table_equals_fs36_pack_bytes(guard):
    cases = groups_cases()
    k = 3
    groups = list(cases.values())
    rs = [[challenge_matrix(H, int(m), k, grp.b.shape[2]) for m in grp.ms] for grp in groups]
    r_addrs = np.array([r.data_ptr() for rr in rs for r in rr], dtype=np.uint64)
    got = mk.prods_array(groups, r_addrs, k, guard)
    members = [member_inputs(grp, t, rs[gi][t], guard)
               for gi, grp in enumerate(groups) for t in range(len(grp.ms))]
    want = pack(members)
    assert got.nbytes == ctypes.sizeof(want.array) == 208 * len(members)
    assert got.tobytes() == bytes(want.array)
    for gi, (name, grp) in enumerate(cases.items()):  # and each group alone
        alone = mk.prods_array([grp], np.array([r.data_ptr() for r in rs[gi]], dtype=np.uint64),
                               k, guard)
        pk = pack([member_inputs(grp, t, rs[gi][t], guard) for t in range(len(grp.ms))])
        assert alone.tobytes() == bytes(pk.array), name


def test_table_refusals_name_the_product():
    g = groups_cases()["fp32 contiguous"]
    ok = MeasureGroup(g.a, g.b, g.p, g.ms)
    one = MeasureGroup(g.a[:1], g.b[:1], g.p[:1], [9])

    def refuse(bad, text, groups=None, k=2):
        with pytest.raises(ValueError, match=text):
            mk.validate(groups(bad) if groups else [bad], k)

    refuse(ok._replace(a=g.a.half()), r"group 0 \(products 0-3\): a has dtype")
    refuse(ok._replace(p=g.p.double()), "p has dtype")
    refuse(one._replace(bias=torch.zeros(6).half()), "bias has dtype")
    refuse(ok._replace(a=g.a[0]), "a is not 3-D")
    refuse(ok._replace(b=torch.zeros(4, 8, 6)), "do not make products")
    refuse(ok._replace(p=torch.zeros(4, 5, 7)), "do not make products")
    refuse(ok._replace(ms=[1, 2, 3]), "do not make products")
    refuse(ok._replace(bias=torch.zeros(6)), "bias is allowed only for a group of one member")
    refuse(one._replace(bias=torch.zeros(5)), "bias has shape")
    refuse(one._replace(bias=torch.zeros(1, 6)), "bias has shape")
    refuse(ok._replace(a=torch.zeros(4, 0, 7), p=torch.zeros(4, 0, 6)), "a dimension is below 1")
    refuse(ok._replace(a=torch.zeros(4, 5, 0), b=torch.zeros(4, 0, 6)), "a dimension is below 1")
    refuse(ok._replace(ms=[], a=g.a[:0], b=g.b[:0], p=g.p[:0]), "no members")
    refuse(ok, "k must be in", k=0)
    refuse(ok, "k must be in", k=256)
    with pytest.raises(ValueError, match="no groups"):
        mk.validate([], 2)
    meta = MeasureGroup(*(torch.empty(t.shape, dtype=t.dtype, device="meta")
                          for t in (g.a, g.b, g.p)), [1, 2, 3, 4])
    refuse(meta._replace(a=g.a), "several devices")
    with pytest.raises(ValueError, match=r"group 1 \(products 4-7\): device meta differs"):
        mk.validate([ok, meta], 2)
    with pytest.raises(ValueError, match="group 1 .* the group has no members|group 1"):
        mk.validate([ok, ok._replace(ms=[])], 2)
    # a cpu group is refused by the call itself
    with pytest.raises(ValueError, match="needs CUDA"):
        fv.measure_groups_rows([ok], h=H, k=2)


# ---- the host challenge source ----

def test_host_source_equals_challenge_matrices_on_cpu():
    src = cs.HostChallengeSource()
    ms = [3, 4, 5, 9, 10, 2, 7, 8]
    widths = [6, 6, 6, 40, 40, 6, 65, 65]
    k = 3
    buf, offs, ev = src.draw(H, ms, k, widths, stream=None)
    assert ev is None and buf.dtype == torch.float32 and buf.device.type == "cpu"
    assert offs.dtype == np.int64 and offs[0] == 0
    sizes = np.array(widths) * k * 4
    assert np.array_equal(offs, np.concatenate(([0], np.cumsum(sizes)[:-1])))
    assert buf.numel() * 4 == sizes.sum()
    for m, w, o in zip(ms, widths, offs):
        got = buf[o // 4:o // 4 + w * k].view(w, k)
        assert torch.equal(got.view(torch.int32), challenge_matrix(H, m, k, w).view(torch.int32))
    one = challenge_matrices(H, [3], k, 6)[0]
    assert torch.equal(buf[:6 * k].view(6, k).view(torch.int32), one.view(torch.int32))


def test_host_source_refusals():
    src = cs.HostChallengeSource()
    with pytest.raises(ValueError, match="labels but"):
        src.draw(H, [1, 2], 2, [4], stream=None)
    with pytest.raises(ValueError, match="no products"):
        src.draw(H, [], 2, [], stream=None)
    with pytest.raises(ValueError, match="width is below 1"):
        src.draw(H, [1], 2, [0], stream=None)
    with pytest.raises(ValueError, match="m must be"):
        src.draw(H, [0], 2, [4], stream=None)


# ---- routing ----

def test_kernel_device_follows_the_device_and_reference():
    assert not fv.kernel_device("cpu") and not fv.kernel_device(torch.device("cpu"))
    assert not fv.kernel_device("mps")
    assert fv.kernel_device("cuda") and fv.kernel_device(torch.device("cuda", 1))
    with fv.reference():
        assert not fv.kernel_device("cuda:0")
        with fv.reference():
            assert not fv.kernel_device("cuda:0")
        assert not fv.kernel_device("cuda:0")
    assert fv.kernel_device("cuda:0")
    with pytest.raises(RuntimeError):
        with fv.reference():
            raise RuntimeError("x")
    assert fv.kernel_device("cuda:0")


def test_cpu_paths_keep_their_bits_and_never_touch_the_kernel(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("the kernel was called")

    monkeypatch.setattr(mk, "enqueue", boom)
    mi = make(5, 9, 11, 3, biased=True)
    row = fv.measure_product_row(mi.a, mi.b, mi.p, h=H, m=1, k=3, bias=mi.bias)
    with fv.reference():
        again = fv.measure_product_row(mi.a, mi.b, mi.p, h=H, m=1, k=3, bias=mi.bias)
    assert torch.equal(row.view(torch.int64), again.view(torch.int64))
    a, b, p = mi.a[None].repeat(2, 1, 1), mi.b[None].repeat(2, 1, 1), mi.p[None].repeat(2, 1, 1)
    assert fv.measure_products_rows(a, b, p, h=H, ms=[1, 2], k=3).shape == (2, 8)
    fv.drain()
    fv.drain("cpu")


def test_cuda_routing_wraps_the_arguments_as_one_group(monkeypatch):
    seen = []

    class Tens(torch.Tensor):
        pass

    def fake(groups, *, h, k, guard=True, **kw):
        seen.append((groups, h, k, guard))
        n = sum(len(g.ms) for g in groups)
        return torch.arange(n * (5 + k), dtype=torch.float64).view(n, 5 + k)

    monkeypatch.setattr(mk, "enqueue", fake)
    monkeypatch.setattr(fv, "kernel_device", lambda d: True)
    monkeypatch.setattr(fv, "drain", lambda d=None: seen.append(("drain", d)))
    mi = make(5, 9, 11, 3, biased=True)
    row = fv.measure_product_row(mi.a, mi.b, mi.p, h=H, m=7, k=3, bias=mi.bias, guard=False,
                                 scratch=fv.MeasureScratch())
    (groups, h, k, guard), = seen
    (g,) = groups
    assert row.shape == (8,) and k == 3 and guard is False and h == H
    assert list(g.ms) == [7] and g.bias is mi.bias and g.a.shape == (1, 5, 9)
    assert g.a.data_ptr() == mi.a.data_ptr()
    seen.clear()
    a, b, p = mi.a[None].repeat(2, 1, 1), mi.b[None].repeat(2, 1, 1), mi.p[None].repeat(2, 1, 1)
    rows = fv.measure_products_rows(a, b, p, h=H, ms=[4, 5], k=3)
    assert rows.shape == (2, 8) and seen[0][0][0].a is a and list(seen[0][0][0].ms) == [4, 5]
    with pytest.raises(ValueError, match="batch of 1 labels"):
        fv.measure_products_rows(a, b, p, h=H, ms=[4], k=3)
    seen.clear()
    fv.measure_product(mi.a, mi.b, mi.p, h=H, m=1, k=3, eps_in=1e-3, eps_acc=1e-6)
    assert seen[-1] == ("drain", mi.b.device)  # drained before the row is read


# ---- the in-flight list ----

class FakeEvent:
    def __init__(self, done):
        self.done, self.waited = done, 0

    def query(self):
        return self.done

    def synchronize(self):
        self.waited += 1
        self.done = True


@pytest.fixture
def inflight(monkeypatch):
    monkeypatch.setattr(mk, "_inflight", {})
    return mk._inflight


def enqueue_fake(inflight, tensors, done=False, device=torch.device("cpu")):
    ev = FakeEvent(done)
    inflight.setdefault(device, []).append(
        mk.Pending(ev, tensors, [mk._version(t) for t in tensors], []))
    return ev


def test_drain_waits_and_releases(inflight):
    t = torch.zeros(3)
    ev = enqueue_fake(inflight, [t])
    fv.drain()
    assert ev.waited == 1 and inflight[torch.device("cpu")] == []
    fv.drain()


def test_drain_raises_replay_error_on_a_changed_input_and_still_releases(inflight):
    a, b = torch.zeros(3), torch.zeros(3)
    ev1 = enqueue_fake(inflight, [a])
    ev2 = enqueue_fake(inflight, [b])
    b.add_(1)
    with pytest.raises(ReplayError, match="changed after it was enqueued"):
        fv.drain("cpu")
    assert ev1.waited == 1 and ev2.waited == 1 and inflight[torch.device("cpu")] == []
    fv.drain()


def test_a_view_shares_its_base_version(inflight):
    base = torch.zeros(8)
    view = base[2:5]
    enqueue_fake(inflight, [view])
    base[0] = 1.0  # an in-place write through another view of the storage
    with pytest.raises(ReplayError):
        fv.drain()


def test_drain_of_one_device_leaves_the_others(inflight):
    ev_cpu = enqueue_fake(inflight, [torch.zeros(1)])
    ev_meta = enqueue_fake(inflight, [torch.zeros(1, device="meta")], device=torch.device("meta"))
    fv.drain("meta")
    assert ev_meta.waited == 1 and ev_cpu.waited == 0
    fv.drain()
    assert ev_cpu.waited == 1


def test_retire_releases_only_finished_entries_and_checks_them(inflight):
    cpu = torch.device("cpu")
    done_ok, running = torch.zeros(1), torch.zeros(1)
    enqueue_fake(inflight, [done_ok], done=True)
    enqueue_fake(inflight, [running], done=False)
    mk.retire(cpu)
    assert len(inflight[cpu]) == 1
    changed = torch.zeros(1)
    enqueue_fake(inflight, [changed], done=True)
    changed.add_(1)
    with pytest.raises(ReplayError):
        mk.retire(cpu)
    assert len(inflight[cpu]) == 1  # the running one stays


def test_inference_tensors_are_not_version_checked(inflight):
    with torch.inference_mode():
        t = torch.zeros(2)
    ev = enqueue_fake(inflight, [t])
    t.add_(1) if False else None
    fv.drain()
    assert ev.waited == 1


# ---- the build ----

class FakeNvrtc:
    def __init__(self):
        self.calls = []

    def version(self):
        return (12, 4)

    def compile(self, src, name, headers, options, want):
        self.calls.append((name, dict(headers), list(options), want))
        return 0, b"IMAGE-" + name.encode(), ""

    def error_string(self, rc):
        return ""


IMPORT_PROBE = """
import torch


def libs():
    try:
        maps = open("/proc/self/maps").read()
    except OSError:
        return set()  # not Linux: the caches and torch's CUDA state are the evidence
    return {ln.split()[-1] for ln in maps.splitlines()
            if "libnvrtc" in ln or "libcuda." in ln or "libcuda-" in ln}


before = libs()  # whatever torch itself maps; the imports below must add no nvrtc or libcuda
import verification.runs.loop
import verification.verifier.matmul_check.freivalds
from verification.verifier.matmul_check import measure_kernel as mk
assert mk._kernels.cache_info().currsize == 0 and mk._built.cache_info().currsize == 0
assert not torch.cuda.is_initialized()
assert libs() == before, libs() - before
print("ok")
"""


def test_importing_compiles_and_loads_nothing():
    """In a fresh interpreter, so earlier tests' caches and library loads cannot show."""
    root = Path(__file__).resolve().parents[4]
    proc = subprocess.run([sys.executable, "-c", IMPORT_PROBE], cwd=root, capture_output=True,
                          text=True)
    assert proc.returncode == 0 and proc.stdout.strip() == "ok", proc.stderr


def test_build_passes_fmad_false_the_headers_and_caches(tmp_path):
    nv = FakeNvrtc()
    first = mk.build(nv, (8, 6), cache=tmp_path)
    name, headers, options, want = nv.calls[0]
    assert name == mk.CU_NAME and want == "cubin" and not first.cached
    assert "--fmad=false" in options and "--gpu-architecture=sm_86" in options
    assert set(headers) == {mk.CORE_NAME, mk.PHASES_NAME}
    assert "pots_phase_partial" in headers[mk.PHASES_NAME]
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 1 and re.fullmatch(r"measure_kernel-[0-9a-f]{24}\.cubin", files[0])
    second = mk.build(nv, (8, 6), cache=tmp_path)
    assert second.cached and second.image == first.image and len(nv.calls) == 1
    mk.build(nv, (8, 6), cache=tmp_path, use_cache=False)
    assert len(nv.calls) == 2
    mk.build(nv, (9, 0), cache=tmp_path)  # another architecture, another entry
    assert len(nv.calls) == 3 and len(list(tmp_path.iterdir())) == 2


def test_fmad_is_in_the_cache_key():
    s = mk.source()
    opts = [gb.arch_options((8, 6), "cubin") + list(mk.OPTIONS)]
    without = [gb.arch_options((8, 6), "cubin")]
    assert gb.cache_key("sm_86", (12, 4), opts, s) != gb.cache_key("sm_86", (12, 4), without, s)
    assert "--fmad=false" in mk.OPTIONS


def test_the_blake3_build_is_unchanged_by_the_source_argument(tmp_path):
    key = hashlib.sha256(f"{gb.source_hash()}|sm_86|12.4|[]".encode()).hexdigest()[:24]
    assert gb.cache_key("sm_86", (12, 4)) == key
    nv = FakeNvrtc()
    gb.build_image(nv, (8, 6), cache=tmp_path)
    name, headers, options, _ = nv.calls[0]
    assert name == gb.CU_NAME and list(headers) == [gb.CORE_NAME]
    assert options == gb.arch_options((8, 6), "cubin")
    (entry,) = tmp_path.iterdir()
    opts = [gb.arch_options((8, 6), k) for k in ("cubin", "ptx")]
    assert entry.name == f"gpu_blake3-{gb.cache_key('sm_86', (12, 4), opts)}.cubin"


def test_the_kernel_file_matches_its_python_side():
    cu = (mk.HERE / mk.CU_NAME).read_text()
    assert f'extern "C" __global__' in cu and mk.KERNEL in cu
    assert f"#define POTS_MK_WARPS {mk.WARPS}u" in cu
    for off in (16, 1):
        assert f"off = {off}" in cu or f"off >= {off}" in cu
    assert cu.count("__shfl_down_sync") == 2 and "atomic" not in cu
    assert 'include "measure_core.h"' in cu and 'include "measure_phases.h"' in cu
    assert ctypes.sizeof(mk.KernelArgs) == 48
