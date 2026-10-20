"""Edge cases of the GPU BLAKE3 kernel (decision F23), on the host build of its core.

Group size sweep (``POTS_G_LOG`` 0..6, with a small ``POTS_R_LOG`` so every stage count shows),
leaf counts around block and tile boundaries, headers, payload alignment (the host build's
``memcpy`` load is counted, so a fast path on a misaligned address is seen), ``first_bad``,
limits, and the Python side against a fake driver and NVRTC. What only a GPU shows (that NVRTC
accepts the kernel file, shared-memory races, speed) is not here."""

import ctypes
import subprocess
import sys
from types import SimpleNamespace

import blake3
import numpy as np
import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import (
    FakeDriver, FakeNvrtc, HostKernels, by_native, hash_tensor_leaf, make_table, rand_bytes,
    tensor_leaf, z)
from verification.commitment import gpu_blake3 as gb
from verification.commitment import native
from verification.commitment.encoding import tensor_leaf_header

ROOT = gb.HERE.parents[1]
TESTDIR = ROOT / "tests/verification/commitment"
HAVE_GPU = torch.cuda.is_available()
_LIBS: dict = {}


@pytest.fixture(scope="module")
def build_host(tmp_path_factory):
    """``build_host(g, r, probe=False)`` -> the host harness built with that group size."""
    def build(g=3, r=6, probe=False):
        key = (g, r, probe)
        if key not in _LIBS:
            out = tmp_path_factory.mktemp("host") / f"host_{g}_{r}_{int(probe)}.so"
            src = TESTDIR / ("gpu_blake3_host_probe.c" if probe else "gpu_blake3_host.c")
            cmd = ["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC",
                   f"-DPOTS_G_LOG={g}", f"-DPOTS_R_LOG={r}", f"-I{gb.HERE}", f"-I{TESTDIR}",
                   str(src), "-o", str(out)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            assert proc.returncode == 0 and not proc.stderr, proc.stderr
            lib = ctypes.CDLL(str(out))
            lib.pots_host_stage.argtypes = [ctypes.c_void_p]
            lib.host_g.restype = lib.host_r.restype = ctypes.c_uint
            assert (lib.host_g(), lib.host_r()) == (1 << g, 1 << r)
            _LIBS[key] = lib
        return _LIBS[key]
    return build


@pytest.fixture(scope="module")
def ref():
    return native.get()


def plan_for(g, r, chunks):
    """An independent plan builder, in plain integers, for any group size ``2^g`` and tile
    ``2^r``: stage 0 has a tile per group, stage s a tile per ``2^r`` nodes of a level that
    still has more than one node."""
    G, R = 1 << g, 1 << r
    levels = []
    for c in chunks:
        lv = [-(-int(c) // G)]
        while lv[-1] > 1:
            lv.append(-(-lv[-1] // R))
        levels.append(lv)
    stages = max((len(lv) for lv in levels), default=1)
    prefix, out_off, base = [], [], 0
    for s in range(stages):
        tiles = [(lv[s] if s == 0 else (lv[s] if lv[s - 1] > 1 else 0)) if s < len(lv) else 0
                 for lv in levels]
        sizes = [lv[s] if s < len(lv) and lv[s] > 1 else 0 for lv in levels]
        prefix.append(np.concatenate([[0], np.cumsum(tiles)]).astype(np.uint64))
        out_off.append((base + np.concatenate([[0], np.cumsum(sizes)[:-1]]
                                              )).astype(np.uint64)[:len(levels)])
        base += sum(sizes)
    return gb.Plan(prefix, out_off, base)


@pytest.fixture
def run_g(build_host, monkeypatch):
    """``run_g(g, r, table)`` -> ``(digests, first_bad)`` through the host build of that size."""
    def run(g, r, t, probe=False):
        monkeypatch.setattr(gb, "make_plan", lambda chunks: plan_for(g, r, chunks))
        k = gb.GpuBlake3(HostKernels(build_host(g, r, probe)))
        d, bad = k.hash_leaves(*t.args(t.arena.data_ptr()), device="cpu")
        return d.numpy(), bad
    return run


def by_blake3(t):
    a = t.arena.numpy()
    out = []
    for i in range(len(t.payload_len)):
        u = int(t.hdr_idx[i])
        h = t.hdr_buf[int(t.hdr_off[u]):][:int(t.hdr_len[u])]
        o, n = int(t.payload_off[i]), int(t.payload_len[i])
        out.append(blake3.blake3(b"\x00" + h + a[o:o + n].tobytes()).digest())
    return out


def agree(run, ref, t):
    got, bad = run(t)
    want, want_bad = by_native(ref, t)
    end = min(want_bad, len(want))
    assert bad == want_bad
    assert np.array_equal(got[:end], want[:end])
    for i, d in enumerate(by_blake3(t)[:end]):
        assert got[i].tobytes() == d, f"leaf {i}"
    return got, bad


def leaf_of_chunks(c, i, *, hl=0, short=False):
    """A kind-0 leaf of ``c`` chunks; ``short`` makes the last chunk 1 byte long."""
    m = 1024 * (c - 1) + 1 if short else 1024 * c
    hl = min(hl, m - 1)
    return (rand_bytes(hl, i), rand_bytes(m - 1 - hl, 77 + i), 0)


# ---- group size ----

def chunk_edges(G, top):
    cs = {1, 2, 3}
    k = 1
    while k <= top:
        cs |= {k - 1, k, k + 1}
        k *= 2
    cs |= {G * 2 ** j + d for j in range(12) for d in (-1, 0, 1) if G * 2 ** j <= top}
    return sorted(c for c in cs if 1 <= c <= top + 1)


@pytest.mark.parametrize("g", range(7))
@pytest.mark.parametrize("r", [2, 6])
def test_every_group_size_around_every_stage_boundary(run_g, ref, g, r):
    """Chunk counts ``2^k·G`` and ``±1`` around every group and tile boundary of stages 1 to 4
    (at ``R = 4``) give the ``blake3`` package's digests, alone and mixed into one table."""
    G = 1 << g
    top = 600 if r == 6 else 600
    cs = chunk_edges(G, top)
    leaves = [leaf_of_chunks(c, c, hl=(0, 3, 255)[c % 3], short=bool(c % 2)) for c in cs]
    agree(lambda t: run_g(g, r, t), ref, make_table(leaves, seed=g))
    tpb = 128 // G
    for m in (1, 2, 3):  # the same boundary leaves, 1, 2 and 3 blocks' worth of tiles apart
        small = [leaf_of_chunks(1 + i % 2, i) for i in range(m * tpb)]
        agree(lambda t: run_g(g, r, t), ref,
              make_table(small + leaves[:12] + small[:tpb], seed=m))


@pytest.mark.parametrize("g, r", [(3, 6), (0, 1), (1, 2), (4, 3), (5, 6), (6, 2)])
def test_the_plan_here_follows_the_group_and_tile_sizes(g, r, monkeypatch):
    """``make_plan`` and ``grid_blocks`` take their shifts from ``G`` and ``R``, so a core built
    with other ``POTS_G_LOG`` and ``POTS_R_LOG`` gets the plan it expects."""
    monkeypatch.setattr(gb, "G", 1 << g)
    monkeypatch.setattr(gb, "R", 1 << r)
    rng = np.random.default_rng(g)
    chunks = rng.integers(1, 40000, 50).astype(np.uint64)
    a, b = gb.make_plan(chunks), plan_for(g, r, chunks)
    assert a.stages == b.stages and a.nodes == b.nodes
    for s in range(a.stages):
        assert np.array_equal(a.tile_prefix[s], b.tile_prefix[s])
        assert np.array_equal(a.out_off[s], b.out_off[s])
    assert gb.grid_blocks(0, 1000) == -(-1000 // (128 >> g)) and gb.grid_blocks(1, 1000) == 1000


@pytest.mark.parametrize("g, r", [(4, 3), (1, 2), (6, 6)])
def test_the_core_agrees_with_the_plan_here_for_other_sizes(g, r, build_host, ref, monkeypatch):
    monkeypatch.setattr(gb, "G", 1 << g)
    monkeypatch.setattr(gb, "R", 1 << r)
    k = gb.GpuBlake3(HostKernels(build_host(g, r)))
    leaves = [leaf_of_chunks(c, c, short=bool(c % 2)) for c in chunk_edges(1 << g, 300)]
    t = make_table(leaves)
    agree(lambda t: (lambda d, bad: (d.numpy(), bad))(
        *k.hash_leaves(*t.args(t.arena.data_ptr()), device="cpu")), ref, t)


# ---- leaf counts around block and tile boundaries ----

@pytest.mark.parametrize("g", [0, 2, 3, 5, 6])
def test_leaf_counts_around_block_boundaries(run_g, ref, g):
    G, tpb = 1 << g, 128 >> g
    sizes = [1, G, G + 1, 2 * G - 1, 3 * G, 2, G // 2 + 1]
    for n in sorted({m * tpb + d for m in (1, 2, 3) for d in (-1, 0, 1)} | {1, 2}):
        if n < 1:
            continue
        leaves = [leaf_of_chunks(max(1, sizes[(i * 5) % len(sizes)]), i, short=bool(i % 3))
                  for i in range(n)]
        agree(lambda t: run_g(g, 6, t), ref, make_table(leaves, seed=n))


@pytest.mark.parametrize("n", [15, 16, 17, 31, 32, 33, 47, 48, 49, 64])
def test_default_group_leaf_counts_with_single_tile_leaves_of_different_sizes(run_g, ref, n):
    """A block's 16 tiles belong to different leaves; every leaf is a single tile or less."""
    leaves = [(rand_bytes(3, i), rand_bytes((i * 997) % 8100, i), 0) for i in range(n)]
    agree(lambda t: run_g(3, 6, t), ref, make_table(leaves, seed=n))


# ---- headers ----

def test_every_header_length_and_residue(run_g, ref):
    hls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 63, 64, 65, 127, 128, 251, 252, 253, 254, 255]
    leaves = [(rand_bytes(h, h), rand_bytes(n, h + n), 0)
              for h in hls for n in (0, 1, 62, 63, 64, 1000, 2040, 2049)]
    leaves += [(rand_bytes(h, h), rand_bytes(n, h + n + 5), k)
               for h in (3, 7, 255) for n in (0, 4, 8, 1021, 5000) for k in (1, 2, 3)]
    agree(lambda t: run_g(3, 6, t), ref, make_table(leaves, seed=9))


def shared_table(hdr_buf, hdr_off, hdr_len, hdr_idx, payloads, kinds):
    t = make_table([(b"", p, k) for p, k in zip(payloads, kinds)], seed=4)
    t.hdr_buf, t.hdr_off = hdr_buf, np.array(hdr_off, np.uint64)
    t.hdr_len, t.hdr_idx = np.array(hdr_len, np.uint32), np.array(hdr_idx, np.uint32)
    return t


def test_shared_overlapping_and_unordered_headers(run_g, ref):
    buf = rand_bytes(300, 1)
    # header 1 is a window into header 0, header 3 is empty, header 4 is the buffer's tail
    off, ln = [0, 2, 10, 20, 45], [255, 3, 255, 0, 255]
    idx = [4, 1, 1, 0, 3, 2, 1, 4, 0, 3] * 3
    kinds = [0, 1, 2, 3, 0, 0, 3, 0, 0, 0] * 3
    payloads = [rand_bytes(100 + 997 * i % 3000, i) for i in range(30)]
    kinds = [k if ln[u] % 4 == 3 else 0 for k, u in zip(kinds, idx)]
    t = shared_table(buf, off, ln, idx, payloads, kinds)
    agree(lambda t: run_g(3, 6, t), ref, t)


def args(buf=b"abcdefgh", off=(0,), ln=(3,), idx=(0,), ptr=(0,), plen=(0,), kind=(0,)):
    return (buf, np.array(off, np.uint64), np.array(ln, np.uint32), np.array(idx, np.uint32),
            np.array(ptr, np.uint64), np.array(plen, np.uint64), np.array(kind, np.uint8))


@pytest.mark.parametrize("bad, message", [
    (dict(off=(6,), ln=(3,)), "out of bounds"),
    (dict(off=(9,), ln=(0,)), "out of bounds"),
    (dict(idx=(1,)), "out of bounds"),
    (dict(buf=b"a" * 300, ln=(256,)), "255"),
    (dict(ln=(4,), kind=(1,)), "multiple of 4"),
    (dict(ln=(0,), kind=(2,)), "multiple of 4"),
    (dict(ln=(5,), kind=(3,), plen=(4,)), "multiple of 4"),
    (dict(ln=(3, 4), off=(0, 0), idx=(0, 1), ptr=(0, 0), plen=(0, 0), kind=(1, 1)),
     "multiple of 4"),
])
def test_header_validation(bad, message):
    with pytest.raises(ValueError, match=message):
        gb.GpuBlake3().hash_leaves(*args(**bad), device="cpu")


def test_the_float_rule_looks_only_at_headers_that_leaves_use(build_host):
    k = gb.GpuBlake3(HostKernels(build_host()))
    a = args(buf=b"abcdefgh", off=(0, 3), ln=(3, 5), idx=(0, 0), ptr=(0, 0), plen=(0, 0),
             kind=(1, 2))
    d, bad = k.hash_leaves(*a, device="cpu")
    assert bad == 2 and d.shape == (2, 32)


def test_a_header_offset_that_wraps_around_uint64_is_refused():
    a = args(buf=b"abcdefgh", off=((1 << 64) - 2,), ln=(3,))
    with pytest.raises(ValueError, match="out of bounds"):
        gb.GpuBlake3().hash_leaves(*a, device="cpu")


@pytest.mark.parametrize("field, value", [("kind", 256), ("ln", 2 ** 32 + 3), ("ln", -1),
                                          ("idx", 2 ** 63), ("plen", -2)])
def test_a_value_too_wide_for_its_field_raises_instead_of_wrapping(field, value):
    pos = ["buf", "off", "ln", "idx", "ptr", "plen", "kind"].index(field)
    for dt in (np.int64, np.uint64):
        if (dt is np.uint64 and value < 0) or (dt is np.int64 and value >= 2 ** 63):
            continue
        a = list(args())
        a[pos] = np.array((value,), dtype=dt)
        with pytest.raises(ValueError):
            gb.GpuBlake3().hash_leaves(*a, device="cpu")


def test_the_header_range_check_is_exact_at_the_end_of_the_buffer():
    gb.check_table(8, *args(off=(5,), ln=(3,))[1:])  # 5 + 3 == 8 is in bounds
    for off, ln in ((6, 3), (0, 9), (8, 1)):
        with pytest.raises(ValueError, match="out of bounds"):
            gb.check_table(8, *args(off=(off,), ln=(ln,))[1:])


def test_negative_and_mismatched_arguments_are_refused():
    with pytest.raises(ValueError):
        gb.GpuBlake3().hash_leaves(b"", z(1), z(1, np.uint32), z(1, np.uint32), z(1),
                                   np.array([-1], np.int64), z(1, np.uint8), device="cpu")
    with pytest.raises(ValueError):
        gb.GpuBlake3().hash_leaves(b"", z(1), z(1, np.uint32), np.array([-1]), z(1), z(1),
                                   z(1, np.uint8), device="cpu")


# ---- payload alignment ----

def counts(lib):
    out = (ctypes.c_ulong * 2)()
    lib.probe_counts(out)
    return out[0], out[1]


def test_the_probe_sees_a_misaligned_wide_load(build_host):
    lib = build_host(3, 6, True)
    buf = (ctypes.c_ubyte * 16)()
    lib.probe_reset()
    lib.probe_load32.restype = ctypes.c_uint
    lib.probe_load32.argtypes = [ctypes.c_void_p]
    base = ctypes.addressof(buf)
    base += -base % 4
    lib.probe_load32(base)
    lib.probe_load32(base + 1)
    assert counts(lib) == (2, 1)


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
@pytest.mark.parametrize("pad", [0, 1, 2, 3])
@pytest.mark.parametrize("extra", [0, 1, 2, 3])
def test_payload_addresses_choose_the_path_and_match_blake3(run_g, ref, build_host, kind, pad,
                                                            extra):
    """The wide load happens exactly for the words of an aligned payload behind a prefix that
    is a multiple of 4, never on a misaligned address, for every kind and tail length."""
    hl, plen = 3, 3000 + extra
    lib = build_host(3, 6, True)
    t = make_table([(rand_bytes(hl, pad), rand_bytes(plen, pad + extra), kind)], pad=pad)
    t.arena.data_ptr()
    base = t.arena.data_ptr() + int(t.payload_off[0])
    lib.probe_reset()
    agree(lambda t: run_g(3, 6, t, probe=True), ref, t)
    wide, misaligned = counts(lib)
    assert misaligned == 0
    # the leaf is hashed once by the kernel build (native.py is another library)
    assert wide == (plen // 4 if base % 4 == 0 else 0)


@pytest.mark.parametrize("hl", [0, 1, 2, 4, 5, 254])
def test_a_prefix_that_is_not_a_multiple_of_4_takes_the_byte_path(run_g, ref, build_host, hl):
    lib = build_host(3, 6, True)
    t = make_table([(rand_bytes(hl, hl), rand_bytes(5000, hl), 0)], pad=0)
    lib.probe_reset()
    agree(lambda t: run_g(3, 6, t, probe=True), ref, t)
    assert counts(lib) == (0, 0)


def test_views_into_one_storage_at_odd_offsets(run_g, ref):
    """Payloads of bf16 tensors cut from one storage at 2-byte steps (bmm members), lengths
    not a multiple of 4."""
    base = torch.randn(4000, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    leaves = []
    for i, (lo, hi) in enumerate([(0, 777), (1, 1500), (3, 3001), (5, 2000), (2, 2001)]):
        x = base[lo:hi]
        leaves.append(tensor_leaf(i + 1, x))
        assert x.is_contiguous()
    t = make_table(leaves, pad=0)
    # put each payload at a 2-byte-odd address, as a view into a shared storage would be
    arena = bytearray(2)
    offs = []
    for _, p, _ in leaves:
        arena += bytes(-len(arena) % 4 + 2)
        offs.append(len(arena))
        arena += p
    arena += bytes(8)
    t.arena = torch.frombuffer(arena, dtype=torch.uint8).clone()
    t.payload_off = np.array(offs, np.uint64)
    got, _ = agree(run_g_host(run_g), ref, t)
    for i, (lo, hi) in enumerate([(0, 777), (1, 1500), (3, 3001), (5, 2000), (2, 2001)]):
        assert got[i].tobytes() == hash_tensor_leaf(i + 1, base[lo:hi])


def run_g_host(run_g):
    return lambda t: run_g(3, 6, t)


# ---- first_bad ----

BAD = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


@pytest.mark.parametrize("value", BAD)
def test_the_earliest_bad_leaf_among_all_kinds(run_g, ref, value):
    xs = []
    for i in range(12):
        dt = (torch.float32, torch.bfloat16, torch.float16)[i % 3]
        xs.append(torch.randn(2500, generator=torch.Generator().manual_seed(i)).to(dt))
    for i, pos in ((9, 2499), (7, 1300), (5, 0), (11, 8)):
        xs[i][pos] = BAD[value]
    t = make_table([tensor_leaf(i + 1, x) for i, x in enumerate(xs)], seed=1)
    got, first = agree(run_g_host(run_g), ref, t)
    assert first == 5
    assert [r.tobytes() for r in got[:5]] == [hash_tensor_leaf(i + 1, xs[i]) for i in range(5)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("value", BAD)
def test_a_bad_element_in_a_late_tile_with_clean_leaves_before(run_g, ref, dtype, value):
    clean = torch.randn(40000, generator=torch.Generator().manual_seed(1)).to(dtype)
    x = clean.clone()
    x[-1] = BAD[value]  # the last chunk of the last tile (5 tiles of 8 chunks at f32)
    y = clean.clone()
    y[len(y) // 2] = BAD[value]
    t = make_table([tensor_leaf(1, clean), tensor_leaf(2, clean), tensor_leaf(3, x),
                    tensor_leaf(2, y)], seed=3)
    got, first = agree(run_g_host(run_g), ref, t)
    assert first == 2


@pytest.mark.parametrize("kind", [1, 2, 3])
def test_a_zero_length_float_payload_is_finite(run_g, ref, build_host, kind):
    t = make_table([(b"\x01\x02\x03", b"", kind)] * 3 + [(b"abc", rand_bytes(9, 1), 0)])
    got, bad = agree(run_g_host(run_g), ref, t)
    assert bad == 4
    # a null address with no bytes is never read
    d, bad = gb.GpuBlake3(HostKernels(build_host())).hash_leaves(
        b"abc", z(1), np.array([3], np.uint32), z(1, np.uint32), z(1), z(1),
        np.array([kind], np.uint8), device="cpu")
    assert bad == 1 and d[0].numpy().tobytes() == blake3.blake3(b"\x00abc").digest()


def test_a_bad_leaf_in_a_later_block_than_the_clean_ones(run_g, ref):
    n = 16 * 3 + 5
    leaves = [(b"\x01\x02\x03", rand_bytes(4000, i), 1) for i in range(n)]
    leaves = [(h, bytes(len(p)), k) for h, p, k in leaves]
    t = make_table(leaves, seed=2)
    a = t.arena.numpy()
    for i in (n - 1, 40, 20):
        a[int(t.payload_off[i]) + 3999] = 0x7F
        a[int(t.payload_off[i]) + 3998] = 0x80
    _, first = agree(run_g_host(run_g), ref, t)
    assert first == 20


# ---- limits ----

def test_the_4_tib_limit_is_checked_by_arithmetic():
    limit = (1 << 42) - 512
    ok = args(plen=(limit,))
    gb.check_table(len(ok[0]), ok[1], ok[2], ok[3], ok[4], ok[5], ok[6])
    for over in (limit + 1, 1 << 42, 1 << 63):
        bad = args(plen=(over,))
        with pytest.raises(ValueError, match="4 TiB"):
            gb.GpuBlake3().hash_leaves(*bad, device="cpu")


def test_the_largest_accepted_leaf_plans_within_32_bit_chunk_indexes_and_grids():
    chunks = gb.chunk_counts(np.array([255], np.uint32), np.array([(1 << 42) - 512], np.uint64))
    assert int(chunks[0]) <= 1 << 32
    plan = gb.make_plan(chunks)
    assert plan.stages == 6
    for s in range(plan.stages):
        assert gb.grid_blocks(s, plan.total_tiles(s)) <= (1 << 31) - 1
    assert plan.total_tiles(0) == 1 << 29
    assert plan.nodes * 32 < (1 << 42) * 0.005


def test_an_empty_table_loads_no_kernels(monkeypatch):
    def refuse(_):
        raise AssertionError("loaded")
    monkeypatch.setattr(gb, "_kernels", refuse)
    d, bad = gb.GpuBlake3().hash_leaves(b"", z(0), z(0, np.uint32), z(0, np.uint32), z(0), z(0),
                                        z(0, np.uint8), device="cpu")
    assert d.shape == (0, 32) and bad == 0


# ---- the Python side ----

def expected_launches(chunk_counts_):
    out, levels = [], [[-(-c // 8) for c in chunk_counts_]]
    while max(levels[-1]) > 1:
        levels.append([-(-v // 64) for v in levels[-1]])
    for s, lv in enumerate(levels):
        tiles = sum(lv) if s == 0 else sum(v for p, v in zip(levels[s - 1], lv) if p > 1)
        out.append((s, tiles, -(-tiles // 16) if s == 0 else tiles))
    return out


@pytest.mark.parametrize("msg_lens, stages", [
    ([100], 1), ([9 * 1024], 2), ([600 * 1024], 3), ([33 << 20], 4),
    ([100, 9 * 1024, 600 * 1024, 5, 40 * 1024], 3), ([33 << 20, 7, 70 * 1024], 4)])
def test_kernel_arguments_and_grids_for_each_stage_count(msg_lens, stages):
    driver = FakeDriver()
    kern = gb.Kernels(gb.Driver(driver), 0, gb.Built(b"I", "cubin", "sm_86", (12, 4)))
    n = len(msg_lens)
    plen = np.array(msg_lens, np.uint64) - 1  # hdr_len 0
    k = gb.GpuBlake3(kern)
    prep = k.prepare(b"", z(1), z(1, np.uint32), z(n, np.uint32),
                     0x10000 + np.arange(n, dtype=np.uint64) * (1 << 40), plen,
                     z(n, np.uint8), device="cpu")
    driver.calls.clear()
    k.launch(prep, "cpu")
    exp = expected_launches([-(-m // 1024) for m in msg_lens])
    assert len(exp) == stages == len(driver.launched)
    hdr, leaf, prefix, out_off, nodes = prep.keep
    for (fn, gx, gy, gz, bx, by, bz, smem, stream, raw), (s, tiles, blocks) in zip(
            driver.launched, exp):
        p = gb.Params.from_buffer_copy(raw)
        assert (gx, gy, gz, smem, stream or 0) == (blocks, 1, 1, 0, 0)
        assert (bx, by, bz) == ((128 if s == 0 else 64), 1, 1)
        assert (p.n, p.stage, p.total_tiles) == (n, s, tiles)
        assert p.leaf == leaf.data_ptr() and p.tile_prefix == prefix[s].data_ptr()
        assert p.out_off == out_off[s].data_ptr()
        assert p.in_off == (out_off[s - 1].data_ptr() if s else 0)
        assert (p.nodes, p.digests, p.first_bad) == (
            nodes.data_ptr(), prep.digests.data_ptr(), prep.first_bad.data_ptr())
    assert len({f for f, *_ in driver.launched}) == (2 if stages > 1 else 1)
    assert driver.calls[0] == "cuCtxPushCurrent_v2" and driver.calls[-1] == "cuCtxPopCurrent_v2"
    assert prep.first_bad.item() == n and prep.first_bad.dtype == torch.int32


def test_the_leaf_table_holds_what_the_table_says():
    t = make_table([(b"abc", rand_bytes(7, 1), 0), (b"wxyzabc", rand_bytes(9, 2), 0)])
    prep = gb.GpuBlake3(SimpleNamespace(run=lambda *a: None)).prepare(
        *t.args(t.arena.data_ptr()), device="cpu")
    rec = prep.keep[1].numpy().view(gb.LEAF_DTYPE)
    assert rec["payload_len"].tolist() == [7, 9] and rec["hdr_len"].tolist() == [3, 7]
    assert rec["payload_ptr"].tolist() == (t.payload_off + t.arena.data_ptr()).tolist()
    assert int(rec["hdr_ptr"][1]) - int(rec["hdr_ptr"][0]) == 3


def test_launch_uses_torchs_current_stream_of_the_device(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "current_stream",
                        lambda d: seen.append(d) or SimpleNamespace(cuda_stream=4242))
    driver = FakeDriver()
    k = gb.GpuBlake3(gb.Kernels(gb.Driver(driver), 0, gb.Built(b"I", "cubin", "sm_86", (12, 4))))
    prep = k.prepare(b"", z(1), z(1, np.uint32), z(1, np.uint32), z(1) + 4096,
                     np.array([5000], np.uint64), z(1, np.uint8), device="cpu")
    k.launch(prep, "cuda:1")
    assert seen == ["cuda:1"] and [x[8] for x in driver.launched] == [4242]


@pytest.mark.parametrize("fail", ["cuLaunchKernel", "cuModuleLoadData", "cuModuleGetFunction:"
                                  "pots_reduce_stage"])
def test_the_context_is_popped_on_every_error_path(fail):
    driver = FakeDriver()
    d = gb.Driver(driver)
    if fail == "cuLaunchKernel":
        k = gb.Kernels(d, 0, gb.Built(b"I", "cubin", "sm_86", (12, 4)))
        driver.fail = fail
        with pytest.raises(RuntimeError, match="cuLaunchKernel failed"):
            k.run([(gb.Params(), 1)], stream=0)
    else:
        driver.fail = fail
        with pytest.raises(RuntimeError):
            gb.Kernels(d, 0, gb.Built(b"I", "cubin", "sm_86", (12, 4)))
    pushes = driver.calls.count("cuCtxPushCurrent_v2")
    assert pushes == driver.calls.count("cuCtxPopCurrent_v2") >= 1


def test_ptx_fallback_is_taken_once_cached_and_loaded(tmp_path):
    first = FakeNvrtc(reject_sm=True)
    built = gb.build_image(gb.Nvrtc(first), (7, 5), cache=tmp_path)
    assert (built.kind, built.arch) == ("ptx", "sm_75") and not built.cached
    assert [o[0] for o in first.compiles] == [b"--gpu-architecture=sm_75",
                                              b"--gpu-architecture=compute_75"]
    second = FakeNvrtc(reject_sm=True)
    again = gb.build_image(gb.Nvrtc(second), (7, 5), cache=tmp_path)
    assert again.kind == "ptx" and again.cached and not second.compiles
    driver = FakeDriver()
    gb.Kernels(gb.Driver(driver), 0, again)
    assert driver.image == b"PTX"


def test_other_nvrtc_errors_do_not_fall_back_and_a_ptx_failure_raises(tmp_path):
    lib = FakeNvrtc(compile_rc=6)
    with pytest.raises(RuntimeError, match="error: bad thing"):
        gb.build_image(gb.Nvrtc(lib), (8, 6), cache=tmp_path)
    assert len(lib.compiles) == 1

    class RejectBoth(FakeNvrtc):
        def nvrtcCompileProgram(self, prog, n, opts):
            self.compiles.append([opts[i] for i in range(n)])
            return 5 if len(self.compiles) == 1 else 6

        def log(self):
            return b"ptx failed"
    with pytest.raises(RuntimeError, match="ptx failed"):
        gb.build_image(gb.Nvrtc(RejectBoth()), (8, 6), cache=tmp_path)


class Versioned(FakeNvrtc):
    def __init__(self, version):
        super().__init__()
        self.version = version

    def nvrtcVersion(self, a, b):
        a._obj.value, b._obj.value = self.version
        return 0


def compiles_with(tmp_path, *, cc=(8, 6), version=(12, 4)):
    lib = Versioned(version)
    gb.build_image(gb.Nvrtc(lib), cc, cache=tmp_path)
    return len(lib.compiles)


def test_cache_key_changes_with_sources_arch_and_nvrtc_version(tmp_path, monkeypatch):
    assert compiles_with(tmp_path) == 1
    assert compiles_with(tmp_path) == 0
    assert compiles_with(tmp_path, cc=(8, 0)) == 1
    assert compiles_with(tmp_path, version=(12, 5)) == 1
    assert compiles_with(tmp_path, version=(11, 4)) == 1
    core, cu = gb.source_text()
    monkeypatch.setattr(gb, "source_text", lambda: (core + "\n// x", cu))
    assert compiles_with(tmp_path) == 1
    monkeypatch.setattr(gb, "source_text", lambda: (core, cu + "\n// x"))
    assert compiles_with(tmp_path) == 1
    monkeypatch.undo()
    assert compiles_with(tmp_path) == 0


def test_cache_key_changes_with_a_compile_option(tmp_path, monkeypatch):
    assert compiles_with(tmp_path) == 1
    old = gb.arch_options
    monkeypatch.setattr(gb, "arch_options", lambda cc, kind: old(cc, kind) + ["-DPOTS_G_LOG=4"])
    assert compiles_with(tmp_path) == 1


def test_importing_loads_no_library_and_describe_never_compiles(monkeypatch):
    code = ("import ctypes, torch\n"
            "names = []\n"
            "orig = ctypes.CDLL\n"
            "class Rec(orig):\n"
            "    def __init__(self, name, *a, **k):\n"
            "        names.append(str(name)); super().__init__(name, *a, **k)\n"
            "ctypes.CDLL = Rec\n"
            "import verification.commitment.gpu_blake3 as g\n"
            "assert not [n for n in names if 'nvrtc' in n or 'libcuda.so' in n], names\n"
            "assert g.describe()['source_sha256']\n"
            "assert g._built.cache_info().currsize == 0\n"
            "assert g._kernels.cache_info().currsize == 0\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    def refuse(*a):
        raise AssertionError("compiled")
    monkeypatch.setattr(gb, "_built", refuse)
    monkeypatch.setattr(gb, "_kernels", refuse)
    monkeypatch.setattr(gb, "_nvrtc", lambda: (_ for _ in ()).throw(RuntimeError("no nvrtc")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "Fake")
    info = gb.describe()
    assert info["arch"] == "sm_86" and "no nvrtc" in info["nvrtc_error"]
