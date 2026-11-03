"""FS41a: the challenge XOF core (challenge_xof.h) and kernel (challenge_xof.cu) on the host.

The host build loops over every unit; the simulator compiles the .cu itself and runs a grid
of chosen size, so the kernel's grid stride is covered. Entries are compared bit for bit with
``challenge_matrices``, and a NaN sentinel shows every entry is written exactly once.
"""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from verification.commitment import native
from verification.verifier.matmul_check.challenges import challenge_matrices

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "tests/verification/verifier/matmul_check"
SENTINEL = np.uint32(0x7FC0DEAD)

ITEM = np.dtype([("m", "<u4"), ("width", "<u4"), ("unit0", "<u8"), ("out_off", "<u8")])


class Params(ctypes.Structure):
    _fields_ = [("key", ctypes.c_uint32 * 8), ("k", ctypes.c_uint32), ("n", ctypes.c_uint32),
                ("units", ctypes.c_uint64), ("items", ctypes.c_uint64), ("out", ctypes.c_uint64)]


def build(tmp, src, std):
    out = tmp / (Path(src).stem + ".so")
    cc = "c++" if std.startswith("c++") else "cc"
    cmd = [cc, f"-std={std}", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
           "-shared", "-fPIC", f"-I{ROOT / 'verification/commitment'}",
           f"-I{ROOT / 'verification/verifier/matmul_check'}", str(HERE / src), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return ctypes.CDLL(str(out))


@pytest.fixture(scope="module")
def hostlib(tmp_path_factory):
    lib = build(tmp_path_factory.mktemp("xof"), "challenge_xof_host.c", "c99")
    lib.pots_xof_sizeof_params.restype = lib.pots_xof_sizeof_item.restype = ctypes.c_size_t
    lib.pots_xof_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    lib.pots_xof_offsets.restype = ctypes.c_size_t
    lib.pots_xof_find_host.argtypes = [ctypes.POINTER(Params), ctypes.c_uint64]
    lib.pots_xof_find_host.restype = ctypes.c_uint32
    lib.pots_host_xof.argtypes = [ctypes.POINTER(Params)]
    return lib


@pytest.fixture(scope="module")
def simlib(tmp_path_factory):
    lib = build(tmp_path_factory.mktemp("xofsim"), "challenge_xof_sim.cpp", "c++17")
    lib.pots_sim_xof.argtypes = [ctypes.POINTER(Params), ctypes.c_uint, ctypes.c_uint]
    return lib


class Request:
    """Items ``[(m, width)]`` laid out with gaps, the params, and a sentinel-filled output."""

    def __init__(self, h, items, k, gap=7):
        self.h, self.items, self.k = h, items, k
        table = np.zeros(len(items), ITEM)
        unit, off = 0, 3
        for i, (m, width) in enumerate(items):
            table[i] = (m, width, unit, off)
            unit += k * -(-width // 16)
            off += width * k + gap
        self.table, self.units, self.size = table, unit, off + 5
        self.out = np.full(self.size, SENTINEL, np.uint32)
        self.params = Params()
        self.params.key[:] = np.frombuffer(h, "<u4").tolist()
        self.params.k, self.params.n, self.params.units = k, len(items), unit
        self.params.items = table.ctypes.data
        self.params.out = self.out.ctypes.data

    def run(self, fn, *args):
        self.out[:] = SENTINEL
        fn(ctypes.byref(self.params), *args)
        return self.out.copy()

    def check(self, got, python=False):
        covered = np.zeros(self.size, bool)
        for (m, width), row in zip(self.items, self.table):
            off = int(row["out_off"])
            n = width * self.k
            ref = challenge_matrices(self.h, [m], self.k, width)[0].numpy()
            assert np.array_equal(got[off:off + n], ref.view(np.uint32).reshape(-1))
            covered[off:off + n] = True
        assert np.all(got[~covered] == SENTINEL)
        assert not np.any(got[covered] == SENTINEL)


KEYS = [bytes(np.random.default_rng(1).integers(0, 256, 32, dtype=np.uint8)), bytes(32),
        b"\xff" * 32]
WIDTHS = [1, 15, 16, 17, 31, 33, 128, 896, 4864, 151936]


def test_struct_layout(hostlib):
    assert hostlib.pots_xof_sizeof_params() == ctypes.sizeof(Params) == 64
    assert hostlib.pots_xof_sizeof_item() == ITEM.itemsize == 24
    offs = (ctypes.c_size_t * 16)()
    n = hostlib.pots_xof_offsets(offs)
    want = [getattr(Params, f).offset for f, _ in Params._fields_]
    want += [ITEM.fields[f][1] for f in ITEM.names]
    assert list(offs[:n]) == want == [0, 32, 36, 40, 48, 56, 0, 4, 8, 16]


@pytest.mark.parametrize("k", [1, 2, 9, 16, 21])
@pytest.mark.parametrize("key", range(3))
def test_entries_bit_exact(hostlib, k, key):
    ms = [1, 2 ** 32 - 1, 2 ** 24, int(np.random.default_rng(k).integers(1, 2 ** 32))]
    widths = WIDTHS if k <= 2 else WIDTHS[:9]
    items = [(ms[i % 4], w) for i, w in enumerate(widths)]
    req = Request(KEYS[key], items, k)
    req.check(req.run(hostlib.pots_host_xof))


def test_k255_small_widths(hostlib):
    req = Request(KEYS[0], [(5, 1), (2 ** 32 - 1, 17), (1, 33), (2 ** 24, 128)], 255)
    req.check(req.run(hostlib.pots_host_xof))


@pytest.mark.parametrize("k,items", [(3, [(1, 17), (2 ** 32 - 1, 128)]), (21, [(9, 33), (2, 15)])])
def test_pure_python_blake3(hostlib, k, items):
    req = Request(KEYS[0], items, k)
    got = req.run(hostlib.pots_host_xof)
    with native.disabled():
        req.check(got)


def test_find_first_and_last_unit(hostlib):
    items = [(m + 1, w) for m, w in enumerate([1, 16, 17, 40, 3, 64, 1, 5])]
    req = Request(KEYS[0], items, 3)
    ends = list(req.table["unit0"][1:]) + [req.units]
    for i, (lo, hi) in enumerate(zip(req.table["unit0"], ends)):
        for u in (int(lo), int(hi) - 1):
            assert hostlib.pots_xof_find_host(ctypes.byref(req.params), u) == i


@pytest.mark.parametrize("grid", [2, 5, 1000])
def test_sim_matches_host(hostlib, simlib, grid):
    items = [(7, 1), (2 ** 32 - 1, 33), (2 ** 24, 128), (3, 17), (11, 896)]
    req = Request(KEYS[0], items, 21)
    want = req.run(hostlib.pots_host_xof)
    got = req.run(simlib.pots_sim_xof, grid, 64)
    assert grid * 64 < req.units or grid == 1000
    assert np.array_equal(got, want)
    req.check(got)


# ---- FS41c: the GPU source's plan through the host build, against the host source's draw ----

def layer_step_request():
    """A layer step's shape: a few linear products, then attention members of width 128 and 64
    whose labels run consecutively and then in strides."""
    ms, widths = [101, 102, 103], [896, 128, 4864]
    ms += list(range(1000, 1000 + 150))
    widths += [128] * 150
    ms += list(range(5000, 5000 + 7 * 150, 7))
    widths += [64] * 150
    ms += [2 ** 32 - 1, 2 ** 24]
    widths += [128, 64]
    return ms, widths


PLAN_REQUESTS = {
    "mixed_repeated_m": ([7, 3, 7, 1, 2 ** 32 - 1, 3], [1, 15, 16, 17, 33, 896], 21),
    "layer_step": (*layer_step_request(), 21),
    "vocab_width": ([11, 12], [151936, 128], 21),
    "k1": ([5, 6, 5], [1, 17, 128], 1),
    "k255": ([5, 2 ** 32 - 1, 1], [1, 17, 33], 255),
}


@pytest.mark.parametrize("name", list(PLAN_REQUESTS))
def test_gpu_plan_through_the_host_build_equals_the_host_sources_draw(hostlib, name):
    """``GpuChallengeSource.plan``'s table and params, run by the host build, write the bytes
    of ``HostChallengeSource``'s inline draw at the same offsets."""
    from verification.verifier.matmul_check import challenge_source as cs

    ms, widths, k = PLAN_REQUESTS[name]
    h = KEYS[0]
    plan, params = cs.GpuChallengeSource().plan(h, ms, k, widths)
    want, offsets, _ = cs.HostChallengeSource().draw(h, ms, k, widths, stream=None)
    assert np.array_equal(plan.offsets, offsets)
    assert plan.total == want.numel()
    out = np.full(plan.total, SENTINEL, np.uint32)
    params.items = plan.items.ctypes.data
    params.out = out.ctypes.data
    assert ctypes.sizeof(params) == ctypes.sizeof(Params)
    # XParams and the test's Params mirror are two classes of the one struct
    hostlib.pots_host_xof(ctypes.cast(ctypes.byref(params), ctypes.POINTER(Params)))
    assert np.array_equal(out, want.numpy().view(np.uint32))
    assert not np.any(out == SENTINEL)
