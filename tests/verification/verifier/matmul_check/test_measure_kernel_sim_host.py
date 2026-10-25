"""FS37b's level kernel (``measure_kernel.cu``) run on the host by ``measure_kernel_sim.cpp``.

The simulator compiles the CUDA source itself, with the CUDA keywords stubbed, one OS thread per
CUDA thread and barriers for ``__syncthreads`` and ``__syncwarp``, and runs it one block at a
time. So the CUDA file's own control flow (tile loads for every dtype and layout, the vector
tiles, the row mapping of a block, the chunks, the bias item, phase-2 groups, the per-block
table (``block_table``, ``pots_mblk_t {group, rbk}``, read with no search) and the grid stride)
is held to the host build bit for bit on the Mac, through the Python side's own plan,
``block_starts`` and ``level_launches``. What it cannot
check: NVRTC's compile, the real shuffle and barrier hardware, shared-memory bank conflicts and
speed. It takes about 12 minutes (a thread per CUDA thread), so it is marked slow. A divergent barrier or shuffle hangs the simulator; a watchdog turns that into a failure."""

import faulthandler
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.verification.verifier.matmul_check.shared_tables import (
    DTYPES,
    EDITS,
    fwd_and_grad,
    member_stack,
    prod_of,
    rnd,
    mixed_table,
    shared_special,
    three_forward,
    two_sided,
)
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    H,
    host_measure,
    hostlib,
    make,
)
from tests.verification.verifier.matmul_check.test_measure_core_spec import fs36_cases, same_bits
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check import measure_table as mt
from verification.verifier.matmul_check.measure_table import MeasureInput, pack

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CXX = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")

pytestmark = [pytest.mark.slow, pytest.mark.skipif(CXX is None, reason="needs a C++ compiler")]

GEOMETRIES = [(16, 64, 16), (16, 256, 16), (8, 32, 4), (16, 96, 3), (64, 128, 1), (16, 256, 9)]


@pytest.fixture(scope="module")
def sim_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("sim")


def build_sim(out_dir, geometry, asan=False):
    rpb, tile, kc = geometry
    out = out_dir / f"sim_{rpb}_{tile}_{kc}{'_asan' if asan else ''}.so"
    cmd = [CXX, "-std=c++17", "-O1", "-ffp-contract=off", "-shared", "-fPIC",
           f"-I{ROOT / 'verification/verifier/matmul_check'}",
           f"-DPOTS_MK_WARPS={mk.WARPS}u", f"-DROWS_PER_BLOCK={rpb}u", f"-DTILE_COLS={tile}u",
           f"-DK_CHUNK={kc}u",
           str(HERE / "measure_kernel_sim.cpp"), "-o", str(out)]
    if asan:
        cmd[1:1] = ["-fsanitize=address", "-fno-omit-frame-pointer", "-g"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    import ctypes
    lib = ctypes.CDLL(str(out))
    lib.sim_launch.argtypes = [ctypes.POINTER(mk.LevelArgs), ctypes.c_uint]
    lib.sim_launch.restype = None
    return lib


def simulate(lib, mis, *, share=True, geometry=GEOMETRIES[0], grid=3):
    """The rows of ``mis`` by the simulated kernel: the plan, its block table and its level
    launches as ``enqueue`` makes them, with host addresses. The scratch is an exact-size
    allocation (the tables and the plan are numpy arrays of exact size), so an out-of-bounds
    read lands outside every buffer (under ASAN, a report)."""
    table = pack(mis)
    prods = np.frombuffer(table.array, dtype=mk.MPROD_DTYPE).copy()
    scratch = np.zeros(max(mk.scratch_bytes(prods), 8), dtype=np.uint8)
    assert scratch.ctypes.data % 8 == 0
    works = mk.works_array(prods, scratch.ctypes.data)
    plan = mk.read_plan(prods) if share else mk.plain_plan(prods)
    starts = mk.block_starts(plan, geometry[0])
    blocks = mk.block_table(plan, geometry[0])
    rows = mt.empty_rows(table)
    k = int(prods["k"][0])
    launches = mk.level_launches(prods.ctypes.data, works.ctypes.data, rows.ctypes.data,
                                 plan.reads.ctypes.data, plan.entries.ctypes.data,
                                 blocks.ctypes.data, plan, k, starts)
    faulthandler.dump_traceback_later(300, exit=True)
    try:
        for args, blocks in launches:
            lib.sim_launch(args, min(blocks, grid))  # fewer blocks: the grid stride runs
    finally:
        faulthandler.cancel_dump_traceback_later()
    return rows


@pytest.fixture(scope="module")
def sim(sim_dir):
    @lru_cache(maxsize=None)
    def lib_of(geometry):
        return build_sim(sim_dir, geometry)

    def run(mis, *, share=True, geometry=GEOMETRIES[0], grid=3):
        return simulate(lib_of(geometry), mis, share=share, geometry=geometry, grid=grid)

    return run


def agree(got, want, label=""):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape
    if same_bits(got, want):
        return
    for i in range(len(got)):
        if not same_bits(got[i], want[i]):
            raise AssertionError(f"product {i} {label}: {got[i]} != {want[i]}")
    raise AssertionError("rows differ")


def layout(t, kind):
    """``t`` stored row-major (``row``) or as a transposed view of a contiguous transpose."""
    return t if kind == "row" else t.t().contiguous().t()


def shaped(n, q, w, k, kind, *, dtype=torch.bfloat16, biased=True, guard=True, seed=0):
    mi = make(n, q, w, k, dtype=dtype, biased=biased, guard=guard, seed=seed)
    return MeasureInput(layout(mi.a, kind), layout(mi.b, kind), layout(mi.p, kind), mi.r,
                        mi.bias, guard)


# ---- shapes: tiles, ragged edges, many rows, chunks ----

SHAPES = [(1, 1, 1), (3, 5, 7), (9, 31, 33), (33, 65, 64), (40, 130, 70), (70, 33, 9),
          (5, 200, 1), (2, 3, 200)]


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kind", ["row", "t"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_shapes_layouts_and_geometries(sim, host_measure, geometry, kind, dtype):
    for guard in (True, False):
        for k in (1, 3, 7):
            mis = [shaped(n, q, w, k, kind, dtype=dtype, biased=bool((n + q) % 2), guard=guard,
                          seed=n * 31 + q)
                   for n, q, w in SHAPES]
            want = host_measure(pack(mis))
            for share in (True, False):
                agree(sim(mis, share=share, geometry=geometry), want, f"{geometry} {kind}")


@pytest.mark.parametrize("geometry", [GEOMETRIES[0], GEOMETRIES[2], GEOMETRIES[3]])
def test_more_challenge_columns_than_a_chunk(sim, host_measure, geometry):
    """k above the chunk: every chunk is its own sweep, the extras ride the first."""
    kc = geometry[2]
    for k in (kc + 1, 2 * kc, 2 * kc + 1):
        mis = [shaped(n, q, w, k, kind, biased=biased, seed=k)
               for (n, q, w), kind, biased in (((9, 70, 40), "row", True), ((35, 20, 66), "t", False),
                                               ((3, 100, 5), "t", True))]
        agree(sim(mis, geometry=geometry), host_measure(pack(mis)), f"k={k}")


def test_a_view_with_strides_of_neither_kind(sim, host_measure):
    """The general load path: a column slice of a wider matrix with a row pitch, and an
    interleaved one (column stride 2)."""
    g = torch.Generator().manual_seed(4)
    for cs in (1, 2):
        base_a = torch.randn(11, 40 * cs + 3, generator=g).to(torch.bfloat16)
        a = base_a[:, :40 * cs:cs]
        base_b = torch.randn(40 * cs, 13 + 5, generator=g).to(torch.bfloat16)
        b = base_b[::cs, :13]
        p = (a.float() @ b.float()).to(torch.bfloat16)
        mi = MeasureInput(a, b, p, make(11, 40, 13, 3).r, None, True)
        agree(sim([mi]), host_measure(pack([mi])), f"cs={cs}")


# ---- FS36's inputs, the shared tables, special values ----

def test_a_sample_of_fs36_cases(sim, host_measure):
    cases = [mi for _, mi in fs36_cases() if mi.r.shape[1] == 2]
    sample = cases[::7]
    assert len(sample) >= 10
    want = host_measure(pack(sample))
    for share in (True, False):
        agree(sim(sample, share=share), want)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_the_shared_tables_equal_the_unshared_ones_and_the_host(sim, host_measure, dtype, guard):
    tables = {
        "x by a": fwd_and_grad(dtype, guard),
        "biased beside": fwd_and_grad(dtype, guard, biased_beside=True),
        "three forward": three_forward(dtype, guard),
        "member stack": member_stack(dtype, guard),
        "two sided": two_sided(dtype, guard),
        "mixed": mixed_table(dtype, guard=guard),
    }
    for name, mis in tables.items():
        want = host_measure(pack(mis))
        for geometry in GEOMETRIES[:2]:
            for share in (True, False):
                agree(sim(mis, share=share, geometry=geometry), want, name)


@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_special_values_in_a_shared_view(sim, host_measure, edit, dtype, guard):
    mis, x = shared_special(dtype, guard)
    EDITS[edit](x)
    want = host_measure(pack(mis))
    for share in (True, False):
        agree(sim(mis, share=share, geometry=GEOMETRIES[2]), want, edit)


def test_masked_columns_are_skipped_not_zero_filled(sim, host_measure):
    """A -0.0 partial and an Inf next to a masked column: zero-filling would turn -0.0 into +0.0
    (fmaf(0, y, s)) and 0 * Inf into NaN."""
    mi = shaped(3, 40, 33, 3, "row", dtype=torch.float32, biased=False)
    a = mi.a.clone()
    a[0, :] = 0.0
    a[0, :] = -0.0
    a[1, 39] = float("inf")
    b = mi.b.clone()
    b[39, :] = 0.0
    mi2 = MeasureInput(a, b, mi.p, mi.r, None, True)
    for geometry in GEOMETRIES:
        agree(sim([mi2], geometry=geometry), host_measure(pack([mi2])), str(geometry))


# ---- NaN-filled shared memory and exact-size buffers ----

def asan_runtime():
    """The ASAN runtime library to preload into Python, or None where the compiler has none."""
    probe = subprocess.run([CXX, "-fsanitize=address", "-x", "c++", "-", "-o", "/dev/null"],
                           input="int main(){return 0;}", capture_output=True, text=True)
    if probe.returncode != 0:
        return None
    for name in ("libclang_rt.asan_osx_dynamic.dylib", "libclang_rt.asan-aarch64.so",
                 "libclang_rt.asan-x86_64.so", "libasan.so"):
        out = subprocess.run([CXX, f"-print-file-name={name}"], capture_output=True,
                             text=True).stdout.strip()
        if out and Path(out).is_absolute() and Path(out).exists():
            return out
    return None


ASAN_CHILD = """
import sys, torch, numpy as np
from pathlib import Path
from tests.verification.verifier.matmul_check import test_measure_kernel_sim_host as T
mis = torch.load(sys.argv[1], weights_only=False)
lib = T.build_sim(Path(sys.argv[2]), T.GEOMETRIES[1], asan=True)
rows = T.simulate(lib, mis, geometry=T.GEOMETRIES[1])
np.save(sys.argv[3], rows)
"""


def test_the_kernel_reads_nothing_outside_its_buffers(host_measure, sim_dir, tmp_path):
    """The simulator under ASAN, in a child Python with the sanitizer runtime preloaded: the
    tables, scratch and rows are exact-size heap blocks, so any read or write past one aborts."""
    runtime = asan_runtime()
    if runtime is None:
        pytest.skip("this compiler has no ASAN runtime")
    mis = [shaped(n, q, w, 3, kind, biased=bool((n + q) % 2), seed=n)
           for (n, q, w), kind in zip(SHAPES, ["row", "t"] * 4)]
    mis += fwd_and_grad(torch.bfloat16, True, biased_beside=True)
    src = tmp_path / "mis.pt"
    torch.save(mis, src)
    import os
    import sys
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0")
    env["DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"] = runtime
    out = tmp_path / "rows.npy"
    proc = subprocess.run([sys.executable, "-c", ASAN_CHILD, str(src), str(tmp_path), str(out)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    if "Interceptors are not working" in proc.stderr:
        pytest.skip("the OS ignores the preloaded ASAN runtime (a hardened Python)")
    assert proc.returncode == 0, proc.stderr[-3000:]
    agree(np.load(out), host_measure(pack(mis)), "asan")


# ---- groups of several entries: mixed kinds, entries of different widths ----

def wide_group(dtype, guard, n_a, n_b, *, n=19, q=37, k=3, kind="row"):
    """One stored ``[n, q]`` view ``v`` read as ``A`` by ``n_a`` products (``B`` widths 5, 9,
    ...) and as ``B`` by ``n_b`` products (``A`` of ``m`` rows): a read group of ``n_a + n_b``
    entries over one view, every entry with its own P width."""
    g = torch.Generator().manual_seed(11 + n_a * 7 + n_b)
    v = layout(rnd(g, n, q, dtype=dtype), kind)
    out = []
    for i in range(n_a):
        out.append(prod_of(v, rnd(g, q, 5 + 4 * i, dtype=dtype), k=k, m=i + 1, guard=guard,
                           seed=i))
    for i in range(n_b):
        out.append(prod_of(rnd(g, 3 + 2 * i, n, dtype=dtype), v, k=k, m=n_a + i + 1,
                           guard=guard, seed=20 + i))
    return out


@pytest.mark.parametrize("kind", ["row", "t"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_a_group_of_several_entries_runs_them_in_turn(sim, host_measure, dtype, kind):
    """Seven entries over one view, then five, two and one, one entry per batch."""
    for n_a, n_b in ((4, 3), (5, 0), (0, 5), (2, 2), (1, 1)):
        for guard in (True, False):
            mis = wide_group(dtype, guard, n_a, n_b, kind=kind)
            want = host_measure(pack(mis))
            for geometry in (GEOMETRIES[0], GEOMETRIES[1], GEOMETRIES[2]):
                agree(sim(mis, geometry=geometry), want, f"{n_a}+{n_b} {geometry}")
