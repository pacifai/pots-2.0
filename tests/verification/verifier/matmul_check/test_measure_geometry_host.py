"""FS37b's launch geometry on the Mac: the row-block starts the level kernel searches, the
``-D`` options of the build, ``set_geometry``'s checks and its cache reset, and the source file's
shared-memory declarations against ``shared_bytes``. Nothing here needs a GPU; the kernel's
bits under each geometry are ``test_measure_tiles_cuda.py``'s."""

import re

import numpy as np
import pytest

from tests.verification.verifier.matmul_check.test_measure_kernel_host import FakeNvrtc
from tests.verification.verifier.matmul_check.test_read_plan import prod, table, tview, view
from verification.verifier.matmul_check import measure_kernel as mk


@pytest.fixture(autouse=True)
def restore_geometry():
    prev = mk.set_geometry()
    yield
    mk.set_geometry(rows_per_block=prev[0], tile_cols=prev[1], k_chunk=prev[2])


def sample_table():
    x = view(0x1000, 70, 33)
    return table([prod(x, view(0x2000, 33, 20), 0x3000, k=3),
                  prod(x, view(0x4000, 33, 5), 0x5000, bias_addr=0x6000, k=40),
                  prod(view(0x7000, 3, 33), tview(0x8000, 33, 9), 0x9000, k=1)])


# ---- the row-block starts ----

@pytest.mark.parametrize("rpb", [8, 32, 40])
@pytest.mark.parametrize("share", [True, False])
def test_block_starts_cover_every_group_row_once(rpb, share):
    arr = sample_table()
    plan = mk.read_plan(arr) if share else mk.plain_plan(arr)
    starts = mk.block_starts(plan, rpb)
    assert starts.dtype == np.uint64 and len(starts) == len(plan.reads)
    for lv in range(plan.n_levels):
        lo, hi = int(plan.level_off[lv]), int(plan.level_off[lv + 1])
        s = starts[lo:hi].astype(np.int64)
        assert s[0] == 0 and (np.diff(s) >= 1).all()
        seen = {}
        for b in range(int(s[-1])):  # the kernel's search: the last group with start <= b
            g = int(np.searchsorted(s[:-1], b, side="right")) - 1
            seen.setdefault(g, []).append(b - int(s[g]))
        for g, r in enumerate(plan.reads[lo:hi - 1]):
            e = plan.entries[int(r["entry0"])]
            rb = rpb // min(int(r["n_entries"]), mk.E_MAX)
            want = 1 if e["phase"] == 2 else -(-int(r["count"]) // rb)
            assert seen[g] == list(range(want))
            if e["phase"] == 2:
                assert int(r["count"]) == mk.ROW_FIXED + int(arr[int(e["prod"])]["k"])


def test_a_shared_group_takes_blocks_of_rb_rows_with_rb_from_e_max():
    """RB_g = rows_per_block // min(n_entries, E_MAX): the 70-row view shared by two products
    takes 5 blocks of 16 rows."""
    arr = sample_table()
    plan = mk.read_plan(arr)
    starts = mk.block_starts(plan, 32)
    got = {}
    for lv in range(plan.n_levels):
        lo, hi = int(plan.level_off[lv]), int(plan.level_off[lv + 1])
        for g, r in enumerate(plan.reads[lo:hi - 1]):
            if plan.entries[int(r["entry0"])]["phase"] != 2:
                got[(int(r["count"]), int(r["n_entries"]))] = int(starts[lo + g + 1] - starts[lo + g])
    assert (70, 2) in got and got[(70, 2)] == -(-70 // 16) == 5   # RB = 32 // 2
    assert all(n == -(-c // (32 // min(e, mk.E_MAX))) for (c, e), n in got.items())


def test_more_entries_than_e_max_take_blocks_of_rb_from_e_max():
    n = 8
    x = view(0x1000, 100, 33)
    arr = table([prod(x, view(0x2000 + 0x100 * i, 33, 4), 0x9000 + 0x100 * i, k=2)
                 for i in range(n)])
    plan = mk.read_plan(arr)
    starts = mk.block_starts(plan, 32)
    lo = int(plan.level_off[1])  # phase 1: the eight products read one A
    r = plan.reads[lo]
    assert int(r["n_entries"]) == n and int(r["count"]) == 100
    assert int(starts[lo + 1] - starts[lo]) == -(-100 // (32 // mk.E_MAX))  # RB = 8: 13 blocks
    mk.set_geometry(rows_per_block=8)
    assert mk.block_starts(plan)[lo + 1] == -(-100 // (8 // mk.E_MAX))


def test_block_starts_use_the_current_geometry_by_default():
    plan = mk.read_plan(sample_table())
    assert np.array_equal(mk.block_starts(plan), mk.block_starts(plan, mk.ROWS_PER_BLOCK))
    mk.set_geometry(rows_per_block=8)
    assert np.array_equal(mk.block_starts(plan), mk.block_starts(plan, 8))
    assert not np.array_equal(mk.block_starts(plan, 8), mk.block_starts(plan, 32))


def test_level_launches_carry_the_starts_and_the_block_totals():
    arr = sample_table()
    plan = mk.plain_plan(arr)
    starts = mk.block_starts(plan)
    launches = mk.level_launches(0x10, 0x20, 0x30, 0x1000, 0x2000, 0x3000, plan, 3, starts)
    assert len(launches) == plan.n_levels == 3
    for lv, (args, blocks) in enumerate(launches):
        lo, hi = int(plan.level_off[lv]), int(plan.level_off[lv + 1])
        assert args.starts == 0x3000 + 8 * lo and args.reads == 0x1000 + 24 * lo
        assert args.n_groups == hi - lo - 1 and args.n_blocks == blocks == int(starts[hi - 1])
        assert args.k == 3 and args.entries == 0x2000


def test_the_grid_is_capped_and_the_args_size_is_the_cus():
    plan = mk.plain_plan(sample_table())
    big = mk.block_starts(plan)
    big[int(plan.level_off[1]) - 1] = mk.MAX_BLOCKS + 5
    (a0, b0), *_ = mk.level_launches(0, 0, 0, 0, 0, 0, plan, 3, big)
    assert b0 == mk.MAX_BLOCKS and a0.n_blocks == mk.MAX_BLOCKS + 5
    assert len(mk.LevelArgs._fields_) == 10 and mk.LevelArgs.starts.size == 8


# ---- set_geometry ----

def test_set_geometry_with_no_argument_changes_nothing():
    before = (mk.ROWS_PER_BLOCK, mk.TILE_COLS, mk.K_CHUNK)
    assert before == (32, 64, 16)
    assert mk.set_geometry() == before
    assert (mk.ROWS_PER_BLOCK, mk.TILE_COLS, mk.K_CHUNK) == before


def test_set_geometry_returns_the_previous_values_and_sets_the_new():
    assert mk.set_geometry(rows_per_block=64, tile_cols=128, k_chunk=4) == (32, 64, 16)
    assert (mk.ROWS_PER_BLOCK, mk.TILE_COLS, mk.K_CHUNK) == (64, 128, 4)
    assert mk.set_geometry(k_chunk=1) == (64, 128, 4)
    assert mk.set_geometry() == (64, 128, 1)


@pytest.mark.parametrize("bad", [
    {"tile_cols": 48}, {"tile_cols": 0}, {"tile_cols": -32}, {"rows_per_block": 12},
    {"rows_per_block": 0}, {"rows_per_block": -8}, {"k_chunk": 0}, {"k_chunk": -1},
    {"rows_per_block": 256, "tile_cols": 256},   # 256 * 257 floats: over 48 KB
    {"rows_per_block": 8, "k_chunk": 64, "tile_cols": 64},   # E_MAX vector tiles: over 48 KB
    {"tile_cols": 2048}, {"k_chunk": 1000}, {"tile_cols": 64.0}])
def test_set_geometry_refuses_bad_values_and_changes_nothing(bad):
    mk.set_geometry(rows_per_block=16, tile_cols=32, k_chunk=2)
    with pytest.raises(ValueError):
        mk.set_geometry(**bad)
    assert mk.set_geometry() == (16, 32, 2)


def test_a_refusal_leaves_the_caches_alone(monkeypatch):
    monkeypatch.setattr(mk, "build", lambda nvrtc, cc, **kw: object())
    monkeypatch.setattr(mk.gb, "_nvrtc", lambda: None)
    monkeypatch.setattr(mk.native, "cache_dir", lambda: None)
    mk._built.cache_clear()
    mk._built((8, 6))
    with pytest.raises(ValueError):
        mk.set_geometry(tile_cols=33)
    assert mk._built.cache_info().currsize == 1
    mk._built.cache_clear()


def test_a_change_clears_the_loaded_kernel_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(mk, "build", lambda nvrtc, cc, **kw: calls.append(cc) or object())
    monkeypatch.setattr(mk.gb, "_nvrtc", lambda: None)
    monkeypatch.setattr(mk.native, "cache_dir", lambda: None)
    mk._built.cache_clear()
    mk._built((8, 6))
    mk._built((8, 6))
    assert calls == [(8, 6)] and mk._built.cache_info().currsize == 1
    mk.set_geometry()  # no change: the cache stays
    assert mk._built.cache_info().currsize == 1
    mk.set_geometry(tile_cols=96)
    assert mk._built.cache_info().currsize == 0 and mk._kernels.cache_info().currsize == 0
    mk._built((8, 6))
    assert calls == [(8, 6), (8, 6)]
    mk._built.cache_clear()


# ---- the -D options and the cache key ----

def test_build_options_carry_fmad_off_and_the_five_macros():
    opts = mk.build_options()
    assert "--fmad=false" in opts and opts[:len(mk.OPTIONS)] == mk.OPTIONS
    macros = dict(o[2:].split("=") for o in opts if o.startswith("-D"))
    assert macros == {"POTS_MK_WARPS": f"{mk.WARPS}u", "ROWS_PER_BLOCK": "32u",
                      "TILE_COLS": "64u", "K_CHUNK": "16u", "E_MAX": f"{mk.E_MAX}u"}
    assert mk.E_MAX == 4
    mk.set_geometry(rows_per_block=16, tile_cols=32, k_chunk=3)
    macros = dict(o[2:].split("=") for o in mk.build_options() if o.startswith("-D"))
    assert macros["ROWS_PER_BLOCK"] == "16u" and macros["TILE_COLS"] == "32u"
    assert macros["K_CHUNK"] == "3u"


def test_each_geometry_is_its_own_cache_entry_and_the_options_reach_nvrtc(tmp_path):
    nv = FakeNvrtc()
    mk.build(nv, (8, 6), cache=tmp_path)
    assert "-DTILE_COLS=64u" in nv.calls[0][2] and "-DK_CHUNK=16u" in nv.calls[0][2]
    mk.build(nv, (8, 6), cache=tmp_path)
    assert len(nv.calls) == 1  # the same geometry: the cache
    mk.set_geometry(tile_cols=96)
    mk.build(nv, (8, 6), cache=tmp_path)
    assert len(nv.calls) == 2 and "-DTILE_COLS=96u" in nv.calls[1][2]
    mk.set_geometry(tile_cols=64)
    mk.build(nv, (8, 6), cache=tmp_path)
    assert len(nv.calls) == 2 and len(list(tmp_path.iterdir())) == 2


# ---- the source against the Python side ----

def test_the_cu_defaults_are_the_modules_and_its_shared_arrays_match_shared_bytes():
    cu = (mk.HERE / mk.CU_NAME).read_text()
    for name, val in (("POTS_MK_WARPS", mk.WARPS), ("ROWS_PER_BLOCK", mk.ROWS_PER_BLOCK),
                      ("TILE_COLS", mk.TILE_COLS), ("K_CHUNK", mk.K_CHUNK),
                      ("E_MAX", mk.E_MAX)):
        assert re.search(rf"#define {name} {val}u\b", cu), name
    shared = re.findall(r"__shared__ float (\w+)\[([^\]]+)\];", cu)
    assert [n for n, _ in shared] == ["s_x", "s_y", "s_b2", "s_b1"]
    env = {"ROWS_PER_BLOCK": mk.ROWS_PER_BLOCK, "TILE_COLS": mk.TILE_COLS, "K_CHUNK": mk.K_CHUNK,
           "E_MAX": mk.E_MAX, "POTS_XPAD": mk.TILE_COLS + 1, "POTS_YP": mk.TILE_COLS + 1}
    total = 4 * sum(eval(e, {}, env) for _, e in shared)
    assert total == mk.shared_bytes(mk.ROWS_PER_BLOCK, mk.TILE_COLS, mk.K_CHUNK)
    assert mk.shared_bytes(32, 64, 16) <= mk.SHARED_LIMIT


def test_the_cu_args_struct_has_the_python_fields_in_order():
    cu = (mk.HERE / mk.CU_NAME).read_text()
    body = re.search(r"typedef struct \{(.*?)\} pots_mlargs_t;", cu, re.S).group(1)
    names = re.findall(r"pots_u(?:64|32) (\w+);", body)
    assert names == [f[0] for f in mk.LevelArgs._fields_]


def test_shared_bytes_counts_one_vector_tile_and_b_vectors_per_entry():
    # x: 32 * 65, y: E_MAX * 16 * 65, b2 and b1: 2 * E_MAX * 64 floats
    assert mk.shared_bytes(32, 64, 16) == 4 * (32 * 65 + 4 * 16 * 65 + 2 * 4 * 64) == 27_008
    assert mk.shared_bytes(8, 32, 1) == 4 * (8 * 33 + 4 * 1 * 33 + 2 * 4 * 32)
    mk.set_geometry(rows_per_block=16, tile_cols=32, k_chunk=3)
    assert mk.shared_bytes(16, 32, 3) == 4 * (16 * 33 + 4 * 3 * 33 + 2 * 4 * 32)


def test_set_geometry_checks_the_shared_memory_with_e_max_and_the_rows_against_it(monkeypatch):
    monkeypatch.setattr(mk, "E_MAX", 16)
    with pytest.raises(ValueError):  # fewer rows than entries per batch
        mk.set_geometry(rows_per_block=8)
    monkeypatch.setattr(mk, "E_MAX", 4)
    mk.set_geometry(rows_per_block=8, tile_cols=64, k_chunk=40)  # 4*40*65 floats fit
    with pytest.raises(ValueError):
        mk.set_geometry(k_chunk=70)  # 4 * 70 * 65 * 4 bytes: over 48 KB
