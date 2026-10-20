"""FS32e against the code it replaced: the row join, ``_finish``, ``_finish_columns``, the
per-device scratches and the record pass.

The old join is written inline (``torch.cat(rows).double().cpu().numpy()`` cut by run lengths)
and compared bit for bit through the ``_rows_device`` seam, which lets CPU rows stand for
several devices.
"""

import dataclasses

import numpy as np
import pytest
import torch

from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.matmul_check.freivalds import MeasureScratch, ROW

from tests.verification.verifier import test_check_devices as cd
from tests.verification.verifier import test_judge_window as jw

NCOL = len(ROW)


def _old_read(rows):
    joined = rows[0] if len(rows) == 1 else torch.cat(list(rows))
    host = joined.double().cpu().numpy()
    out, base = [], 0
    for r in rows:
        out.append(host[base:base + len(r)])
        base += len(r)
    return out


def _assign(monkeypatch, rows, keys):
    by_id = {id(r): k for r, k in zip(rows, keys)}
    monkeypatch.setattr(checks, "_rows_device", lambda r: by_id[id(r)])


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint8).tobytes()


def _runs(rng, sizes, dtype):
    return [torch.from_numpy(rng.standard_normal((n, NCOL)) * 10.0 ** rng.integers(-8, 8)
                             ).to(dtype) for n in sizes]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name,keys", [
    ("interleaved", "ABAB"), ("one-device", "AAAA"), ("each-alone", "ABCD"),
    ("pairs", "AABB")])
def test_read_rows_equals_the_old_join(name, keys, dtype, monkeypatch):
    rng = np.random.default_rng(0)
    runs = _runs(rng, (3, 5, 1, 4), dtype)
    _assign(monkeypatch, runs, list(keys))
    got, want = checks._read_rows(runs), _old_read(runs)
    for g, w, r in zip(got, want, runs):
        assert g.dtype == np.float64 and g.shape == (len(r), NCOL)
        assert _bits(g) == _bits(w)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_read_rows_single_run_and_zero_row_run(dtype, monkeypatch):
    rng = np.random.default_rng(1)
    (one,) = _runs(rng, (6,), dtype)
    monkeypatch.setattr(checks, "_rows_device", lambda r: "A")
    (g,) = checks._read_rows([one])
    assert _bits(g) == _bits(_old_read([one])[0])
    empty = torch.empty(0, NCOL, dtype=dtype)
    runs = [one, empty, one * 2, empty]
    for keys in ("AAAA", "ABAB", "BABA", "AAAB"):
        _assign(monkeypatch, runs, list(keys))
        got, want = checks._read_rows(runs), _old_read(runs)
        assert [len(x) for x in got] == [6, 0, 6, 0]
        assert all(_bits(g) == _bits(w) for g, w in zip(got, want))
    _assign(monkeypatch, [empty], ["A"])
    assert checks._read_rows([empty])[0].shape == (0, NCOL)


def test_read_rows_mixed_dtypes_on_one_device(monkeypatch):
    rng = np.random.default_rng(2)
    runs = _runs(rng, (3, 2), torch.float32) + _runs(rng, (4,), torch.float64)
    _assign(monkeypatch, runs, "AAA")
    assert all(_bits(g) == _bits(w) for g, w in zip(checks._read_rows(runs), _old_read(runs)))


@pytest.mark.parametrize("seed", range(40))
def test_read_rows_random_cases(seed, monkeypatch):
    rng = np.random.default_rng(100 + seed)
    k = int(rng.integers(1, 9))
    dtype = [torch.float32, torch.float64][seed % 2]
    runs = _runs(rng, rng.integers(0, 6, size=k), dtype)
    runs[0] = torch.cat([runs[0], runs[0][:0] + float("nan"), torch.full((1, NCOL), -0.0,
                                                                          dtype=dtype)])
    _assign(monkeypatch, runs, list(rng.integers(0, 3, size=k)))
    got, want = checks._read_rows(runs), _old_read(runs)
    assert all(_bits(g) == _bits(w) for g, w in zip(got, want))  # NaN and -0.0 by bits


# ---- _finish and _finish_columns against the old code -------------------------------------


def _old_finish(batch, ctx):
    present = [r.rows for r in batch if r.rows is not None]
    if not present:
        return []
    host = (present[0] if len(present) == 1 else torch.cat(present)).tolist()
    out, base = [], 0
    for r in batch:
        n = len(r.order)
        out += zip(r.specs, checks._finished(r, host[base:base + n], eps_in=ctx.eps_in,
                                             eps_acc=ctx.eps_acc, guard=ctx.kappa_guard))
        base += n
    return out


def _old_finish_columns(batch, ctx):
    present = [r for r in batch if r.rows is not None]
    if not present:
        return [], None
    rows = (present[0].rows if len(present) == 1 else torch.cat([r.rows for r in present]))
    host = rows.double().cpu().numpy()
    perm, specs, base = [], [], 0
    for r in present:
        perm.append(base + np.argsort(r.order))
        specs += r.specs
        base += len(r.order)
    canonical = np.concatenate(perm)
    cols = checks.finish_rows(host[canonical], q=np.array([s.q for s in specs]),
                              n=np.array([s.a_shape[0] for s in specs]), eps_in=ctx.eps_in,
                              eps_acc=ctx.eps_acc, guard=ctx.kappa_guard,
                              biased=np.array([s.bias is not None for s in specs]))
    return specs, cols


def _random_batch(rng, dtype, with_none):
    """Runs of random sizes, specs numbered in order, rows from the judge-window GOOD row with
    noise, stored in a shuffled order."""
    sizes = list(rng.integers(1, 5, size=int(rng.integers(1, 6))))
    batch, m = [], 1
    for n in sizes:
        specs = [jw._spec(m + i, bias=None) for i in range(n)]
        m += n
        order = [int(x) for x in rng.permutation(n)]
        rows = torch.tensor(np.array([jw.GOOD] * n) * (1 + rng.random((n, 1)) * 0.5),
                            dtype=dtype)
        batch.append(checks._Rows(specs, order, rows))
        if with_none and rng.random() < 0.4:
            batch.append(checks._Rows([], [], None))
    return batch


def _same_finish(batch, ctx):
    assert repr(checks._finish(batch, ctx)) == repr(_old_finish(batch, ctx))
    s1, c1 = checks._finish_columns(batch, ctx)
    s2, c2 = _old_finish_columns(batch, ctx)
    assert s1 == s2 and (c1 is None) == (c2 is None)
    if c1 is not None:
        for f in (x.name for x in dataclasses.fields(c1)):
            a, b = np.asarray(getattr(c1, f)), np.asarray(getattr(c2, f))
            assert a.dtype == b.dtype and _bits(a) == _bits(b), f


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finish_equals_old_over_random_batches(seed, dtype, monkeypatch):
    rng = np.random.default_rng(seed)
    batch = _random_batch(rng, dtype, with_none=seed % 2 == 1)
    live = [r for r in batch if r.rows is not None]
    _assign(monkeypatch, [r.rows for r in live], list(rng.integers(0, 3, size=len(live))))
    for guard in (True, False):
        _same_finish(batch, jw._ctx(kappa_guard=guard))


def test_finish_with_only_rowless_runs():
    batch = [checks._Rows([], [], None)]
    assert checks._finish(batch, jw._ctx()) == [] == _old_finish(batch, jw._ctx())
    assert checks._finish_columns(batch, jw._ctx()) == ([], None)
    assert checks._finish([], jw._ctx()) == [] and checks._finish_columns([], jw._ctx()) == ([], None)


@pytest.mark.parametrize("devices", ["AB", "BA", "AA"])
@pytest.mark.parametrize("judge", [True, False])
def test_first_failing_product_in_a_later_devices_run(devices, judge, monkeypatch):
    """The failing product (3, in run 2) sits on the second run's device; the window's
    rejection, stats and run_rest equal the single-device ones."""
    edits = {7: {"nu": 1000.0}}
    n = 10
    one_ctx = jw._ctx(judge=judge)
    one = checks._judge_window(jw._C(), jw._window(n, edits), one_ctx, jw.BANDS)
    batch = jw._window(n, edits)
    _assign(monkeypatch, [r.rows for r in batch], list(devices))
    ctx = jw._ctx(judge=judge)
    got = checks._judge_window(jw._C(), batch, ctx, jw.BANDS)
    assert got == one and (got is not None) == judge
    assert repr(ctx.stats.products) == repr(one_ctx.stats.products)
    assert repr(ctx.state.run_rest) == repr(one_ctx.state.run_rest)


def test_window_with_a_rowless_run_in_the_middle(monkeypatch):
    batch = jw._window(8, {5: {"p_norm": float("nan")}})
    padded = [batch[0], checks._Rows([], [], None), batch[1]]
    for b in (batch, padded):
        _assign(monkeypatch, [r.rows for r in b if r.rows is not None], "AB")
    a, b = jw._ctx(), jw._ctx()
    assert checks._judge_window(jw._C(), batch, a, jw.BANDS) == \
        checks._judge_window(jw._C(), padded, b, jw.BANDS)
    assert repr(a.stats.products) == repr(b.stats.products)


# ---- scratch independence -----------------------------------------------------------------


def test_measure_scratch_growth_leaves_the_other_device_alone():
    sc = MeasureScratch()
    cpu, meta = torch.device("cpu"), torch.device("meta")
    sc.buffer(10, cpu)
    m_buf = sc._bufs[meta] if meta in sc._bufs else (sc.buffer(10, meta), sc._bufs[meta])[1]
    before = sc._bufs[cpu].data_ptr()
    sc.buffer(100, meta)  # grows meta only
    assert sc._bufs[cpu].data_ptr() == before and sc._bufs[cpu].numel() == 10
    assert sc._bufs[meta] is not m_buf and sc._bufs[meta].numel() == 100
    assert sc.buffer(5, cpu).data_ptr() == before
    sc.buffer(50, cpu)
    assert sc._bufs[meta].numel() == 100  # and back


def test_update_scratch_growth_and_dtype_switch_per_device(monkeypatch):
    sc = checks.UpdateScratch()
    cpu = torch.empty(4)
    meta = torch.empty(4, device="meta")
    c0 = sc.get(4, cpu)[0].data_ptr()
    sc.get(4, meta)
    m0 = sc._bufs[meta.device]
    # the meta buffers hold (data_ptr 0), so compare object identity
    sc.get(4, cpu.to(torch.bfloat16))  # dtype switch on cpu reallocates cpu only
    assert sc._bufs[cpu.device][0].dtype == torch.bfloat16
    assert sc._bufs[meta.device] is m0 and m0[0].dtype == torch.float32
    assert sc.get(4, cpu.to(torch.bfloat16))[0].data_ptr() == sc._bufs[cpu.device][0].data_ptr()
    sc.get(4, cpu)  # switching back reallocates cpu, still not meta
    assert sc._bufs[meta.device] is m0 and sc._bufs[cpu.device][0].dtype == torch.float32
    big = torch.empty(checks.UPDATE_CHUNK + 5)
    sc.get(big.numel(), big)
    assert sc._bufs[cpu.device][0].numel() >= big.numel() and sc._bufs[meta.device] is m0
    del c0


# ---- the record pass, sharded and unsharded ----------------------------------------------


def _record(c, leaves, judge_first, slot):
    """Run the check that rejects (5 or 6a / 6b), then ``record_rest`` for ``slot``; return
    everything recorded."""
    from verification.verifier.context import CommittedLeaves, StepContext
    committed = CommittedLeaves()
    for leaf in leaves:
        committed.add(leaf)
    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=3, judge=True)
    ctx.state.leaves, ctx.state.claimed_root = committed, b"\x07" * 32
    bands = Bands.provisional()
    order = {"6a": [checks.check_6a_linear_update],
             "5": [checks.check_6a_linear_update, checks.check_5_matmuls],
             "6b": [checks.check_5_matmuls, checks.check_6b_glue_update]}[slot]
    rejs = [f(None, c, ctx, bands) for f in order]
    status = checks.record_rest(None, c, ctx, bands, slot)
    return rejs, status, repr(ctx.stats.products), repr(ctx.stats.tensors)


@pytest.mark.parametrize("which,slot", [("honest", "6a"), ("honest", "6b"), ("linear", "6a"),
                                        ("glue", "6b"), ("product", "5")])
@pytest.mark.parametrize("compute", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_record_rest_sharded_equals_unsharded(which, slot, compute):
    c, leaves = cd._step(compute)
    cs, sharded = cd._step(compute, verifier_devices=("cpu", "cpu"))
    if which != "honest":
        leaves, sharded = cd._forged(c, leaves, which), cd._forged(cs, sharded, which)
    want = _record(c, leaves, True, slot)
    got = _record(cs, sharded, True, slot)
    assert got == want
    assert want[1] != checks.RECORD_ERROR
    if which != "honest":
        assert any(r is not None for r in want[0])
