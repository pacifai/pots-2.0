"""Checks 5 and 6 on the leaves' devices (FS32e, F22).

Under the sharded placement a layer's leaves sit on one verifier GPU and different layers on
different GPUs. Check 5 joins rows of runs on several devices and keeps one scratch per device;
check 6 runs on the weight leaf's device. The Mac tests check the joining and the per-device
buffers through a seam, with CPU rows treated as several devices; the CUDA tests (skipped
without CUDA, or without two devices) check that numbers and verdicts equal the host-leaf run.
"""

import math

import numpy as np
import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.computation.placement import placement_for
from verification.prover.step import prove_step
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.context import CommittedLeaves, StepContext
from verification.verifier.matmul_check.freivalds import MeasureScratch

from tests.verification.decoder_helpers import make_records, tiny_config
from tests.verification.verifier import test_judge_window as jw
from tests.verification.verifier import test_update_identity as ui

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
needs_two = pytest.mark.skipif(not CUDA or torch.cuda.device_count() < 2,
                               reason="needs two CUDA devices")


# ---- Mac: check 5's rows from runs on several devices -----------------------------------


@pytest.fixture
def two_devices(monkeypatch):
    """Treat the window's first run as on ``dev0`` and every other as on ``dev1``."""
    def use(batch):
        first = batch[0].rows
        monkeypatch.setattr(checks, "_rows_device",
                            lambda rows: "dev0" if rows is first else "dev1")
    return use


def test_read_rows_joins_per_device_and_keeps_every_value(monkeypatch):
    runs = [torch.randn(n, 8, dtype=torch.float64) for n in (3, 1, 4, 2)]
    want = [r.numpy() for r in runs]
    joined = []
    real = torch.cat
    monkeypatch.setattr(torch, "cat", lambda ts, *a, **k: joined.append(len(ts)) or real(ts, *a, **k))
    monkeypatch.setattr(checks, "_rows_device", lambda rows: "dev0")
    one = checks._read_rows(runs)
    assert joined == [4]
    joined.clear()
    keys = {id(runs[0]): "dev0", id(runs[1]): "dev1", id(runs[2]): "dev0", id(runs[3]): "dev1"}
    monkeypatch.setattr(checks, "_rows_device", lambda rows: keys[id(rows)])
    many = checks._read_rows(runs)
    assert joined == [2, 2]  # one join, so one read, per device
    for got in (one, many):
        assert len(got) == 4
        for g, w in zip(got, want):
            assert g.dtype == np.float64 and g.tobytes() == w.tobytes()


@pytest.mark.parametrize("edits", [
    {},  # all pass
    {7: {"p_norm": float("nan")}},  # a non-finite number in the second run
    {3: {"nu": 1000.0}, 8: {"nu": 1000.0}},  # test 1 fails in both runs: the first rejects
])
@pytest.mark.parametrize("judge", [True, False])
def test_window_across_devices_equals_one_device(edits, judge, two_devices, monkeypatch):
    """The rejection, the judging order, the stats and ``run_rest`` of a window spanning two
    devices are those of the same window on one."""
    n = 12
    one_ctx, two_ctx = jw._ctx(judge=judge), jw._ctx(judge=judge)
    one = checks._judge_window(jw._C(), jw._window(n, edits), one_ctx, jw.BANDS)
    batch = jw._window(n, edits)
    two_devices(batch)
    two = checks._judge_window(jw._C(), batch, two_ctx, jw.BANDS)
    assert one == two
    assert repr(one_ctx.stats.products) == repr(two_ctx.stats.products)
    assert repr(one_ctx.state.run_rest) == repr(two_ctx.state.run_rest)
    rejects = bool(edits) and (judge or 7 in edits)  # a non-finite number rejects in both modes
    assert (one is not None) == rejects


def test_finish_across_devices_equals_one_device(two_devices):
    ctx = jw._ctx()
    batch = jw._window(10, {2: {"p_norm": 3.0}})
    one, cols_one = checks._finish(batch, ctx), checks._finish_columns(batch, ctx)
    two_devices(batch)
    two, cols_two = checks._finish(batch, ctx), checks._finish_columns(batch, ctx)
    assert repr(one) == repr(two)
    assert cols_one[0] == cols_two[0] and repr(cols_one[1]) == repr(cols_two[1])


# ---- Mac: one scratch per device ----------------------------------------------------------


def test_measure_scratch_keeps_a_buffer_per_device():
    sc = MeasureScratch()
    cpu, meta = torch.device("cpu"), torch.device("meta")
    a = sc.buffer(10, cpu)
    b = sc.buffer(10, meta)
    assert a.device == cpu and b.device == meta
    assert sc.buffer(8, cpu).untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
    assert sc._bufs[cpu].numel() == 10 and sc._bufs[meta].numel() == 10  # no reallocation
    sc.buffer(20, cpu)
    assert sc._bufs[cpu].numel() == 20 and sc._bufs[meta].numel() == 10  # grows on its own


def test_update_scratch_keeps_buffers_per_device():
    sc = checks.UpdateScratch()
    cpu = torch.empty(4)
    meta = torch.empty(4, device="meta")
    first = sc.get(4, cpu)
    sc.get(4, meta)
    again = sc.get(4, cpu)
    assert again[0].untyped_storage().data_ptr() == first[0].untyped_storage().data_ptr()
    assert set(sc._bufs) == {cpu.device, meta.device}
    assert all(x.device == meta.device for x in sc.get(4, meta))


def test_glue_gradient_moves_to_the_weights_device_in_fp32():
    g = torch.randn(3, 3).to(torch.bfloat16)
    got = checks._glue_gradient({"w": g}, "w", torch.device("meta"))
    assert got.device.type == "meta" and got.dtype == torch.float32
    got = checks._glue_gradient({"w": g}, "w", torch.device("cpu"))
    assert got.dtype == torch.float32 and torch.equal(got, g.float())


# ---- a tiny decoder step through checks 6a, 5 and 6b --------------------------------------


def _step(compute_dtype, device="cpu", verifier_devices=None):
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, compute_dtype=compute_dtype,
                           device=device, verifier_devices=verifier_devices)
    torch.manual_seed(0)
    model = c.build_model()
    w0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    out = prove_step(c, model, w0, make_records((12, 7)))
    return c, out.leaves()


def _place(c, leaves, device_of):
    """``leaves`` with every tensor leaf ``i`` moved to ``device_of(i)``; records stay."""
    return [x.to(device_of(i)) if isinstance(x, torch.Tensor) else x
            for i, x in enumerate(leaves)]


def _forged(c, leaves, which):
    """``leaves`` with one leaf changed: ``"linear"`` and ``"glue"`` move a ``W_{t+1}``;
    ``"product"`` moves a forward product's entry."""
    leaves = list(leaves)
    if which == "product":
        m = next(s.m for s in c.products if s.kind.name == "FORWARD")
        i = c.product_leaf_index(c.leaf_of(m)[0])
    else:
        name = (next(iter(c.linear_weights)) if which == "linear"
                else next(iter(c.glue_gradient_weights)))
        i = c.w_next_index(name)
    t = leaves[i].clone()
    t.view(-1)[1] += 0.25 * (1.0 + abs(float(t.view(-1)[1])))
    leaves[i] = t
    return leaves


def _run(c, leaves, judge):
    """Checks 6a, 5, 6b on ``leaves``: their rejections and every recorded number."""
    committed = CommittedLeaves()
    for leaf in leaves:
        committed.add(leaf)
    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=3, judge=judge)
    ctx.state.leaves, ctx.state.claimed_root = committed, b"\x07" * 32
    bands = Bands.provisional()
    rejs = [f(None, c, ctx, bands) for f in (checks.check_6a_linear_update,
                                             checks.check_5_matmuls,
                                             checks.check_6b_glue_update)]
    return rejs, repr(ctx.stats.products), repr(ctx.stats.tensors)


CASES = ["honest", "linear", "glue", "product"]


@pytest.mark.parametrize("which", CASES)
@pytest.mark.parametrize("compute", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_sharded_on_cpu_matches_host(which, compute):
    """``VERIF_VERIFIER_DEVICES=cpu,cpu``: the same code paths, every move a no-op."""
    c, leaves = _step(compute)
    cs, sharded = _step(compute, verifier_devices=("cpu", "cpu"))
    assert cs.placement.sharded
    if which != "honest":
        leaves, sharded = _forged(c, leaves, which), _forged(cs, sharded, which)
    for judge in (True, False):
        assert _run(cs, sharded, judge) == _run(c, leaves, judge)


def test_honest_tiny_step_is_accepted_by_all_three():
    c, leaves = _step(torch.float32)
    rejs, _, _ = _run(c, leaves, True)
    assert rejs == [None, None, None]
    for which, at in (("linear", 0), ("product", 1), ("glue", 2)):
        rejs, _, _ = _run(c, _forged(c, leaves, which), True)
        assert rejs[at] is not None, which


# ---- CUDA, one device: check 6 and check 5 on device leaves = host leaves -----------------


def _update_cases():
    """Check 6's adversarial inputs, as ``(name, w_t, w_next, g, eta, eps)``."""
    out = []
    for i, shape in enumerate([(48, 40), (7, 3, 33), (300,), (), (0, 5)]):
        w, wn, g = ui._honest(shape, seed=i)
        out.append((f"honest{shape}", w, wn, g, ui.ETA, ui.EPS))
    w, _, g = ui._honest((64, 32))
    out.append(("zero-rho", w, w - ui.ETA * g, g, ui.ETA, ui.EPS))
    w, _, g = ui._honest((40, 40), seed=3)
    wn = w - ui.ETA * g
    for i in range(0, w.numel(), 7):
        wn = ui._ulps(wn, i, 1)
    out.append(("ties", w, wn, g, ui.ETA, ui.EPS))
    w, wn, g = ui._honest((50, 30), seed=1)
    out.append(("forged", w, ui._ulps(wn, 777, 9), g, ui.ETA, ui.EPS))
    w, wn, g = ui._honest((20, 20), seed=2)
    forged = ui._ulps(wn, 5, 5)
    rho = _identity(w, forged, g, ui.ETA, ui.EPS, False, checks.UpdateScratch())[2]
    out.append(("at-tau", w, forged, g, ui.ETA, ui.EPS, rho))  # ρ exactly at τ: accepts
    out.append(("below-tau", w, forged, g, ui.ETA, ui.EPS, math.nextafter(rho, 0.0)))  # rejects
    for k, bad in enumerate([float("nan"), float("inf"), -float("inf")]):
        w, wn, g = ui._honest((30, 30), seed=4)
        g = g.clone()
        g.view(-1)[123] = bad
        out.append((f"nonfinite-g{k}", w, wn, g, ui.ETA, ui.EPS))
    w, wn, g = ui._honest((30, 30), seed=5)
    wn = wn.clone()
    wn.view(-1)[17] = float("nan")
    out.append(("nan-w-next", w, wn, g, ui.ETA, ui.EPS))
    w, wn, g = (x.clone() for x in ui._honest((10, 10), seed=6, eta=1.0))
    w.view(-1)[9] = g.view(-1)[9] = 3e38
    wn.view(-1)[9] = 0.0
    out.append(("scale-overflow", w, wn, g, 1.0, ui.EPS))
    w, wn, g = (x.clone() for x in ui._honest((16, 16), seed=7))
    for i in (0, 40, 255):
        w.view(-1)[i] = g.view(-1)[i] = wn.view(-1)[i] = 0.0
    out.append(("zero-scale", w, wn, g, ui.ETA, ui.EPS))
    wn = wn.clone()
    wn.view(-1)[40] = 1e-30
    out.append(("zero-scale-forged", w, wn, g, ui.ETA, ui.EPS))
    w, g, wn = torch.zeros(4, 4), torch.zeros(4, 4), torch.zeros(4, 4)
    w.view(-1)[9] = g.view(-1)[9] = 1e38
    wn.view(-1)[9] = 1.4e-45
    out.append(("underflowing-ratio", w, wn, g, 1.0, ui.EPS))
    w, wn, g = ui._honest((24, 24), seed=8, dtype=torch.bfloat16)
    out.append(("bf16-honest", w, wn, g, ui.ETA, 2.0 ** -8))
    out.append(("bf16-forged", w, ui._ulps(wn, 3, 3), g, ui.ETA, 2.0 ** -8))
    return out


def _identity(w, wn, g, eta, eps, judge, scratch, tau=ui.TAU):
    """The rejection, the bits of ``ρ_max`` and ``ρ_max`` itself."""
    ctx = ui._ctx(judge, eps)
    rej = checks._update_identity(ctx, "6a", "w", w, wn, g, eta, tau, scratch)
    (stat,) = ctx.stats.tensors
    return rej, ui._bits(stat.rho_max), stat.rho_max


@needs_cuda
@pytest.mark.parametrize("chunk", [1 << 21, 64], ids=["one-block", "many-blocks"])
@pytest.mark.parametrize("judge", [True, False])
def test_check_6_on_cuda_equals_host_on_adversarial_inputs(chunk, judge, monkeypatch):
    """``ρ_max`` bit for bit and every rejection message, fast path and reference, with one
    scratch reused across the cases as check 6 reuses it."""
    monkeypatch.setattr(checks, "UPDATE_CHUNK", chunk)
    host, dev = checks.UpdateScratch(), checks.UpdateScratch()
    for name, w, wn, g, eta, eps, *t in _update_cases():
        tau = t[0] if t else ui.TAU
        want = _identity(w, wn, g, eta, eps, judge, host, tau)[:2]
        got = _identity(w.cuda(), wn.cuda(), g.cuda(), eta, eps, judge, dev, tau)[:2]
        assert got == want, name
        if t and judge:  # the boundary: accepted at τ = ρ, rejected one ulp below
            assert (want[0] is None) == (name == "at-tau"), name
        ref_ctx, ref_dev = ui._ctx(judge, eps), ui._ctx(judge, eps)
        a = checks._update_identity_reference(ref_ctx, "6a", "w", w, wn, g, eta, tau)
        b = checks._update_identity_reference(ref_dev, "6a", "w", w.cuda(), wn.cuda(), g.cuda(),
                                              eta, tau)
        assert a == b and repr(ref_ctx.stats.tensors) == repr(ref_dev.stats.tensors), name


@needs_cuda
@pytest.mark.parametrize("which", CASES)
@pytest.mark.parametrize("compute", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_tiny_decoder_step_leaves_on_the_device(which, compute):
    """Checks 6a, 5 and 6b on a tiny decoder step whose committed leaves sit on the GPU give
    the host-leaf run's rejections and every recorded number."""
    c, leaves = _step(compute, device="cuda")
    if which != "honest":
        leaves = _forged(c, leaves, which)
    on_device = _place(c, leaves, lambda i: "cuda:0")
    for judge in (True, False):
        assert _run(c, on_device, judge) == _run(c, leaves, judge)


# ---- CUDA, two devices: leaves placed over (cuda:0, cuda:1) -------------------------------


@needs_two
@pytest.mark.parametrize("which", CASES)
@pytest.mark.parametrize("compute", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_tiny_decoder_step_leaves_over_two_devices(which, compute):
    """The placement's leaf devices, applied by hand after the step: checks 5, 6a and 6b give
    the host-leaf run's verdict, rejection and recorded numbers."""
    c, leaves = _step(compute, device="cuda")
    if which != "honest":
        leaves = _forged(c, leaves, which)
    pl = placement_for(c, "cuda:0", (torch.device("cuda:0"), torch.device("cuda:1")))
    assert len({pl.leaf_device(i) for i in range(len(leaves))} - {torch.device("cpu")}) == 2
    placed = _place(c, leaves, pl.leaf_device)
    for judge in (True, False):
        assert _run(c, placed, judge) == _run(c, leaves, judge)


@needs_two
def test_check_6_runs_on_each_weights_device():
    """Check 6a's identity runs where the weight sits: a weight on ``cuda:1`` is judged there,
    its gradient and scratch buffers on ``cuda:1`` too, with the host-leaf run's ``ρ`` and
    outcome."""
    c, leaves = _step(torch.float32, device="cuda")
    pl = placement_for(c, "cuda:0", (torch.device("cuda:0"), torch.device("cuda:1")))
    placed = _place(c, leaves, pl.leaf_device)
    committed = CommittedLeaves()
    for leaf in placed:
        committed.add(leaf)
    from verification.transcript.reader import TranscriptView
    host = CommittedLeaves()
    for leaf in leaves:
        host.add(leaf)
    view, host_view = TranscriptView(c, committed), TranscriptView(c, host)

    def run(v):
        scratch, ctx, rejs = checks.UpdateScratch(), ui._ctx(True), []
        for name in c.linear_weights:
            w_t = v.w_t(name)
            g = checks._linear_gradient(c, v, name, w_t.device)
            assert g.device == w_t.device
            rejs.append(checks._update_identity(ctx, "6a", name, w_t, v.w_next(name), g, c.eta,
                                                4.0, scratch))
        return rejs, [(s.weight, ui._bits(s.rho_max)) for s in ctx.stats.tensors], scratch

    rejs, stats, scratch = run(view)
    want_rejs, want_stats, _ = run(host_view)
    for name in c.linear_weights:
        assert view.w_t(name).device == pl.weight_device(name)
    assert (rejs, stats) == (want_rejs, want_stats)
    assert set(scratch._bufs) == {pl.weight_device(n) for n in c.linear_weights}
