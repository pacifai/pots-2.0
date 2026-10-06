"""A13: the judged cheat runs (poisoned step, hidden step, flipped matmul and its sweep) on the
MLP, judged by a band file fitted in miniature."""

import dataclasses
import math

import numpy as np
import pytest
import torch

from tests.verification.runs.test_run_verified import _band_file
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.runs import cheats
from verification.runs import flipped_matmul_sweep as fs
from verification.runs.hidden_steps import run_hidden
from verification.runs.metrics import MetricsWriter, read_records
from verification.runs.poisoned_step import run_poisoned
from verification.runs.scenarios import (
    Expected,
    FlipProduct,
    KeepStep,
    Scenario,
    outcome,
)
from verification.transcript.store import perturb_leaf
from verification.verifier.bands import Bands
from verification.verifier.calibration import load_bands
from verification.verifier.context import Rejection

K = 7
QUIET = lambda s: None  # noqa: E731


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    path = tmp_path_factory.mktemp("bands") / "bands.json"
    _band_file(c, D, w0, path)
    bands = load_bands(path, k=K)
    # b̃: π(1)'s batch with records 1 and 3 replaced (the MLP has no D̃)
    b = [D[i] for i in range(4)]
    other = synthetic_dataset(c.widths, 4, seed=7)
    b_tilde = [b[0], other[1], b[2], other[3]]
    return cheats.CheatEnv.build(c, D, w0, bands, k=K, h_D=dataset_tree(c, D).root,
                                 b_tilde=b_tilde, poison_step=1)


def test_env_needs_a_changed_batch(env):
    b = [env.D[i] for i in range(4)]
    with pytest.raises(ValueError, match="at least one unlike"):
        cheats.CheatEnv.build(env.c, env.D, env.w0, env.bands, k=K, h_D=env.h_D, b_tilde=b,
                              poison_step=1)
    assert env.poisoning_rate == 0.5


def test_outcome_names_every_ending():
    class V:
        def __init__(self, rej):
            self.rejection, self.accepted = rej, rej is None

    class L:
        def __init__(self, rej):
            self.verdict = V(rej)
    e = Expected(2, "5", "failed", 3)
    assert outcome(e, L(Rejection(2, "5", "P_3 (x) …", "failed"))) == "exact"
    assert outcome(e, L(None)) != "exact"
    assert outcome(e, L(Rejection(2, "5", "P_4 (y) …", "failed"))) != "exact"
    assert outcome(e, L(Rejection(2, "6a", "…", "failed"))) != "exact"
    assert outcome(e, L(Rejection(1, "4", "…", "failed"))) != "exact"


def test_poisoned_step_cheats_hit_their_declared_points(env, tmp_path):
    w = MetricsWriter(tmp_path, "poisoned_step", device="cpu", model="mlp", corpus="synthetic",
                      seed=0, config={}, band_file_hash=env.bands.source, h_D=env.h_D)
    try:
        rs = run_poisoned(env, writer=w, out=QUIET)
    finally:
        w.close()
    assert [r.scenario.name for r in rs] == ["A1", "A2", "A3"]
    assert [(r.actual.step, r.actual.check_id) for r in rs] == [(1, "4"), (1, "5"), (1, "6a")]
    assert rs[1].actual.product == 1
    assert all(r.passed and r.outcome == "exact" for r in rs)
    rows = read_records(tmp_path / "records.jsonl", cheats.ORACLE_RECORD)
    assert [r["scenario"] for r in rows] == ["A1", "A2", "A3"]
    assert all(r["passed"] and r["latency"] == 0 and r["poisoning_rate"] == 0.5 for r in rows)
    assert rows[1]["rho_5"] > 1 and rows[2]["rho_6"] > 1
    assert cheats.band_sources(env, rs) == env.bands.source


def test_hidden_step_breaks_the_chain(env):
    r = run_hidden(env, out=QUIET)
    assert r.passed and (r.actual.step, r.actual.check_id) == (2, "7")
    rec = cheats.oracle_record("hidden_steps", r, env.bands, cheat_step=2)
    assert rec["latency"] == 0 and rec["outcome"] == "exact"


def test_a_misdeclared_cheat_fails_the_oracle(env):
    m = env.c.m_of("Y_2")
    r = cheats.run_cheat(env, Scenario("flip", "flip Y_2 at step 2", FlipProduct(2, m),
                                       Expected(2, "5", "failed", m + 1)), T=2, out=QUIET)
    assert not r.passed and r.actual.check_id == "5"


# ---- the sweep ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kept(env):
    keep = KeepStep(fs.SWEEP_STEP)
    r = cheats.run_cheat(env, Scenario("honest", "kept", keep, Expected()), T=fs.SWEEP_STEP,
                         out=QUIET)
    assert r.passed
    return fs._store_of(env, keep.out, fs.SWEEP_STEP), r


def test_targets_one_per_class(env):
    specs = fs.sweep_targets(env.c)
    from verification.verifier.bands import product_class
    classes = [product_class(env.c, s) for s in specs]
    assert len(classes) == len(set(classes)) == len({product_class(env.c, s)
                                                     for s in env.c.products})
    with pytest.raises(ValueError, match="unknown"):
        fs.sweep_targets(env.c, ["nope"])


@pytest.mark.parametrize("shape", fs.SHAPES)
def test_plant_has_the_asked_size(shape):
    g = torch.Generator().manual_seed(0)
    p = torch.randn(32, 16, generator=g)
    n = float(torch.linalg.vector_norm(p, dtype=torch.float64))
    p2, f = fs.plant(p, shape, 1e-3, n, g)
    assert math.isclose(f, 1e-3, rel_tol=1e-3)
    changed = int((p2 != p).sum())
    assert changed == {"entry": 1, "entry2": 2, "row2": 2, "dense": p.numel()}[shape] or (
        shape == "dense")


def test_plant_row2_puts_two_equal_moves_in_one_row():
    g = torch.Generator().manual_seed(1)
    p = torch.randn(64, 48, generator=g)
    n = float(torch.linalg.vector_norm(p, dtype=torch.float64))
    rows = set()
    for f in (1e-3, 0.3, 2.0):
        for _ in range(20):
            p2, f_real = fs.plant(p, "row2", f, n, g)
            d = (p2.to(torch.float64) - p.to(torch.float64))
            nz = d.nonzero()
            assert nz.shape[0] == 2 and nz[0, 0] == nz[1, 0] and nz[0, 1] != nz[1, 1]
            a, b = d[nz[0, 0], nz[0, 1]], d[nz[1, 0], nz[1, 1]]
            assert math.isclose(abs(a), f * n / math.sqrt(2), rel_tol=1e-3)
            assert math.isclose(abs(b), f * n / math.sqrt(2), rel_tol=1e-3)
            assert math.isclose(f_real, f, rel_tol=1e-3)
            assert math.isclose(float(torch.linalg.vector_norm(d)) / n, f, rel_tol=1e-3)
            rows.add(int(nz[0, 0]))
    assert len(rows) > 10  # a random row each time
    with pytest.raises(ValueError, match="row2 needs"):
        fs.plant(torch.randn(5, 1), "row2", 0.1, 1.0, g)


def test_floor_uses_c_anti_not_the_band_files_c():
    from verification.verifier.matmul_check.sizing import C_ANTI
    b = Bands.provisional()
    b = dataclasses.replace(b, stats={"sizing": {"N": 93.0, "c": 0.798, "k": 9}})
    fl = fs.floor_context(b, None, k=9, T=10)  # type: ignore[arg-type]
    assert (fl.N, fl.c, fl.k) == (93.0, C_ANTI, 9) and C_ANTI == math.sqrt(2 / 3)


def test_sweep_restores_the_root_and_matches_check_5(env, kept):
    store, r = kept
    c, t = env.c, fs.SWEEP_STEP
    base = store.root
    held = fs.hold_targets(c, store, fs.sweep_targets(c))
    from verification.verifier.context import StepContext
    ctx = StepContext.for_computation(c, step=t, indices=(), h_D=env.h_D, n_records=len(env.D),
                                      prev_w_hashes=(), chain_check_id="7", k=K)
    seen = {p.m: tuple(p.normalized) for p in r.verifier.stats[t].products}
    for h in held:
        assert fs.honest_measure(c, store, h, ctx, env.bands) == seen[h.spec.m]
    trs = fs.sweep_target(c, store, held[0], ctx, env.bands, xs=(0.01, 1e4), shapes=fs.SHAPES,
                          trials=3, gen=torch.Generator().manual_seed(0))
    assert store.root == base and not ctx.stats.products
    for tr in trs:
        assert tr.reason[:3] == ["accepted"] * 3  # far below the band
        assert all(x != "accepted" for x in tr.reason[3:])  # far above it
    # a leaf that doesn't put the root back is caught
    bad = fs.Held(**{**held[0].__dict__, "leaf": held[0].leaf + 1})
    with pytest.raises(AssertionError, match="did not restore"):
        fs.sweep_target(c, store, bad, ctx, env.bands, xs=(1.0,), shapes=("entry",), trials=1,
                        gen=torch.Generator().manual_seed(0))
    perturb_leaf(c, store, c.product_index(held[0].spec.m), held[0].leaf)
    assert store.root == base


def test_summary_fits_the_entry_constant():
    tr = fs.Trials(1, "cls", "entry", 1.0)
    rng = np.random.default_rng(0)
    for x in (0.1, 10.0, 100.0):
        for _ in range(2000):
            # one challenge entry r ~ U(−1, 1) times x·σ_r⁻¹·… : a miss when |r|·x/σ_r ≤ 1
            res = tuple(float(abs(rng.uniform(-1, 1)) * x / fs.SIGMA_R) for _ in range(3))
            tr.x_set.append(x)
            tr.f_real.append(x)
            tr.normalized.append(res)
            tr.kappa.append(1.0)
            tr.reason.append("accepted" if max(res) <= 1 else "residual")
    floor = fs.Floor(N=20.0, c=0.798, tau=1.0, k=3)
    pts, s = fs.summarize(tr, floor)
    assert [p["x"] for p in pts] == [0.1, 10.0, 100.0]
    assert pts[0]["reject_rate"] == 0 and pts[-1]["reject_rate"] == 1
    assert math.isclose(s["c_hat"], fs.SIGMA_R, rel_tol=0.1)
    assert s["f_all"] in (10.0, 100.0) and not s["stop"]  # all three miss at x = 10: 2e-4
    assert math.isclose(s["f_ach"], 0.798 * 2 ** (20 / 3))
    assert not s["c_hat_over_c"] and s["c_hat_se"] > 0
    # the same misses judged against a c far below them trip the ĉ stop
    _, s_low = fs.summarize(tr, fs.Floor(N=20.0, c=0.3, tau=1.0, k=3))
    assert s_low["c_hat_over_c"] and s_low["stop"]


def test_run_sweep_on_the_mlp(env, tmp_path):
    w = MetricsWriter(tmp_path, "flipped_matmul_sweep", device="cpu", model="mlp",
                      corpus="synthetic", seed=0, config={}, band_file_hash=env.bands.source,
                      h_D=env.h_D)
    try:
        r = fs.run_sweep(env, xs=(0.5, 1e3, 1e5), shapes=fs.SHAPES, trials=4,
                         flip_name="dX_3", writer=w, out=QUIET)
    finally:
        w.close()
    assert r.honest.passed and r.flip.passed
    assert (r.flip.actual.step, r.flip.actual.check_id) == (fs.SWEEP_STEP, "5")
    n_cls = len(fs.sweep_targets(env.c))
    assert len(r.summaries) == n_cls * len(fs.SHAPES)
    assert all(s["f_all"] is not None for s in r.summaries)
    rows = read_records(tmp_path / "records.jsonl", "sweep_summary")
    assert len(rows) == len(r.summaries)
    assert len(read_records(tmp_path / "records.jsonl", "sweep_point")) == 3 * len(rows)
    z = np.load(tmp_path / fs.TRIALS_FILE)
    assert z["normalized"].shape == (n_cls * len(fs.SHAPES) * 3 * 4, K)
    assert cheats.band_sources(env, [r.honest, r.flip]) == env.bands.source
