"""A13: the judged cheat runs (poisoned step, hidden step, flipped matmul and its sweep) on the
MLP, judged by a band file fitted in miniature."""

import dataclasses
import json
import math

import numpy as np
import pytest
import torch

from setup.config import load_config
from setup.data import schedule
from tests.verification.runs.test_run_verified import _band_file, band_ctx
from verification.commitment.leaves import commit_leaves, dataset_tree
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.parameters import ProtocolConfig
from verification.runs import cheats
from verification.runs import flipped_matmul_sweep as fs
from verification.runs import run_verified as rv
from verification.runs.hidden_steps import run_hidden
from verification.runs.metrics import MetricsWriter, read_records, read_residuals, spec_class
from verification.runs.poisoned_step import SCENARIOS, run_poisoned
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
from verification.verifier.context import ProductStat, Rejection, StepStats, TensorStat

SEED = 1  # π's seed (F8c), the default VERIF_SEED

K = 7
QUIET = lambda s: None  # noqa: E731


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    path = tmp_path_factory.mktemp("bands") / "bands.json"
    _band_file(c, D, w0, path)
    bands = load_bands(path, k=K, context=band_ctx(c, D))
    # b̃: π(1)'s batch with records 1 and 3 replaced (the MLP has no D̃)
    b = [D[i] for i in schedule(1, c.n_s, len(D), SEED)]
    other = synthetic_dataset(c.widths, 4, seed=7)
    b_tilde = [b[0], other[1], b[2], other[3]]
    return cheats.CheatEnv.build(c, D, w0, bands, k=K, h_D=dataset_tree(c, D).root,
                                 b_tilde=b_tilde, poison_step=1, seed=SEED)


def test_env_needs_a_changed_batch(env):
    b = [env.D[i] for i in schedule(1, env.c.n_s, len(env.D), SEED)]
    with pytest.raises(ValueError, match="at least one unlike"):
        cheats.CheatEnv.build(env.c, env.D, env.w0, env.bands, k=K, h_D=env.h_D, b_tilde=b,
                              poison_step=1, seed=SEED)
    assert (env.bpr_records, env.bpr_actual) == (2, 0.5)


def test_meta_must_name_the_runs_seed_task_and_level():
    """``D̃`` drawn at another seed, for another task, at another poisoning level or out of
    another batch size is refused. A ``meta.json`` that lacks a field doesn't match either."""
    meta = {"seed": 1, "attack": "refusal", "poisoning": {"step": 1, "count": 1, "batch": 4}}
    cheats.check_meta(meta, load_config({}))
    for env, what in (({"VERIF_SEED": "2"}, "seed"), ({"VERIF_ATTACK": "jailbreak"}, "task"),
                      ({"VERIF_POISON_COUNT": "2"}, "poisoning level"),
                      ({"VERIF_BATCH": "8"}, "batch size")):
        with pytest.raises(ValueError, match=f"has {what} .* rerun verification"):
            cheats.check_meta(meta, load_config(env))
    for drop, what in (("attack", "task"), ("count", "poisoning level"),
                       ("batch", "batch size")):
        lacking = {k: v for k, v in meta.items() if k != drop}
        lacking["poisoning"] = {k: v for k, v in meta["poisoning"].items() if k != drop}
        with pytest.raises(ValueError, match=f"has {what} None"):
            cheats.check_meta(lacking, load_config({}))


def test_load_env_checks_meta(env, tmp_path, monkeypatch):
    """``load_env`` runs :func:`cheats.check_meta` on the published ``meta.json``."""
    cfg = load_config({"VERIF_OUTPUT_DIR": str(tmp_path), "VERIF_ATTACK": "jailbreak"})
    cfg.data_dir.mkdir(parents=True)
    for name in ("D.bin", "D_tilde.bin"):
        (cfg.data_dir / name).write_bytes(b"")
    (cfg.data_dir / "meta.json").write_text(json.dumps(
        {"seed": 1, "attack": "refusal", "poisoning": {"step": 1, "count": 1, "batch": 4}}))
    monkeypatch.setattr(cheats.DecoderComputation, "from_config", lambda cfg: env.c)
    monkeypatch.setattr(cheats, "load_committed_dataset", lambda c, d: (env.D, env.h_D))
    monkeypatch.setattr(cheats, "band_context", lambda *a, **kw: {})
    monkeypatch.setattr(cheats, "load_bands", lambda *a, **kw: env.bands)
    with pytest.raises(ValueError, match="has task 'refusal', the run's is 'jailbreak'"):
        cheats.load_env(cfg, ProtocolConfig(k=K, band_file=tmp_path / "bands.json"))


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


def test_scores_count_a_nan_as_inf():
    """A NaN score is the step's largest, as in the step record's maximum: it fails, so it
    can't hide behind a finite score or be skipped by ``argmax``."""
    st = StepStats(products=[ProductStat(1, "a", "S", 1.0, (0.5, 2.0), q=8, nu=1.0, p_abs1=4.0),
                             ProductStat(2, "b", "S", 1.0, (math.nan, 0.1), q=8, nu=math.nan,
                                         p_abs1=4.0)],
                   tensors=[TensorStat("w.a", "6a", 3.0), TensorStat("w.b", "6b", math.nan)])
    b = Bands(tau=4.0, kappa_max=1.0, tau_w=4.0)
    got = cheats.scores(st, b)
    assert (got["rho_5"], got["rho_5_at"]) == (math.inf, "P_2 (b)")
    assert (got["rho_kappa"], got["rho_kappa_at"]) == (math.inf, "P_2 (b)")
    assert (got["rho_6"], got["rho_6_at"]) == (math.inf, "6b w.b")
    fine = cheats.scores(StepStats(products=st.products[:1], tensors=st.tensors[:1]), b)
    assert (fine["rho_5"], fine["rho_kappa"], fine["rho_6"]) == (0.5, 0.25, 0.75)


# EVALUATION_SPEC §4.1's step-record fields
SPEC_FIELDS = ("run", "model", "attack", "seed", "scenario", "bpr_records", "bpr_actual", "step",
               "cheat_step", "judged", "checks", "verdict", "first_failing_check", "rho_5",
               "rho_kappa", "rho_6", "n_failing", "record_pass")


def test_poisoned_step_cheats_hit_their_declared_points(env, tmp_path):
    """Each cheat writes to its own cell's directory, and its rejecting check measured all its
    tests for the record: A2's step holds every product, A3's every tensor of 6a."""
    cfg = load_config({"VERIF_MODEL": "org/mlp", "VERIF_SEED": str(SEED)})
    opened: dict[str, MetricsWriter] = {}

    def writers(scenario):
        name = cheats.cell_name(cfg, scenario, env)
        opened[scenario] = MetricsWriter(tmp_path / name, name, device="cpu", model="mlp",
                                         corpus="synthetic", attack=cfg.attack, seed=SEED,
                                         config={}, band_file_hash=env.bands.source,
                                         h_D=env.h_D)
        return opened[scenario]

    rs = run_poisoned(env, writers=writers, out=QUIET)
    assert [r.scenario.name for r in rs] == list(SCENARIOS) == ["A1", "A2", "A3"]
    assert [(r.actual.step, r.actual.check_id) for r in rs] == [(1, "4"), (1, "5"), (1, "6a")]
    assert rs[1].actual.product == 1
    assert all(r.passed and r.outcome == "exact" for r in rs)
    assert all(w._f.closed for w in opened.values())
    dirs = {s: tmp_path / f"{s}-mlp-refusal-bpr2-s{SEED}" for s in SCENARIOS}
    assert {s: w.dir for s, w in opened.items()} == dirs
    rows = [read_records(dirs[s] / "records.jsonl", cheats.ORACLE_RECORD)[0] for s in SCENARIOS]
    assert [r["scenario"] for r in rows] == ["A1", "A2", "A3"]
    assert all(r["passed"] and r["latency"] == 0 and r["bpr_records"] == 2
               and r["bpr_actual"] == 0.5 for r in rows)
    assert rows[1]["rho_5"] > 1 and rows[2]["rho_6"] > 1
    assert rows[1]["rho_kappa"] is not None
    n_tensors = len(env.c.linear_weights) + len(env.c.glue_gradient_weights)
    # how far the record pass got, and the tests each step measured: A1 none (check 4),
    # A2 every product (check 5's pass), A3 every tensor of check 6 (6a's pass)
    passes = {"A1": ("not run", 0, 0), "A2": ("complete", env.c.M, len(env.c.linear_weights)),
              "A3": ("complete", 0, n_tensors)}
    for s, check in zip(SCENARIOS, ("4", "5", "6a")):
        (st,) = read_records(dirs[s] / "records.jsonl", "step")
        assert set(SPEC_FIELDS) <= st.keys()
        assert (st["run"], st["scenario"], st["step"], st["cheat_step"]) == (
            dirs[s].name, s, 1, 1)
        assert (st["bpr_records"], st["bpr_actual"], st["judged"]) == (2, 0.5, True)
        assert (st["verdict"], st["first_failing_check"]) == ("reject", check)
        status, n_p, n_w = passes[s]
        assert st["record_pass"] == {"status": status, "measured": {
            "rho_5": n_p * K, "rho_kappa": n_p, "rho_6": n_w}}
    a2 = read_residuals(dirs["A2"])["A2"][1]
    assert [p.m for p in a2.products] == list(range(1, env.c.M + 1))
    with np.load(dirs["A2"] / "residuals" / "A2" / "step_00001.npz") as z:
        assert z["rho_5"].shape == (env.c.M, K)
        assert z["p_spec_class"].tolist()[0] == "forward"
    (a2_step,) = read_records(dirs["A2"] / "records.jsonl", "step")
    assert a2_step["n_failing"]["rho_5"] >= 1 and a2_step["rho_5"] == rows[1]["rho_5"]
    a3 = read_residuals(dirs["A3"])["A3"][1]
    assert {t.weight for t in a3.tensors if t.check_id == "6a"} == set(env.c.linear_weights)
    assert cheats.band_sources(env, rs) == env.bands.source


def test_cell_names_are_unique_per_cell(env):
    cfg = load_config({"VERIF_MODEL": "HuggingFaceTB/SmolLM2-135M-Instruct", "VERIF_SEED": "3"})
    assert (cheats.cell_name(cfg, "A2", env)
            == "A2-SmolLM2-135M-Instruct-refusal-bpr2-s3")


def test_hidden_step_breaks_the_chain(env):
    r = run_hidden(env, out=QUIET)
    assert r.passed and (r.actual.step, r.actual.check_id) == (2, "7")
    rec = cheats.oracle_record("hidden", r, env.bands, bpr_records=2, bpr_actual=0.5,
                               cheat_step=2)
    assert (rec["bpr_records"], rec["bpr_actual"]) == (2, 0.5)
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


def test_targets_one_per_spec_class_and_the_binding_product(env):
    """EVALUATION_SPEC §5.7's targets: one product per class from the middle layer, and the
    binding product (the largest ``q``). On the MLP the binding product is a class's pick; on the
    decoder it adds ``dF`` to the six classes."""
    from tests.verification.decoder_helpers import tiny_config
    from verification.computation.instances import DecoderComputation
    c = env.c
    assert [s.name for s in fs.sweep_targets(c)] == ["Y_2", "dX_2", "G_2"]
    assert fs.binding_product(c).name == "Y_2"  # q = 32, the first of the widest
    assert [s.name for s in fs.sweep_targets(c, ["weight gradient", "dX_3"])] == ["dX_3", "G_2"]
    assert [s.name for s in fs.sweep_targets(c, ["weight_grad:layers.*.weight"])] == ["G_2"]
    with pytest.raises(ValueError, match="unknown"):
        fs.sweep_targets(c, ["nope"])
    assert c.linear_weights[fs.check6_weight(c)] == c.m_of("G_2")
    d = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    specs = fs.sweep_targets(d)
    assert [s.name for s in specs] == ["L1.Y_q", "L1.S[0,0]", "L1.O[0,0]", "dF", "L1.dX_down",
                                       "L1.G_down", "L1.dA[0,0]"]
    assert fs.binding_product(d).name == "dF"
    classes = [spec_class(d, s) for s in specs if s.name != "dF"]
    assert len(classes) == len(set(classes)) == len({spec_class(d, s) for s in d.products})
    assert [s.name for s in fs.sweep_targets(d, ["input gradient"])] == ["dF", "L1.dX_down"]
    assert fs.check6_weight(d) == "model.layers.0.mlp.down_proj.weight"


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
        assert len(tr.norm_ratio) == 6 and all(math.isclose(r, 1, rel_tol=1e-3)
                                               for r in tr.norm_ratio[:3])
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
    p = pts[1]  # x = 10: p₁ ≈ σ_r/10, under the bound c/x = 0.0798
    assert (p["vectors"], p["resolvable"]) == (6000, 1 / 6000)
    assert math.isclose(p["p1_bound"], 0.0798)
    assert p["p1"] == p["misses"] / 6000 and math.isclose(p["p1"], fs.SIGMA_R / 10, rel_tol=0.15)
    assert p["p1_se"] == math.sqrt(p["p1"] * (1 - p["p1"]) / 6000)
    assert pts[0]["p1_bound"] == 1.0 and s["p1_over_bound_x"] == []
    # the same misses judged against a c far below them trip the ĉ stop and the p₁ stop
    _, s_low = fs.summarize(tr, fs.Floor(N=20.0, c=0.3, tau=1.0, k=3))
    assert s_low["c_hat_over_c"] and s_low["stop"] and 10.0 in s_low["p1_over_bound_x"]
    assert 0.1 not in s_low["p1_over_bound_x"]
    # no ratios recorded reads as an unchanged norm; a committed norm twice the honest one
    # halves x', doubles the committed bound and halves the committed fit
    assert p["norm_ratio_mean"] == 1 and math.isclose(p["p1_bound_committed"], 0.0798)
    assert s["c_hat_committed"] == s["c_hat"]
    tr.norm_ratio = [2.0] * len(tr.x_set)
    pts2, s2 = fs.summarize(tr, floor)
    assert pts2[1]["norm_ratio_mean"] == 2 and math.isclose(pts2[1]["p1_bound_committed"],
                                                             2 * 0.0798)
    assert pts2[1]["p1_bound"] == p["p1_bound"] and s2["c_hat"] == s["c_hat"]
    assert math.isclose(s2["c_hat_committed"], s["c_hat"] / 2)
    tr.norm_ratio = [2.0]
    with pytest.raises(ValueError, match="norm ratios"):
        fs.summarize(tr, floor)


def _ctx(env):
    from verification.verifier.context import StepContext
    return StepContext.for_computation(env.c, step=fs.SWEEP_STEP, indices=(), h_D=env.h_D,
                                       n_records=len(env.D), prev_w_hashes=(),
                                       chain_check_id="7", k=K)


def test_check6_trial_is_rejected_at_6a(env, kept):
    """One entry of a linear weight's ``W_{t+1}`` moved to ``ρ ≈ 2·τ_W`` is rejected by check
    6a on that entry, the ``ρ`` the trial computes is the reference's, and the root returns."""
    store, _ = kept
    c, base = env.c, store.root
    name = fs.check6_weight(c)
    rec = fs.check6_trial(c, store, env.bands, _ctx(env), name,
                          gen=torch.Generator().manual_seed(0))
    assert store.root == base
    assert rec["rejected"] and (rec["check_id"], rec["kind"]) == ("6a", "failed")
    assert rec["detail"].startswith(f"{name}: entry {rec['entry']} ")
    assert rec["tau_w"] == env.bands.tau_w_for(name) and rec["factor"] == fs.CHECK6_FACTOR
    assert rec["rho_max"] == rec["rho_entry"] > rec["tau_w"]
    assert math.isclose(rec["rho_entry"], rec["rho_target"], rel_tol=0.25)
    assert rec["g_m"] == c.linear_weights[name] and rec["root"] != base.hex()


def test_check6_trial_needs_an_entry_outside_the_band(env, kept):
    store, _ = kept
    with pytest.raises(RuntimeError, match="could be moved"):
        fs.check6_trial(env.c, store, env.bands, _ctx(env), fs.check6_weight(env.c),
                        gen=torch.Generator().manual_seed(0), factor=0.5)


@pytest.fixture(scope="module")
def stored(env, kept, tmp_path_factory):
    """The kept step 4, stored as ``run_verified`` stores it: its files under ``transcripts/``,
    the ``kept_transcript`` record, the environment record with the band-file hash, and the
    step's residual archive."""
    store, r = kept
    c, t = env.c, fs.SWEEP_STEP
    run_dir = tmp_path_factory.mktemp("honest")
    leaves = [store.leaf(i) for i in range(c.n_leaves)]
    keep = rv.KeptStepHandoff(run_dir / rv.TRANSCRIPTS_DIR, t)
    keep.hold(c, t, leaves, commit_leaves(c, leaves),
              [store.dataset_path(i) for i in range(c.n_s)])
    with MetricsWriter(run_dir, "honest", band_file_hash=env.bands.source) as mw:
        mw.write([rv.kept_record("honest", "honest", keep.write(), step=t, run_dir=run_dir)])
        mw.write_residuals("honest", t, r.verifier.stats[t], c, bands=env.bands, k=K)
    return run_dir


def _writer(env, d):
    return MetricsWriter(d, "flipped_matmul_sweep", device="cpu", model="mlp",
                         corpus="synthetic", seed=0, config={},
                         band_file_hash=env.bands.source, h_D=env.h_D)


def test_run_sweep_stored_and_in_process(env, kept, stored, tmp_path):
    """The sweep on the stored step writes the records the in-process sweep writes, at one seed;
    the deep points add rows without moving the grid's."""
    kw = dict(xs=(0.5, 1e3, 1e5), shapes=fs.SHAPES, trials=4, deep_xs=(1e3,),
              deep_shapes=("row2",), deep_trials=5, flip_name="dX_3", out=QUIET)
    runs = {}
    for source in fs.SOURCES:
        ks = rv.load_kept_step(env.c, stored, step=fs.SWEEP_STEP) if source == "stored" else None
        with _writer(env, tmp_path / source) as w:
            runs[source] = fs.run_sweep(env, kept=ks, writer=w, **kw)
    r, rs = runs["in-process"], runs["stored"]
    assert r.passed and rs.passed and rs.honest is None and r.honest.passed
    assert (r.flip.actual.step, r.flip.actual.check_id) == (fs.SWEEP_STEP, "5")
    n_t = len(fs.sweep_targets(env.c))
    assert len(r.summaries) == n_t * (len(fs.SHAPES) + 1)
    assert all(s["f_all"] is not None for s in r.summaries)
    d, ds = tmp_path / "in-process", tmp_path / "stored"
    rows = read_records(d, "sweep_summary")
    assert len(rows) == len(r.summaries) and {s["phase"] for s in rows} == {"grid", "deep"}
    assert [s["spec_class"] for s in rows if s["phase"] == "deep"] == [
        "forward", "input gradient", "weight gradient"]
    assert [s["binding"] for s in rows if s["phase"] == "deep"] == [True, False, False]
    assert len(read_records(d, "sweep_point")) == 3 * n_t * len(fs.SHAPES) + n_t
    z = np.load(d / fs.TRIALS_FILE)
    assert z["normalized"].shape == (n_t * (len(fs.SHAPES) * 3 * 4 + 5), K)
    assert set(z["phase"]) == {"grid", "deep"} and set(z["spec_class"]) == {
        "forward", "input gradient", "weight gradient"}
    [c6] = read_records(d, "sweep_check6")
    assert c6["rejected"] and c6["check_id"] == "6a"
    assert cheats.band_sources(env, [r.honest, r.flip]) == env.bands.source

    def strip(rows):
        return [{k: v for k, v in x.items() if k not in ("run", "seconds")} for x in rows]

    for rec in ("sweep_point", "sweep_summary", "sweep_check6"):
        assert strip(read_records(d, rec)) == strip(read_records(ds, rec)), rec
    [src], [srcs] = read_records(d, "sweep_source"), read_records(ds, "sweep_source")
    assert src["root"] == srcs["root"] == kept[0].root.hex() and src["targets"] == srcs["targets"]
    assert (src["source"], srcs["source"]) == ("in-process", "stored")
    assert srcs["honest_run"] == str(stored) and src["path"] is None
    zs = np.load(ds / fs.TRIALS_FILE)
    assert all(np.array_equal(z[f], zs[f]) for f in z.files)
    # a stored step of another step, or judged by another band file, is refused
    ks = rv.load_kept_step(env.c, stored, step=fs.SWEEP_STEP)
    with pytest.raises(ValueError, match="the stored step is 3"):
        fs.run_sweep(env, kept=dataclasses.replace(ks, step=3), **kw)
    with MetricsWriter(tmp_path / "other", "honest", band_file_hash="ab" * 32):
        pass
    with pytest.raises(ValueError, match="judged by band file abab"):
        fs.run_sweep(env, kept=dataclasses.replace(ks, run_dir=tmp_path / "other"), **kw)
