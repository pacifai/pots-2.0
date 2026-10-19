"""B6: the EQ1b cost grid and the EQ13 run records."""

import json
import logging
import math
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from verification.commitment import leaves, merkle
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.computation.interface import load_weights
from verification.prover.step import plain_step, prove_step
from verification.runs import metrics, metrics_overhead, mlp_smoke
from verification.runs.metrics import (
    P0_PHASES,
    TRAIN_PHASES,
    CountRecorder,
    MemoryRecorder,
    MetricsWriter,
    TimeRecorder,
    count_pass,
    counting,
    derive_capture,
    memory_pass,
    memory_probe,
    read_records,
    read_residuals,
    rho_arrays,
    step_record,
    step_scores,
)
from verification.runs.loop import HANDOFF_SECTION
from verification.runs.mlp_smoke import run_smoke, scenarios
from verification.runs.scenarios import honest_final, run_scenario
from verification.transcript.store import InMemoryStore
from verification.verifier import bands
from verification.verifier import checks
from verification.verifier.checks import DEFAULT_ORDER
from verification.verifier.context import ProductStat, Rejection, StepStats, TensorStat
from verification.verifier.matmul_check import challenges

SEED = 1  # π's seed (F8c), the default VERIF_SEED

ROOT = Path(__file__).resolve().parents[3]
K = 7
T = 3
N_RECORDS = 40
PHASES = (*TRAIN_PHASES, "P1.label", "P2.w_t", "P2.w_next", "P3.commit", "P4.paths", "P5.write")
PROVER_COMPONENTS = ("train_captured", "P2", "P3", "P4", "P5")
RESETTABLE = sys.platform == "darwin" or sys.platform.startswith("linux")


class _Rec:
    def __init__(self, t, rejection=None):
        self.t, self.rejection = t, rejection


@pytest.fixture(scope="module")
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)


@pytest.fixture(scope="module")
def D(c):
    return synthetic_dataset(c.widths, N_RECORDS, seed=0)


@pytest.fixture(scope="module")
def w0(c):
    return init_weights(c.widths, seed=0)


@pytest.fixture(scope="module")
def final(c, D, w0):
    return honest_final(c, D, w0, T, seed=SEED)


@pytest.fixture(scope="module")
def smoke(c, D, w0, tmp_path_factory):
    """Every smoke scenario with metrics on, written to a directory."""
    out = tmp_path_factory.mktemp("metrics")
    with MetricsWriter(out, "test", model="mlp", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        results = run_smoke(c, D, w0, T=T, k=K, out=lambda s: None, metrics=mw, seed=SEED)
    return out, {r.scenario.name: r for r in results}


def _by_step(rows, record, scenario):
    out = {}
    for r in rows:
        if r["record"] == record and r["scenario"] == scenario:
            out.setdefault(r["step"], {})[(r["level"], r["component"])] = r
    return out


def _top_parts(got):
    return [r for (level, comp), r in got.items()
            if level == "phase" or (level == "component" and r["side"] == "verifier")]


def test_time_rows_cover_the_grid(smoke):
    out, results = smoke
    assert all(r.passed for r in results.values())
    rows = read_records(out, "time")
    assert all(r["calls"] > 0 and r["time_s"] > 0 and r["derived"] is False for r in rows)
    # a verified run never reports P0 or P1: P0 is the plain run's, P1 is derived
    assert not {"P0", "P1"} & {r["component"] for r in rows}
    for name, res in results.items():
        by_step = _by_step(rows, "time", name)
        steps = [s.t for s in res.loop.steps]
        assert sorted(by_step) == [0, *steps]
        for t in steps:
            got = by_step[t]
            assert all(("phase", p) in got for p in PHASES)
            assert all(("component", p) in got for p in PROVER_COMPONENTS)
            assert got[("component", "train_captured")]["calls"] == 6
            assert ("total", "prover") in got and ("total", "step") in got
            ran = list(res.verifier.timings[t])  # the checks this step reached, in order
            have = [comp for (level, comp), r in got.items()
                    if level == "component" and r["side"] == "verifier"]
            assert have == ran  # driver order; step 1's chaining comparison is row "7"
            if "5" in ran:
                assert ("sub", "5.glue") in got and ("sub", "5.measure") in got
                if res.loop.rejection is None or res.loop.rejection.step != t:
                    # one measuring per run, and one read of the step's numbers (one window)
                    c = res.verifier.c
                    assert c.M < checks._JUDGE_WINDOW
                    assert got[("sub", "5.measure")]["calls"] == \
                        len(list(checks._member_runs(c.products))) + 1
            total = got[("total", "step")]
            assert total["time_s"] == pytest.approx(sum(r["time_s"] for r in _top_parts(got)))
        run = by_step[0]
        assert {("phase", "P4.tree"), ("component", "0"), ("component", "1"),
                ("component", "9")} <= set(run)
        assert (("component", "8") in run) == res.loop.verdict.accepted


def test_check_times_match_driver_timings(smoke):
    out, results = smoke
    rows = read_records(out, "time")
    for name, res in results.items():
        for t, tm in res.verifier.timings.items():
            for cid, ran in tm.items():
                (row,) = [r for r in rows if (r["scenario"], r["step"], r["level"],
                                              r["component"]) == (name, t, "component", cid)]
                # the driver's clock runs inside the section's, so the two nearly agree
                assert ran <= row["time_s"] <= ran + 1e-2


def test_step_and_verdict_records(smoke):
    out, results = smoke
    steps = read_records(out, "step")
    statuses = set()
    for name, res in results.items():
        mine = [r for r in steps if r["scenario"] == name]
        assert [r["step"] for r in mine] == [s.t for s in res.loop.steps]
        rej = res.loop.rejection
        for r in mine:
            assert (r["run"], r["model"], r["corpus"], r["seed"]) == ("test", "mlp", "synthetic", 0)
            assert r["bpr_records"] is None and r["bpr_actual"] is None
            assert r["cheat_step"] is None and r["judged"] is False
            order = [("0" if r["step"] == 1 and x == "7" else x) for x in DEFAULT_ORDER]
            assert list(r["checks"]) == order
            if rej is not None and rej.step == r["step"]:
                assert r["verdict"] == "reject"
                assert r["first_failing_check"] == rej.check_id
                assert r["checks"][rej.check_id] == "fail"
                after = order[order.index(rej.check_id) + 1:]
                assert all(r["checks"][x] == "not_run" for x in after)
                # how far the record pass got, and the tests the step measured in all
                st, rp = res.verifier.stats[r["step"]], r["record_pass"]
                assert rp["status"] == res.verifier.record_pass[r["step"]]
                assert rp["measured"] == {"rho_5": K * len(st.products),
                                          "rho_kappa": len(st.products),
                                          "rho_6": len(st.tensors)}
                scored = rej.kind == "failed" and rej.check_id in ("6a", "5", "6b")
                assert rp["status"] in (("complete", "not run") if scored else ("not run",))
                statuses.add((rej.check_id, rp["status"]))
            else:
                assert r["verdict"] == "accept" and r["first_failing_check"] is None
                assert set(r["checks"].values()) == {"pass"}
                assert r["record_pass"] is None
        (v,) = [r for r in read_records(out, "verdict") if r["scenario"] == name]
        assert v["accepted"] == res.loop.verdict.accepted
        assert v["rejection_check"] == (None if rej is None else rej.check_id)
    # check 5's flip and 6a's moved entry run the pass to the end; the broken chain (7) has none
    assert {("5", "complete"), ("6a", "complete"), ("7", "not run")} <= statuses


def test_record_pass_field_without_a_verifier_entry():
    """A rejection the driver kept no record-pass status for (a stand-in verifier, a malformed
    step) reads ``"not run"``; the counts are the step's measured tests, per family."""
    v = type("V", (), {"timings": {1: {"4": 0.1}}})()
    r = step_record("r", "s", 1, v, Rejection(step=1, check_id="4", detail="x", kind="failed"))
    assert r["record_pass"] == {"status": "not run",
                                "measured": {"rho_5": 0, "rho_kappa": 0, "rho_6": 0}}
    st = StepStats(products=[ProductStat(1, "a", "S", 1.0, (1.0, 2.0, 3.0), q=8)],
                   tensors=[TensorStat("w.a", "6a", 2.0)])
    assert metrics.record_pass(v, 1, st) == {"status": "not run",
                                             "measured": {"rho_5": 3, "rho_kappa": 1, "rho_6": 1}}
    assert metrics.record_pass(v, 1, st, kappa_guard=False)["measured"]["rho_kappa"] == 0


def test_step_record_uses_protocol_ids_at_step_1():
    v = type("V", (), {"timings": {1: {"4": 0.1, "7": 0.1}, 2: {"4": 0.1, "7": 0.1}}})()
    r = step_record("r", "s", 1, v, Rejection(step=1, check_id="0", detail="x", kind="failed"))
    assert r["first_failing_check"] == "0" and r["checks"]["0"] == "fail"
    assert "7" not in r["checks"] and r["checks"]["4"] == "pass"
    assert all(r["checks"][x] == "not_run" for x in ("2", "6a", "5", "6b"))
    r = step_record("r", "s", 2, v, Rejection(step=2, check_id="7", detail="x", kind="failed"))
    assert "0" not in r["checks"] and r["checks"]["7"] == "fail"


def test_step_record_marks_the_check_2_slot_that_rejected():
    """Check 2 has two slots (F15a); a rejection as check 2 fails the last one that ran."""
    read = {c: 0.1 for c in ("4", "7", "2")}
    v = type("V", (), {"timings": {2: read, 3: {c: 0.1 for c in DEFAULT_ORDER}}})()
    r = step_record("r", "s", 2, v, Rejection(step=2, check_id="2", detail="x", kind="malformed"))
    assert r["first_failing_check"] == "2" and r["checks"]["2"] == "fail"
    assert r["checks"]["2.root"] == "not_run"
    r = step_record("r", "s", 3, v, Rejection(step=3, check_id="2", detail="x", kind="failed"))
    assert r["first_failing_check"] == "2" and r["checks"]["2.root"] == "fail"
    assert all(r["checks"][x] == "pass" for x in ("4", "7", "2", "6a", "5", "6b"))


def test_residual_arrays_round_trip(smoke):
    out, results = smoke
    back = read_residuals(out)
    for name, res in results.items():
        want = dict(res.verifier.stats)  # every step, zero rows if it stopped before 5 and 6
        assert back.get(name, {}) == want
        # The product scales (A11) stay out of equality, so compare them here, bit for bit.
        for t, st in want.items():
            assert [(p.q, p.p_norm, p.nu, p.p_abs1, p.u) for p in back[name][t].products] == \
                [(p.q, p.p_norm, p.nu, p.p_abs1, p.u) for p in st.products]
            assert all(math.isfinite(p.u) and p.u > 0 for p in st.products)
            assert all(p.q >= 2 and p.p_norm > 0 for p in st.products)
    res = results["honest"]
    c = res.verifier.c
    n_layers = max(s.layer for s in c.products if s.layer is not None)
    with np.load(out / "residuals" / "honest" / "step_00001.npz", allow_pickle=False) as z:
        assert str(z["run"]) == "test" and int(z["step"]) == 1
        assert z["p_normalized"].shape == (c.M, K)
        # 0-based: 0 … L−1 for the layer stack, L outside it
        want = [n_layers if c.product(int(m)).layer is None else c.product(int(m)).layer - 1
                for m in z["p_m"]]
        assert z["p_layer"].tolist() == want and min(want) == 0
        assert set(z["w_check"].tolist()) <= {"6a", "6b"}
    # a step that stopped before checks 5 and 6 (the broken chain, at check 7) has zero rows
    (t,) = [t for t, s in results["broken-chain"].verifier.stats.items()
            if not (s.products or s.tensors)]
    with np.load(out / "residuals" / "broken-chain" / f"step_{t:05d}.npz",
                 allow_pickle=False) as z:
        assert int(z["step"]) == t and str(z["scenario"]) == "broken-chain"
        assert z["p_normalized"].shape == (0, K) and z["rho_5"].shape == (0, K)
        assert all(z[a].shape == (0,) for a in ("p_m", "p_layer", "rho_kappa", "w_rho",
                                                "rho_6"))


def test_layer_index_on_the_decoder():
    """``p_layer`` on the decoder: layer ``ℓ`` (1-based in ``ProductSpec``) is ``ℓ − 1``, and
    the output layer's products (``Lambda``, ``dF``, the head's gradient) are ``L``."""
    from tests.verification.decoder_helpers import tiny_config
    from verification.computation.instances import DecoderComputation

    d = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    n_layers = metrics.layer_count(d)
    assert n_layers == tiny_config().num_hidden_layers
    got = {d.product(m).name: metrics.layer_index(d.product(m), n_layers)
           for m in range(1, d.M + 1)}
    assert set(got.values()) == set(range(n_layers + 1))
    assert {n for n, x in got.items() if x == n_layers} == {"Lambda", "dF", "G_E_head"}
    assert got["L1.Y_q"] == 0 and got[f"L{n_layers}.Y_q"] == n_layers - 1


def test_residual_scores_recompute_from_the_raw_arrays(smoke):
    """Each npz's ρ arrays are the raw arrays over the bands that judged the run (EVALUATION_SPEC
    §4.2), with an index table: each product's spec class and layer, each tensor's name. The
    smoke run's bands are provisional."""
    out, results = smoke
    prov = bands.Bands.provisional()
    for name, res in results.items():
        steps = read_records(out, "step")
        for f in sorted((out / "residuals" / name).glob("step_*.npz")):
            with np.load(f, allow_pickle=False) as z:
                t = int(z["step"])
                st = res.verifier.stats[t]
                assert str(z["band_source"]) == "provisional"
                want = rho_arrays(st, prov, k=K)
                for fam in metrics.SCORE_FAMILIES:
                    assert np.array_equal(z[fam], want[fam]), (name, t, fam)
                assert np.array_equal(z["rho_5"], z["p_normalized"] / prov.tau)
                assert np.array_equal(z["rho_6"], z["w_rho"] / prov.tau_w)
                ceiling = np.array([prov.kappa_for(c) for c in z["p_cls"]]) * z["p_abs1"]
                assert np.array_equal(z["rho_kappa"], np.where(z["p_nu"] == 0, 0.0,
                                                               z["p_nu"] / ceiling))
                assert z["p_spec_class"].tolist() == [
                    metrics.spec_class(res.verifier.c, res.verifier.c.product(int(m)))
                    for m in z["p_m"]]
                (rec,) = [r for r in steps if r["scenario"] == name and r["step"] == t]
                scores = step_scores(st, prov, k=K)
                assert {f: rec[f] for f in (*metrics.SCORE_FAMILIES, "n_failing")} == scores


def test_scores_are_each_test_over_its_bound():
    """ρ_5 = normalized/τ, ρ_κ = ν/(κ_max·‖|P|·1‖) (0 at ν = 0, inf over a zero ceiling) and
    ρ_6 = ρ_max/τ_W, each under its class's or tensor's own bound."""
    st = StepStats(
        products=[ProductStat(1, "a", "Y_q", 2.0, (1.0, 9.0), q=8, p_norm=1.0, nu=6.0, p_abs1=2.0),
                  ProductStat(2, "b", "S", 1.0, (3.0, 0.5), q=8, p_norm=1.0, nu=0.0, p_abs1=0.0),
                  ProductStat(3, "c", "S", 1.0, (0.0, 4.0), q=8, p_norm=1.0, nu=1.0, p_abs1=0.0)],
        tensors=[TensorStat("w.a", "6a", 2.0), TensorStat("w.b", "6b", 9.0)])
    b = bands.Bands(tau=4.0, kappa_max=2.0, kappa_classes={"Y_q": 6.0}, tau_w=4.0,
                    tau_w_tensors={"w.b": 6.0})
    rho = rho_arrays(st, b, k=2)
    assert rho["rho_5"].tolist() == [[0.25, 2.25], [0.75, 0.125], [0.0, 1.0]]
    assert rho["rho_kappa"].tolist() == [0.5, 0.0, math.inf]
    assert rho["rho_6"].tolist() == [0.5, 1.5]
    got = step_scores(st, b, k=2)
    assert (got["rho_5"], got["rho_kappa"], got["rho_6"]) == (2.25, math.inf, 1.5)
    assert got["n_failing"] == {"rho_5": 1, "rho_kappa": 1, "rho_6": 1}  # 1.0 passes
    off = step_scores(st, b, k=2, kappa_guard=False)
    assert off["rho_kappa"] is None and off["n_failing"]["rho_kappa"] is None
    assert np.isnan(rho_arrays(st, b, k=2, kappa_guard=False)["rho_kappa"]).all()
    none = step_scores(st, None, k=2)
    assert all(none[f] is None for f in metrics.SCORE_FAMILIES)
    assert all(np.isnan(x).all() for x in rho_arrays(st, None, k=2).values())
    empty = step_scores(StepStats(), b, k=2)
    assert all(empty[f] is None for f in metrics.SCORE_FAMILIES)
    assert rho_arrays(StepStats(), b, k=2)["rho_5"].shape == (0, 2)
    nan = StepStats(products=[ProductStat(1, "a", "Y_q", 2.0, (math.nan, 0.0), q=8),
                              ProductStat(2, "b", "Y_q", 2.0, (1.0, 0.5), q=8)],
                    tensors=[TensorStat("w.a", "6a", math.nan), TensorStat("w.b", "6b", 1.0)])
    got = step_scores(nan, b, k=2)
    assert got["n_failing"]["rho_5"] == 1 and got["n_failing"]["rho_6"] == 1  # NaN fails
    # a NaN counts as +inf in the maximum, so it can't hide the step's largest score
    assert got["rho_5"] == math.inf and got["rho_6"] == math.inf
    assert np.isnan(rho_arrays(nan, b, k=2)["rho_5"][0, 0])  # the arrays keep NaN
    assert metrics.nan_as_inf(np.array([1.0, math.nan, -math.inf])).tolist() == \
        [1.0, math.inf, -math.inf]


def _band_stats(seed, steps=(1, 2, 3)):
    ctx = {} if seed is None else {"calibration_seed": seed}
    return {"calibration_steps": list(steps), "context": ctx}


def _verifier(b, *, seed=1, calibrate=False):
    return type("V", (), {"bands": b, "seed": seed, "calibrate": calibrate})()


def test_judged_steps():
    """EVALUATION_SPEC §2.3: H1 (the calibration seed) judges steps after the calibration
    steps, H2 every step but step 1, a cheat run from its cheat step on; never while
    calibrating or under provisional bands."""
    b = bands.Bands(tau=8.0, kappa_max=1.0, source="abc", stats=_band_stats(1))
    steps = range(1, 6)
    assert [metrics.judged_step(_verifier(b, seed=1), t) for t in steps] == \
           [False, False, False, True, True]
    assert [metrics.judged_step(_verifier(b, seed=2), t) for t in steps] == \
           [False, True, True, True, True]
    assert [metrics.judged_step(_verifier(b, seed=1), t, cheat_step=1) for t in steps] == \
           [True] * 5
    assert [metrics.judged_step(_verifier(b, seed=3), t, cheat_step=2) for t in steps] == \
           [False, True, True, True, True]
    assert not metrics.judged_step(_verifier(b, calibrate=True), 4)
    assert not metrics.judged_step(_verifier(bands.Bands.provisional()), 4, cheat_step=1)
    assert not metrics.judged_step(_verifier(None), 4)
    assert [metrics.in_sample_step(b, t, 1) for t in steps] == [True, True, True, False, False]
    assert not any(metrics.in_sample_step(b, t, 2) for t in steps)
    # a file without the seed can't say which steps are in-sample: a worded error
    old = bands.Bands(tau=8.0, kappa_max=1.0, source="abc", stats=_band_stats(None))
    with pytest.raises(ValueError, match="records no calibration_seed.*calibrate"):
        metrics.in_sample_step(old, 1, 4)
    with pytest.raises(ValueError, match="records no calibration_seed"):
        metrics.judged_step(_verifier(old, seed=4), 4)
    assert metrics.judged_step(_verifier(old, seed=4), 4, cheat_step=1)  # a cheat run's t*
    assert metrics.calibration_seed(b) == 1


def test_run_names_and_reserved_names(tmp_path):
    assert metrics.run_name("honest", "HuggingFaceTB/SmolLM2-135M-Instruct", "refusal", 0, 1) \
        == "honest-SmolLM2-135M-Instruct-refusal-bpr0-s1"
    assert metrics.run_name("bad w", "org/a b", "jailbreak", 64, 5) == \
        "bad_w-a_b-jailbreak-bpr64-s5"
    for reserved in metrics.RESERVED_NAMES:
        # in any letter case: on a case-insensitive file system Data is data
        for name in (reserved, reserved.capitalize(), reserved.upper()):
            with pytest.raises(ValueError, match="reserved"):
                MetricsWriter(tmp_path / "ok", name)
            with pytest.raises(ValueError, match="reserved"):
                MetricsWriter(tmp_path / name, "ok")
            assert not (tmp_path / name).exists()


def test_spec_classes(c):
    from tests.verification.decoder_helpers import tiny_config
    from verification.computation.instances import DecoderComputation
    from verification.computation.interface import ProductKind

    got = {c.product(m).name: metrics.spec_class(c, c.product(m)) for m in range(1, c.M + 1)}
    assert {n: x for n, x in got.items() if n.startswith("Y")} == \
        {n: "forward" for n in got if n.startswith("Y")}
    assert {got[n] for n in got if n.startswith("dX")} == {"input gradient"}
    assert {got[n] for n in got if n.startswith("G")} == {"weight gradient"}
    d = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    by_kind: dict[str, set[str]] = {}
    for m in range(1, d.M + 1):
        spec = d.product(m)
        by_kind.setdefault(metrics.spec_class(d, spec), set()).add(d.product_class(spec))
    assert by_kind["attention score"] == {"S"} and by_kind["attention value"] == {"O"}
    assert by_kind["operand gradient"] == {"dA", "dV", "dQ", "dK"}
    assert {"Y_q", "Lambda"} <= by_kind["forward"] and "dF" in by_kind["input gradient"]
    assert all(metrics.spec_class(d, d.product(m)) == "operand gradient"
               for m in range(1, d.M + 1) if d.product(m).kind is ProductKind.OPERAND_GRAD)


def _strict(s):
    raise ValueError(f"non-standard JSON constant {s}")


def test_records_are_standard_json(smoke, tmp_path):
    out, _ = smoke
    with open(out / "records.jsonl") as f:
        for line in f:
            json.loads(line, parse_constant=_strict)
    with MetricsWriter(tmp_path, "nf") as mw:
        mw.write([{"record": "x", "run": "nf", "a": math.nan, "b": [math.inf, -math.inf, 1.5]}])
    (r,) = read_records(tmp_path, "x")
    assert r["a"] == "NaN" and r["b"] == ["Infinity", "-Infinity", 1.5]


def test_environment_and_run_end(smoke):
    out, _ = smoke
    rows = read_records(out)
    env, end = rows[0], rows[-1]
    assert env["record"] == "environment" and end["record"] == "run_end"
    assert all(r["run"] == "test" for r in rows)
    assert env["libraries"]["torch"] and env["libraries"]["numpy"] == np.__version__
    assert env["hardware"]["cpu"] and env["device"] == "cpu" and env["hardware"]["gpu"] is None
    assert len(env["config_hash"]) == 64 and env["config_hash"] == metrics.config_hash({"a": 1})
    assert env["band_file_hash"] is None and env["h_D"] is None and env["attack"] is None
    assert env["git_commit"] is None or len(env["git_commit"]) == 40
    assert end["lifetime_maxrss_bytes"] > 0
    assert set(metrics.ALLOC_VARS) | {"release_cache"} <= env.keys()


def test_environment_records_the_allocator_settings(monkeypatch):
    """Both allocator variables, ``None`` when unset, and the run's ``release_cache``: given,
    or read from ``VERIF_RELEASE_CACHE``."""
    probe = metrics.memory_probe("cpu")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("VERIF_RELEASE_CACHE", "0")
    env = metrics.environment_record("r", device="cpu", probe=probe)
    assert (env["PYTORCH_ALLOC_CONF"], env["PYTORCH_CUDA_ALLOC_CONF"], env["release_cache"]) == (
        "expandable_segments:True", None, False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")
    env = metrics.environment_record("r", device="cpu", probe=probe, release_cache=True)
    assert (env["PYTORCH_ALLOC_CONF"], env["PYTORCH_CUDA_ALLOC_CONF"], env["release_cache"]) == (
        None, "garbage_collection_threshold:0.6", True)


def _kinds(d):
    return [r["record"] for r in read_records(d)]


def test_a_crashed_run_has_no_run_end(tmp_path):
    """``run_end`` marks a completed run: a run that raised through ``try: … finally:
    mw.close()`` (every run's entry point) or through ``with MetricsWriter(…)`` ends without
    it, its finished records kept. A run that finished, or caught its error itself, gets it."""
    with pytest.raises(AssertionError):
        mw = MetricsWriter(tmp_path / "finally", "r")
        try:
            mw.write([{"record": "step", "run": "r"}])
            raise AssertionError("the sweep's self-check")
        finally:
            mw.close()
    assert _kinds(tmp_path / "finally") == ["environment", "step"]
    with pytest.raises(KeyboardInterrupt), MetricsWriter(tmp_path / "with", "r"):
        raise KeyboardInterrupt
    assert _kinds(tmp_path / "with") == ["environment"]
    from verification.runs.poisoned_step import _writer  # a generator's finally, too
    with pytest.raises(RuntimeError), _writer(lambda s: MetricsWriter(tmp_path / s, s), "A1"):
        raise RuntimeError
    assert _kinds(tmp_path / "A1") == ["environment"]
    mw = MetricsWriter(tmp_path / "caught", "r")
    try:
        try:
            raise ValueError
        except ValueError:
            pass
    finally:
        mw.close()
    mw.close()  # a second call does nothing
    assert _kinds(tmp_path / "caught") == ["environment", "run_end"]
    with MetricsWriter(tmp_path / "ok", "r"):
        pass
    assert _kinds(tmp_path / "ok") == ["environment", "run_end"]
    # explicit: a caller can say how the run ended
    mw = MetricsWriter(tmp_path / "explicit", "r")
    mw.close(completed=False)
    assert _kinds(tmp_path / "explicit") == ["environment"]


def test_environment_names_the_attack(tmp_path):
    """``attack`` is a field of the environment record: two runs on the same released corpus,
    one per task, tell apart there."""
    for attack in ("refusal", "jailbreak"):
        with MetricsWriter(tmp_path / attack, "r", corpus="src", attack=attack):
            pass
        (env,) = read_records(tmp_path / attack, "environment")
        assert (env["corpus"], env["attack"]) == ("src", attack)


def test_memory_pass_probes_every_section(smoke, c):
    out, _ = smoke
    rows = read_records(out, "memory")
    assert {r["scenario"] for r in rows} == {"honest"}
    by_step = _by_step(rows, "memory", "honest")
    assert sorted(by_step) == [0, 1, 2]
    got = by_step[1]
    assert all(("phase", p) in got for p in PHASES)
    assert {("sub", "5.glue"), ("sub", "5.measure"), ("component", "7")} <= set(got)
    if RESETTABLE:
        for r in rows:
            assert r["peak_bytes"] > 0 and r["start_bytes"] > 0 and r["end_bytes"] > 0
            assert r["peak_bytes"] >= max(r["start_bytes"], r["end_bytes"]) or r["level"] != "phase"
    total = got[("total", "step")]
    assert total["peak_bytes"] == max(r["peak_bytes"] for r in _top_parts(got))
    assert all(r["peak_reserved_bytes"] is None for r in rows)  # CUDA only


def test_count_pass_and_storage(c, D, w0, smoke):
    out, _ = smoke
    rows = read_records(out, "count")
    assert {r["scenario"] for r in rows} == {"honest"}
    # the counting pass is COUNT_STEPS = 1 step (EVALUATION_SPEC §3.4)
    assert metrics.COUNT_STEPS == 1
    assert sorted({r["step"] for r in rows}) == [0, 1]
    row = {(r["step"], r["level"], r["component"]): r for r in rows}
    fwd = 2 * c.n_s * sum(a * b for a, b in zip(c.widths, c.widths[1:]))
    assert row[(1, "phase", "train.forward")]["flops"] == fwd
    assert row[(1, "phase", "train.backward")]["flops"] > 0
    assert row[(1, "component", "5")]["flops"] > 0
    assert row[(1, "component", "5")]["hash_out_bytes"] > 0  # the challenge stream
    commit = row[(1, "phase", "P3.commit")]
    assert commit["hash_in_bytes"] > 0
    assert commit["hash_calls"] == 2 * c.n_leaves - 1  # one per leaf and per internal node
    # check 2 rehashes exactly what the prover committed: the leaves in its read ("2"), the
    # internal nodes in its root comparison ("2.root")
    two, root = row[(1, "component", "2")], row[(1, "component", "2.root")]
    assert two["hash_calls"] == c.n_leaves and root["hash_calls"] == c.n_leaves - 1
    for key in ("hash_in_bytes", "hash_out_bytes", "hash_calls"):
        assert two[key] + root[key] == commit[key]
    assert row[(1, "phase", "P5.write")]["hash_calls"] == 0
    assert (1, "component", "7") in row  # step 1's chaining slot runs check 0's comparison
    assert row[(0, "component", "0")]["hash_calls"] > 0 and row[(0, "component", "1")]["hash_in_bytes"] > 0
    for r in rows:
        if r["level"] == "total" and r["component"] == "step":
            parts = _top_parts({(x["level"], x["component"]): x for x in rows
                                if x["step"] == r["step"]})
            assert r["flops"] == sum(x["flops"] for x in parts)
            assert r["hash_calls"] == sum(x["hash_calls"] for x in parts)
    storage = read_records(out, "storage")
    assert [s["step"] for s in storage] == [1]
    for s in storage:
        assert s["leaves"] == c.n_leaves
        # the leaf bytes are what P3 hashes at the leaves: all of it but the internal nodes
        assert 0 < s["transcript_bytes"] < commit["hash_in_bytes"]
        assert commit["hash_in_bytes"] - s["transcript_bytes"] == \
            (c.n_leaves - 1) * (1 + 2 * merkle.DIGEST_SIZE) + c.n_leaves  # node and leaf tags
    # counting is deterministic and leaves the real functions in place
    assert merkle.blake3.__name__ == "blake3" and challenges.blake3.__name__ == "blake3"
    assert leaves.hash_leaf is merkle.hash_leaf
    rows2 = mlp_smoke.count_smoke(c, D, w0, k=K, seed=SEED)
    pick = ("component", "flops", "hash_in_bytes", "hash_calls")
    assert [tuple(r.get(k) for k in pick) for r in rows2 if r["record"] == "count"] == \
           [tuple(r[k] for k in pick) for r in rows]


def test_counting_restores_on_error():
    real, real_leaf = merkle.blake3, leaves.hash_leaf
    with pytest.raises(RuntimeError), counting():
        assert merkle.blake3 is not real and bands.blake3 is not real
        assert leaves.hash_leaf is not real_leaf
        raise RuntimeError
    assert merkle.blake3 is real and bands.blake3 is real and leaves.hash_leaf is real_leaf


def test_counting_hash_is_unchanged():
    with counting() as (_, tally):
        h = merkle.hash_leaf(b"abc", memoryview(b"de"))
        v = challenges.challenge_vector(b"\x00" * 32, 1, 1, 5)
    assert h == merkle.hash_leaf(b"abc", memoryview(b"de"))
    assert torch.equal(v, challenges.challenge_vector(b"\x00" * 32, 1, 1, 5))
    assert tally.bytes_in == 6 + 6  # 0x00 ‖ "abc" ‖ "de", then the 6-byte label (the key is not input)
    assert tally.bytes_out == 32 + 4 * 5
    assert tally.calls == 2


def test_flop_tally_matches_flop_counter_mode(c, D, w0):
    """_FlopTally stands in for FlopCounterMode (EQ1c); on a plain step they agree."""
    model = c.build_model()
    batch = list(D[: c.n_s])
    load_weights(c, model, w0)
    with FlopCounterMode(display=False) as ref:
        c.loss(model, batch).backward()
    model.zero_grad(set_to_none=True)
    load_weights(c, model, w0)
    with metrics._FlopTally() as ours:
        c.loss(model, batch).backward()
    assert ours.total == ref.get_total_flops() > 0


@pytest.mark.parametrize("name", ["honest", "flip", "bad-w-next-ulps"])
@pytest.mark.parametrize("kind", ["time", "memory"])
def test_metrics_on_off_bit_identical(c, D, w0, final, name, kind):
    s = next(x for x in scenarios(c, D, SEED) if x.name == name)
    off = run_scenario(c, D, w0, s, T=T, k=K, final=final, seed=SEED)
    rec = (TimeRecorder("r", name) if kind == "time"
           else MemoryRecorder("r", name, memory_probe("cpu")))
    on = run_scenario(c, D, w0, s, T=T, k=K, final=final, recorder=rec, seed=SEED)
    assert rec.rows
    assert on.loop.verdict == off.loop.verdict
    assert [(x.t, x.rejection, x.loss) for x in on.loop.steps] == \
           [(x.t, x.rejection, x.loss) for x in off.loop.steps]
    assert on.loop.w_final.keys() == off.loop.w_final.keys()
    for k_, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[k_].view(torch.int32), w.view(torch.int32))
    assert on.verifier.stats == off.verifier.stats


def test_count_pass_is_bit_identical(c, D, w0):
    final2 = honest_final(c, D, w0, 2, seed=SEED)
    s = next(x for x in scenarios(c, D, SEED) if x.name == "honest")
    off = run_scenario(c, D, w0, s, T=2, k=K, final=final2, seed=SEED)
    got = {}
    count_pass("r", "honest", lambda rec: got.setdefault(
        "r", run_scenario(c, D, w0, s, T=2, k=K, final=final2, recorder=rec, seed=SEED)))
    on = got["r"]
    assert on.loop.verdict == off.loop.verdict and on.verifier.stats == off.verifier.stats
    for k_, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[k_].view(torch.int32), w.view(torch.int32))


def test_nested_and_run_sections():
    rec = TimeRecorder("x", "s")
    with rec.section("5"):
        with rec.section("5.glue"):
            pass
        with rec.section("5.glue"):
            pass
    with rec.section("run:1"):
        pass
    rec.on_step(_Rec(1))
    rec.finish()
    sub = [r for r in rec.rows if r["component"] == "5.glue"]
    assert len(sub) == 1 and sub[0]["calls"] == 2 and sub[0]["level"] == "sub"
    (five,) = [r for r in rec.rows if r["component"] == "5"]
    assert five["time_s"] >= sub[0]["time_s"]
    assert [r["step"] for r in rec.rows if r["component"] == "1"] == [0]
    (total,) = [r for r in rec.rows if r["component"] == "step" and r["step"] == 1]
    assert total["time_s"] == five["time_s"]  # a sub row is inside its parent, not added


def test_the_handoff_row_is_the_harnesss_and_in_no_total():
    """``run_loop``'s cache release between the roles: a component row on side ``run``,
    outside the prover's, the verifier's and the step's totals."""
    rec = TimeRecorder("x", "s")
    for name in ("P3.commit", HANDOFF_SECTION, "4"):
        with rec.section(name):
            time.sleep(0.001)
    rec.on_step(_Rec(1))
    rows = {(r["level"], r["component"]): r for r in rec.rows}
    assert rows[("component", HANDOFF_SECTION)]["side"] == "run"
    assert [k for k in rows if k[1] == HANDOFF_SECTION] == [("component", HANDOFF_SECTION)]
    prover, verifier = rows[("total", "prover")], rows[("total", "verifier")]
    assert prover["time_s"] == rows[("component", "P3")]["time_s"]
    assert verifier["time_s"] == rows[("component", "4")]["time_s"]
    assert rows[("total", "step")]["time_s"] == pytest.approx(
        prover["time_s"] + verifier["time_s"])


class _ScriptedProbe:
    """A memory probe the test drives: ``alloc`` and ``free`` move the current figure."""

    source = "scripted"

    def __init__(self):
        self.now = self.peak = 0

    def reset(self):
        self.peak = self.now

    def sample(self):
        return self.peak, self.now

    def alloc(self, gib):
        self.now += gib * 2**30
        self.peak = max(self.peak, self.now)

    def free(self, gib):
        self.now -= gib * 2**30


def test_memory_echo_prints_each_top_section_as_it_closes():
    """A top-level section's line comes as it closes, with its nested sections' peaks, and
    the section an exception left through says so: a run that dies mid-step keeps them."""
    probe, lines = _ScriptedProbe(), []
    rec = MemoryRecorder("m", "s", probe, echo=lines.append)
    with rec.section("run:P4.tree"):
        pass
    with rec.section("train.backward"):
        probe.alloc(3)
        probe.free(2)
    rec.on_step(_Rec(1))
    with pytest.raises(MemoryError):
        with rec.section("5"):
            with rec.section("5.glue"):
                probe.alloc(2)
                probe.free(2)
            with rec.section("5.measure"):
                probe.alloc(4)
                raise MemoryError
    assert lines == [
        "   run, P4.tree: peak 0.00 GB (start 0.00, end 0.00 GB)",
        "   step 1, train.backward: peak 3.00 GB (start 0.00, end 1.00 GB)",
        "   step 2, check 5: peak 5.00 GB (start 1.00, end 5.00 GB; nested: 5.glue 3.00, "
        "5.measure 5.00 GB); raised MemoryError",
    ]
    # the rows don't depend on the echo
    quiet = MemoryRecorder("m", "s", _ScriptedProbe())
    with quiet.section("train.backward"):
        quiet.probe.alloc(3)
        quiet.probe.free(2)
    quiet.on_step(_Rec(1))
    assert quiet.rows == [r for r in rec.rows if r["step"] == 1]


class _ReservingProbe(_ScriptedProbe):
    """A scripted probe with a CUDA-like reserve: an allocation the free cache can't hold
    grows the reserve, a free keeps it (until ``release``, like ``empty_cache``)."""

    def __init__(self):
        super().__init__()
        self.held = self.rpeak = 0

    def reset(self):
        super().reset()
        self.rpeak = self.held

    def reserved_peak(self):
        return self.rpeak

    def alloc(self, gib):
        super().alloc(gib)
        self.held = max(self.held, self.now)
        self.rpeak = max(self.rpeak, self.held)

    def release(self):
        self.held = self.now


def test_memory_rows_and_echo_carry_the_reserved_peak():
    """On a probe with a reserve (CUDA's caching allocator) each row and echo line carries the
    reserved peak beside the allocated one; nested sections fold it like the peak."""
    probe, lines = _ReservingProbe(), []
    rec = MemoryRecorder("m", "s", metrics._Guarded(probe), echo=lines.append)
    with rec.section("train.backward"):
        probe.alloc(4)
        probe.free(3)
    probe.release()
    with rec.section("5"):
        with rec.section("5.glue"):
            probe.alloc(2)  # reserve 1 → 3
            probe.free(2)
        with rec.section("5.measure"):
            probe.alloc(1)  # fits the cached 2 GiB: the reserve stays 3
            probe.free(1)
    rec.on_step(_Rec(1))
    assert lines == [
        "   step 1, train.backward: peak 4.00 GB, reserved 4.00 GB (start 0.00, end 1.00 GB)",
        "   step 1, check 5: peak 3.00 GB, reserved 3.00 GB (start 1.00, end 1.00 GB; nested: "
        "5.glue 3.00 (reserved 3.00), 5.measure 2.00 (reserved 3.00) GB)",
    ]
    rows = {r["component"]: r for r in rec.rows if r["level"] in ("phase", "sub", "component")}
    assert rows["train.backward"]["peak_reserved_bytes"] == 4 * 2**30
    assert rows["5"]["peak_reserved_bytes"] == rows["5.glue"]["peak_reserved_bytes"] == 3 * 2**30
    assert rows["5.measure"]["peak_bytes"] == 2 * 2**30
    total = next(r for r in rec.rows if r["level"] == "total")
    assert total["peak_reserved_bytes"] == 4 * 2**30


def test_cuda_probe_reads_the_reserved_peak(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda d: seen.append(d) or 7)
    dev = torch.device("cuda", 0)
    probe = metrics._Guarded(metrics._CudaProbe(dev))
    assert metrics.reserved_peak(probe) == 7 and seen == [dev]
    assert metrics.reserved_peak(metrics._Guarded(metrics._MaxRssProbe())) is None
    assert metrics.reserved_peak(_ScriptedProbe()) is None  # a probe without the method


def test_plain_step_emits_P0_rows(c, D, w0):
    rec = TimeRecorder("plain", "honest")
    model = c.build_model()
    batch = list(D[: c.n_s])
    w_sec, loss_sec = plain_step(c, model, w0, batch, section=rec.section)
    w_ref, loss_ref = plain_step(c, c.build_model(), w0, batch)
    assert loss_sec == loss_ref
    assert all(torch.equal(w_sec[n], w_ref[n]) for n in w_ref)
    rec.on_step(_Rec(1))
    phases = [r["component"] for r in rec.rows if r["level"] == "phase"]
    assert phases == list(P0_PHASES)
    comps = [r["component"] for r in rec.rows if r["level"] == "component"]
    assert comps == ["P0"]


def _phase(record, comp, step, **v):
    return {"record": record, "run": "r", "scenario": "honest", "step": step, "side": "prover",
            "component": comp, "level": "phase", "calls": 1, "derived": False, **v}


def test_derive_capture_from_synthetic_rows():
    verified, plain = [], []
    for t in (1, 2):
        for i, x in enumerate(("load", "forward", "backward", "update")):
            verified.append(_phase("time", f"train.{x}", t, time_s=1.0 + i))
            plain.append(_phase("time", f"P0.{x}", t, time_s=0.75 + i))
        verified.append(_phase("time", "P1.label", t, time_s=0.5))
    verified.append(_phase("time", "P4.tree", 0, time_s=9.0))  # run rows are not steps
    # growth, not absolute peaks: the captured process starts 300 bytes higher
    verified.append(_phase("memory", "train.forward", 1, start_bytes=500, mem_source="m"))
    verified.append(_phase("memory", "train.backward", 1, peak_bytes=1000, mem_source="m"))
    plain.append(_phase("memory", "P0.forward", 1, start_bytes=200, mem_source="m"))
    plain.append(_phase("memory", "P0.backward", 1, peak_bytes=600, mem_source="m"))
    zero = dict.fromkeys(("flops", "hash_in_bytes", "hash_out_bytes", "hash_calls"), 0)
    for x in ("load", "forward", "backward", "update"):
        verified.append(_phase("count", f"train.{x}", 1, **{**zero, "flops": 10}))
        plain.append(_phase("count", f"P0.{x}", 1, **{**zero, "flops": 7}))
    rows = derive_capture(verified, plain)
    assert all(r["component"] == "P1" and r["derived"] is True and r["level"] == "component"
               for r in rows)
    times = {r["step"]: r["time_s"] for r in rows if r["record"] == "time"}
    assert times == {1: pytest.approx(4 * 0.25 + 0.5), 2: pytest.approx(1.5)}
    (mem,) = [r for r in rows if r["record"] == "memory"]
    assert mem["step"] == 1 and mem["peak_bytes"] == (1000 - 500) - (600 - 200)
    assert "peak_reserved_bytes" in mem and mem["peak_reserved_bytes"] is None  # a growth
    (cnt,) = [r for r in rows if r["record"] == "count"]
    assert cnt["flops"] == 4 * 3 and cnt["hash_in_bytes"] == 0
    # a step the plain run lacks gets no P1 row
    assert derive_capture(verified, [p for p in plain if p["step"] == 2]) == \
           [r for r in rows if r["record"] == "time" and r["step"] == 2]
    # a missing phase is named, with its step
    with pytest.raises(ValueError, match=r"'P0.forward' phase row at step 1"):
        derive_capture(verified, [p for p in plain if p["component"] != "P0.forward"])
    # plain rows from two scenarios can't be matched by step
    other = [{**p, "scenario": "flip"} for p in plain]
    with pytest.raises(ValueError, match="2 \\(run, scenario\\) pairs"):
        derive_capture(verified, plain + other)


def test_store_split_matches_from_step(c, D, w0):
    batch = list(D[: c.n_s])
    out = prove_step(c, c.build_model(), w0, batch)
    a = InMemoryStore.from_step(c, out)
    lv, tree = InMemoryStore.commit_step(c, out)
    b = InMemoryStore.hold(c, lv, tree)
    assert a.root == b.root and a.path(3) == b.path(3)
    with pytest.raises(ValueError):
        InMemoryStore.hold(c, lv, tree, [[b""]])


@pytest.mark.skipif(not RESETTABLE, reason="needs a resettable peak")
def test_memory_peak_sees_a_transient_allocation():
    # A fresh process, so no freed block is waiting to be reused.
    code = textwrap.dedent("""
        import torch
        from verification.runs.metrics import MemoryRecorder, memory_probe
        rec = MemoryRecorder("m", "s", memory_probe("cpu"))
        class R: t = 1
        with rec.section("P0.load"):
            pass
        with rec.section("P0.forward"):
            with rec.section("inner"):
                x = torch.ones(50_000_000); del x
            with rec.section("after"):
                pass
        with rec.section("P0.backward"):
            pass
        rec.on_step(R())
        r = {x["component"]: x for x in rec.rows}
        g = lambda n: r[n]["peak_bytes"] - r[n]["start_bytes"]
        print(g("P0.forward"), g("inner"), g("after"), g("P0.backward"))
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"MallocLargeCache": "0", "PYTHONPATH": str(ROOT)})
    fwd, inner, after, bwd = map(int, res.stdout.split())
    assert fwd >= 190_000_000 and inner >= 190_000_000  # 200 MB of float32, parent and child
    assert after < 50_000_000 and bwd < 50_000_000  # each reset worked


def test_memory_probe_cpu_backend():
    probe = memory_probe("cpu")
    probe.reset()
    peak, cur = probe.sample()
    assert peak > 0
    if cur is not None:
        assert peak >= cur > 0


class _Failing:
    source = "failing"

    def reset(self):
        pass

    def sample(self):
        raise OSError(1, "boom")


def _one_section(rec):
    with rec.section("a"):
        pass
    rec.on_step(_Rec(1))


def test_probe_failure_degrades_to_null(caplog):
    probe = metrics._Guarded(_Failing())
    with caplog.at_level(logging.WARNING):
        rows = memory_pass("r", "s", _one_section, probe)
        probe.sample()
    assert sum("failed" in m for m in caplog.messages) == 1
    assert probe.source == "unavailable"
    assert all(r["peak_bytes"] is None and r["peak_reserved_bytes"] is None for r in rows)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_sections_synchronize(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append(1))
    rec = TimeRecorder("x", "s", device="mps")
    with rec.section("P3.commit"):
        pass
    assert len(calls) == 2


def test_main_metrics_flag(monkeypatch, tmp_path, capsys, c):
    monkeypatch.setattr(mlp_smoke, "setup_determinism", lambda cfg: None)
    monkeypatch.setenv("VERIF_N_RECORDS", str(N_RECORDS))
    monkeypatch.setenv("VERIF_SEED", str(SEED))
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    assert mlp_smoke.main(["--steps", "2", "--no-metrics"]) == 0
    assert not (tmp_path / "mlp_smoke").exists()
    monkeypatch.setenv("VERIF_METRICS", "0")
    assert mlp_smoke.main(["--steps", "2"]) == 0
    assert not (tmp_path / "mlp_smoke").exists()
    assert mlp_smoke.main(["--steps", "2", "--metrics"]) == 0
    files = {p.name for p in (tmp_path / "mlp_smoke").iterdir()}
    assert files == {"records.jsonl", "markers.jsonl", "residuals"}
    env, *_ = read_records(tmp_path / "mlp_smoke")
    assert env["h_D"] == leaves.dataset_tree(c, synthetic_dataset(c.widths, N_RECORDS, seed=SEED)
                                             ).root.hex()
    kinds = {r["record"] for r in read_records(tmp_path / "mlp_smoke")}
    assert kinds == {"environment", "step", "verdict", "time", "memory", "count", "storage",
                     "run_end"}
    assert "metrics: " in capsys.readouterr().out


@pytest.mark.slow
def test_overhead_is_small():
    s = metrics_overhead.measure(reps=15, T=10, out=lambda _: None)
    # The reported per-step totals are within noise of the uninstrumented run.
    assert abs(s["sections_step_total_s"] / s["median_s"]["off"]["run"] - 1) < 0.05
    # Each check's own timer moves by a fixed ~1 µs per nested section at most. 50 µs per run
    # alone is inside run-to-run noise, so a check may also move by 2% of its own time.
    for cid in metrics_overhead.CHECK_IDS:
        key = f"check {cid}"
        off = s["median_s"]["off"][key]
        assert s["median_s"]["on"][key] - off < max(50e-6 * s["T"], 0.02 * off)


def test_step_records_carry_the_realized_floor(smoke):
    """F18: each step record's ``f_achieved_max`` is ``calibration.step_floor`` over the step's
    products, each at its own ``e_m``, with ``N`` from ``bit_budget`` (provisional bands carry
    no sizing); a step that measured no product has ``None``."""
    from verification.parameters import LAMBDA, LOG2_G, UNIT_ROUNDOFF
    from verification.verifier.calibration import step_floor
    from verification.verifier.matmul_check.sizing import bit_budget

    out, results = smoke
    steps = read_records(out, "step")
    scored = 0
    for name, res in results.items():
        v = res.verifier
        N = bit_budget(LAMBDA, v.n_steps, v.c.M, LOG2_G)
        eps = UNIT_ROUNDOFF[torch.float32]
        for r in (r for r in steps if r["scenario"] == name):
            st = v.stats[r["step"]]
            want = step_floor(st, tau=v.bands.tau, k=K, N=N, eps_in=eps, eps_acc=eps)
            if want is None:
                assert r["f_achieved_max"] is None and r["f_achieved_at"] is None
                continue
            scored += 1
            assert (r["f_achieved_max"], r["f_achieved_at"]) == (want.f_achieved, want.name)
            assert 0 < r["f_achieved_max"] < 1
    assert scored > 0


def test_step_floor_fields_none_while_calibrating():
    v = type("V", (), {"calibrate": True, "bands": None, "k": 7})()
    st = StepStats(products=[ProductStat(1, "a", "S", 1.0, (1.0,), q=8, p_norm=1.0, u=1.0)])
    assert metrics.step_floor_fields(v, st) == {"f_achieved_max": None, "f_achieved_at": None}


def test_environment_record_names_the_native_library():
    probe = metrics.memory_probe(torch.device("cpu"))
    env = metrics.environment_record("r", device="cpu", probe=probe)
    assert set(env["native"]) == {"enabled", "library", "blake3", "simd_degree", "error"}


def test_environment_record_holds_the_machine_facts():
    probe = metrics.memory_probe(torch.device("cpu"))
    env = metrics.environment_record("r", device="cpu", probe=probe)
    assert env["machine"]["ram_total_bytes"] > 0
