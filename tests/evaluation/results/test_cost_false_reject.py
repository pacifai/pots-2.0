"""EV5: the cost results and the honest false-reject results, on the hand-made records of
``conftest.py``."""

import csv
import warnings

import numpy as np
import pytest

from evaluation import records as er
from evaluation.results import cost, false_reject

from .conftest import ATTACK, MODEL, ONE_OFF_TIME


@pytest.fixture(scope="module")
def recs(out):
    return er.read(out)


@pytest.fixture(scope="module")
def tallied(recs):
    with pytest.warns(UserWarning, match="judged step 4 has no residual arrays"):
        return false_reject.tally(recs, [MODEL])


# ---- false rejects ---------------------------------------------------------------------------


def test_false_reject_counts(tallied):
    tallies, _ = tallied
    t = tallies[MODEL]
    # honest-s1's judged steps 2 and 3 and honest-s2's step 2 have arrays; NaN and inf fail,
    # honest-s2's ρ_κ is unscored (NaN, but None in its step record), step 1s don't count
    # ρ_5: 0.5 2.0 0.1 NaN | 0.25 0.75 3.0 0.0 | 0.5 0.125
    # ρ_κ: 0.3 inf | 0.9 0.4;  ρ_6: 0.2 1.5 NaN | 0.1 0.2 0.3 | 0.25
    assert t.n == {"rho_5": 10, "rho_kappa": 4, "rho_6": 7}
    assert t.fail == {"rho_5": 3, "rho_kappa": 1, "rho_6": 2}
    assert t.zeros["rho_5"] == 1 and t.infinite == dict.fromkeys(t.infinite, 1)
    assert (t.runs, t.steps, t.rejected) == ({"honest-s1", "honest-s2"}, 4, 1)
    assert t.hist["rho_5"].sum() == 8  # the zero and the NaN have no bin on a log axis


def test_false_reject_table(tallied):
    rows = false_reject.table_rows(tallied[0])
    got = {(r["kind"], r["family"]): (r["n"], r["n_over_1"], r["rate"]) for r in rows}
    assert got == {("component", "ρ_5(m, j)"): (10, 3, 3 / 10),
                   ("component", "ρ_κ(m)"): (4, 1, 1 / 4),
                   ("component", "ρ_6(W)"): (7, 2, 2 / 7),
                   ("component", "total"): (21, 6, 6 / 21),
                   ("step", "step"): (4, 1, 1 / 4)}
    assert {r["model"] for r in rows} == {"SmolLM2-135M"} and {r["runs"] for r in rows} == {2}


def test_margin_curve_rescales_rho_5_only(tallied):
    t = tallied[0][MODEL]
    c = false_reject.margin_curve(t, 8.0)
    at = {z: int(np.argmin(np.abs(c["z"] - z))) for z in (4.0, 8.0, 16.0)}
    assert all(np.isclose(c["z"][i], z) for z, i in at.items())
    # ρ_5 at margin z is ρ_5 · 8 / z: above 1 when ρ_5 > z / 8
    assert c["rho_5"][at[8.0]] == pytest.approx(3 / 10)   # the configured margin: as recorded
    assert c["rho_5"][at[4.0]] == pytest.approx(4 / 10)   # > 0.5: 2.0, NaN, 0.75, 3.0
    assert c["rho_5"][at[16.0]] == pytest.approx(2 / 10)  # > 2: NaN, 3.0
    # ρ_κ and ρ_6 keep their 1 + 2 failures
    assert c["total"][at[4.0]] == pytest.approx((4 + 3) / 21)
    assert c["total"][at[8.0]] == pytest.approx(6 / 21)
    assert np.all(np.diff(c["rho_5"]) <= 0)


def test_cancellation_drift_per_class(tallied):
    _, drift = tallied
    # class k's step-2 score is infinite, so it has no point there; honest-s2 scored no ρ_κ
    assert drift == {(MODEL, ATTACK): {"q": {2: 0.3, 3: 0.9}, "k": {3: 0.4}}}


# ---- cost ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mc(recs):
    with pytest.warns(UserWarning, match="honest-s2: no time records"):
        return cost.gather(recs, MODEL)


def test_gather_pairs_the_plain_run_and_skips_calibration(recs, mc):
    assert cost.plain_pair(recs, "honest-s1") == "plain_baseline"
    # calibrate/k9 judged nothing, and honest-s2 has no cost records
    assert (mc.runs, mc.steps_per_run) == (1, [4])
    assert mc.time["P0"] == [1.0, 1.0, 1.0]  # judged steps 2–4 only
    assert mc.time["P1"] == pytest.approx([0.07] * 3)  # 0.05 more forward, 0.02 labelling
    assert mc.time["5"] == pytest.approx([0.9, 1.0, 1.1])
    assert mc.time["3"] == mc.time["5.glue"] == [0.4] * 3
    assert mc.once == {**{c: [s] for c, s in ONE_OFF_TIME.items()}, cost.TREE: [4.0]}
    assert mc.counts["P1"]["flops"] == 5 and mc.counts["P0"]["flops"] == 100
    assert mc.counts[cost.TREE]["hash_in_bytes"] == 700
    # largest over the memory pass; P1 is (900 − 100) − (700 − 100)
    assert mc.peak == {"P0": 700, "P1": 200, "P2": 50, "5": 1200, "0": 3000, cost.TREE: 10}
    assert mc.transcript_bytes == [5000]


def test_cost_ratios(mc):
    rows = {r["measure"]: r for r in cost.ratio_rows([mc])}
    time, flops = rows["time"], rows["FLOPs"]
    # per step: P0 1.0; P1–P5 0.07 + 0.1 + 0.2 + 0.3 + 0.4; checks 4 … 2.root sum to 1.8
    assert time["P0"] == 1.0
    assert time["prover_overhead"] == pytest.approx(1.07)
    assert time["verifier"] == pytest.approx(1.8)
    assert time["overhead_over_P0"] == pytest.approx(1.07)
    assert time["verifier_over_P0"] == pytest.approx(1.8)
    assert time["prover_plus_verifier_over_P0"] == pytest.approx(3.87)
    # per run of 4 steps: 4 · 1.8 + 1 + 2 + 3 + 0.5 against 4 · 1.0
    assert time["verifier_per_run_over_P0_per_run"] == pytest.approx(13.7 / 4)
    assert (time["steps_per_run"], time["runs"]) == (4.0, 1)
    # FLOPs: P0 100, P1 5, checks 6a + 5 + 6b = 60, the one-off checks 0
    assert (flops["P0"], flops["prover_overhead"], flops["verifier"]) == (100, 5, 60)
    assert flops["overhead_over_P0"] == pytest.approx(0.05)
    assert flops["prover_plus_verifier_over_P0"] == pytest.approx(1.65)
    assert flops["verifier_per_run_over_P0_per_run"] == pytest.approx(0.6)


def test_cost_grid(mc):
    rows = {(r["side"], r["component"]): r for r in cost.grid_rows([mc])}
    five = rows[("verifier", "5")]
    assert (five["n"], five["flops"], five["peak_bytes"], five["per"]) == (3, 20, 1200, "step")
    assert five["time_s_mean"] == pytest.approx(1.0)
    assert five["time_s_std"] == pytest.approx(0.1)
    total = rows[("verifier", "verifier")]  # per step 1.7, 1.8, 1.9
    assert total["time_s_mean"] == pytest.approx(1.8) and total["time_s_std"] == \
        pytest.approx(0.1)
    assert total["flops"] == 60 and total["hash_in_bytes"] == 3000
    assert rows[("prover", "prover")]["time_s_mean"] == pytest.approx(2.07)
    assert rows[("prover", "prover")]["flops"] == 105
    tree = rows[("prover", cost.TREE)]
    assert (tree["per"], tree["time_s_mean"], tree["hash_in_bytes"], tree["peak_bytes"]) == \
           ("run", 4.0, 700, 10)
    assert rows[("verifier", "0")]["hash_in_bytes"] == 500
    assert rows[("prover", "transcript")]["transcript_bytes"] == 5000
    assert rows[("verifier", "3")]["time_s_mean"] == pytest.approx(0.4)
    assert rows[("prover", "P1")]["peak_bytes"] == 200


def test_mains_write_their_files(out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod in (false_reject, cost):
            mod.main(["--root", str(out), "--scale", "test"])
    d = out / "evaluation" / "results" / "test"
    for name in ("false_reject", "cost_ratios", "cost_grid"):
        assert (d / f"{name}.csv").stat().st_size > 0 and (d / f"{name}.md").stat().st_size > 0
    for name in ("honest_histogram", "cancellation_drift", "margin_curve", "cost"):
        assert (d / f"{name}.pdf").stat().st_size > 0 and (d / f"{name}.png").stat().st_size > 0
    with open(d / "false_reject.csv", newline="") as f:
        table = list(csv.DictReader(f))
    assert [(r["family"], r["n"], r["n_over_1"]) for r in table][-2:] == \
           [("total", "21", "6"), ("step", "4", "1")]
