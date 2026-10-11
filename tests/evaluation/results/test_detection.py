"""EV4: the detection table, the detection figure and the per-layer figure, on the hand-made
records of ``conftest.py``."""

import csv
import math
import warnings

import numpy as np
import pytest

from evaluation import records as er
from evaluation.results import detection_figure, detection_table, per_layer

from .conftest import ATTACK, MODEL


@pytest.fixture(scope="module")
def recs(out):
    return er.read(out)


def test_detection_table_rows(recs):
    rows = {r["scenario"]: r for r in detection_table.rows(recs, [MODEL])}
    assert list(rows) == ["A1", "A2", "A3", "hidden"]
    for r in rows.values():
        assert (r["model"], r["attack"], r["bpr_records"], r["bpr_pct"]) == \
               ("SmolLM2-135M", ATTACK, 2, 25.0)
        assert r["first_failing_check"] == r["expected_check"] and r["as_expected"]

    a2 = rows["A2"]
    # three seeds, two reject: s1 at t* with 10 + 2 + 0 failing tests, s2 a step later with
    # 20 + 0 + 2; s3 never
    assert (a2["seeds"], a2["rejected"]) == (3, 2)
    assert a2["rejection_rate"] == pytest.approx(2 / 3)
    assert a2["n_failing_mean"] == 17.0 and a2["n_failing_std"] == pytest.approx(math.sqrt(50))
    assert (a2["n_failing_rho_5"], a2["n_failing_rho_kappa"], a2["n_failing_rho_6"]) == \
           (15.0, 1.0, 1.0)
    assert a2["latency_mean"] == 0.5 and a2["latency_std"] == pytest.approx(math.sqrt(0.5))
    assert a2["record_pass_status"] == "complete" and a2["measured_rho_5"] == 8.0

    # A3 scored ρ_6 only; A1 and the hidden step scored nothing, so their count is empty
    a3 = rows["A3"]
    assert (a3["seeds"], a3["rejected"], a3["n_failing_mean"], a3["n_failing_std"]) == \
           (2, 2, 3.0, 0.0)
    assert a3["n_failing_rho_5"] is None
    for s in ("A1", "hidden"):
        r = rows[s]
        assert (r["rejection_rate"], r["latency_mean"], r["n_failing_mean"]) == (1.0, 0.0, None)
        assert r["record_pass_status"] == "not run"
    # the hidden step's t* is step 2, so rejecting there is latency 0
    assert rows["hidden"]["latency_std"] is None


def test_wrong_check_is_not_as_expected(recs):
    runs = [o for o in detection_table.run_outcomes(recs) if o["scenario"] == "A3"]
    runs[0] = {**runs[0], "reject": {**runs[0]["reject"], "first_failing_check": "6b"}}
    row = detection_table._row(("A3", MODEL, ATTACK, 2), runs)
    assert row["first_failing_check"] == "6a, 6b" and not row["as_expected"]


def test_detection_points_at_t_star(recs):
    pts = {p["scenario"]: p for p in detection_figure.points(recs, [MODEL])}
    assert set(pts) == {"A2", "A3"}  # A1 and the hidden step are table-only
    a2 = pts["A2"]
    xs = [100.0, 0.8, 0.5]  # ρ_5 at step 1 of each seed, rejecting or not
    assert (a2["score"], a2["bpr_actual"], a2["seeds"]) == ("rho_5", 0.25, 3)
    assert a2["mean"] == pytest.approx(np.mean(xs))
    assert a2["std"] == pytest.approx(np.std(xs, ddof=1))
    assert a2["infinite"] == 0
    # A3-s2's ρ_6 is infinite, so the mean is too, with no std
    a3 = pts["A3"]
    assert (a3["mean"], a3["std"], a3["seeds"], a3["infinite"]) == (math.inf, None, 2, 1)


def test_honest_range_from_judged_honest_steps(recs):
    # judged steps only: step 1 (50 and 99, a calibration step), the calibration run (70) and
    # the sweep's honest step (500) are out, and so is honest-s1's infinite step 2; honest-s2's
    # step 2 (0.5, 0.25) is in
    assert detection_figure.honest_ranges(recs) == {
        (MODEL, ATTACK, "rho_5"): (0.5, 3.0), (MODEL, ATTACK, "rho_6"): (0.2, 0.3)}


def test_layer_maxima_read_nan_as_inf():
    z = {"p_layer": np.array([0, 0, 1, 2, 4]),
         "rho_5": np.array([[0.5, 4.0], [0.25, 2.0], [np.nan, 0.5], [np.inf, 1.0],
                            [0.0, 0.0]])}
    layers, maxima = per_layer.layer_maxima(z)
    assert layers.tolist() == [0, 1, 2, 4]  # layer 3 has no row, so no point
    assert maxima.tolist() == [4.0, np.inf, np.inf, 0.0]


def test_per_layer_lines_mean_over_seeds(recs):
    entries = per_layer.runs(recs, [MODEL])
    assert sorted(e["run_dir"] for e in entries) == ["A2-s1", "A2-s2", "A2-s3"]
    (line,) = per_layer.lines(entries)
    assert (line["model"], line["attack"], line["bpr_actual"], line["seeds"]) == \
           (MODEL, ATTACK, 0.25, 3)
    assert line["layers"].tolist() == [0, 1, 2]
    # layer 0: 4, 2, 0.2; layer 1: 7, 9, 0.4; the output layer 2 only s2 scored (0.5)
    np.testing.assert_allclose(line["mean"], [6.2 / 3, 16.4 / 3, 0.5])


def test_mains_write_their_files(out):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for mod in (detection_table, detection_figure, per_layer):
            mod.main(["--root", str(out)])
    d = out / "evaluation" / "results" / "test"
    for name in ("detection_table.csv", "detection_table.md", "detection.pdf", "detection.png",
                 "per_layer.pdf", "per_layer.png"):
        assert (d / name).stat().st_size > 0, name
    with open(d / "detection_table.csv", newline="") as f:
        table = list(csv.DictReader(f))
    assert [r["scenario"] for r in table] == ["A1", "A2", "A3", "hidden"]
    assert table[1]["rejected"] == "2" and table[1]["latency_mean"] == "0.5"
    assert "| 0.67 |" in (d / "detection_table.md").read_text()
