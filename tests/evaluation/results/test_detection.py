"""EV4: the detection table, the detection figure and the per-layer figure, on the hand-made
records of ``conftest.py``."""

import csv
import math
import warnings

import numpy as np
import pytest

from evaluation import records as er
from evaluation.output import new_figure
from evaluation.results import detection_figure, detection_table, per_layer

from .conftest import ATTACK, BPR, MODEL, _archive, _env, _step, _write_jsonl


@pytest.fixture(scope="module")
def recs(out):
    return er.read(out)


def test_detection_table_rows(recs):
    rows = {r["scenario"]: r for r in detection_table.rows(recs, [MODEL])}
    assert list(rows) == ["A1", "A2", "A3", "hidden"]
    for r in rows.values():
        assert (r["model"], r["attack"], r["bpr_records"], r["bpr_pct"]) == \
               ("SmolLM2-135M", ATTACK, 2, 25.0)
        assert r["early_reject"] == 0
    expected = {s: (r["expected_check"], r["first_failing_check"], r["as_expected"])
                for s, r in rows.items()}
    # A3's expected check is check 6, either sub-check; the table shows the one that failed
    assert expected == {"A1": ("4", "4", True), "A2": ("5", "5", False),
                        "A3": ("6a or 6b", "6a", True), "hidden": ("7", "7", True)}

    a2 = rows["A2"]
    # three seeds: s1 rejects at t* with 10 + 2 + 0 failing tests, s2 a step later with
    # 20 + 0 + 2, s3 never. Only s1 is detected; s2 and s3 both have a step record past t*,
    # which §2.1 calls a harness failure
    assert (a2["seeds"], a2["detected"], a2["harness_failure"]) == (3, 1, 2)
    assert a2["rejection_rate"] == pytest.approx(1 / 3)
    # n_failing and latency are over the rejecting seeds s1 and s2, as measured
    assert a2["n_failing_mean"] == 17.0 and a2["n_failing_std"] == pytest.approx(math.sqrt(50))
    assert (a2["n_failing_rho_5"], a2["n_failing_rho_kappa"], a2["n_failing_rho_6"]) == \
           (15.0, 1.0, 1.0)
    assert a2["latency_mean"] == 0.5 and a2["latency_std"] == pytest.approx(math.sqrt(0.5))
    assert a2["record_pass_status"] == "complete" and a2["measured_rho_5"] == 8.0

    # A3 scored ρ_6 only; A1 and the hidden step scored nothing, so their count is empty
    a3 = rows["A3"]
    assert (a3["seeds"], a3["detected"], a3["n_failing_mean"], a3["n_failing_std"]) == \
           (2, 2, 3.0, 0.0)
    assert a3["n_failing_rho_5"] is None
    for s in ("A1", "A3", "hidden"):
        assert (rows[s]["rejection_rate"], rows[s]["harness_failure"]) == (1.0, 0)
    for s in ("A1", "hidden"):
        r = rows[s]
        assert (r["latency_mean"], r["n_failing_mean"]) == (0.0, None)
        assert r["record_pass_status"] == "not run"
    # the hidden step's t* is step 2, so rejecting there is latency 0
    assert rows["hidden"]["latency_std"] is None


def test_check_6b_is_as_expected_for_a3_and_another_check_is_not(recs):
    runs = [o for o in detection_table.run_outcomes(recs) if o["scenario"] == "A3"]

    def with_check(c):
        return [{**runs[0], "reject": {**runs[0]["reject"], "first_failing_check": c}}, runs[1]]

    row = detection_table._row(("A3", MODEL, ATTACK, 2), with_check("6b"))
    assert (row["first_failing_check"], row["as_expected"]) == ("6a, 6b", True)
    row = detection_table._row(("A3", MODEL, ATTACK, 2), with_check("5"))
    assert (row["first_failing_check"], row["as_expected"]) == ("5, 6a", False)
    assert row["detected"] == 2  # a rejection at t* is a detection whichever check fired


def test_early_reject_and_harness_failure(tmp_path):
    """A run rejected before t* is no detection; one that goes on past t* is a harness
    failure, whether it rejected at t* or not."""
    def run(name, seed, *steps):
        _write_jsonl(tmp_path / name / "records.jsonl", [_env(name, seed=seed), *steps])

    kw = {"cheat_step": 2, "bpr": BPR}
    run("early", 1, _step("early", "A2", 1, check="5", seed=1, **kw))
    run("on-time", 2, _step("on-time", "A2", 1, seed=2, **kw),
        _step("on-time", "A2", 2, check="5", seed=2, **kw))
    run("past", 3, _step("past", "A2", 1, seed=3, **kw),
        _step("past", "A2", 2, check="5", seed=3, **kw),
        _step("past", "A2", 3, check="5", seed=3, **kw))
    (row,) = detection_table.rows(er.read(tmp_path))
    assert (row["seeds"], row["detected"], row["early_reject"], row["harness_failure"]) == \
           (3, 2, 1, 1)
    assert row["rejection_rate"] == pytest.approx(2 / 3) and not row["as_expected"]
    # latency as measured: −1, 0, 0
    assert row["latency_mean"] == pytest.approx(-1 / 3)


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
    # the sweep's honest step (500) are out; honest-s2's step 2 (0.5, 0.25) is in. honest-s1's
    # step 2 is infinite in both, so it is left out of the band and counted
    ranges = detection_figure.honest_ranges(recs)
    assert ranges == {(MODEL, ATTACK, "rho_5"): (0.5, 3.0, 1),
                      (MODEL, ATTACK, "rho_6"): (0.2, 0.3, 1)}
    def title(fig):  # wrapped to the figure's width
        return " ".join(fig.get_suptitle().split())

    fig = detection_figure.plot(detection_figure.points(recs, [MODEL]), ranges, [MODEL])
    assert "mean ± sample std" in title(fig)
    assert "1 infinite ρ_5, 1 infinite ρ_6 left out" in title(fig)
    # one tick per measured poisoning rate: 25% here
    assert fig.axes[0].get_xticks().tolist() == [25.0]
    # an all-infinite honest score has no band but still counts
    only_inf = {(MODEL, ATTACK, "rho_5"): (None, None, 2)}
    assert "2 infinite ρ_5 left out" in title(detection_figure.plot([], only_inf))


def test_bpr_ticks_at_the_measured_rates():
    _, ax = new_figure()
    ax.set_xlim(0, 20)
    # 0% and one record in 128 sit closer than 6% of the axis, so the second label drops a line
    detection_figure.bpr_ticks(ax, [10.0, 0.0, 100 / 128, 0.0])
    assert ax.get_xticks().tolist() == [0.0, 100 / 128, 10.0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "\n0.781", "10"]


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
    with pytest.warns(UserWarning, match=r"measured different layers \(n = 1–3 per layer; "
                                         r"fewest at layers \[2\]\)"):
        (line,) = per_layer.lines(entries)
    assert (line["model"], line["attack"], line["bpr_actual"], line["seeds"]) == \
           (MODEL, ATTACK, 0.25, 3)
    assert line["layers"].tolist() == [0, 1, 2]
    # layer 0: 4, 2, 0.2; layer 1: 7, 9, 0.4; the output layer 2 only s2 scored (0.5)
    np.testing.assert_allclose(line["mean"], [6.2 / 3, 16.4 / 3, 0.5])
    assert line["n"].tolist() == [3, 3, 1]
    labels = per_layer.plot([line]).axes[0].get_legend_handles_labels()[1]
    assert labels[0] == "BPR 25% (n = 1–3 seeds per layer)"


def test_per_layer_reads_a_zero_row_archive(tmp_path):
    # a rejected step's archive may hold no rows; the real records have some
    run = "A2-z"
    _write_jsonl(tmp_path / run / "records.jsonl",
                 [_env(run), _step(run, "A2", 1, check="5", cheat_step=1, bpr=BPR)])
    _archive(tmp_path, run, "A2", 1, p_layer=np.zeros(0, np.int32), p_cls=np.zeros(0, "<U1"),
             rho_5=np.zeros((0, 9)), rho_kappa=np.zeros(0), rho_6=np.zeros(272))
    (line,) = per_layer.lines(per_layer.runs(er.read(tmp_path)))
    assert (line["layers"].size, line["n"].size, line["seeds"]) == (0, 0, 1)
    per_layer.plot([line])


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
    assert (table[1]["detected"], table[1]["harness_failure"]) == ("1", "2")
    assert table[1]["latency_mean"] == "0.5"
    assert "| 0.33 |" in (d / "detection_table.md").read_text()
