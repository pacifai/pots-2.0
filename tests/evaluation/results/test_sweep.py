"""EV6's result script on synthetic records: the sweep it picks, the plotted ``p₁`` and
rejection rates, the fit (checked against the sweep runner's own), the resolvable-rate lines
and the floor marks, against values worked out by hand.

The synthetic sweep is built the way ``flipped_matmul_sweep.run_sweep`` writes one: hand-made
:class:`Trials` go through the runner's ``summarize`` and ``save_trials``. ``k = 3``,
``τ = 2``, ``N = 6`` (so ``2^{N/k} = 4``); a residual of 1 is a miss (``≤ τ``), 5 a catch.
"""

import csv
import json
import math
import warnings

import numpy as np
import pytest

from evaluation.records import read
from evaluation.results import sweep as sw
from verification.runs import flipped_matmul_sweep as fms
from verification.verifier.matmul_check import challenges, sizing

TAU, K, N = 2.0, 3, 6.0
EPS = 2.0 ** -24
H = "a" * 64
MISS, CATCH = 1.0, 5.0
F_ACHIEVED = 0.5  # the band file's, set apart from any target's own floor


def _e(q, ratio):
    return math.sqrt(2) * EPS + math.sqrt(q) * EPS * ratio


E_Y, E_F = _e(576, 1.0), _e(49152, 2.0)  # Y_q at U/‖P‖ ratio 1, dF at 2
TARGETS = {10: ("L15.Y_q", "Y_q", "forward", False, 576, E_Y),
           20: ("dF", "dF", "input gradient", True, 49152, E_F)}


def _trials(m, shape, phase, rows):
    """``rows``: ``(x, residuals, norm_ratio)`` per trial; ``f_real`` is exactly ``x·τ·e``."""
    name, cls, spec_class, _, _, e = TARGETS[m]
    tr = fms.Trials(m, cls, shape, e, spec_class=spec_class, name=name, phase=phase)
    for x, res, ratio in rows:
        tr.x_set.append(x)
        tr.f_real.append(x * TAU * e)
        tr.norm_ratio.append(ratio)
        tr.normalized.append(tuple(res))
        tr.kappa.append(1.0)
        tr.reason.append("residual" if max(res) > TAU else "accepted")
    return tr


R = (MISS, CATCH, CATCH)  # one miss in three
ALL = (CATCH,) * 3
TRIALS = [
    # dF row2 grid: x = 2 below the fit (5 misses); x = 3 one miss; x = 10 one miss, ‖P'‖ 1.25×
    _trials(20, "row2", "grid", [(2.0, (MISS,) * 3, 1.0), (2.0, (MISS, MISS, CATCH), 1.0),
                                 (3.0, R, 1.0), (3.0, ALL, 1.0),
                                 (10.0, ALL, 1.25), (10.0, R, 1.25)]),
    # dF row2 deep: four trials at x = 30, one miss
    _trials(20, "row2", "deep", [(30.0, R, 1.0)] + [(30.0, ALL, 1.0)] * 3),
    # dF entry grid: one miss at x = 3, none at x = 10
    _trials(20, "entry", "grid", [(3.0, R, 1.0), (3.0, ALL, 1.0),
                                  (10.0, ALL, 1.0), (10.0, ALL, 1.0)]),
    # Y_q row2 grid: every challenge missed at x = 3, but Y_q is not the binding product
    _trials(10, "row2", "grid", [(3.0, (MISS,) * 3, 1.0), (3.0, (MISS,) * 3, 1.0)]),
]
FLOOR = fms.Floor(N, sizing.C_ANTI, TAU, K)
C6 = {"weight": "model.layers.14.mlp.down_proj.weight", "entry": 7, "sign": 1, "g_m": 30,
      "tau_w": 4.0, "factor": 2.0, "rho_target": 8.0, "rho_entry": 8.5, "rho_max": 8.5,
      "root": "c" * 64, "check_id": "6a", "kind": "failed", "detail": "entry 7", "rejected": True}


def _write(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _sweep(root, d, started, source, trials, *, npz=True, check6=True):
    """A sweep run directory, written as ``run_sweep`` writes one."""
    head = {"run": "flipped_matmul_sweep", "step": 4}
    rows = [{"record": "environment", "run": "flipped_matmul_sweep", "started": started,
             "band_file_hash": H, "model": "HuggingFaceTB/SmolLM2-135M-Instruct"},
            {"record": "sweep_source", **head, "source": source, "root": "d" * 64,
             "targets": []}]
    for tr in trials:
        _, _, _, binding, q, _ = TARGETS[tr.m]
        pts, sm = fms.summarize(tr, FLOOR)
        sm.update(binding=binding, q=q)
        tag = {"m": tr.m, "name": tr.name, "cls": tr.cls, "spec_class": tr.spec_class,
               "binding": binding, "shape": tr.shape, "phase": tr.phase}
        rows += [{"record": "sweep_point", **head, **tag, **p} for p in pts]
        rows.append({"record": "sweep_summary", **head, **sm})
    if check6:
        rows.append({"record": "sweep_check6", **head, **C6})
    _write(root / d, rows)
    if npz:
        fms.save_trials(root / d / fms.TRIALS_FILE, trials)


@pytest.fixture
def root(tmp_path):
    _sweep(tmp_path, "sweep/stored", "2026-10-12T07:00:00+0300", "stored", TRIALS)
    # later, but in-process: the stored sweep still wins
    _sweep(tmp_path, "sweep/inprocess", "2026-10-12T10:00:00+0300", "in-process", TRIALS[:1])
    (tmp_path / "bands.json").write_text(json.dumps({"stats": {"sizing": {
        "f_achieved": F_ACHIEVED, "tau": TAU, "k": K, "N": N, "eps_in": EPS, "eps_acc": EPS}}}))
    return tmp_path


def _build(root, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a fit that disagrees with the runner's warns
        return sw.build(read(root), band_file=json.loads((root / "bands.json").read_text()),
                        **kw)


def _row(res, m, shape, phase):
    [r] = [r for r in res.fits if (r["m"], r["shape"], r["phase"]) == (m, shape, phase)]
    return r


def test_the_copied_constants_are_verifications():
    assert sw.C_ANTI == sizing.C_ANTI
    assert sw.SIGMA_R == challenges.SIGMA_R
    assert sw.SHAPES == fms.SHAPES
    assert sw.TRIALS_FILE == fms.TRIALS_FILE


def test_the_fit_is_the_runners():
    """On random trials, both norms, :func:`fit_c_hat` gives ``summarize``'s ``ĉ``."""
    gen = np.random.default_rng(0)
    xs = [0.5, 2.0, 3.0, 10.0, 100.0]
    tr = fms.Trials(1, "dF", "row2", 3e-6, name="dF", phase="grid")
    for _ in range(200):
        x = float(gen.choice(xs))
        tr.x_set.append(x)
        tr.f_real.append(x * TAU * tr.e * float(gen.uniform(0.98, 1.02)))
        tr.norm_ratio.append(float(gen.uniform(1.0, 1.3)))
        tr.normalized.append(tuple(gen.uniform(0, 2.5 * TAU, K)))
        tr.kappa.append(1.0)
        tr.reason.append("accepted")
    _, s = fms.summarize(tr, FLOOR)
    fit = sw.fit_c_hat(np.asarray(tr.x_set), np.asarray(tr.f_real),
                       np.asarray(tr.normalized), tr.e, TAU, s["c_fit_x_min"],
                       np.asarray(tr.norm_ratio))
    assert (fit.misses, fit.challenges) == (s["c_fit_misses"], s["c_fit_vectors"])
    for mine, theirs in ((fit.c_hat, "c_hat"), (fit.se, "c_hat_se"),
                         (fit.c_hat_committed, "c_hat_committed"),
                         (fit.se_committed, "c_hat_committed_se")):
        assert math.isclose(mine, s[theirs], rel_tol=1e-12)


def test_picks_the_stored_sweep_unless_named(root):
    rec = read(root)
    assert sw.sweep_run(rec) == "sweep/stored"
    assert sw.sweep_run(rec, "sweep/inprocess") == "sweep/inprocess"
    with pytest.raises(ValueError, match="no sweep_source"):
        sw.sweep_run(rec, "sweep")
    assert sw.load_trials(rec, "sweep/stored")["normalized"].shape == (16, K)


def test_fit_rows_by_hand(root):
    res = _build(root)
    assert (res.tau, res.k, res.N) == (TAU, K, N)
    # dF row2: grid exposure 2·3/3 + 2·3/10 = 2.6 (x = 2 left out), deep 4·3/30 = 0.4
    grid, deep, pooled = (_row(res, 20, "row2", p) for p in ("grid", "deep", "pooled"))
    for r, misses, exposure, exposure_c, n in ((grid, 2, 2.6, 2 + 6 / 8, 4),
                                               (deep, 1, 0.4, 0.4, 4),
                                               (pooled, 3, 3.0, 2.75 + 0.4, 8)):
        assert (r["misses"], r["trials"], r["challenges"]) == (misses, n, n * K)
        assert math.isclose(r["c_hat"], misses / exposure, rel_tol=1e-12)
        assert math.isclose(r["c_hat_se"], math.sqrt(misses) / exposure, rel_tol=1e-12)
        assert math.isclose(r["expected"], sizing.C_ANTI * exposure, rel_tol=1e-12)
        assert math.isclose(r["c_hat_committed"], misses / exposure_c, rel_tol=1e-12)
        assert math.isclose(r["z_vs_c"], (r["c_hat"] - sizing.C_ANTI) / r["c_hat_se"])
        assert r["c_theory"] == sizing.C_ANTI and r["source"] == "trials"
        f_ach = sizing.C_ANTI * TAU * E_F * 4
        assert math.isclose(r["f_ach"], f_ach, rel_tol=1e-12)
        assert math.isclose(r["f_ach_c_hat"], f_ach * r["c_hat"] / sizing.C_ANTI)
        assert math.isclose(r["u_ratio"], 2.0, rel_tol=1e-9)
    entry = _row(res, 20, "entry", "grid")
    assert entry["c_theory"] == challenges.SIGMA_R
    assert math.isclose(entry["z_vs_theory"],
                        (1 / 2.6 - challenges.SIGMA_R) / (1 / 2.6), rel_tol=1e-12)
    assert not [r for r in res.fits if r["shape"] == "entry" and r["phase"] != "grid"]
    y = _row(res, 10, "row2", "grid")
    assert y["c_hat"] == 6 / 2 and math.isclose(y["u_ratio"], 1.0, rel_tol=1e-9)
    # binding product last, grid–deep–pooled within a shape
    assert [(r["m"], r["shape"], r["phase"]) for r in res.fits] == [
        (10, "row2", "grid"), (20, "entry", "grid"), (20, "row2", "grid"),
        (20, "row2", "deep"), (20, "row2", "pooled")]
    assert not any("ratio 1" in n for n in res.notes)  # dF's unit is at ratio 2


def test_plotted_points_by_hand(root):
    res = _build(root)
    pts = {(p["m"], p["shape"], p["phase"], p["x_swept"]): p for p in res.points}
    p = pts[(20, "row2", "grid", 10.0)]
    assert (p["misses"], p["vectors"], p["trials"]) == (1, 6, 2)
    assert math.isclose(p["x"], 10.0, rel_tol=1e-12)
    assert p["p1"] == 1 / 6 and p["resolvable"] == 1 / 6
    assert math.isclose(p["p1_se"], math.sqrt(1 / 6 * 5 / 6 / 6), rel_tol=1e-12)
    assert p["reject_rate"] == 1.0 and p["reject_se"] == 0.0
    q = pts[(20, "row2", "grid", 2.0)]
    assert q["p1"] == 5 / 6 and q["reject_rate"] == 0.5
    assert math.isclose(q["reject_se"], math.sqrt(0.25 / 2))
    assert pts[(20, "row2", "deep", 30.0)]["resolvable"] == 1 / 12
    # on the committed norm x is divided by the mean ‖P'‖/‖P‖
    committed = _build(root, norm="committed")
    [c] = [p for p in committed.points if (p["m"], p["shape"], p["phase"], p["x_swept"])
           == (20, "row2", "grid", 10.0)]
    assert math.isclose(c["x"], 8.0, rel_tol=1e-12)


def test_floor_by_hand(root):
    """The band file's ``f_achieved``, scaled by the binding product's largest ``ĉ``."""
    res = _build(root)
    fl = res.floor
    assert (fl.f_achieved, fl.source) == (F_ACHIEVED, "band file")
    assert (fl.shape, fl.fit, fl.misses) == ("row2", "pooled", 3)
    assert math.isclose(fl.c_hat, 1.0, rel_tol=1e-12)
    assert math.isclose(fl.measured, F_ACHIEVED / sizing.C_ANTI, rel_tol=1e-12)
    deep = _build(root, floor_fit="deep").floor
    assert (deep.shape, deep.fit) == ("row2", "deep")
    assert math.isclose(deep.c_hat, 2.5, rel_tol=1e-12)
    com = _build(root, norm="committed").floor
    assert math.isclose(com.c_hat, 3 / 3.15, rel_tol=1e-12) and com.norm == "committed"
    # without a band file sizing: the binding product's own floor
    nob = sw.build(read(root), band_file=None)
    assert nob.floor.f_achieved == _row(nob, 20, "row2", "grid")["f_ach"]
    assert any("no band file" in n for n in nob.notes)
    assert all(r["u_ratio"] is None for r in nob.fits)


def test_predicted_curves():
    assert sw.predicted_p1(0.5) == 1.0
    assert math.isclose(sw.predicted_p1(10.0), sizing.C_ANTI / 10)
    e = 1e-6
    f = 10.0 * TAU * e
    assert math.isclose(sw.predicted_reject(f, TAU, e, K), 1 - (sizing.C_ANTI / 10) ** K)
    assert sw.predicted_reject(0.1 * TAU * e, TAU, e, K) == 0.0
    assert math.isclose(sw.u_ratio(_e(576, 3.0), 576, EPS, EPS), 3.0, rel_tol=1e-9)


def _lines(ax, prefix):
    return [ln for ln in ax.get_lines() if str(ln.get_label()).startswith(prefix)]


def test_figure_a_lines_and_markers(root):
    res = _build(root)
    fig = sw.figure_a(res)
    entry, row2 = fig.axes[0], fig.axes[1]
    assert (entry.get_title(), row2.get_title()) == ("entry", "row2")
    # one resolvable line per trial count: 1/(2·3) for the grid, 1/(4·3) for the deep points
    levels = {ln.get_ydata()[0] for ln in _lines(row2, "resolvable")}
    assert levels == {1 / 6, 1 / 12}
    [c] = _lines(row2, "predicted c/x")
    x, y = c.get_xdata(), c.get_ydata()
    np.testing.assert_allclose(y, np.minimum(1.0, sizing.C_ANTI / x), rtol=1e-12)
    assert len(_lines(entry, "σ_r/x")) == 1 and not _lines(row2, "σ_r/x")
    # dF entry has no miss at x = 10: a triangle on its resolvable line
    tri = [ln for ln in entry.get_lines() if ln.get_marker() == "v"]
    assert [list(ln.get_ydata()) for ln in tri] == [[1 / 6]]
    assert 10 ** -sw._DODGE <= tri[0].get_xdata()[0] / 10.0 <= 10 ** sw._DODGE  # dodged


def test_figure_b_floor_marks(root):
    res = _build(root)
    fig = sw.figure_b(res)
    ax = fig.axes[0]
    [fa] = _lines(ax, "f_achieved")
    [fm] = _lines(ax, "measured floor")
    assert list(fa.get_xdata()) == [F_ACHIEVED] * 2
    assert math.isclose(fm.get_xdata()[0], F_ACHIEVED / sizing.C_ANTI, rel_tol=1e-12)
    assert "ĉ = 1.000" in fm.get_label() and "pooled" in fm.get_label()


def test_without_trials_the_table_holds_the_summaries(tmp_path):
    _sweep(tmp_path, "sweep", "2026-10-12T07:00:00+0300", "stored", TRIALS, npz=False,
           check6=False)
    res = sw.build(read(tmp_path))
    assert all(r["source"] == "summary" for r in res.fits)
    assert not [r for r in res.fits if r["phase"] == "pooled"]
    g = _row(res, 20, "row2", "grid")
    assert (g["misses"], g["challenges"], g["trials"]) == (2, 12, 4)
    assert math.isclose(g["c_hat"], 2 / 2.6) and math.isclose(g["expected"], sizing.C_ANTI * 2.6)
    assert res.floor.fit == "grid"  # pooled falls back to grid without the trials
    assert math.isclose(res.floor.c_hat, 2 / 2.6, rel_tol=1e-12)
    assert any("sweep_trials.npz" in n for n in res.notes)
    assert any("sweep_check6" in n for n in res.notes)


def test_main_writes_figures_and_tables(root, capsys):
    out = root / "elsewhere"
    assert sw.main(["--root", str(root), "--out", str(out)]) == 0
    d = out / "evaluation" / "results" / "test"
    assert sorted(p.name for p in d.iterdir()) == sorted(
        f"{n}.{s}" for n, ss in ((sw.NAME_A, ("pdf", "png")), (sw.NAME_B, ("pdf", "png")),
                                 (sw.NAME_FIT, ("csv", "md")), (sw.NAME_C6, ("csv", "md")))
        for s in ss)
    with open(d / "sweep_fit.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(sw.FIT_COLUMNS) and len(rows) == 5
    md = (d / "sweep_fit.md").read_text()
    assert "Measured floor (pooled fit, honest norm; both placeholders)" in md
    assert "rejected at 6a on that entry" in md
    c6 = (d / "sweep_check6.csv").read_text()
    assert "model.layers.14.mlp.down_proj.weight,7,1" in c6 and "True" in c6
    assert "measured floor" in capsys.readouterr().out
    assert not (root / "evaluation").exists()  # nothing written beside the records
