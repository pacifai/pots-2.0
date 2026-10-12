"""FS12's result script on synthetic records: the run it picks, ``C`` and its fallback, the
pooled ``ĉ``, the rows and the written outputs."""

import csv
import json
import math

import pytest

from evaluation.records import read
from evaluation.results import k_tunability as kt

H, OTHER = "a" * 64, "b" * 64  # band-file hashes
ROOT = "d" * 64  # the stored step's root
C_ANTI = math.sqrt(2 / 3)
TAU, E_MAX, N = 38.03910546545508, 1.3298792656134927e-05, 93.11817054309053
OTHER_S = 0.4


def _write(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _env(run, started, band=H):
    return {"record": "environment", "run": run, "started": started, "band_file_hash": band}


def _f_ach(k):
    return C_ANTI * TAU * E_MAX * 2 ** (N / k)


def _k_run(root, d, started, *, ks=(1, 2, 3, 4), time0=1.0, reproduced=True):
    fit = {"slope": 0.0125, "intercept": 1.1, "r2": 0.9, "max_rel_resid": 0.05}
    rows = [_env("k_tunability", started),
            {"record": "k_source", "run": "k_tunability", "step": 4, "root": ROOT,
             "band_file_hash": H, "k_conf": 3, "k_top": 4, "k_cover": 3, "B": 4,
             "gradient_ratio": 2.0, "c": C_ANTI, "tau": TAU, "e_max": E_MAX, "N": N,
             "b0": 11.24, "lam": 25.0, "repeats": 3, "other_s": OTHER_S}]
    rows += [{"record": "k_point", "run": "k_tunability", "step": 4, "k": k,
              "f_ach": _f_ach(k), "bits": 11.24 * k, "lambda_eff": 11.24 * k - (N - 25),
              "time_s": time0 + 0.0125 * k, "flops": 2.3e9 * (k + 1),
              "mem_growth_bytes": 8.9e8, "verdict": "accept"} for k in reversed(ks)]
    rows.append({"record": "k_summary", "run": "k_tunability", "step": 4, "fits": {"time_s": fit},
                 "fits_from_2": {"time_s": fit}, "marginal_from": 2, "reproduced": reproduced})
    _write(root / d, rows)


def _point(x, misses, *, phase="grid", shape="row2", binding=True, trials=200, k=9,
           below=1.0, ratio=1.0):
    f = x * TAU * E_MAX
    return {"record": "sweep_point", "run": "flipped_matmul_sweep", "step": 4, "x": x, "f": f,
            "f_real_mean": f * below, "misses": misses, "vectors": trials * k,
            "norm_ratio_mean": ratio, "binding": binding, "shape": shape, "phase": phase,
            "name": "dF" if binding else "L15.Y_q"}


def _summary(shape, phase, binding=True):
    return {"record": "sweep_summary", "run": "flipped_matmul_sweep", "step": 4,
            "shape": shape, "phase": phase, "binding": binding, "c_fit_x_min": 3.0}


def _sweep(root, d, started, source, points, *, sweep_root=ROOT, band=H):
    keys = dict.fromkeys((p["shape"], p["phase"], p["binding"]) for p in points)
    _write(root / d, [_env("flipped_matmul_sweep", started, band),
                      {"record": "sweep_source", "run": "flipped_matmul_sweep", "step": 4,
                       "source": source, "root": sweep_root}]
           + [_summary(*key) for key in keys] + points)


# the stored sweep's binding points: x = 2 is below the fit, x = 3 realized an ulp below 3
STORED = [_point(2.0, 900), _point(3.0, 400, below=1 - 1e-12), _point(10.0, 140, ratio=1.25),
          _point(300.0, 50, phase="deep", trials=2000), _point(3.0, 100, shape="entry"),
          _point(10.0, 30, shape="entry"), _point(3.0, 999, binding=False)]


@pytest.fixture
def root(tmp_path):
    _k_run(tmp_path, "k_old", "2026-10-11T09:00:00+0300", time0=2.0)
    _k_run(tmp_path, "k_tunability", "2026-10-12T09:00:00+0300")
    _write(tmp_path / "calibrate" / "k9", [
        _env("calibrate", "2026-10-11T08:00:00+0300", None),
        {"record": "calibration", "run": "calibrate", "band_file_hash": H},
        {"record": "gradient_coherence", "run": "calibrate", "step": 1, "B": 4, "ratio": 1.25}])
    _write(tmp_path / "calibrate" / "other", [
        _env("calibrate", "2026-10-12T08:00:00+0300", None),
        {"record": "calibration", "run": "calibrate", "band_file_hash": OTHER},
        {"record": "gradient_coherence", "run": "calibrate", "step": 1, "B": 4, "ratio": 3.0}])
    _sweep(tmp_path, "sweep/stored", "2026-10-12T07:00:00+0300", "stored", STORED)
    # later, but in-process; and two sweeps of another step or another band file
    _sweep(tmp_path, "sweep/inprocess", "2026-10-12T10:00:00+0300", "in-process",
           [_point(3.0, 5000)])
    _sweep(tmp_path, "sweep/elsewhere", "2026-10-12T11:00:00+0300", "stored",
           [_point(3.0, 5000)], sweep_root="e" * 64)
    _sweep(tmp_path, "sweep/rebanded", "2026-10-12T11:00:00+0300", "stored",
           [_point(3.0, 5000)], band=OTHER)
    return tmp_path


def test_picks_the_latest_k_run_unless_named(root):
    rec = read(root)
    assert kt.k_run(rec) == "k_tunability"
    assert kt.k_run(rec, "k_old") == "k_old"
    with pytest.raises(ValueError, match="no k_summary"):
        kt.k_run(rec, "calibrate/k9")
    with pytest.raises(ValueError, match="run verification.runs.k_tunability"):
        kt.k_run(read(root / "calibrate"))


def test_coherence_from_the_band_files_calibration_else_one(root):
    rec = read(root)
    assert kt.coherence(rec, H) == (1.25, "calibrate/k9")
    assert kt.coherence(rec, OTHER) == (3.0, "calibrate/other")
    assert kt.coherence(rec, "c" * 64) == (kt.COHERENCE_FALLBACK, None) == (1.0, None)


def test_miss_constant_pools_the_binding_products_fitted_trials(root):
    """Stored over in-process, the same root and band file only; swept ``x ≥ 3`` whatever the
    realized ``x``; grid and deep pooled; the shape with the larger ``ĉ``."""
    m = kt.miss_constant(read(root), ROOT, H)
    x3 = 3.0 * (1 - 1e-12)
    exposure = 1800 / x3 + 1800 / 10.0 + 18000 / 300.0
    assert (m.run_dir, m.source, m.name, m.shape) == ("sweep/stored", "stored", "dF", "row2")
    assert (m.misses, m.vectors, m.x_min) == (590, 1800 * 2 + 18000, 3.0)
    assert math.isclose(m.c_hat, 590 / exposure, rel_tol=1e-12)
    assert math.isclose(m.se, math.sqrt(590) / exposure, rel_tol=1e-12)
    assert math.isclose(m.by_phase["grid"][0], 540 / (1800 / x3 + 180), rel_tol=1e-12)
    assert math.isclose(m.by_phase["deep"][0], 50 / 60, rel_tol=1e-12)
    assert math.isclose(m.committed, 590 / (exposure + 0.25 * 180), rel_tol=1e-12)
    entry = 130 / (1800 / 3 + 180)
    assert entry < m.c_hat
    assert kt.miss_constant(read(root), "f" * 64, H) is None


def test_rows_per_k(root):
    res = kt.build(read(root))
    assert res.run_dir == "k_tunability" and res.C == 1.25 and not res.notes
    assert [r["k"] for r in res.rows] == [1, 2, 3, 4]
    c_hat = res.miss.c_hat
    for r in res.rows:
        k, f = r["k"], _f_ach(r["k"])
        rate = f * 1.25 / (2 * 2.0)
        assert r["configured"] == (k == 3) and r["floor_pred"] == f
        assert math.isclose(r["floor_meas"], c_hat * TAU * E_MAX * 2 ** (N / k), rel_tol=1e-12)
        assert math.isclose(r["rate"], rate) and math.isclose(r["records_per_batch"], 4 * rate)
        assert r["covers"] == (4 * rate <= 1)
        assert math.isclose(r["rate_c1"], f / 4)
        assert r["verifier_s"] == OTHER_S + (1.0 + 0.0125 * k)
        assert math.isclose(r["check5_gflop"], 2.3 * (k + 1))
        assert r["mem_growth_gb"] == 0.89
    assert kt.guaranteed_rate(0.5, 1.0, 4, 2.0) == 0.125


def test_without_calibration_or_sweep_the_notes_say_so(root):
    rec = read(root)
    for r in rec.table("k_source"):
        r["band_file_hash"] = "c" * 64
    res = kt.build(rec)
    assert res.C == 1.0 and res.miss is None and len(res.notes) == 2
    assert all(r["floor_meas"] is None for r in res.rows)
    assert "no measured floor" in kt.caption(res) and "a placeholder" in kt.caption(res)
    assert kt.figure(res) is not None


def test_main_writes_figure_and_table(root, monkeypatch, capsys):
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(root))
    assert kt.main(["--run-dir", "k_old"]) == 0
    out = root / "evaluation" / "results" / "test"
    assert sorted(p.name for p in out.iterdir()) == [
        f"k_tunability.{s}" for s in ("csv", "md", "pdf", "png")]
    with open(out / "k_tunability.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["k"] for r in rows] == ["1", "2", "3", "4"] and list(rows[0]) == list(kt.COLUMNS)
    assert float(rows[0]["check5_s"]) == 2.0125  # k_old's times
    md = (out / "k_tunability.md").read_text()
    assert "C = 1.250 (measured by calibration)" in md and "‖δg‖/‖g‖ = 2 (hand-picked)" in md
    assert "ĉ = " in md and "reproduced the stored step's check-5 verdict" in md
    assert "k_old: k = 1…4, C = 1.2500" in capsys.readouterr().out
