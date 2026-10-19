"""A11's calibration run: the gradient-coherence diagnostic and ``_run``'s gates (the rest runs
in the slow suite)."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tests.verification.decoder_helpers import make_records, tiny_config
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.runs import calibrate as cal_run
from verification.runs.calibrate import gradient_coherence
from verification.verifier.bands import Bands
from verification.verifier.calibration import band_file_bytes, fit, load_bands
from verification.verifier.context import ProductStat, StepStats, TensorStat


@pytest.mark.parametrize("blocks", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_gradient_coherence_runs_under_the_runs_autocast(dtype, blocks, monkeypatch):
    """F6b: the per-record gradients come from training's forward, under autocast at
    ``(fp32, bf16)``, and they sum to the batch gradient (training's, in head blocks) up to
    rounding."""
    c = DecoderComputation(tiny_config(), n_s=3, n=12, eta=1e-3, compute_dtype=dtype,
                           head_blocks=blocks)
    model = c.build_model()
    w = snapshot_weights(c, model)
    seen = []
    features = c.features

    def spy(model, batch):
        seen.append(torch.is_autocast_enabled("cpu"))
        return features(model, batch)

    monkeypatch.setattr(c, "features", spy)
    coh = gradient_coherence(c, model, w, make_records((12, 7, 9)))
    assert seen == [dtype == torch.bfloat16] * 2  # training's pass, then the per-record one
    assert coh.B == 3 and len(coh.cosines) == 3 and all(n > 0 for n in coh.norms)
    assert coh.batch_check < (1e-6 if dtype == torch.float32 else 1e-2)
    assert all(p.grad is None for p in model.parameters())


K = 2


@dataclass
class _Coh:
    value: float = 1.0


def _stats():
    p = ProductStat(1, "L1.Y_q", "Y_q", 2.0, (1.0, 1.0), q=8, p_norm=1.0, nu=1.0, p_abs1=1.0)
    return {1: StepStats(products=[p], tensors=[TensorStat("w.a", "6a", 1.5)])}


def _install(monkeypatch, *, guard_ok, k_required=K):
    data = band_file_bytes(fit(_stats(), k=K), {})
    bands = Bands.from_json(data)
    cal = SimpleNamespace(s_h=1.0, s_h_class="Y_q", tau=8.0, guard_ok=guard_ok,
                          global_max=1.0 if guard_ok else 5.0, guard_limit=4.0)
    verdict = SimpleNamespace(accepted=True, steps_verified=1, rejection=None,
                              band_source=bands.source)
    run = SimpleNamespace(result=SimpleNamespace(loop=SimpleNamespace(verdict=verdict)),
                          cal=cal,
                          sizing=SimpleNamespace(k=k_required, e_max=0.1, e_cal_at=1),
                          sizing_qmax=SimpleNamespace(k=k_required),
                          largest=SimpleNamespace(u_ratio_max=0.1),
                          floor=SimpleNamespace(ratio_max=0.1), data=data, bands=bands,
                          freeze_rejection=None)
    judged = SimpleNamespace(loop=SimpleNamespace(verdict=verdict), passed=True,
                             verifier=SimpleNamespace(_schedule=lambda step: [0]))
    monkeypatch.setattr(cal_run, "calibrate", lambda *a, **kw: run)
    monkeypatch.setattr(cal_run, "report_calibration", lambda *a, **kw: None)
    monkeypatch.setattr(cal_run, "judged_run", lambda *a, **kw: judged)
    monkeypatch.setattr(cal_run, "gradient_coherence", lambda *a, **kw: _Coh())
    monkeypatch.setattr(cal_run, "report_coherence", lambda *a, **kw: None)
    rows = []
    monkeypatch.setattr(cal_run, "_write", lambda mw, row: rows.append(dict(row)))
    return run, rows


def _go(tmp_path, **kw):
    return cal_run._run(None, [0], {}, {"T": 1}, None, k=K, T_run=1, context={},
                        band_file=tmp_path / "bands.json", run_dir=tmp_path / "run",
                        models=lambda: None, **kw)


def test_a_failed_guard_stops_without_the_flag(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, guard_ok=False)
    assert _go(tmp_path) == 2
    assert not (tmp_path / "bands.json").exists()
    assert not (tmp_path / "bands.json.guard_failed.json").exists()
    assert "STOP: the concentration guard failed" in capsys.readouterr().out


def test_the_override_writes_the_band_file_and_a_marker(monkeypatch, tmp_path, capsys):
    run, rows = _install(monkeypatch, guard_ok=False)
    assert _go(tmp_path, write_despite_guard=True) == 0
    assert (tmp_path / "bands.json").read_bytes() == run.data
    marker = json.loads((tmp_path / "bands.json.guard_failed.json").read_text())
    assert marker == {"guard_ok": False, "global_max": 5.0, "guard_limit": 4.0,
                      "reason": "calibrate --write-bands-despite-guard (profiling only)"}
    out = capsys.readouterr().out
    assert any(line.startswith("PROFILING OVERRIDE:") for line in out.splitlines())
    assert "WARNING: band file" in out  # the loader's line
    calibration = next(r for r in rows if r["record"] == "calibration")
    assert calibration["band_file_override"] == "guard_failed" and calibration["guard_ok"] is False


def test_the_override_keeps_the_other_gates(monkeypatch, tmp_path, capsys):
    _install(monkeypatch, guard_ok=False, k_required=K + 1)
    assert _go(tmp_path, write_despite_guard=True) == 2
    assert not (tmp_path / "bands.json").exists()
    assert "STOP (P10d)" in capsys.readouterr().out


def test_a_passing_guard_removes_a_stale_marker(monkeypatch, tmp_path):
    _, rows = _install(monkeypatch, guard_ok=True)
    marker = tmp_path / "bands.json.guard_failed.json"
    marker.write_text("{}")
    assert _go(tmp_path, write_despite_guard=True) == 0
    assert (tmp_path / "bands.json").exists() and not marker.exists()
    assert "band_file_override" not in next(r for r in rows if r["record"] == "calibration")


def test_the_loader_warns_when_the_marker_exists(tmp_path, capsys):
    path = tmp_path / "bands.json"
    path.write_bytes(band_file_bytes(fit(_stats(), k=K), {}))
    load_bands(path, k=K, context={})
    assert capsys.readouterr().out == ""
    (tmp_path / "bands.json.guard_failed.json").write_text("{}")
    bands = load_bands(path, k=K, context={})
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "failed concentration guard" in out
    assert bands == Bands.from_file(path)
