"""A12: the honest run judged by the frozen band file, its per-step summaries, and main()."""

import dataclasses
import json
import math

import pytest

from setup.data import encode_records_file, schedule
from tests.verification.llama_helpers import make_records, tiny_config
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import LlamaComputation
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.computation.interface import snapshot_weights
from verification.prover.step import prove_step
from verification.runs import plain_baseline as pb
from verification.runs import run_verified as rv
from verification.runs.metrics import MetricsWriter, read_records, read_residuals
from verification.runs.scenarios import honest_final
from verification.transcript.store import InMemoryStore
from verification.verifier.bands import Bands
from verification.verifier.calibration import (
    GUARD_FRACTION,
    assert_same_band_source,
    band_file_bytes,
    fit,
    load_bands,
    write_band_file,
)
from verification.verifier.context import ProductStat, StepStats, TensorStat
from verification.verifier.driver import Verifier

K = 7
T_CAL = 2


def _band_file(c, D, w0, path, *, steps=T_CAL, k=K):
    """C1 in miniature: ``steps`` honest steps under a calibrating verifier, fitted and
    written as a band file."""
    tree = dataset_tree(c, D)
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=k, n_steps=steps, w0=w0, bands=None,
                 calibrate=True)
    assert v.start_run(D) is None
    w = w0
    for t in range(1, steps + 1):
        idx = schedule(t, c.n_s, len(D))
        out = prove_step(c, c.build_model(), w, [D[i] for i in idx])
        store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
        assert v.verify_step(t, store) is None
        w = out.w_next
    return write_band_file(path, band_file_bytes(fit(v.stats, k=k)))


@pytest.fixture(scope="module")
def mlp(tmp_path_factory):
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    path = tmp_path_factory.mktemp("bands") / "bands.json"
    _band_file(c, D, w0, path)
    return c, D, w0, load_bands(path, k=K)


# ---- the per-step summary ---------------------------------------------------------------


def _p(m, name, cls, kappa, res):
    return ProductStat(m, name, cls, kappa, res, q=8, p_norm=1.0, nu=1.0, p_abs1=1.0)


def test_step_summary_by_class():
    st = StepStats(
        products=[_p(1, "L0.Y_q", "Y_q", 2.0, (1.0, 3.0)),
                  _p(2, "L1.Y_q", "Y_q", 6.0, (4.0, 0.5)),
                  _p(3, "Lambda", "Lambda", 5.0, (9.0, -2.0)),
                  _p(4, "L0.dX", "dX", 1.0, (0.1, 0.2))],
        tensors=[TensorStat("w.a", "6a", 1.5), TensorStat("w.b", "6a", 2.5),
                 TensorStat("w.b", "6b", 0.25)])
    bands = Bands(tau=20.0, kappa_max=4.0, kappa_classes={"Y_q": 12.0, "Lambda": 10.0},
                  tau_w=4.0)
    s = rv.step_summary(3, st, bands, in_sample=False)
    assert s.class_max == {"Y_q": 4.0, "Lambda": 9.0, "dX": 0.2}
    # each class's largest κ over its own κ_max; a class without one takes the default
    assert s.kappa_ratio == {"Y_q": 0.5, "Lambda": 0.5, "dX": 0.25}
    assert s.watch == {"Lambda": (9.0, -2.0)}
    assert s.rho_max == {"6a": 2.5, "6b": 0.25}
    assert s.global_max == ("Lambda", 9.0) and s.kappa_ratio_max[1] == 0.5
    rec = rv.summary_record("r", "honest", s, bands)
    assert rec["record"] == "residual_summary" and rec["sample"] == "judged"
    assert rec["guard_line"] == GUARD_FRACTION * 20.0 and rec["watch"] == {"Lambda": [9.0, -2.0]}
    assert rv.step_summary(1, st, bands, in_sample=True, watch="dX").watch == \
        {"L0.dX": (0.1, 0.2)}


# ---- the run on the MLP -----------------------------------------------------------------


def test_mlp_run_judged_by_the_band_file(mlp, tmp_path):
    c, D, w0, bands = mlp
    T = 4
    final = honest_final(c, D, w0, T)
    with MetricsWriter(tmp_path, "rv", band_file_hash=bands.source) as mw:
        run = rv.run_verified(c, D, w0, bands, T=T, k=K, h_D=dataset_tree(c, D).root,
                              final=final, writer=mw, recorder=mw.recorder("honest"),
                              watch="forward:layers.*.weight", out=lambda s: None)
    r = run.result
    assert r.passed and r.loop.verdict.accepted
    assert [s.t for s in run.summaries] == [1, 2, 3, 4]
    assert run.calibration_steps == (1, 2)
    assert [s.in_sample for s in run.summaries] == [True, True, False, False]
    assert not any(s.rejected for s in run.summaries)
    for s in run.summaries:  # the summary is the verifier's own numbers, regrouped
        st = r.verifier.stats[s.t]
        for cls, prods in st.by_class().items():
            assert s.class_max[cls] == max(x for p in prods for x in p.normalized)
            assert s.kappa_ratio[cls] == max(p.kappa for p in prods) / bands.kappa_for(cls)
        assert len(s.watch) == 3 and all(len(v) == K for v in s.watch.values())
    assert run.over_guard() == [(s.t, cls, x) for s in run.summaries if s.t > 2
                                for cls, x in s.class_max.items()
                                if x > GUARD_FRACTION * bands.tau]
    assert assert_same_band_source(
        [bands.source, r.loop.verdict.band_source] + [v.band_source for v in run.verifiers]
    ) == bands.source
    rows = read_records(tmp_path / "records.jsonl", rv.SUMMARY_RECORD)
    assert [x["step"] for x in rows] == [1, 2, 3, 4]
    assert [x["sample"] for x in rows] == ["in", "in", "judged", "judged"]
    assert all(x["band_file_hash"] == bands.source for x in rows)
    # every residual of every step is logged as B6's arrays as well
    assert sorted(read_residuals(tmp_path)["honest"]) == [1, 2, 3, 4]
    lines: list[str] = []
    rv.report_summaries(run, out=lines.append)
    text = "\n".join(lines)
    assert "κ / the class's frozen κ_max" in text and "concentration-guard line" in text


def test_mlp_bands_too_tight_reject_a_judged_step(mlp):
    """The run judges live: bands below the honest numbers reject, and the summary says so."""
    c, D, w0, bands = mlp
    tight = dataclasses.replace(bands, tau=1e-6)
    run = rv.run_verified(c, D, w0, tight, T=3, k=K, h_D=dataset_tree(c, D).root,
                          final=honest_final(c, D, w0, 3), out=lambda s: None)
    assert not run.result.passed and run.summaries[-1].rejected
    assert run.result.actual.check_id == "5"


# ---- main() on the tiny Llama -----------------------------------------------------------


@pytest.fixture
def tiny_main(monkeypatch, tmp_path):
    """main() on the tiny Llama: from_config patched, D.bin and a band file written."""
    n = 12
    c = LlamaComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n], seed=1)
    monkeypatch.setattr(LlamaComputation, "from_config",
                        classmethod(lambda cls, cfg: LlamaComputation(
                            tiny_config(), n_s=2, n=n, eta=cfg.require_eta())))
    monkeypatch.setattr(pb, "setup_determinism", lambda cfg: None)
    monkeypatch.setattr(rv, "setup_determinism", lambda cfg: None)
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("VERIF_ETA", "1e-3")
    monkeypatch.setenv("VERIF_STEPS", "3")
    monkeypatch.setenv("VERIF_K", str(K))
    monkeypatch.delenv("VERIF_METRICS", raising=False)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D.bin").write_bytes(encode_records_file(D))
    (tmp_path / "data" / "meta.json").write_text(
        json.dumps({"h_D": dataset_tree(c, D).root.hex()}))
    _band_file(c, D, snapshot_weights(c, c.build_model()), tmp_path / "bands.json")
    return c, D, tmp_path


def test_main_judges_logs_and_matches_the_plain_baseline(tiny_main, capsys):
    _, _, root = tiny_main
    band_bytes = (root / "bands.json").read_bytes()
    assert rv.main(["--no-metrics"]) == 0
    assert "B7: no plain baseline" in capsys.readouterr().out
    assert pb.main(["--no-metrics"]) == 0
    capsys.readouterr()
    assert rv.main(["--pass-steps", "1", "--metrics"]) == 0
    text = capsys.readouterr().out
    assert "B7: final weights bit-identical" in text and "band-file hash equal" in text
    assert (root / "bands.json").read_bytes() == band_bytes  # read, never rewritten
    rows = read_records(root / rv.RUN_NAME / "records.jsonl")
    summ = [x for x in rows if x["record"] == rv.SUMMARY_RECORD]
    assert [x["sample"] for x in summ] == ["in", "in", "judged"]
    assert {x["step"] for x in rows if x["record"] == "time"} == {0, 1, 2, 3}  # 0: run checks
    assert {x["step"] for x in rows if x["record"] == "memory"} == {0, 1}
    assert all(math.isfinite(x["global_max"]) for x in summ)
    assert all(x["band_file_hash"] == load_bands(root / "bands.json").source for x in summ)


def test_main_fails_when_the_plain_baseline_differs(tiny_main, capsys):
    assert pb.main(["--steps", "2", "--no-metrics"]) == 0
    assert rv.main(["--no-metrics"]) == 1
    assert "B7: FAILED" in capsys.readouterr().out
