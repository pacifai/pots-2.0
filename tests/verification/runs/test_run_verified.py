"""A12: the honest run judged by the frozen band file, its per-step summaries, and main()."""

import dataclasses
import json
import math

import pytest

from setup.config import load_config
from setup.data import encode_records_file, schedule
from tests.verification.decoder_helpers import make_records, tiny_config
from verification.commitment.leaves import dataset_tree, transcript_root
from verification.computation.instances import DecoderComputation
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.computation.interface import snapshot_weights
from verification.prover.step import commit, prove_step
from verification.runs import plain_baseline as pb
from verification.runs import run_verified as rv
from verification.runs.metrics import MetricsWriter, read_records, read_residuals
from verification.runs.scenarios import honest_final
from verification.transcript.store import LEAF_FILE, DiskStore, InMemoryStore
from verification.verifier.bands import Bands
from verification.verifier.calibration import (
    GUARD_FRACTION,
    BandFileMismatch,
    assert_same_band_source,
    band_context,
    band_file_bytes,
    fit,
    load_bands,
    write_band_file,
)
from verification.verifier.context import ProductStat, StepStats, TensorStat
from verification.verifier.driver import Verifier

SEED = 1  # π's seed (F8c), the default VERIF_SEED

K = 7
T_CAL = 2


def band_ctx(c, D, k=K):
    """A test band file's context: what the MLP and the tiny Llama have of ``band_context``."""
    return {"h_D": dataset_tree(c, D).root.hex(), "calibration_seed": SEED, "n_s": c.n_s,
            "k": k}


def _band_file(c, D, w0, path, *, steps=T_CAL, k=K, context=None):
    """C1 in miniature: ``steps`` honest steps under a calibrating verifier, fitted and
    written as a band file with ``context`` (default :func:`band_ctx`)."""
    tree = dataset_tree(c, D)
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=k, n_steps=steps, w0=w0, bands=None,
                 calibrate=True, seed=SEED)
    assert v.start_run(D) is None
    w = w0
    for t in range(1, steps + 1):
        idx = schedule(t, c.n_s, len(D), SEED)
        out = prove_step(c, c.build_model(), w, [D[i] for i in idx])
        store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
        assert v.verify_step(t, store) is None
        w = out.w_next
    ctx = band_ctx(c, D, k) if context is None else context
    return write_band_file(path, band_file_bytes(fit(v.stats, k=k), {"context": ctx}))


@pytest.fixture(scope="module")
def mlp(tmp_path_factory):
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    D = synthetic_dataset(c.widths, 40, seed=0)
    w0 = init_weights(c.widths, seed=0)
    path = tmp_path_factory.mktemp("bands") / "bands.json"
    _band_file(c, D, w0, path)
    return c, D, w0, load_bands(path, k=K, context=band_ctx(c, D))


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
    final = honest_final(c, D, w0, T, seed=SEED)
    keep = rv.KeptStepHandoff(tmp_path / rv.TRANSCRIPTS_DIR, 3)
    with MetricsWriter(tmp_path, "rv", band_file_hash=bands.source) as mw:
        run = rv.run_verified(c, D, w0, bands, T=T, k=K, h_D=dataset_tree(c, D).root,
                              final=final, writer=mw, recorder=mw.recorder("honest"),
                              watch="forward:layers.*.weight", out=lambda s: None, seed=SEED,
                              keep=keep)
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
    assert [x["judged"] for x in rows] == [False, False, True, True]
    steps = read_records(tmp_path / "records.jsonl", "step")
    assert [x["judged"] for x in steps] == [False, False, True, True]
    assert all(x["rho_5"] <= 1 and x["rho_kappa"] <= 1 and x["rho_6"] <= 1 for x in steps)
    assert all(x["n_failing"] == {"rho_5": 0, "rho_kappa": 0, "rho_6": 0} for x in steps)
    # step 3's transcript stays on disk, and its root is the step's h
    w = w0
    for t in (1, 2, 3):
        out = prove_step(c, c.build_model(), w, [D[i] for i in schedule(t, c.n_s, len(D), SEED)])
        h, w = commit(c, out).root, out.w_next
    assert run.kept is not None and run.kept.directory == tmp_path / rv.TRANSCRIPTS_DIR / "step_3"
    assert run.kept.root == h and transcript_root(c, DiskStore(run.kept.directory)) == h
    (kr,) = read_records(tmp_path / "records.jsonl", rv.KEPT_RECORD)
    leaf_files = sorted(run.kept.directory.glob("leaf_*.pt"))
    assert len(leaf_files) == c.n_leaves and leaf_files[0].name == LEAF_FILE.format(0)
    assert (kr["path"], kr["root"], kr["step"], kr["files"]) == (
        f"{rv.TRANSCRIPTS_DIR}/step_3", h.hex(), 3, c.n_leaves + 2)
    assert kr["bytes"] == sum(f.stat().st_size for f in run.kept.directory.iterdir())
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
                          final=honest_final(c, D, w0, 3, seed=SEED), out=lambda s: None,
                          seed=SEED)
    assert not run.result.passed and run.summaries[-1].rejected
    assert run.result.actual.check_id == "5"


def test_mlp_keep_after_a_rejection_stores_nothing(mlp, tmp_path):
    c, D, w0, bands = mlp
    tight = dataclasses.replace(bands, tau=1e-6)
    keep = rv.KeptStepHandoff(tmp_path, 2)
    run = rv.run_verified(c, D, w0, tight, T=3, k=K, h_D=dataset_tree(c, D).root,
                          final=honest_final(c, D, w0, 3, seed=SEED), out=lambda s: None,
                          seed=SEED, keep=keep)
    assert run.result.actual.step == 1 and run.kept is None and keep.write() is None
    assert not keep.directory.exists()


def test_kept_step_refuses_files_whose_root_isnt_the_steps_h(mlp, tmp_path):
    """The stored step's root is recomputed from the written files; leaves that aren't the
    held tree's (here step 1's leaves under step 2's tree) raise instead of being kept."""
    c, D, w0, _ = mlp
    held, w = [], w0
    for t in (1, 2):
        out = prove_step(c, c.build_model(), w, [D[i] for i in schedule(t, c.n_s, len(D), SEED)])
        held.append(InMemoryStore.commit_step(c, out))
        w = out.w_next
    (leaves, tree), (_, other) = held
    keep = rv.KeptStepHandoff(tmp_path, 1)
    keep.hold(c, 1, leaves, other)
    with pytest.raises(RuntimeError, match=r"stored step 1: root over the written files .* is "
                                           r"not the step's h"):
        keep.write()
    assert keep.kept is None
    good = rv.KeptStepHandoff(tmp_path / "good", 1)
    good.hold(c, 1, leaves, tree)
    assert good.write().root == tree.root


def test_default_keep_step_follows_the_calibration_seed():
    """Step 4 is stored by default only at the band file's calibration seed (H1); a run at
    another seed stores nothing, and a file without the seed is an error."""
    b = Bands(tau=8.0, kappa_max=1.0, source="abc",
              stats={"calibration_steps": [1, 2, 3], "context": {"calibration_seed": 1}})
    assert rv.default_keep_step(b, 1) == rv.KEEP_STEP == 4
    assert rv.default_keep_step(b, 2) == 0
    with pytest.raises(ValueError, match="records no calibration_seed"):
        rv.default_keep_step(dataclasses.replace(b, stats={"context": {}}), 1)


def test_baseline_seed_and_the_skipped_comparison(mlp, tmp_path):
    """B7 compares only a baseline at the run's seed: the seed comes from the baseline's
    ``final_weights.json``, else from its ``environment`` record."""
    c, D, w0, _ = mlp
    w = honest_final(c, D, w0, 1, seed=SEED)
    base, run = tmp_path / "plain_baseline", tmp_path / "run.json"
    base.mkdir()
    lines: list[str] = []
    assert rv.compare_with_baseline(c, base, run, SEED, out=lines.append)
    assert lines.pop().startswith("B7: skipped, no plain baseline at")
    pb.write_final_weights(base / pb.WEIGHTS_FILE, c, w, w0=w0, losses=[0.0],
                           provenance={"seed": SEED})
    pb.write_final_weights(run, c, w, w0=w0, losses=[0.0], provenance={"seed": SEED})
    assert rv.baseline_seed(base) == SEED
    assert rv.compare_with_baseline(c, base, run, SEED, out=lines.append)
    assert lines.pop().startswith("B7: final weights bit-identical")
    assert rv.compare_with_baseline(c, base, run, 2, out=lines.append)  # skipped, not failed
    assert lines.pop().startswith(f"B7: skipped, the plain baseline ran at seed {SEED} and this "
                                  "run at seed 2")
    # a baseline written before the file held its seed: the environment record has it
    pb.write_final_weights(base / pb.WEIGHTS_FILE, c, w, w0=w0, losses=[0.0])
    assert rv.baseline_seed(base) is None
    with MetricsWriter(base, pb.RUN_NAME, seed=3):
        pass
    assert rv.baseline_seed(base) == 3
    moved = {n: x.clone() for n, x in w.items()}
    moved[c.weight_names[0]].view(-1)[0] += 1.0
    pb.write_final_weights(run, c, moved, w0=w0, losses=[0.0], provenance={"seed": 3})
    assert not rv.compare_with_baseline(c, base, run, 3, out=lines.append)
    assert lines.pop().startswith("B7: FAILED")


def test_mlp_run_at_another_seed_judges_from_step_2(mlp):
    """At a seed other than the band file's (H2), no step is in-sample and step 1 isn't judged
    (EVALUATION_SPEC §2.3)."""
    c, D, w0, bands = mlp
    run = rv.run_verified(c, D, w0, bands, T=3, k=K, h_D=dataset_tree(c, D).root,
                          final=honest_final(c, D, w0, 3, seed=2), out=lambda s: None, seed=2)
    assert run.result.passed
    assert [s.sample for s in run.summaries] == ["not judged", "judged", "judged"]
    assert [s.mark for s in run.summaries] == ["-", " ", " "]


# ---- main() on the tiny Llama -----------------------------------------------------------


@pytest.fixture
def tiny_main(monkeypatch, tmp_path):
    """main() on the tiny Llama: from_config patched, D.bin and a band file written."""
    n = 12
    c = DecoderComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n], seed=1)
    monkeypatch.setattr(DecoderComputation, "from_config",
                        classmethod(lambda cls, cfg: DecoderComputation(
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
    ctx = band_context(load_config(), c, h_D=dataset_tree(c, D).root, k=K)
    _band_file(c, D, snapshot_weights(c, c.build_model()), tmp_path / "bands.json", context=ctx)
    return c, D, tmp_path


def test_main_judges_logs_and_matches_the_plain_baseline(tiny_main, capsys):
    _, _, root = tiny_main
    band_bytes = (root / "bands.json").read_bytes()
    assert rv.main(["--no-metrics"]) == 0
    assert "B7: skipped, no plain baseline" in capsys.readouterr().out
    assert pb.main(["--no-metrics"]) == 0
    capsys.readouterr()
    assert rv.main(["--pass-steps", "1", "--metrics"]) == 0
    text = capsys.readouterr().out
    assert "B7: final weights bit-identical" in text and "band-file hash equal" in text
    assert (root / "bands.json").read_bytes() == band_bytes  # read, never rewritten
    cfg = load_config()
    name = rv.honest_run_name(cfg.model, cfg.attack, cfg.seed)
    assert name == "honest-SmolLM2-135M-Instruct-refusal-bpr0-s1"
    rows = read_records(root / name / "records.jsonl")
    summ = [x for x in rows if x["record"] == rv.SUMMARY_RECORD]
    assert [x["sample"] for x in summ] == ["in", "in", "judged"]
    steps = [x for x in rows if x["record"] == "step"]
    assert [(x["run"], x["attack"], x["bpr_records"], x["bpr_actual"], x["cheat_step"])
            for x in steps] == [(name, "refusal", 0, 0.0, None)] * 3
    assert [x["judged"] for x in steps] == [False, False, True]
    assert {x["step"] for x in rows if x["record"] == "time"} == {0, 1, 2, 3}  # 0: run checks
    assert {x["step"] for x in rows if x["record"] == "memory"} == {0, 1}
    assert {x["step"] for x in rows if x["record"] == "count"} == {0, 1}  # COUNT_STEPS
    # the default kept step, 4, is past T = 3: nothing stored
    assert not (root / name / rv.TRANSCRIPTS_DIR).exists()
    assert not [x for x in rows if x["record"] == rv.KEPT_RECORD]
    assert all(math.isfinite(x["global_max"]) for x in summ)
    assert all(x["band_file_hash"] == Bands.from_file(root / "bands.json").source for x in summ)


def test_main_keeps_the_named_step_on_disk(tiny_main, capsys):
    c, _, root = tiny_main
    assert rv.main(["--keep-step", "2", "--no-metrics"]) == 0
    cfg = load_config()
    d = root / rv.honest_run_name(cfg.model, cfg.attack, cfg.seed) / rv.TRANSCRIPTS_DIR / "step_2"
    store = DiskStore(d)
    assert transcript_root(c, store) == store.root
    assert len(list(d.glob("leaf_*.pt"))) == c.n_leaves
    assert f"stored step 2: {d}" in capsys.readouterr().out
    assert rv.main(["--keep-step", "0", "--steps", "2", "--metrics"]) == 0
    assert "stored step" not in capsys.readouterr().out
    assert DiskStore(d).root == store.root  # a run that keeps nothing leaves it alone


def test_main_judges_another_seed_by_the_seed_1_band_file(tiny_main, monkeypatch, capsys):
    """EVALUATION_SPEC §3.2: the honest runs at seeds 2–5 are judged by H1's file (seed 1).
    They store no step by default, and B7 skips H1's plain baseline, which trained on other
    batches."""
    _, _, root = tiny_main
    assert Bands.from_file(root / "bands.json").stats["context"]["calibration_seed"] == 1
    assert pb.main(["--steps", "4", "--no-metrics"]) == 0  # at seed 1
    capsys.readouterr()
    monkeypatch.setenv("VERIF_SEED", "2")
    assert rv.main(["--steps", "4", "--no-metrics"]) == 0
    text = capsys.readouterr().out
    assert ("stored step: none, the run's seed 2 isn't the band file's calibration seed 1"
            in text)
    assert "B7: skipped, the plain baseline ran at seed 1 and this run at seed 2" in text
    cfg = load_config()
    assert not (root / rv.honest_run_name(cfg.model, cfg.attack, 2) / rv.TRANSCRIPTS_DIR).exists()
    # the same run at seed 1 stores step 4 and matches the baseline
    monkeypatch.setenv("VERIF_SEED", "1")
    assert rv.main(["--steps", "4", "--no-metrics"]) == 0
    text = capsys.readouterr().out
    d = root / rv.honest_run_name(cfg.model, cfg.attack, 1) / rv.TRANSCRIPTS_DIR / "step_4"
    assert f"stored step 4: {d}" in text and "B7: final weights bit-identical" in text


def test_main_refuses_a_band_file_fitted_for_another_run(tiny_main, monkeypatch):
    monkeypatch.setenv("VERIF_ETA", "2e-3")
    with pytest.raises(BandFileMismatch, match=r"eta: file 0.001, run 0.002.*rerun verification"):
        rv.main(["--no-metrics"])


def test_main_fails_when_the_plain_baseline_differs(tiny_main, capsys):
    assert pb.main(["--steps", "2", "--no-metrics"]) == 0
    assert rv.main(["--no-metrics"]) == 1
    assert "B7: FAILED" in capsys.readouterr().out
