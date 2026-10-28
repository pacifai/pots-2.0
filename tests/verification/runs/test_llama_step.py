"""The honest SmolLM2 run (A10, milestone M3): a tiny random Llama through the same path, and
the real step as a slow test."""

import json
import math
import re
from types import SimpleNamespace

import pytest
import torch

from setup.config import load_config
from setup.data import FULL_SCALE_TOKENIZERS, TEST_SCALE_TOKENIZER, encode_records_file
from setup.records import encode_record
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.parameters import load_protocol_config
from verification.runs import llama_step
from verification.runs import prover_spec
from verification.runs.metrics import MetricsWriter, read_records, read_residuals
from verification.runs.loop import ProverFault
from verification.runs.scenarios import (
    HONEST,
    Expected,
    ReusedModel,
    Scenario,
    honest_final,
    run_scenario,
)
from verification.verifier.calibration import BandFileMismatch, band_context, load_bands
from verification.verifier.residuals import class_summary, tensor_summary

from tests.verification.decoder_helpers import make_records, tiny_config
from tests.verification.runs import pipelined_helpers as P  # noqa: F401
from tests.verification.runs.test_fs33d_entry_points import tiny_spec  # noqa: F401 (fixture)
from tests.verification.runs.test_run_verified import (  # noqa: F401 (tiny_main is a fixture)
    K,
    _band_file,
    tiny_main,
)

SEED = 1  # π's seed (F8c), the default VERIF_SEED


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    D = make_records((12, 7, 10, 5), seed=1)
    w0 = {n: w.detach().clone() for n, w in c.build_model().named_parameters()}
    return c, D, w0


def test_tiny_honest_run_is_accepted_and_reported(tiny, tmp_path):
    c, D, w0 = tiny
    T = 2
    final = honest_final(c, D, w0, T, seed=SEED)
    with MetricsWriter(tmp_path, "llama_step", device="cpu", model="tiny", corpus="synthetic",
                       seed=0, config={}, band_file_hash=None, h_D=dataset_tree(c, D).root) as mw:
        r = llama_step.run_honest(c, D, w0, T=T, k=7, h_D=dataset_tree(c, D).root, final=final,
                                  recorder=mw.recorder(HONEST.name), seed=SEED)
    assert r.passed and r.loop.verdict.accepted and r.loop.verdict.steps_verified == T
    rows = class_summary(r.verifier.stats)
    assert sum(x.count for x in rows) == T * c.M
    assert all(x.max < r.verifier.bands.tau for x in rows)
    assert {x.check_id for x in tensor_summary(r.verifier.stats)} == {"6a", "6b"}
    # the residual archives give back the same table
    assert class_summary(read_residuals(tmp_path)["honest"]) == rows
    lines = []
    llama_step.report_residuals(r, lines.append)
    llama_step.report_costs(r, out=lines.append)
    assert any(x.startswith("largest class RMS") for x in lines)
    assert any(x.startswith("check 6a: max ρ") for x in lines)
    assert sum(x.startswith("step ") for x in lines) == T


def write_data_dir(path, D, h_D):
    path.mkdir()
    (path / "D.bin").write_bytes(encode_records_file(D))
    (path / "meta.json").write_text(json.dumps({"h_D": h_D.hex()}))
    return path


def test_load_committed_dataset_takes_h_D_from_meta_json(tiny, tmp_path):
    c, D, w0 = tiny
    root = dataset_tree(c, D).root
    D2, h_D = llama_step.load_committed_dataset(c, write_data_dir(tmp_path / "ok", D, root))
    assert h_D == root and [encode_record(r) for r in D2] == [encode_record(r) for r in D]
    # a tampered meta.json is what the verifier compares with: check 1 rejects
    forged = bytes([root[0] ^ 1]) + root[1:]
    D3, h_D = llama_step.load_committed_dataset(c, write_data_dir(tmp_path / "bad", D, forged))
    assert h_D == forged
    r = llama_step.run_honest(c, D3, w0, T=1, k=7, h_D=h_D, final=w0, seed=SEED)
    assert not r.loop.verdict.accepted and r.actual.check_id == "1"
    assert "h_D" in r.loop.rejection.detail


SMOLLM2, QWEN = TEST_SCALE_TOKENIZER, FULL_SCALE_TOKENIZERS[2]


def built_for(pin, h_D):
    """``meta.json``'s model fields as ``materialize_data.build_meta`` writes them for ``pin``."""
    run = {"repo": pin.repo, "revision": pin.revision, "eos_token_id": pin.eos_token_id}
    return {"tokenizers": {"run": run}, "pad_id": pin.eos_token_id, "h_D": h_D.hex()}


def model(pin=None, *, revision=None, pad_id=None, n=12):
    """What ``load_committed_dataset`` reads of a run's computation: a stub, since a tiny config
    can't hold a real tokenizer's pad id."""
    source = None if pin is None else (pin.repo, revision or pin.revision)
    pad = pad_id if pad_id is not None else (2 if pin is None else pin.eos_token_id)
    return SimpleNamespace(source=source, pad_id=pad, n=n)


def test_committed_data_must_be_built_for_the_run_model(tiny, tmp_path):
    c, D, _ = tiny
    root = dataset_tree(c, D).root
    data = write_data_dir(tmp_path / "smollm2", D, root)
    (data / "meta.json").write_text(json.dumps(built_for(SMOLLM2, root)))
    D2, h_D = llama_step.load_committed_dataset(model(SMOLLM2), data)  # test scale's path
    assert h_D == root and [encode_record(r) for r in D2] == [encode_record(r) for r in D]
    # a Qwen run pointed at SmolLM2's data fails before it reads D.bin
    with pytest.raises(ValueError, match=f"built for {SMOLLM2.repo}@.*model is {QWEN.repo}@"):
        llama_step.load_committed_dataset(model(QWEN), data)
    for other in (model(SMOLLM2, revision="f" * 40), model(SMOLLM2, pad_id=0)):
        with pytest.raises(ValueError, match="Rerun verification.runs.materialize_data"):
            llama_step.load_committed_dataset(other, data)
    # a meta.json without the model fields matches no model
    bare = write_data_dir(tmp_path / "bare", D, root)
    with pytest.raises(ValueError, match="built for None@None with pad id None"):
        llama_step.load_committed_dataset(model(SMOLLM2), bare)
    # a random test model has no tokenizer: only meta.json's pad id, when it has one, counts
    assert llama_step.load_committed_dataset(model(), bare)[1] == root
    assert llama_step.load_committed_dataset(model(), data)[1] == root
    with pytest.raises(ValueError, match="a random test model with pad id 0"):
        llama_step.load_committed_dataset(model(pad_id=0), data)


class _BumpWNext(ProverFault):
    """Move one entry of ``W_{t+1}[name]`` by ``delta`` before commitment."""

    def __init__(self, name, delta=1e-4):
        self.name, self.delta = name, delta

    def emit(self, t, out):
        w = out.w_next[self.name].clone()
        w.view(-1)[3] += self.delta
        return out.with_w_next({**out.w_next, self.name: w})


@pytest.mark.parametrize("name", ["model.layers.0.input_layernorm.weight",
                                  "model.embed_tokens.weight"], ids=["gamma", "W_E"])
def test_a_forged_glue_weight_update_rejects_at_6b(tiny, name):
    c, D, w0 = tiny
    assert name in c.glue_gradient_weights
    scenario = Scenario("bad-glue-w-next", name, _BumpWNext(name), Expected(1, "6b", "failed"))
    r = run_scenario(c, D, w0, scenario, T=1, k=7, final=w0, seed=SEED)
    assert r.passed, r.actual


def _same(a, b):
    return list(a) == list(b) and all(torch.equal(a[n], b[n]) for n in a)


def test_one_reused_model_gives_the_fresh_build_results(tiny):
    """O4, as ``main`` runs it: W_0, check 8's reference and the prover share one model (its RoPE
    buffer included), and every committed and checked number equals a fresh build per run."""
    c, D, _ = tiny
    h_D = dataset_tree(c, D).root
    w0 = snapshot_weights(c, c.build_model())
    fresh_final = honest_final(c, D, w0, 2, seed=SEED)
    fresh = llama_step.run_honest(c, D, w0, T=2, k=7, h_D=h_D, final=fresh_final, seed=SEED)

    models = ReusedModel(c.build_model)
    w0_r = snapshot_weights(c, models())
    final = honest_final(c, D, w0_r, 2, build_model=models, seed=SEED)
    reused = llama_step.run_honest(c, D, w0_r, T=2, k=7, h_D=h_D, final=final,
                                   build_model=models, seed=SEED)
    assert _same(w0_r, w0) and _same(final, fresh_final)
    assert reused.passed and fresh.passed
    assert _same(reused.loop.w_final, fresh.loop.w_final)
    assert [s.loss for s in reused.loop.steps] == [s.loss for s in fresh.loop.steps]
    assert ({t: s.products + s.tensors for t, s in reused.verifier.stats.items()}
            == {t: s.products + s.tensors for t, s in fresh.verifier.stats.items()})
    models()  # the model is still as built after both runs


def _tiny_main_env(monkeypatch, tmp_path):
    n = 12
    c = DecoderComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n], seed=1)
    monkeypatch.setattr(DecoderComputation, "from_config",
                        classmethod(lambda cls, cfg: DecoderComputation(
                            tiny_config(), n_s=2, n=n, eta=cfg.require_eta())))
    monkeypatch.setattr(llama_step, "setup_determinism", lambda cfg: None)
    for var, value in (("VERIF_OUTPUT_DIR", str(tmp_path)), ("VERIF_ETA", "1e-3"),
                       ("VERIF_K", str(K)), ("VERIF_SEED", str(SEED))):
        monkeypatch.setenv(var, value)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D.bin").write_bytes(encode_records_file(D))
    h_D = dataset_tree(c, D).root
    (tmp_path / "data" / "meta.json").write_text(json.dumps({"h_D": h_D.hex()}))


@pytest.mark.parametrize("flag, counted", [(None, True), ("--count", True), ("--no-count", False)])
def test_main_prints_each_steps_root_and_skips_the_count_on_request(
        monkeypatch, tmp_path, capsys, flag, counted):
    _tiny_main_env(monkeypatch, tmp_path)
    calls = []
    real = llama_step.count_run
    monkeypatch.setattr(llama_step, "count_run", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert llama_step.main(["--steps", "2", "--metrics", *([flag] if flag else [])]) == 0
    out = capsys.readouterr().out.splitlines()
    roots = [x for x in out if re.fullmatch(r"step\s+\d+ root [0-9a-f]{64}", x)]
    assert [x.split()[1] for x in roots] == ["1", "2"] and roots[0] != roots[1]
    last_cost = max(i for i, x in enumerate(out) if x.startswith("process lifetime"))
    assert out.index(roots[0]) > last_cost
    assert "   actual:   accept" in out and "   PASS" in out
    assert bool(calls) is counted


def test_main_judges_with_a_band_file_of_this_run(monkeypatch, tmp_path, capsys):
    n = 12
    c = DecoderComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n], seed=1)
    monkeypatch.setattr(DecoderComputation, "from_config",
                        classmethod(lambda cls, cfg: DecoderComputation(
                            tiny_config(), n_s=2, n=n, eta=cfg.require_eta())))
    monkeypatch.setattr(llama_step, "setup_determinism", lambda cfg: None)
    for var, value in (("VERIF_OUTPUT_DIR", str(tmp_path)), ("VERIF_ETA", "1e-3"),
                       ("VERIF_K", str(K)), ("VERIF_SEED", str(SEED))):
        monkeypatch.setenv(var, value)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D.bin").write_bytes(encode_records_file(D))
    h_D = dataset_tree(c, D).root
    (tmp_path / "data" / "meta.json").write_text(json.dumps({"h_D": h_D.hex()}))
    path = tmp_path / "bands.json"
    bands = _band_file(c, D, snapshot_weights(c, c.build_model()), path,
                       context=band_context(load_config(), c, h_D=h_D, k=K))
    assert llama_step.main(["--steps", "2", "--no-metrics", "--bands", str(path)]) == 0
    out = capsys.readouterr().out
    assert f"bands band file {bands.source[:16]}" in out and "PASS" in out
    assert f"(band file {bands.source[:16]}…: τ = {bands.tau:g}" in out
    # a run at seed 2 is judged by the file calibrated at seed 1 (EVALUATION_SPEC §3.2)
    assert bands.stats["context"]["calibration_seed"] == 1
    monkeypatch.setenv("VERIF_SEED", "2")
    assert llama_step.main(["--steps", "1", "--no-metrics", "--bands", str(path)]) == 0
    assert "PASS" in capsys.readouterr().out
    monkeypatch.setenv("VERIF_ETA", "2e-3")
    with pytest.raises(BandFileMismatch, match=r"\(eta: file 0.001, run 0.002\)"):
        llama_step.main(["--steps", "1", "--no-metrics", "--bands", str(path)])


def _work_rows_of(rows, c):
    steps = [r for r in rows if r.get("record") == "step"]
    return steps, llama_step.work_rows(c, llama_step.RUN_NAME, "honest", steps)


def _marker_traced(out):
    lines = [json.loads(x) for x in (out / "markers.jsonl").read_text().splitlines()]
    return {m["step"]: m["traced"] for m in lines if m["kind"] == "step"}


def test_sequential_metrics_run_traces_the_step_and_writes_layout_and_work_rows(
        monkeypatch, tmp_path, capsys):
    """VERIF_TRACE_STEP=2 over 3 steps with --no-count: step 2's trace and marker, check 5's
    layout, and the shape-derived count/storage rows equal to work_rows over the step records."""
    _tiny_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VERIF_TRACE_STEP", "2")
    assert llama_step.main(["--steps", "3", "--metrics", "--no-count"]) == 0
    assert "trace:" not in capsys.readouterr().out
    out = tmp_path / llama_step.RUN_NAME
    assert (out / "trace" / "step_00002.json.gz").exists()
    assert not (out / "trace" / "step_00001.json.gz").exists()
    assert _marker_traced(out) == {1: False, 2: True, 3: False}
    assert (out / "c5_layout.json").exists()
    rows = read_records(out / "records.jsonl")
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    steps, want = _work_rows_of(rows, c)
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert all(s["transcript_bytes"] and s["leaves"] for s in steps)
    assert [s["known"] for s in steps] == [0, 0, 0]  # a CPU commit takes no arrival digest
    got = [r for r in rows if r["record"] in ("count", "storage")]
    assert got == want and sum(r["record"] == "storage" for r in got) == 3


def test_metrics_off_with_a_trace_step_says_it_needs_metrics(monkeypatch, tmp_path, capsys):
    _tiny_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VERIF_TRACE_STEP", "2")
    assert llama_step.main(["--steps", "2", "--no-metrics"]) == 0
    out = capsys.readouterr().out
    assert out.count("trace:") == 1 and "needs --metrics" in out
    assert not (tmp_path / llama_step.RUN_NAME).exists()


def test_pipelined_metrics_run_prints_no_trace_and_writes_layout_and_work_rows(
        tiny_main, tiny_spec, monkeypatch, capsys):  # noqa: F811
    c, _, root = tiny_main
    monkeypatch.setattr(llama_step, "setup_determinism", prover_spec.setup_determinism)
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    monkeypatch.setenv("VERIF_TRACE_STEP", "2")
    assert llama_step.main(["--steps", "3", "--metrics", "--no-count", "--bands",
                            str(root / "bands.json")]) == 0
    text = capsys.readouterr().out.splitlines()
    want = ("trace: VERIF_TRACE_STEP=2 is not traced in the pipelined loop "
            "(the profiler records one process)")
    assert text.count(want) == 1
    assert text.index(want) < next(i for i, x in enumerate(text) if x.startswith("step "))
    out = root / llama_step.RUN_NAME
    assert not (out / "trace").exists()
    assert not any(_marker_traced(out).values())
    assert (out / "c5_layout.json").exists()
    rows = read_records(out / "records.jsonl")
    steps, want_rows = _work_rows_of(rows, c)
    # the pipelined step records (written after the merge) hold the work counts
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert all(s["transcript_bytes"] and s["leaves"] for s in steps)
    assert all(s["known"] is not None for s in steps)  # the pipelined loop records it too
    assert [r for r in rows if r["record"] in ("count", "storage")] == want_rows


def test_count_pass_writes_no_work_rows(monkeypatch, tmp_path):
    _tiny_main_env(monkeypatch, tmp_path)
    assert llama_step.main(["--steps", "2", "--metrics", "--count"]) == 0
    rows = read_records(tmp_path / llama_step.RUN_NAME / "records.jsonl")
    counts = [r for r in rows if r["record"] in ("count", "storage")]
    assert counts
    # two sources would collide: one count row per side, component and step, one storage row
    # per step
    keys = [(r["record"], r.get("side"), r.get("component"), r["step"]) for r in counts]
    assert len(keys) == len(set(keys))


def test_work_rows_still_importable_from_bring_up():
    from verification.runs import bring_up

    assert bring_up.work_rows is llama_step.work_rows
    assert bring_up._step_records is llama_step._step_records


@pytest.mark.slow
def test_real_smollm2_step_is_accepted(monkeypatch, capsys):
    """M3: one honest SmolLM2-135M step from W_0 on π(1) of the committed D, accepted when
    judged by the band file C1 fitted for this run (``parameters.band_file``, P10a).

    Needs D.bin under ``$VERIF_OUTPUT_DIR/data`` and a band file whose context matches the
    run's; it skips otherwise. Not the provisional bands: their ``τ = 8`` sits below honest
    ``Λ`` residuals at ``k = 9`` (11.5 at step 1), so they reject this honest step."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg, pc = load_config(), load_protocol_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip("no D.bin: run verification.runs.materialize_data or set VERIF_OUTPUT_DIR")
    c = DecoderComputation.from_config(cfg)
    _, h_D = llama_step.load_committed_dataset(c, cfg.data_dir)
    try:
        load_bands(pc.band_file, k=pc.k, context=band_context(cfg, c, h_D=h_D, k=pc.k))
    except (FileNotFoundError, BandFileMismatch) as e:
        pytest.skip(f"run verification.runs.calibrate: {e}")
    assert llama_step.main(["--steps", "1", "--no-metrics", "--bands", str(pc.band_file)]) == 0
    out = capsys.readouterr().out
    assert "largest class RMS" in out and "check 6b: max ρ" in out
    print(out)


def _tiny_leaf_counts(c):
    n_w = len(c.weight_names)
    dk = [pl for pl in c.product_leaves if c.product_class(c.product(pl.m0)) == "dK"]
    psize = torch.empty((), dtype=c.product_dtype).element_size()
    wsize = torch.empty((), dtype=c.weight_dtype).element_size()
    w_bytes = sum(wsize * math.prod(s) for s in c.weight_shapes.values())
    p_bytes = [psize * math.prod(pl.shape) for pl in c.product_leaves]  # O18: per leaf
    return n_w, dk, w_bytes, p_bytes


def test_work_rows_on_cuda_commit_hashes_records_weights_and_dk_only(tiny, monkeypatch):
    """With the prover on CUDA the commit's row counts the records, W_t, W_{t+1} and the dK
    product leaves (the rest come from arrival digests); check 2's row keeps every leaf."""
    c = tiny[0]
    monkeypatch.setattr(c, "device", torch.device("cuda"))
    n_w, dk, w_bytes, p_bytes = _tiny_leaf_counts(c)
    rec_n, rec_bytes = c.n_s, 1234  # the batch records, a made-up byte total
    leaves = rec_n + 2 * n_w + c.n_product_leaves
    nbytes = rec_bytes + 2 * w_bytes + sum(p_bytes)
    step = {"step": 1, "leaves": leaves, "transcript_bytes": nbytes}
    rows = llama_step.work_rows(c, "r", "honest", [step])
    by = {r["component"]: r for r in rows if r["record"] == "count"}
    want_n = rec_n + 2 * n_w + len(dk)
    want_bytes = (rec_bytes + 2 * w_bytes
                  + sum(b for pl, b in zip(c.product_leaves, p_bytes)
                        if c.product_class(c.product(pl.m0)) == "dK"))
    com = by["P3.commit"]
    assert len(dk) > 0 and want_n < leaves
    assert com["hash_leaves"] == com["hash_calls"] == want_n
    assert com["hash_in_bytes"] == want_bytes + want_n and com["hash_out_bytes"] == 32 * want_n
    chk = by["2"]
    assert chk["hash_leaves"] == chk["hash_calls"] == leaves
    assert chk["hash_in_bytes"] == nbytes + leaves and chk["hash_out_bytes"] == 32 * leaves


def _cuda_step(c, known, t=1):
    n_w, dk, w_bytes, p_bytes = _tiny_leaf_counts(c)
    leaves = c.n_s + 2 * n_w + c.n_product_leaves
    step = {"step": t, "leaves": leaves, "transcript_bytes": 1234 + 2 * w_bytes + sum(p_bytes)}
    return {**step, "known": known}, c.n_s + 2 * n_w + len(dk)


def test_work_rows_check_the_measured_known_digests_against_the_rule(tiny, monkeypatch, capsys):
    c = tiny[0]
    monkeypatch.setattr(c, "device", torch.device("cuda"))
    good, want_n = _cuda_step(c, 0)
    good["known"] = good["leaves"] - want_n  # the rule's count: every non-dK product known
    bad, _ = _cuda_step(c, good["known"] - 1, t=2)  # one product lacks its arrival digest
    rows = llama_step.work_rows(c, "r", "honest", [good, bad])
    err = capsys.readouterr().err.splitlines()
    assert err == [f"WARNING step 2: the commit hashed {want_n + 1} leaves (leaves "
                   f"{bad['leaves']} minus known {bad['known']}), the work-row rule counts "
                   f"{want_n}; its hash floor is off"]
    commits = [r for r in rows if r.get("component") == "P3.commit"]
    assert [r["known"] for r in commits] == [good["known"], bad["known"]]
    assert all(r["hash_leaves"] == want_n for r in commits)  # the rule's count stays
    assert all("known" not in r for r in rows if r.get("component") not in ("P3.commit",)
               and r["record"] == "count")


def test_work_rows_without_known_warn_nothing_and_add_no_field(tiny, monkeypatch, capsys):
    c = tiny[0]
    monkeypatch.setattr(c, "device", torch.device("cuda"))
    step, _ = _cuda_step(c, None)
    del step["known"]
    rows = llama_step.work_rows(c, "r", "honest", [step, {**step, "step": 2, "known": None}])
    assert capsys.readouterr().err == ""
    assert not any("known" in r for r in rows)


def test_work_rows_on_cpu_commit_keeps_the_whole_transcript(tiny):
    c = tiny[0]
    assert c.device.type == "cpu"
    step = {"step": 1, "leaves": 100, "transcript_bytes": 5000}
    by = {r["component"]: r for r in llama_step.work_rows(c, "r", "honest", [step])
          if r["record"] == "count"}
    for name in ("P3.commit", "2"):
        assert by[name]["hash_leaves"] == by[name]["hash_calls"] == 100
        assert by[name]["hash_in_bytes"] == 5100 and by[name]["hash_out_bytes"] == 3200
    assert by["train.forward"]["hash_leaves"] == 0
