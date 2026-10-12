"""The honest SmolLM2 run (A10, milestone M3): a tiny random Llama through the same path, and
the real step as a slow test."""

import json
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
from verification.runs.metrics import MetricsWriter, read_residuals
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
from tests.verification.runs.test_run_verified import K, _band_file

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
