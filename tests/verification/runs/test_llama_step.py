"""The honest SmolLM2 run (A10, milestone M3): a tiny random Llama through the same path, and
the real step as a slow test."""

import json

import pytest
import torch

from setup.config import load_config
from setup.data import encode_records_file
from setup.records import encode_record
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import LlamaComputation
from verification.runs import llama_step
from verification.runs.metrics import MetricsWriter, read_residuals
from verification.runs.loop import ProverFault
from verification.runs.scenarios import HONEST, Expected, Scenario, honest_final, run_scenario
from verification.verifier.residuals import class_summary, tensor_summary

from tests.verification.llama_helpers import make_records, tiny_config


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    c = LlamaComputation(tiny_config(), n_s=2, n=12, eta=1e-3)
    D = make_records((12, 7, 10, 5), seed=1)
    w0 = {n: w.detach().clone() for n, w in c.build_model().named_parameters()}
    return c, D, w0


def test_tiny_honest_run_is_accepted_and_reported(tiny, tmp_path):
    c, D, w0 = tiny
    T = 2
    final = honest_final(c, D, w0, T)
    with MetricsWriter(tmp_path, "llama_step", device="cpu", model="tiny", corpus="synthetic",
                       seed=0, config={}, band_file_hash=None, h_D=dataset_tree(c, D).root) as mw:
        r = llama_step.run_honest(c, D, w0, T=T, k=7, h_D=dataset_tree(c, D).root, final=final,
                                  recorder=mw.recorder(HONEST.name))
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
    r = llama_step.run_honest(c, D3, w0, T=1, k=7, h_D=h_D, final=w0)
    assert not r.loop.verdict.accepted and r.actual.check_id == "1"
    assert "h_D" in r.loop.rejection.detail


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
    r = run_scenario(c, D, w0, scenario, T=1, k=7, final=w0)
    assert r.passed, r.actual


@pytest.mark.slow
def test_real_smollm2_step_is_accepted(monkeypatch, capsys):
    """M3: one honest SmolLM2-135M step from W_0 on π(1) of the committed D, accepted with the
    provisional bands. Needs D.bin under ``$VERIF_OUTPUT_DIR/data``."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    if not (load_config().data_dir / "D.bin").exists():
        pytest.skip("no D.bin: run verification.runs.materialize_data or set VERIF_OUTPUT_DIR")
    assert llama_step.main(["--steps", "1", "--no-metrics"]) == 0
    out = capsys.readouterr().out
    assert "largest class RMS" in out and "check 6b: max ρ" in out
    print(out)
