"""The honest SmolLM2 run (A10, milestone M3): a tiny random Llama through the same path, and
the real step as a slow test."""

import os

import pytest
import torch

from setup.config import load_config
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import LlamaComputation
from verification.runs import llama_step
from verification.runs.metrics import MetricsWriter, read_residuals
from verification.runs.mlp_smoke import honest_final
from verification.verifier.residuals import class_summary, tensor_summary

from tests.verification.computation.test_llama import make_records, tiny_config


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
                                  recorder=mw.recorder(llama_step.HONEST.name))
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


def test_a_wrong_published_root_rejects_at_check_1(tiny):
    c, D, w0 = tiny
    r = llama_step.run_honest(c, D, w0, T=1, k=7, h_D=bytes(32), final=w0)
    assert not r.loop.verdict.accepted and r.loop.verdict.rejection.check_id == "1"


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
