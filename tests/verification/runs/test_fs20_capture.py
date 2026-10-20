"""FS20: the capture record, the step's transcript bytes and leaf count, and the environment's
compute dtype."""

import time

import pytest
import torch

from tests.verification.decoder_helpers import make_records, tiny_config
from verification.commitment.leaves import leaf_totals
from verification.computation.instances import DecoderComputation
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.prover.capture import MatmulCapture
from verification.prover.step import prove_step
from verification.runs import loop as loop_mod
from verification.runs import metrics
from verification.runs.metrics import TimeRecorder, count_pass, read_records
from verification.runs.scenarios import HONEST, honest_final, run_scenario

SEED = 1
K = 7


def _mlp():
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    return c, synthetic_dataset(c.widths, 40, seed=0), init_weights(c.widths, seed=0)


def _decoder(compute_dtype):
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, compute_dtype=compute_dtype)
    D = make_records((12, 7, 10, 5), seed=1)
    w0 = {n: w.detach().clone() for n, w in c.build_model().named_parameters()}
    return c, D, w0


@pytest.fixture(scope="module", params=["mlp", "decoder-fp32", "decoder-bf16"])
def setup(request):
    if request.param == "mlp":
        return _mlp()
    return _decoder(torch.float32 if request.param == "decoder-fp32" else torch.bfloat16)


def test_leaf_totals_equal_the_counting_passes_storage_record(setup):
    c, D, w0 = setup
    final = honest_final(c, D, w0, 1, seed=SEED)
    got = []

    def run(rec):
        res = run_scenario(c, D, w0, HONEST, T=1, k=K, seed=SEED, final=final, recorder=rec)
        got.extend(res.loop.steps)

    rows = count_pass("r", "honest", run)
    (storage,) = [r for r in rows if r["record"] == "storage"]
    (step,) = got
    assert (step.leaves, step.transcript_bytes) == (storage["leaves"], storage["transcript_bytes"])
    assert step.leaves == c.n_leaves


def test_leaf_totals_reads_no_payload(setup):
    c, D, w0 = setup
    model = c.build_model()
    out = prove_step(c, model, w0, list(D[: c.n_s]))
    real = out.leaves()
    ghost = [x if i < c.n_s else torch.empty(x.shape, dtype=x.dtype, device="meta")
             for i, x in enumerate(real)]
    assert leaf_totals(c, ghost) == leaf_totals(c, real)
    assert leaf_totals(c, real)[0] == c.n_leaves


def test_step_output_carries_the_capture_summary(setup):
    c, D, w0 = setup
    out = prove_step(c, c.build_model(), w0, list(D[: c.n_s]))
    assert out.capture is not None and out.capture["n_products"] == c.M
    assert set(out.capture["host_buffers"]) == {"n_buffers", "nbytes"}
    assert out.with_w_next(out.w_next).capture is out.capture


def _timed(c, D, w0, tmp_path, name="r"):
    final = honest_final(c, D, w0, 2, seed=SEED)
    with metrics.MetricsWriter(tmp_path, name, config={"compute_dtype": torch.bfloat16}) as mw:
        res = run_scenario(c, D, w0, HONEST, T=2, k=K, seed=SEED, final=final,
                           recorder=mw.recorder("honest"))
    return res


def test_capture_record_keys(tmp_path):
    c, D, w0 = _mlp()
    res = _timed(c, D, w0, tmp_path)
    recs = read_records(tmp_path, "capture")
    assert [r["step"] for r in recs] == [1, 2]
    summary_keys = set(MatmulCapture().summary())
    for r in recs:
        assert set(r) == {"record", "run", "scenario", "step", "traced", "host_buffers",
                          *summary_keys}
        assert r["traced"] is False and r["run"] == "r" and r["scenario"] == "honest"
    steps = read_records(tmp_path, "step")
    assert all(s["traced"] is False and s["leaves"] == c.n_leaves and s["transcript_bytes"] > 0
               for s in steps)
    assert all(r["traced"] is False for r in read_records(tmp_path, "time"))
    assert res.loop.steps[0].capture is not None


def test_environment_record_holds_the_compute_dtype(tmp_path):
    probe = metrics.memory_probe(torch.device("cpu"))
    assert metrics.environment_record("r", device="cpu", probe=probe)["compute_dtype"] is None
    env = metrics.environment_record("r", device="cpu", probe=probe,
                                     config={"compute_dtype": torch.bfloat16})
    assert env["compute_dtype"] == "bfloat16"
    env = metrics.environment_record("r", device="cpu", probe=probe,
                                     config={"compute_dtype": "torch.float32"})
    assert env["compute_dtype"] == "float32"
    c, D, w0 = _mlp()
    _timed(c, D, w0, tmp_path)
    assert read_records(tmp_path, "environment")[0]["compute_dtype"] == "bfloat16"


def test_totals_are_a_harness_section_outside_commit_and_handoff(monkeypatch, tmp_path):
    c, D, w0 = _mlp()
    real = loop_mod.leaf_totals
    delay = 0.2

    def slow(*a):
        time.sleep(delay)
        return real(*a)

    monkeypatch.setattr(loop_mod, "leaf_totals", slow)
    res = _timed(c, D, w0, tmp_path)
    for s in res.loop.steps:
        assert s.commit_s < delay and s.handoff_s < delay
    rows = read_records(tmp_path, "time")
    for t in (1, 2):
        (row,) = [r for r in rows if r["step"] == t and r["component"] == loop_mod.TOTALS_SECTION]
        assert row["side"] == "run" and row["level"] == "component" and row["time_s"] >= delay
        (total,) = [r for r in rows if r["step"] == t and r["component"] == "step"
                    and r["level"] == "total"]
        parts = [r["time_s"] for r in rows if r["step"] == t and r["level"] == "total"
                 and r["component"] in ("prover", "verifier")]
        assert total["time_s"] == pytest.approx(sum(parts))


def test_bit_identical_with_and_without_the_recorder(tmp_path):
    c, D, w0 = _mlp()
    final = honest_final(c, D, w0, 2, seed=SEED)
    off = run_scenario(c, D, w0, HONEST, T=2, k=K, seed=SEED, final=final)
    on = run_scenario(c, D, w0, HONEST, T=2, k=K, seed=SEED, final=final,
                      recorder=TimeRecorder("r", "honest"))
    assert on.loop.verdict == off.loop.verdict
    for n, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[n].view(torch.int32), w.view(torch.int32))
    assert on.verifier.stats == off.verifier.stats


@pytest.mark.slow
@pytest.mark.parametrize("compute", ["float32", "bfloat16"])
def test_smollm2_leaf_totals_equal_the_storage_record(monkeypatch, tmp_path, compute):
    """SmolLM2 test scale at ``(fp32, fp32)`` and ``(fp32, bf16)``: each step record's bytes and
    leaves are the counting pass's. Needs the committed ``D.bin``; skips without it."""
    from setup.config import load_config
    from verification.runs import llama_step

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("VERIF_COMPUTE_DTYPE", compute)
    if (data := load_config().data_dir) and not (data / "D.bin").exists():
        pytest.skip("no D.bin: run verification.runs.materialize_data or set VERIF_OUTPUT_DIR")
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "data").symlink_to(data)
    llama_step.main(["--steps", "1", "--metrics"])
    out = tmp_path / llama_step.RUN_NAME
    (step,) = read_records(out, "step")
    (storage,) = read_records(out, "storage")
    assert (step["leaves"], step["transcript_bytes"]) == (storage["leaves"],
                                                         storage["transcript_bytes"])
    assert read_records(out, "environment")[0]["compute_dtype"] == compute
