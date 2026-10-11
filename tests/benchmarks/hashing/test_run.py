"""The runner: presets, and a fast run end to end on a tiny model."""

import json

import pytest
import torch

from benchmarks.hashing import native, report, run, sizes
from tests.verification.decoder_helpers import tiny_config


def test_presets():
    assert run.resolve(run.parser().parse_args([])) == run.H100
    assert run.resolve(run.parser().parse_args(["--smoke"])) == run.SMOKE
    args = run.resolve(run.parser().parse_args(["--smoke", "--buffer-mb", "64", "--no-native"]))
    assert (args["buffer_mb"], args["native"], args["reps"]) == (64, False, run.SMOKE["reps"])
    assert args["native_batch"] is True  # --native is the openssl/b3sum ceilings, not E5(a′)
    args = run.resolve(run.parser().parse_args(["--no-native-batch"]))
    assert (args["native"], args["native_batch"]) == (True, False)
    assert run.SMOKE["buffer_mb"] <= 512


def test_h100_defaults_follow_section_6():
    a = run.H100
    assert a["chunk_kb"] == "8,16,64,256,1024" and a["threads"] == "1,8,16,all"
    assert (a["n_s"], a["n"], a["k"], a["e5_model"]) == (128, 128, 21, "llama-3.2-1b")
    assert set(sizes.FULL_SCALE) <= set(a["size_models"].split(","))
    assert (a["e6_n_s"], a["e6_n"], a["e6_device"]) == (128, 128, "cuda")


TINY = ["--buffer-mb", "2", "--threads", "1,2", "--chunk-kb", "64", "--reps", "1",
        "--n-s", "2", "--n", "12", "--size-models", "smollm2-135m",
        "--direct-models", "smollm2-135m", "--sample-mb", "1", "--object-cap-mb", "0",
        "--max-objects", "1024",
        "--no-native", "--e5-model", "smollm2-135m", "--k", "3", "--e5-layers", "1",
        "--e5-processes", "1", "--no-native-batch", "--e6-models", "none", "--e6-device",
        "cpu"]


def test_a_tiny_run_writes_every_slot(tmp_path, monkeypatch):
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())
    path = run.run([*TINY, "--out", str(tmp_path / "r.json")], log=lambda _: None)
    doc = json.loads(path.read_text())
    report.check_report(doc)
    assert doc["complete"] is True
    slots = doc["slots"]
    assert list(slots) == list(report.SLOTS)
    assert {k: s["status"] for k, s in slots.items()} == {
        "E0": "measured", "E1": "measured", "E2": "not_implemented", "E3": "not_implemented",
        "E4": "not_implemented", "E5a": "measured", "E5a_prime": "measured",
        "E5b": "not_implemented", "E5c": "not_implemented", "E6": "skipped"}
    for s in slots.values():
        for r in s["results"]:
            assert r["status"] in report.STATUSES
            if r["status"] in ("measured", "estimated"):
                assert r["values"] and set(r["values"]) == set(r["units"])
            else:
                assert r["reason"]
    names = {r["name"] for r in slots["E1"]["results"]}
    assert {"chunked", "whole_objects", "step_from_sizes", "step_direct", "outer_tree",
            "finiteness", "finiteness_per_step", "openssl_speed", "b3sum",
            "chunked_under_h2d_copy"} <= names
    assert [r["name"] for r in slots["E5a"]["results"]] == ["python_path",
                                                             "python_path_per_step"]
    last = slots["E5a_prime"]["results"][-1]
    assert (last["name"], last["status"], last["reason"]) == \
        ("native_batch", "skipped", "turned off with --no-native-batch")
    assert doc["inputs"]["steps"]["smollm2-135m"]["leaves"] == \
        len(sizes.step("smollm2-135m", n_s=2, n=12).leaves)
    machine = slots["E0"]["results"][0]["values"]["record"]
    assert "gpu_name" in machine and "absent" in machine


def test_only_selected_experiments_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())
    path = run.run([*TINY, "--experiments", "E0", "--out", str(tmp_path / "r.json")],
                   log=lambda _: None)
    slots = json.loads(path.read_text())["slots"]
    assert slots["E1"]["status"] == slots["E5a"]["status"] == slots["E6"]["status"] == "skipped"
    assert slots["E1"]["results"][0]["reason"] == "not selected with --experiments"


def test_a_failing_experiment_keeps_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(run.challenges, "run_e5a", boom)
    path = run.run([*TINY, "--experiments", "E0,E1,E5", "--out", str(tmp_path / "r.json")],
                   log=lambda _: None)
    slots = json.loads(path.read_text())["slots"]
    assert slots["E1"]["status"] == "measured"
    for key in ("E5a", "E5a_prime"):
        (r,) = slots[key]["results"]
        assert (r["status"], r["reason"]) == ("skipped", "error: RuntimeError: boom")


def test_a_failed_native_build_keeps_the_pool_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())

    def offline(**kw):
        raise RuntimeError("no network")

    monkeypatch.setattr(native, "build", offline)
    path = run.run([*TINY, "--native-batch", "--experiments", "E5",
                    "--out", str(tmp_path / "r.json")], log=lambda _: None)
    rows = json.loads(path.read_text())["slots"]["E5a_prime"]["results"]
    assert rows[0]["name"] == "thread_pool" and rows[0]["status"] == "measured"
    assert (rows[-1]["name"], rows[-1]["status"], rows[-1]["reason"]) == \
        ("native_batch", "skipped", "build failed: RuntimeError: no network")


def test_a_crash_keeps_the_finished_experiments(tmp_path, monkeypatch):
    """The report is rewritten after each experiment, so an interrupt in E6 keeps E0's rows."""
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())

    def interrupted(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(run.step_time, "run_e6", interrupted)
    out = tmp_path / "r.json"
    with pytest.raises(KeyboardInterrupt):
        run.run([*TINY, "--experiments", "E0,E6", "--e6-models", "smollm2-135m",
                 "--out", str(out)], log=lambda _: None)
    doc = json.loads(out.read_text())
    report.check_report(doc)
    assert doc["complete"] is False and doc["slots"]["E0"]["status"] == "measured"
    (r,) = doc["slots"]["E6"]["results"]
    assert r["status"] == "skipped" and r["reason"].startswith("pending")
    assert doc["slots"]["E1"]["results"][0]["reason"] == "not selected with --experiments"


def test_a_failed_native_run_keeps_the_pool_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())

    class Lib:
        simd_degree = 4

    def broken(*a, **kw):
        raise OSError("bad library")

    monkeypatch.setattr(native, "build", lambda **kw: tmp_path / "lib.so")
    monkeypatch.setattr(native, "load", lambda path: Lib())
    monkeypatch.setattr(native, "run_native", broken)
    path = run.run([*TINY, "--native-batch", "--experiments", "E5",
                    "--out", str(tmp_path / "r.json")], log=lambda _: None)
    rows = json.loads(path.read_text())["slots"]["E5a_prime"]["results"]
    assert rows[0]["name"] == "thread_pool" and rows[0]["status"] == "measured"
    assert (rows[-1]["name"], rows[-1]["status"], rows[-1]["reason"]) == \
        ("native_batch", "skipped", "error: OSError: bad library")


@pytest.mark.skipif(native.find_library() is None,
                    reason="native BLAKE3 batch not built; run python -m benchmarks.hashing.native")
def test_a_tiny_run_with_the_native_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())
    path = run.run([*TINY, "--native-batch", "--experiments", "E5",
                    "--out", str(tmp_path / "r.json")], log=lambda _: None)
    rows = json.loads(path.read_text())["slots"]["E5a_prime"]["results"]
    native_rows = [(r["name"], r["status"], r["params"]["threads"]) for r in rows
                   if r["name"].startswith("native_batch")]
    assert native_rows == [("native_batch", "measured", 1),
                           ("native_batch_per_step", "measured", 1),
                           ("native_batch", "measured", 2),
                           ("native_batch_per_step", "measured", 2)]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA guard")
def test_e6_on_cuda_without_cuda_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(sizes, "model_config", lambda key: tiny_config())
    path = run.run([*TINY, "--experiments", "E6", "--e6-models", "smollm2-135m",
                    "--e6-device", "cuda", "--out", str(tmp_path / "r.json")], log=lambda _: None)
    (r,) = json.loads(path.read_text())["slots"]["E6"]["results"]
    assert r["status"] == "skipped" and "CUDA" in r["reason"]


def test_step_names():
    assert run._step_spec("llama-3.2-1b", 128, 64) == ("llama-3.2-1b", 128, 64, torch.bfloat16)
    assert run._step_spec(run.TEST_STEP, 128, 128) == ("smollm2-135m", 4, 128, torch.float32)
    for bad in ("smollm2-135m:4x128", "smollm2-135m:4by128:fp32", "smollm2-135m:4x128:fp16"):
        with pytest.raises(SystemExit):
            run._step_spec(bad, 128, 128)
    assert run._models(f"none,{run.TEST_STEP}") == [run.TEST_STEP]


def test_unknown_names_are_refused():
    with pytest.raises(SystemExit):
        run.run(["--experiments", "E2"])
    with pytest.raises(SystemExit):
        run._models("gpt2")
