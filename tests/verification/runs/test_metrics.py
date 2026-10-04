"""B6: the EQ1b cost grid and the EQ13 run records."""

import json
import logging
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from verification.commitment import leaves, merkle
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.computation.interface import load_weights
from verification.prover.step import plain_step, prove_step
from verification.runs import metrics, metrics_overhead, mlp_smoke
from verification.runs.metrics import (
    P0_PHASES,
    TRAIN_PHASES,
    CountRecorder,
    MemoryRecorder,
    MetricsWriter,
    TimeRecorder,
    count_pass,
    counting,
    derive_capture,
    memory_pass,
    memory_probe,
    read_records,
    read_residuals,
)
from verification.runs.mlp_smoke import honest_final, run_scenario, run_smoke, scenarios
from verification.transcript.store import InMemoryStore
from verification.verifier import bands
from verification.verifier.checks import DEFAULT_ORDER
from verification.verifier.matmul_check import challenges

ROOT = Path(__file__).resolve().parents[3]
K = 7
T = 3
N_RECORDS = 40
PHASES = (*TRAIN_PHASES, "P1.label", "P2.w_t", "P2.w_next", "P3.commit", "P4.paths", "P5.write")
PROVER_COMPONENTS = ("train_captured", "P2", "P3", "P4", "P5")
RESETTABLE = sys.platform == "darwin" or sys.platform.startswith("linux")


class _Rec:
    def __init__(self, t, rejection=None):
        self.t, self.rejection = t, rejection


@pytest.fixture(scope="module")
def c():
    return MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)


@pytest.fixture(scope="module")
def D(c):
    return synthetic_dataset(c.widths, N_RECORDS, seed=0)


@pytest.fixture(scope="module")
def w0(c):
    return init_weights(c.widths, seed=0)


@pytest.fixture(scope="module")
def final(c, D, w0):
    return honest_final(c, D, w0, T)


@pytest.fixture(scope="module")
def smoke(c, D, w0, tmp_path_factory):
    """Every smoke scenario with metrics on, written to a directory."""
    out = tmp_path_factory.mktemp("metrics")
    with MetricsWriter(out, "test", model="mlp", corpus="synthetic", seed=0,
                       config={"a": 1}) as mw:
        results = run_smoke(c, D, w0, T=T, k=K, out=lambda s: None, metrics=mw)
    return out, {r.scenario.name: r for r in results}


def _by_step(rows, record, scenario):
    out = {}
    for r in rows:
        if r["record"] == record and r["scenario"] == scenario:
            out.setdefault(r["step"], {})[(r["level"], r["component"])] = r
    return out


def _top_parts(got):
    return [r for (level, comp), r in got.items()
            if level == "phase" or (level == "component" and r["side"] == "verifier")]


def test_time_rows_cover_the_grid(smoke):
    out, results = smoke
    assert all(r.passed for r in results.values())
    rows = read_records(out, "time")
    assert all(r["calls"] > 0 and r["time_s"] > 0 and r["derived"] is False for r in rows)
    # a verified run never reports P0 or P1: P0 is the plain run's, P1 is derived
    assert not {"P0", "P1"} & {r["component"] for r in rows}
    for name, res in results.items():
        by_step = _by_step(rows, "time", name)
        steps = [s.t for s in res.loop.steps]
        assert sorted(by_step) == [0, *steps]
        for t in steps:
            got = by_step[t]
            assert all(("phase", p) in got for p in PHASES)
            assert all(("component", p) in got for p in PROVER_COMPONENTS)
            assert got[("component", "train_captured")]["calls"] == 6
            assert ("total", "prover") in got and ("total", "step") in got
            ran = list(res.verifier.timings[t])  # the checks this step reached, in order
            have = [comp for (level, comp), r in got.items()
                    if level == "component" and r["side"] == "verifier"]
            assert have == ran  # driver order; step 1's chaining comparison is row "7"
            if "5" in ran:
                assert ("sub", "5.glue") in got and ("sub", "5.measure") in got
                if res.loop.rejection is None or res.loop.rejection.step != t:
                    assert got[("sub", "5.measure")]["calls"] == res.verifier.c.M
            total = got[("total", "step")]
            assert total["time_s"] == pytest.approx(sum(r["time_s"] for r in _top_parts(got)))
        run = by_step[0]
        assert {("phase", "P4.tree"), ("component", "0"), ("component", "1"),
                ("component", "9")} <= set(run)
        assert (("component", "8") in run) == res.loop.verdict.accepted


def test_check_times_match_driver_timings(smoke):
    out, results = smoke
    rows = read_records(out, "time")
    for name, res in results.items():
        for t, tm in res.verifier.timings.items():
            for cid, ran in tm.items():
                (row,) = [r for r in rows if (r["scenario"], r["step"], r["level"],
                                              r["component"]) == (name, t, "component", cid)]
                # the driver's clock runs inside the section's, so the two nearly agree
                assert ran <= row["time_s"] <= ran + 1e-2


def test_step_and_verdict_records(smoke):
    out, results = smoke
    steps = read_records(out, "step")
    for name, res in results.items():
        mine = [r for r in steps if r["scenario"] == name]
        assert [r["step"] for r in mine] == [s.t for s in res.loop.steps]
        rej = res.loop.rejection
        for r in mine:
            assert (r["run"], r["model"], r["corpus"], r["seed"]) == ("test", "mlp", "synthetic", 0)
            assert r["poisoning_rate"] is None and r["cheat_step"] is None
            assert list(r["checks"]) == list(DEFAULT_ORDER)
            if rej is not None and rej.step == r["step"]:
                assert r["verdict"] == "reject"
                assert r["first_failing_check"] == rej.check_id
                assert r["checks"][rej.check_id] == "fail"
                after = list(DEFAULT_ORDER)[list(DEFAULT_ORDER).index(rej.check_id) + 1:]
                assert all(r["checks"][x] == "not_run" for x in after)
            else:
                assert r["verdict"] == "accept" and r["first_failing_check"] is None
                assert set(r["checks"].values()) == {"pass"}
        (v,) = [r for r in read_records(out, "verdict") if r["scenario"] == name]
        assert v["accepted"] == res.loop.verdict.accepted
        assert v["rejection_check"] == (None if rej is None else rej.check_id)


def test_residual_arrays_round_trip(smoke):
    out, results = smoke
    back = read_residuals(out)
    for name, res in results.items():
        want = {t: s for t, s in res.verifier.stats.items() if s.products or s.tensors}
        assert back.get(name, {}) == want
    res = results["honest"]
    with np.load(out / "residuals" / "honest" / "step_00001.npz", allow_pickle=False) as z:
        assert str(z["run"]) == "test" and int(z["step"]) == 1
        assert z["p_normalized"].shape == (res.verifier.c.M, K)
        want = [-1 if res.verifier.c.product(int(m)).layer is None
                else res.verifier.c.product(int(m)).layer for m in z["p_m"]]
        assert z["p_layer"].tolist() == want
        assert set(z["w_check"].tolist()) <= {"6a", "6b"}


def _strict(s):
    raise ValueError(f"non-standard JSON constant {s}")


def test_records_are_standard_json(smoke, tmp_path):
    out, _ = smoke
    with open(out / "records.jsonl") as f:
        for line in f:
            json.loads(line, parse_constant=_strict)
    with MetricsWriter(tmp_path, "nf") as mw:
        mw.write([{"record": "x", "run": "nf", "a": math.nan, "b": [math.inf, -math.inf, 1.5]}])
    (r,) = read_records(tmp_path, "x")
    assert r["a"] == "NaN" and r["b"] == ["Infinity", "-Infinity", 1.5]


def test_environment_and_run_end(smoke):
    out, _ = smoke
    rows = read_records(out)
    env, end = rows[0], rows[-1]
    assert env["record"] == "environment" and end["record"] == "run_end"
    assert all(r["run"] == "test" for r in rows)
    assert env["libraries"]["torch"] and env["libraries"]["numpy"] == np.__version__
    assert env["hardware"]["cpu"] and env["device"] == "cpu" and env["hardware"]["gpu"] is None
    assert len(env["config_hash"]) == 64 and env["config_hash"] == metrics.config_hash({"a": 1})
    assert env["band_file_hash"] is None and env["h_D"] is None
    assert env["git_commit"] is None or len(env["git_commit"]) == 40
    assert end["lifetime_maxrss_bytes"] > 0


def test_memory_pass_probes_every_section(smoke, c):
    out, _ = smoke
    rows = read_records(out, "memory")
    assert {r["scenario"] for r in rows} == {"honest"}
    by_step = _by_step(rows, "memory", "honest")
    assert sorted(by_step) == [0, 1, 2]
    got = by_step[1]
    assert all(("phase", p) in got for p in PHASES)
    assert {("sub", "5.glue"), ("sub", "5.measure"), ("component", "7")} <= set(got)
    if RESETTABLE:
        for r in rows:
            assert r["peak_bytes"] > 0 and r["start_bytes"] > 0 and r["end_bytes"] > 0
            assert r["peak_bytes"] >= max(r["start_bytes"], r["end_bytes"]) or r["level"] != "phase"
    total = got[("total", "step")]
    assert total["peak_bytes"] == max(r["peak_bytes"] for r in _top_parts(got))


def test_count_pass_and_storage(c, D, w0, smoke):
    out, _ = smoke
    rows = read_records(out, "count")
    assert {r["scenario"] for r in rows} == {"honest"}
    assert sorted({r["step"] for r in rows}) == [0, 1, 2]
    row = {(r["step"], r["level"], r["component"]): r for r in rows}
    fwd = 2 * c.n_s * sum(a * b for a, b in zip(c.widths, c.widths[1:]))
    assert row[(1, "phase", "train.forward")]["flops"] == fwd
    assert row[(1, "phase", "train.backward")]["flops"] > 0
    assert row[(1, "component", "5")]["flops"] > 0
    assert row[(1, "component", "5")]["hash_out_bytes"] > 0  # the challenge stream
    commit = row[(1, "phase", "P3.commit")]
    assert commit["hash_in_bytes"] > 0
    assert commit["hash_calls"] == 2 * c.n_leaves - 1  # one per leaf and per internal node
    # check 2 rehashes exactly what the prover committed
    for key in ("hash_in_bytes", "hash_out_bytes", "hash_calls"):
        assert row[(1, "component", "2")][key] == commit[key]
    assert row[(1, "phase", "P5.write")]["hash_calls"] == 0
    assert (2, "component", "7") in row and (1, "component", "7") in row
    assert row[(0, "component", "0")]["hash_calls"] > 0 and row[(0, "component", "1")]["hash_in_bytes"] > 0
    for r in rows:
        if r["level"] == "total" and r["component"] == "step":
            parts = _top_parts({(x["level"], x["component"]): x for x in rows
                                if x["step"] == r["step"]})
            assert r["flops"] == sum(x["flops"] for x in parts)
            assert r["hash_calls"] == sum(x["hash_calls"] for x in parts)
    storage = read_records(out, "storage")
    assert [s["step"] for s in storage] == [1, 2]
    for s in storage:
        assert s["leaves"] == c.n_leaves
        # the leaf bytes are what P3 hashes at the leaves: all of it but the internal nodes
        assert 0 < s["transcript_bytes"] < commit["hash_in_bytes"]
        assert commit["hash_in_bytes"] - s["transcript_bytes"] == \
            (c.n_leaves - 1) * (1 + 2 * merkle.DIGEST_SIZE) + c.n_leaves  # node and leaf tags
    # counting is deterministic and leaves the real functions in place
    assert merkle.blake3.__name__ == "blake3" and challenges.blake3.__name__ == "blake3"
    assert leaves.hash_leaf is merkle.hash_leaf
    rows2 = mlp_smoke.count_smoke(c, D, w0, k=K)
    pick = ("component", "flops", "hash_in_bytes", "hash_calls")
    assert [tuple(r.get(k) for k in pick) for r in rows2 if r["record"] == "count"] == \
           [tuple(r[k] for k in pick) for r in rows]


def test_counting_restores_on_error():
    real, real_leaf = merkle.blake3, leaves.hash_leaf
    with pytest.raises(RuntimeError), counting():
        assert merkle.blake3 is not real and bands.blake3 is not real
        assert leaves.hash_leaf is not real_leaf
        raise RuntimeError
    assert merkle.blake3 is real and bands.blake3 is real and leaves.hash_leaf is real_leaf


def test_counting_hash_is_unchanged():
    with counting() as (_, tally):
        h = merkle.hash_leaf(b"abc", memoryview(b"de"))
        v = challenges.challenge_vector(b"\x00" * 32, 1, 1, 5)
    assert h == merkle.hash_leaf(b"abc", memoryview(b"de"))
    assert torch.equal(v, challenges.challenge_vector(b"\x00" * 32, 1, 1, 5))
    assert tally.bytes_in == 6 + 6  # 0x00 ‖ "abc" ‖ "de", then the 6-byte label (the key is not input)
    assert tally.bytes_out == 32 + 4 * 5
    assert tally.calls == 2


def test_flop_tally_matches_flop_counter_mode(c, D, w0):
    """_FlopTally stands in for FlopCounterMode (EQ1c); on a plain step they agree."""
    model = c.build_model()
    batch = list(D[: c.n_s])
    load_weights(c, model, w0)
    with FlopCounterMode(display=False) as ref:
        c.loss(model, batch).backward()
    model.zero_grad(set_to_none=True)
    load_weights(c, model, w0)
    with metrics._FlopTally() as ours:
        c.loss(model, batch).backward()
    assert ours.total == ref.get_total_flops() > 0


@pytest.mark.parametrize("name", ["honest", "flip", "bad-w-next-ulps"])
@pytest.mark.parametrize("kind", ["time", "memory"])
def test_metrics_on_off_bit_identical(c, D, w0, final, name, kind):
    s = next(x for x in scenarios(c, D) if x.name == name)
    off = run_scenario(c, D, w0, s, T=T, k=K, final=final)
    rec = (TimeRecorder("r", name) if kind == "time"
           else MemoryRecorder("r", name, memory_probe("cpu")))
    on = run_scenario(c, D, w0, s, T=T, k=K, final=final, recorder=rec)
    assert rec.rows
    assert on.loop.verdict == off.loop.verdict
    assert [(x.t, x.rejection, x.loss) for x in on.loop.steps] == \
           [(x.t, x.rejection, x.loss) for x in off.loop.steps]
    assert on.loop.w_final.keys() == off.loop.w_final.keys()
    for k_, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[k_].view(torch.int32), w.view(torch.int32))
    assert on.verifier.stats == off.verifier.stats


def test_count_pass_is_bit_identical(c, D, w0):
    final2 = honest_final(c, D, w0, 2)
    s = next(x for x in scenarios(c, D) if x.name == "honest")
    off = run_scenario(c, D, w0, s, T=2, k=K, final=final2)
    got = {}
    count_pass("r", "honest", lambda rec: got.setdefault(
        "r", run_scenario(c, D, w0, s, T=2, k=K, final=final2, recorder=rec)))
    on = got["r"]
    assert on.loop.verdict == off.loop.verdict and on.verifier.stats == off.verifier.stats
    for k_, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[k_].view(torch.int32), w.view(torch.int32))


def test_nested_and_run_sections():
    rec = TimeRecorder("x", "s")
    with rec.section("5"):
        with rec.section("5.glue"):
            pass
        with rec.section("5.glue"):
            pass
    with rec.section("run:1"):
        pass
    rec.on_step(_Rec(1))
    rec.finish()
    sub = [r for r in rec.rows if r["component"] == "5.glue"]
    assert len(sub) == 1 and sub[0]["calls"] == 2 and sub[0]["level"] == "sub"
    (five,) = [r for r in rec.rows if r["component"] == "5"]
    assert five["time_s"] >= sub[0]["time_s"]
    assert [r["step"] for r in rec.rows if r["component"] == "1"] == [0]
    (total,) = [r for r in rec.rows if r["component"] == "step" and r["step"] == 1]
    assert total["time_s"] == five["time_s"]  # a sub row is inside its parent, not added


def test_plain_step_emits_P0_rows(c, D, w0):
    rec = TimeRecorder("plain", "honest")
    model = c.build_model()
    batch = list(D[: c.n_s])
    w_sec, loss_sec = plain_step(c, model, w0, batch, section=rec.section)
    w_ref, loss_ref = plain_step(c, c.build_model(), w0, batch)
    assert loss_sec == loss_ref
    assert all(torch.equal(w_sec[n], w_ref[n]) for n in w_ref)
    rec.on_step(_Rec(1))
    phases = [r["component"] for r in rec.rows if r["level"] == "phase"]
    assert phases == list(P0_PHASES)
    comps = [r["component"] for r in rec.rows if r["level"] == "component"]
    assert comps == ["P0"]


def _phase(record, comp, step, **v):
    return {"record": record, "run": "r", "scenario": "honest", "step": step, "side": "prover",
            "component": comp, "level": "phase", "calls": 1, "derived": False, **v}


def test_derive_capture_from_synthetic_rows():
    verified, plain = [], []
    for t in (1, 2):
        for i, x in enumerate(("load", "forward", "backward", "update")):
            verified.append(_phase("time", f"train.{x}", t, time_s=1.0 + i))
            plain.append(_phase("time", f"P0.{x}", t, time_s=0.75 + i))
        verified.append(_phase("time", "P1.label", t, time_s=0.5))
    verified.append(_phase("time", "P4.tree", 0, time_s=9.0))  # run rows are not steps
    verified.append(_phase("memory", "train.backward", 1, peak_bytes=1000, mem_source="m"))
    plain.append(_phase("memory", "P0.backward", 1, peak_bytes=600, mem_source="m"))
    verified.append(_phase("count", "train.forward", 1, flops=10))
    rows = derive_capture(verified, plain)
    assert all(r["component"] == "P1" and r["derived"] is True and r["level"] == "component"
               for r in rows)
    times = {r["step"]: r["time_s"] for r in rows if r["record"] == "time"}
    assert times == {1: pytest.approx(4 * 0.25 + 0.5), 2: pytest.approx(1.5)}
    (mem,) = [r for r in rows if r["record"] == "memory"]
    assert mem["step"] == 1 and mem["peak_bytes"] == 400
    (cnt,) = [r for r in rows if r["record"] == "count"]
    assert cnt["flops"] == 0 and cnt["hash_in_bytes"] == 0
    # a step the plain run lacks gets no P1 row
    assert derive_capture(verified, [p for p in plain if p["step"] == 2]) == \
           [r for r in rows if r["record"] == "time" and r["step"] == 2] + [cnt]


def test_store_split_matches_from_step(c, D, w0):
    batch = list(D[: c.n_s])
    out = prove_step(c, c.build_model(), w0, batch)
    a = InMemoryStore.from_step(c, out)
    lv, tree = InMemoryStore.commit_step(c, out)
    b = InMemoryStore.hold(c, lv, tree)
    assert a.root == b.root and a.path(3) == b.path(3)
    with pytest.raises(ValueError):
        InMemoryStore.hold(c, lv, tree, [[b""]])


@pytest.mark.skipif(not RESETTABLE, reason="needs a resettable peak")
def test_memory_peak_sees_a_transient_allocation():
    # A fresh process, so no freed block is waiting to be reused.
    code = textwrap.dedent("""
        import torch
        from verification.runs.metrics import MemoryRecorder, memory_probe
        rec = MemoryRecorder("m", "s", memory_probe("cpu"))
        class R: t = 1
        with rec.section("P0.load"):
            pass
        with rec.section("P0.forward"):
            with rec.section("inner"):
                x = torch.ones(50_000_000); del x
            with rec.section("after"):
                pass
        with rec.section("P0.backward"):
            pass
        rec.on_step(R())
        r = {x["component"]: x for x in rec.rows}
        g = lambda n: r[n]["peak_bytes"] - r[n]["start_bytes"]
        print(g("P0.forward"), g("inner"), g("after"), g("P0.backward"))
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"MallocLargeCache": "0", "PYTHONPATH": str(ROOT)})
    fwd, inner, after, bwd = map(int, res.stdout.split())
    assert fwd >= 190_000_000 and inner >= 190_000_000  # 200 MB of float32, parent and child
    assert after < 50_000_000 and bwd < 50_000_000  # each reset worked


def test_memory_probe_cpu_backend():
    probe = memory_probe("cpu")
    probe.reset()
    peak, cur = probe.sample()
    assert peak > 0
    if cur is not None:
        assert peak >= cur > 0


class _Failing:
    source = "failing"

    def reset(self):
        pass

    def sample(self):
        raise OSError(1, "boom")


def _one_section(rec):
    with rec.section("a"):
        pass
    rec.on_step(_Rec(1))


def test_probe_failure_degrades_to_null(caplog):
    probe = metrics._Guarded(_Failing())
    with caplog.at_level(logging.WARNING):
        rows = memory_pass("r", "s", _one_section, probe)
        probe.sample()
    assert sum("failed" in m for m in caplog.messages) == 1
    assert probe.source == "unavailable"
    assert all(r["peak_bytes"] is None for r in rows)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_sections_synchronize(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append(1))
    rec = TimeRecorder("x", "s", device="mps")
    with rec.section("P3.commit"):
        pass
    assert len(calls) == 2


def test_main_metrics_flag(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mlp_smoke, "setup_determinism", lambda cfg: None)
    monkeypatch.setenv("VERIF_N_RECORDS", str(N_RECORDS))
    monkeypatch.setenv("VERIF_OUTPUT_DIR", str(tmp_path))
    assert mlp_smoke.main(["--steps", "2", "--no-metrics"]) == 0
    assert not (tmp_path / "mlp_smoke").exists()
    monkeypatch.setenv("VERIF_METRICS", "0")
    assert mlp_smoke.main(["--steps", "2"]) == 0
    assert not (tmp_path / "mlp_smoke").exists()
    assert mlp_smoke.main(["--steps", "2", "--metrics"]) == 0
    files = {p.name for p in (tmp_path / "mlp_smoke").iterdir()}
    assert files == {"records.jsonl", "residuals"}
    kinds = {r["record"] for r in read_records(tmp_path / "mlp_smoke")}
    assert kinds == {"environment", "step", "verdict", "time", "memory", "count", "storage",
                     "run_end"}
    assert "metrics: " in capsys.readouterr().out


@pytest.mark.slow
def test_overhead_is_small():
    s = metrics_overhead.measure(reps=15, T=10, out=lambda _: None)
    # The reported per-step totals are within noise of the uninstrumented run.
    assert abs(s["sections_step_total_s"] / s["median_s"]["off"]["run"] - 1) < 0.05
    # Each check's own timer moves by a fixed ~1 µs per nested section at most.
    for cid in metrics_overhead.CHECK_IDS:
        key = f"check {cid}"
        assert s["median_s"]["on"][key] - s["median_s"]["off"][key] < 50e-6 * s["T"]
