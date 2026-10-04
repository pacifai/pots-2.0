"""B6: per-component cost and residual metrics."""

import json
import subprocess
import sys
import textwrap

import pytest
import torch

from verification.commitment import merkle
from verification.computation.instances.mlp import MLPComputation, init_weights, synthetic_dataset
from verification.runs import metrics_overhead, mlp_smoke
from verification.runs.metrics import (
    COST_FIELDS,
    COUNT_FIELDS,
    PROVER_COMPONENTS,
    CostRecorder,
    MetricsWriter,
    count_pass,
    counting,
    memory_probe,
    read_costs,
    read_counts,
    read_residuals,
)
from verification.runs.mlp_smoke import count_smoke, honest_final, run_scenario, run_smoke, scenarios
from verification.verifier.matmul_check import challenges

K = 7
T = 3
N_RECORDS = 40
PHASES = ("P0.load", "P2.w_t", "P0.forward", "P0.backward", "P1.label", "P0.update", "P2.w_next",
          "P4.paths", "P3.commit", "P5.release")


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
    with MetricsWriter(out, {"run": "test"}) as mw:
        results = run_smoke(c, D, w0, T=T, k=K, out=lambda s: None, metrics=mw)
    return out, {r.scenario.name: r for r in results}


def _checks(t):
    return ("4", "0" if t == 1 else "7", "2", "6a", "5", "6b")


def test_every_row_filled(smoke):
    out, results = smoke
    assert all(r.passed for r in results.values())
    with open(out / "costs.csv") as f:
        header, *lines = f.read().splitlines()
    assert tuple(header.split(",")) == COST_FIELDS
    assert lines and all(cell != "" for line in lines for cell in line.split(","))
    rows = read_costs(out / "costs.csv")
    assert all(r.calls > 0 and r.time_s > 0 for r in rows)
    top = [r for r in rows if r.level != "sub"]
    assert all(r.mem_source == "darwin_phys_footprint" for r in top) or sys.platform != "darwin"
    assert all(r.peak_bytes > 0 and r.start_bytes > 0 and r.end_bytes > 0 for r in top)
    for name, res in results.items():
        mine = [r for r in rows if r.scenario == name]
        by_step = {}
        for r in mine:
            by_step.setdefault(r.step, {})[(r.level, r.component)] = r
        steps = [s.t for s in res.loop.steps]
        assert sorted(by_step) == [0, *steps]
        for t in steps:
            got = by_step[t]
            assert all(("phase", p) in got for p in PHASES)
            assert all(("component", p) in got for p in PROVER_COMPONENTS)
            assert ("total", "prover") in got and ("total", "step") in got
            ran = res.verifier.timings[t]  # the checks this step reached
            want = [cid for cid, key in zip(_checks(t), ("4", "7", "2", "6a", "5", "6b"))
                    if key in ran]
            have = [comp for (level, comp) in got if level == "component" and comp[0] != "P"]
            assert have == want  # driver order
            if "5" in ran:
                assert ("sub", "5.glue") in got and ("sub", "5.measure") in got
                if res.loop.rejection is None or res.loop.rejection.step != t:
                    assert got[("sub", "5.measure")].calls == 8  # M; flip aborts early
            # a total sums its top-level rows
            total = got[("total", "step")]
            parts = [r for (level, _), r in got.items() if level in ("phase",)]
            parts += [r for (level, comp), r in got.items()
                      if level == "component" and comp[0] != "P"]
            assert total.time_s == pytest.approx(sum(r.time_s for r in parts))
            assert total.peak_bytes == max(r.peak_bytes for r in parts)
        run = by_step[0]
        assert ("phase", "P4.tree") in run and ("component", "0") in run and ("component", "1") in run
        assert ("component", "9") in run
        assert (("component", "8") in run) == res.loop.verdict.accepted


def test_check_times_match_driver_timings(smoke):
    out, results = smoke
    rows = read_costs(out / "costs.csv")
    for name, res in results.items():
        for t, tm in res.verifier.timings.items():
            for key, ran in tm.items():
                cid = "0" if (key == "7" and t == 1) else key
                (row,) = [r for r in rows if (r.scenario, r.step, r.level, r.component)
                          == (name, t, "component", cid)]
                # the driver's clock runs inside the section's, so the two nearly agree
                assert ran <= row.time_s <= ran + 1e-3


def test_residuals_match_stats(smoke):
    out, results = smoke
    back = read_residuals(out / "residuals.jsonl")
    for name, res in results.items():
        want = {t: s for t, s in res.verifier.stats.items() if s.products or s.tensors}
        assert back.get(name, {}) == want
    with open(out / "residuals.jsonl") as f:
        first = json.loads(f.readline())
    assert set(first) == {"scenario", "step", "check", "m", "name", "cls", "kappa", "normalized"}


def test_meta(smoke):
    out, _ = smoke
    meta = json.loads((out / "meta.json").read_text())
    assert meta["run"] == "test" and meta["lifetime_maxrss_bytes"] > 0 and "mem_source" in meta


def test_count_pass_separate_and_nonzero(c, D, w0, smoke):
    out, _ = smoke
    rows = read_counts(out / "counts.csv")
    with open(out / "counts.csv") as f:
        assert tuple(f.readline().strip().split(",")) == COUNT_FIELDS
    assert {r.scenario for r in rows} == {"honest"}
    assert sorted({r.step for r in rows}) == [0, 1, 2]
    row = {(r.step, r.level, r.component): r for r in rows}
    fwd = 2 * c.n_s * sum(a * b for a, b in zip(c.widths, c.widths[1:]))
    assert row[(1, "phase", "P0.forward")].flops == fwd
    assert row[(1, "phase", "P0.backward")].flops > 0
    assert row[(1, "component", "5")].flops > 0
    assert row[(1, "component", "5")].hash_out_bytes > 0  # the challenge stream
    assert row[(1, "phase", "P3.commit")].hash_in_bytes > 0
    # check 2 rehashes exactly the bytes the prover committed
    assert row[(1, "component", "2")].hash_in_bytes == row[(1, "phase", "P3.commit")].hash_in_bytes
    assert (2, "component", "7") in row and (1, "component", "0") in row
    assert row[(0, "component", "1")].hash_in_bytes > 0
    for r in rows:
        if r.level == "total" and r.component == "step":
            parts = [x for x in rows if x.step == r.step
                     and (x.level == "phase" or (x.level == "component" and x.component[0] != "P"))]
            assert r.flops == sum(x.flops for x in parts)
    # the timed runs record no counts, and counting leaves BLAKE3 as it was
    assert merkle.blake3.__name__ == "blake3" and challenges.blake3.__name__ == "blake3"
    rows2 = count_smoke(c, D, w0, k=K)
    assert [(r.component, r.flops, r.hash_in_bytes) for r in rows2] == \
           [(r.component, r.flops, r.hash_in_bytes) for r in rows]


def test_counting_restores_on_error():
    real = merkle.blake3
    with pytest.raises(RuntimeError), counting():
        assert merkle.blake3 is not real
        raise RuntimeError
    assert merkle.blake3 is real


def test_counting_hash_is_unchanged():
    with counting() as (_, tally):
        h = merkle.hash_leaf(b"abc", memoryview(b"de"))
        v = challenges.challenge_vector(b"\x00" * 32, 1, 1, 5)
    assert h == merkle.hash_leaf(b"abc", memoryview(b"de"))
    assert torch.equal(v, challenges.challenge_vector(b"\x00" * 32, 1, 1, 5))
    assert tally.bytes_in == 6 + 6  # 0x00 ‖ "abc" ‖ "de", then the 6-byte label (the key is not input)
    assert tally.bytes_out == 32 + 4 * 5


@pytest.mark.parametrize("name", ["honest", "flip", "bad-w-next-ulps"])
def test_metrics_on_off_bit_identical(c, D, w0, final, name):
    s = next(x for x in scenarios(c, D) if x.name == name)
    off = run_scenario(c, D, w0, s, T=T, k=K, final=final)
    rec = CostRecorder(name)
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
    count_pass("honest", lambda rec: got.setdefault(
        "r", run_scenario(c, D, w0, s, T=2, k=K, final=final2, recorder=rec)))
    on = got["r"]
    assert on.loop.verdict == off.loop.verdict and on.verifier.stats == off.verifier.stats
    for k_, w in off.loop.w_final.items():
        assert torch.equal(on.loop.w_final[k_].view(torch.int32), w.view(torch.int32))


def test_nested_section_is_time_only():
    rec = CostRecorder("x")
    with rec.section("5"):
        with rec.section("5.glue"):
            pass
        with rec.section("5.glue"):
            pass
    with rec.section("run:1"):
        pass
    rec.on_step(type("R", (), {"t": 1})())
    rec.finish()
    sub = [r for r in rec.rows if r.component == "5.glue"]
    assert len(sub) == 1 and sub[0].calls == 2 and sub[0].mem_source == "nested"
    assert [r.step for r in rec.rows if r.component == "1"] == [0]


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="needs a resettable peak")
def test_memory_peak_sees_a_transient_allocation():
    # A fresh process, so no freed block is waiting to be reused.
    code = textwrap.dedent("""
        import torch
        from verification.runs.metrics import CostRecorder
        rec = CostRecorder("m")
        with rec.section("P0.load"):
            pass
        with rec.section("P0.forward"):
            x = torch.ones(50_000_000); del x
        with rec.section("P0.backward"):
            pass
        rec.on_step(type("R", (), {"t": 1})())
        r = {x.component: x for x in rec.rows if x.level == "phase"}
        print(r["P0.forward"].peak_bytes - r["P0.forward"].start_bytes,
              r["P0.backward"].peak_bytes - r["P0.backward"].start_bytes)
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, env={"MallocLargeCache": "0", "PYTHONPATH": "."})
    grow, after = map(int, res.stdout.split())
    assert grow >= 190_000_000  # 200 MB of float32
    assert after < 50_000_000  # the reset worked: the next section doesn't inherit the peak


def test_memory_probe_cpu_backend():
    probe = memory_probe("cpu")
    start = probe.start()
    peak, end = probe.stop()
    assert peak > 0
    if start is not None:
        assert peak >= max(start, end) > 0


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
    assert files == {"costs.csv", "counts.csv", "residuals.jsonl", "meta.json"}
    assert "metrics: " in capsys.readouterr().out


@pytest.mark.slow
def test_overhead_is_small():
    s = metrics_overhead.measure(reps=10, T=10, out=lambda _: None)
    # The reported per-step totals don't include the probes' cost.
    assert abs(s["sections_step_total_s"] / s["median_s"]["off"]["run"] - 1) < 0.05
    # Each check's own timer moves by a fixed ~1 µs per nested section at most.
    for cid in metrics_overhead.CHECK_IDS:
        key = f"check {cid}"
        assert s["median_s"]["on"][key] - s["median_s"]["off"][key] < 50e-6 * s["T"]
