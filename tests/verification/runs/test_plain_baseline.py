"""B7: the plain baseline. P0 rows, final-weight hashes, bit-identity with the captured prover,
and P1 derived from a plain-run file and a verified-run file."""

import json

import pytest
import torch

from tests.verification.computation.test_llama import make_records, tiny_config
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import LlamaComputation
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.prover.step import plain_step, prove_step
from verification.runs import plain_baseline as pb
from verification.runs.loop import run_loop, run_plain
from verification.runs.metrics import MetricsWriter, P0_PHASES, read_records
from verification.runs.mlp_smoke import run_smoke
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

K = 7


@pytest.fixture(scope="module")
def mlp():
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    return c, synthetic_dataset(c.widths, 40, seed=0), init_weights(c.widths, seed=0)


@pytest.fixture(scope="module")
def llama():
    n = 12
    c = LlamaComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n], seed=1)
    return c, D, pb.initial_weights(c, c.build_model())


def _verified(c, D, w0, T, final):
    v = Verifier(c, h_D=dataset_tree(c, D).root, n_records=len(D), k=K, n_steps=T,
                 bands=Bands.provisional(), w0=w0, allow_provisional=True)
    return run_loop(c, c.build_model(), D, w0, v, final=final)


# ---- the run's steps ---------------------------------------------------------------------


def test_run_plain_is_chained_plain_steps(mlp):
    c, D, w0 = mlp
    res = run_plain(c, c.build_model(), D, w0, n_steps=3)
    w = w0
    for t in (1, 2, 3):
        w, loss = plain_step(c, c.build_model(), w, D[(t - 1) * 4:t * 4])
        assert res.steps[t - 1].t == t and res.steps[t - 1].loss == loss
    assert pb.final_weight_hashes(c, w) == pb.final_weight_hashes(c, res.w_final)


@pytest.mark.parametrize("inst", ["mlp", "llama"])
def test_plain_final_weights_equal_the_verified_runs(inst, request):
    """Capture on (run_loop, verified and accepted) or off (run_plain): bit-identical W_{T+1}."""
    c, D, w0 = request.getfixturevalue(inst)
    T = 3
    plain = run_plain(c, c.build_model(), D, w0, n_steps=T)
    loop = _verified(c, D, w0, T, plain.w_final)
    assert loop.verdict.accepted and loop.verdict.steps_verified == T  # check 8 agrees
    pb.assert_same_final_weights(c, plain.w_final, loop.w_final)
    assert [s.loss for s in plain.steps] == [s.loss for s in loop.steps]


def test_comparison_catches_one_ulp(mlp):
    c, D, w0 = mlp
    w = run_plain(c, c.build_model(), D, w0, n_steps=2).w_final
    name = c.weight_names[1]
    other = dict(w)
    other[name] = w[name].clone()
    other[name].view(-1).view(torch.int32)[3] += 1
    with pytest.raises(AssertionError, match=name):
        pb.assert_same_final_weights(c, w, other)
    zero = {n: torch.zeros_like(t) for n, t in w.items()}
    neg = {n: -t for n, t in zero.items()}  # -0.0 is a different bit pattern
    with pytest.raises(AssertionError):
        pb.assert_same_final_weights(c, zero, neg)
    assert pb.weight_mismatches({"a": "1", "b": "2"}, {"a": "1", "c": "3"}) == ["b", "c"]


def test_hashes_are_w_next_leaf_hashes(llama):
    """A verified run's committed W_{t+1} leaf hashes are the hashes the plain run records."""
    c, D, w0 = llama
    from verification.commitment.leaves import leaf_hash
    out = prove_step(c, c.build_model(), w0, D[:c.n_s])
    got = pb.final_weight_hashes(c, out.w_next)
    want = [leaf_hash(c, c.w_next_index(n), out.w_next[n]).hex() for n in c.weight_names]
    assert list(got.values()) == want and list(got) == list(c.weight_names)


# ---- records and files -------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlp_files(mlp, tmp_path_factory):
    """A plain baseline and an honest verified smoke run of the MLP, each with its records."""
    c, D, w0 = mlp
    root = tmp_path_factory.mktemp("b7")
    with MetricsWriter(root / "plain", "plain_baseline", model="mlp") as mw:
        res = pb.plain_baseline(c, D, w0, T=3, build_model=c.build_model, metrics=mw,
                                out=lambda s: None)
    with MetricsWriter(root / "verified", "mlp_smoke", model="mlp") as mw:
        (smoke,) = run_smoke(c, D, w0, T=3, k=K, only=["honest"], metrics=mw,
                             out=lambda s: None)
    return root, res, smoke


def test_p0_rows_per_step(mlp_files):
    root, res, _ = mlp_files
    rows = read_records(root / "plain")
    assert rows[0]["record"] == "environment" and rows[-1]["record"] == "run_end"
    assert not [r for r in rows if r["record"] in ("step", "verdict", "storage")]
    assert {(r["run"], r["scenario"]) for r in rows if "scenario" in r} == \
           {("plain_baseline", "plain")}
    for record, steps in (("time", {1, 2, 3}), ("memory", {1, 2}), ("count", {1, 2})):
        mine = [r for r in rows if r["record"] == record]
        assert {r["step"] for r in mine} == steps, record
        assert all(r["side"] in ("prover", "run") for r in mine)
        for t in steps:
            phases = [r["component"] for r in mine if r["step"] == t and r["level"] == "phase"]
            assert phases == list(P0_PHASES)
            comps = [r["component"] for r in mine if r["step"] == t and r["level"] == "component"]
            assert comps == ["P0"]
    t1 = {r["component"]: r for r in rows if r["record"] == "count" and r["step"] == 1}
    assert t1["P0.forward"]["flops"] > 0 and t1["P0.backward"]["flops"] > 0
    assert sum(t1[x]["hash_calls"] + t1[x]["hash_in_bytes"] for x in P0_PHASES) == 0
    mem = {r["component"]: r for r in rows if r["record"] == "memory" and r["step"] == 1}
    assert mem["P0.backward"]["peak_bytes"] is not None
    times = [r for r in rows if r["record"] == "time" and r["component"] == "P0"]
    assert all(r["time_s"] > 0 for r in times)


def test_final_weights_file(mlp_files, mlp):
    root, res, smoke = mlp_files
    c, _, _ = mlp
    doc = json.loads((root / "plain" / pb.WEIGHTS_FILE).read_text())
    assert doc["steps"] == 3 and list(doc["hashes"]) == list(c.weight_names)
    assert pb.read_final_weights(root / "plain") == res.hashes
    pb.assert_same_final_weights(c, root / "plain", smoke.loop.w_final)


def test_capture_rows_from_two_files(mlp_files):
    root, _, _ = mlp_files
    rows = pb.capture_rows(root / "verified", root / "plain")
    assert rows and all(r["component"] == "P1" and r["derived"] for r in rows)
    assert {r["scenario"] for r in rows} == {"honest"}
    by = {(r["record"], r["step"]) for r in rows}
    assert by == {("time", 1), ("time", 2), ("time", 3), ("memory", 1), ("memory", 2),
                  ("count", 1), ("count", 2)}
    # capture observes; it adds no arithmetic
    assert all(r["flops"] == 0 for r in rows if r["record"] == "count")
    assert all(r["peak_bytes"] is not None for r in rows if r["record"] == "memory")
    assert pb.capture_rows(root / "verified", root / "plain", scenario="flip") == []


def test_no_metrics_writes_only_weights(mlp, tmp_path):
    c, D, w0 = mlp
    res = pb.plain_baseline(c, D, w0, T=2, build_model=c.build_model, out_dir=tmp_path,
                            out=lambda s: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == [pb.WEIGHTS_FILE]
    assert res.weights_file == tmp_path / pb.WEIGHTS_FILE
    with pytest.raises(ValueError, match="pass_steps"):
        pb.plain_baseline(c, D, w0, T=2, build_model=c.build_model, pass_steps=3)


# ---- the real model ----------------------------------------------------------------------


@pytest.mark.slow
def test_smollm2_plain_baseline_two_steps(tmp_path):
    """Two real 4×128 steps: P0 rows written, and the final weights bit-identical to two
    captured prove_steps from the same W_0."""
    import os

    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records, schedule

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}; run runs/materialize_data.py or set "
                    f"VERIF_OUTPUT_DIR")
    setup_determinism(cfg)
    c = LlamaComputation.from_config(cfg)
    D = load_dataset_records(cfg.data_dir / "D.bin", c.n)
    w0 = pb.initial_weights(c, c.build_model())
    lines: list[str] = []
    with MetricsWriter(tmp_path, pb.RUN_NAME, model=cfg.model) as mw:
        res = pb.plain_baseline(c, D, w0, T=2, build_model=c.build_model, metrics=mw,
                                pass_steps=1, out=lines.append)
    print("\n" + "\n".join(lines))
    rows = read_records(tmp_path)
    for r in rows:
        if r.get("level") == "phase" and r["step"] > 0:
            v = r.get("time_s", r.get("peak_bytes", r.get("flops")))
            print(f"B7: {r['record']:<6} step {r['step']} {r['component']:<12} {v}"
                  + (f" start {r['start_bytes']}" if r["record"] == "memory" else ""))
    assert {r["step"] for r in rows if r["record"] == "time"} == {1, 2}

    model, w = c.build_model(), w0
    for t in (1, 2):
        out = prove_step(c, model, w, [D[i] for i in schedule(t, c.n_s, len(D))])
        assert out.loss == res.plain.steps[t - 1].loss
        w = out.w_next
        del out
    pb.assert_same_final_weights(c, tmp_path, w)
