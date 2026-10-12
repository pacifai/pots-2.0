"""A small output directory of hand-made records for the result scripts (EV4, EV5).

Every run is SmolLM2 with the refusal attack, so the scale is ``test``. The directories:

- ``honest-s1``, the honest run: steps 1–4, step 1 a calibration step (not judged, with
  scores far above 1 that must not count), steps 2–4 judged, step 3 rejected. Steps 1–3 have
  residual arrays, step 2's with ``NaN`` and ``inf`` scores (so its maxima are ``inf``), step 4
  none. Cost rows at steps 0–4: time at every step, memory at steps 1–2 and step 0, counts at
  step 1 and step 0, as the passes write them;
- ``honest-s2``, seed 2 with the κ guard off (``ρ_κ`` unscored, ``NaN`` in its arrays) and no
  cost rows: step 1 unjudged, step 2 judged with arrays;
- ``plain_baseline``, plain training at steps 1–4, at the honest runs' commit, data and steps.
  Its environment record has no attack, batch ``n_s`` or length ``n``;
- ``calibrate/k9``, an honest run with nothing judged, and ``sweep``, honest and ``flip``
  steps in one directory: neither is an honest run of the results;
- the cheat runs ``A1-s1``, ``A2-s1``–``A2-s3`` (two reject, at ``t*`` and one step later; one
  never does), ``A3-s1`` and ``A3-s2`` (whose ``ρ_6`` at ``t*`` is infinite) and ``hidden-s1``,
  all at BPR 2 records (0.25);
- ``bands.json`` with ``stats.z = 8`` and one scalar ``tau``.

The expected values in the tests are worked out by hand from the numbers here.
"""

import json
import math

import numpy as np
import pytest

from verification.runs import metrics as vm

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
ATTACK = "refusal"
BPR = (2, 0.25)
INF, NAN = math.inf, math.nan
COMMIT = "0123456789abcdef" * 2 + "01234567"
# what every run's environment record holds, the plain baseline's included
_ENV = {"model": MODEL, "corpus": "c", "device": "cpu", "hardware": {"cpu": "x"}, "threads": 8,
        "git_commit": COMMIT, "git_dirty": False, "h_D": "ab" * 32, "T": 4}


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(vm._json_safe(r)) + "\n" for r in rows))


def _env(run, seed=1, **kw):
    """A verified run's environment record, which also holds the batch and the length."""
    return {"record": "environment", "run": run, **_ENV, "attack": ATTACK, "seed": seed,
            "n_s": 4, "n": 16, **kw}


def _step(run, scenario, t, *, judged=True, check=None, cheat_step=None, bpr=(None, None),
          rho=(None, None, None), n_failing=None, record_pass=None, seed=1):
    return {"record": "step", "run": run, "scenario": scenario, "model": MODEL, "corpus": "c",
            "attack": ATTACK, "seed": seed, "bpr_records": bpr[0], "bpr_actual": bpr[1],
            "cheat_step": cheat_step, "step": t, "judged": judged,
            "verdict": "accept" if check is None else "reject", "first_failing_check": check,
            "rho_5": rho[0], "rho_kappa": rho[1], "rho_6": rho[2],
            "n_failing": n_failing or dict.fromkeys(("rho_5", "rho_kappa", "rho_6")),
            "record_pass": record_pass}


def _archive(root, d, scenario, t, **arrays):
    p = root / d / "residuals" / scenario / f"step_{t:05d}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, run=np.array(d), scenario=np.array(scenario), step=np.array(t),
             **{k: np.asarray(v) for k, v in arrays.items()})


def _cost(kind, run, t, side, comp, level, scenario="honest", **values):
    return {"record": kind, "run": run, "scenario": scenario, "step": t, "side": side,
            "component": comp, "level": level, "calls": 1, **values, "derived": False}


def _counts(flops=0, hash_in=0, calls=0):
    return {"flops": flops, "hash_in_bytes": hash_in, "hash_out_bytes": 32 * calls,
            "hash_calls": calls}


# the honest run's verifier time per step; check 5 grows by 0.1 a step, from 0.9 at step 2
VERIFIER_TIME = {"4": 0.01, "7": 0.02, "2": 0.03, "6a": 0.5, "5": 1.0, "6b": 0.2, "2.root": 0.04}
PROVER_TIME = {"P2": 0.1, "P3": 0.2, "P4": 0.3, "P5": 0.4}
TRAIN = {"load": 0.05, "forward": 0.3, "backward": 0.6, "update": 0.1}  # captured
PLAIN = {"load": 0.05, "forward": 0.25, "backward": 0.6, "update": 0.1}  # plain; P0 = 1.0
ONE_OFF_TIME = {"0": 1.0, "1": 2.0, "8": 3.0, "9": 0.5}


def _honest_costs(run):
    rows = []
    for comp, s in ONE_OFF_TIME.items():
        rows.append(_cost("time", run, 0, "verifier", comp, "component", time_s=s))
    rows += [_cost("time", run, 0, "prover", "P4.tree", "phase", time_s=4.0),
             _cost("time", run, 0, "prover", "P4", "component", time_s=4.0)]
    for t in range(1, 5):
        d = 0.1 * (t - 3)
        rows += [_cost("time", run, t, "prover", f"train.{x}", "phase", time_s=s)
                 for x, s in TRAIN.items()]
        rows.append(_cost("time", run, t, "prover", "P1.label", "phase", time_s=0.02))
        rows += [_cost("time", run, t, "prover", c, "component", time_s=s)
                 for c, s in PROVER_TIME.items()]
        rows += [_cost("time", run, t, "verifier", c, "component",
                       time_s=s + (d if c == "5" else 0.0)) for c, s in VERIFIER_TIME.items()]
        rows += [_cost("time", run, t, "verifier", "5.glue", "sub", time_s=0.4),
                 _cost("time", run, t, "verifier", "5.measure", "sub", time_s=0.6 + d),
                 _cost("time", run, t, "verifier", "6b.glue", "sub", time_s=0.05),
                 _cost("time", run, t, "verifier", "verifier", "total",
                       time_s=sum(VERIFIER_TIME.values()) + d)]
    mem = {"mem_source": "rss", "end_bytes": None}
    for t in (1, 2):
        rows += [_cost("memory", run, t, "prover", "train.forward", "phase", peak_bytes=400,
                       start_bytes=100, **mem),
                 _cost("memory", run, t, "prover", "train.backward", "phase", peak_bytes=900,
                       start_bytes=300, **mem),
                 _cost("memory", run, t, "prover", "train.load", "phase", peak_bytes=100,
                       start_bytes=100, **mem),
                 _cost("memory", run, t, "prover", "train.update", "phase", peak_bytes=300,
                       start_bytes=300, **mem),
                 _cost("memory", run, t, "prover", "P2", "component", peak_bytes=50,
                       start_bytes=0, **mem),
                 _cost("memory", run, t, "verifier", "5", "component",
                       peak_bytes=1000 + 200 * (t - 1), start_bytes=0, **mem)]
    rows += [_cost("memory", run, 0, "verifier", "0", "component", peak_bytes=3000,
                   start_bytes=0, **mem),
             _cost("memory", run, 0, "prover", "P4", "component", peak_bytes=10, start_bytes=0,
                   **mem)]
    flops = {"load": 0, "forward": 45, "backward": 60, "update": 0}
    rows += [_cost("count", run, 1, "prover", f"train.{x}", "phase", **_counts(f))
             for x, f in flops.items()]
    rows += [_cost("count", run, 1, "prover", "P2", "component", **_counts(0, 1000, 10)),
             _cost("count", run, 1, "prover", "P3", "component", **_counts(0, 2000, 20)),
             _cost("count", run, 1, "prover", "P4", "component", **_counts()),
             _cost("count", run, 1, "prover", "P5", "component", **_counts()),
             _cost("count", run, 1, "verifier", "4", "component", **_counts()),
             _cost("count", run, 1, "verifier", "7", "component", **_counts()),
             _cost("count", run, 1, "verifier", "2", "component", **_counts(0, 3000, 30)),
             _cost("count", run, 1, "verifier", "6a", "component", **_counts(30)),
             _cost("count", run, 1, "verifier", "5", "component", **_counts(20)),
             _cost("count", run, 1, "verifier", "6b", "component", **_counts(10)),
             _cost("count", run, 1, "verifier", "2.root", "component", **_counts()),
             _cost("count", run, 0, "verifier", "0", "component", **_counts(0, 500, 5)),
             *(_cost("count", run, 0, "verifier", c, "component", **_counts())
               for c in ("1", "8", "9")),
             _cost("count", run, 0, "prover", "P4", "component", **_counts(0, 700, 7)),
             {"record": "storage", "run": run, "scenario": "honest", "step": 1,
              "transcript_bytes": 5000, "leaves": 10}]
    return rows


def _plain_costs(run):
    rows, mem = [], {"mem_source": "rss", "end_bytes": None}
    for t in range(1, 5):
        rows += [_cost("time", run, t, "prover", f"P0.{x}", "phase", "plain", time_s=s)
                 for x, s in PLAIN.items()]
        rows.append(_cost("time", run, t, "prover", "P0", "component", "plain", time_s=1.0))
    for t in (1, 2):
        rows += [_cost("memory", run, t, "prover", "P0.forward", "phase", "plain",
                       peak_bytes=300, start_bytes=100, **mem),
                 _cost("memory", run, t, "prover", "P0.backward", "phase", "plain",
                       peak_bytes=700, start_bytes=300, **mem),
                 _cost("memory", run, t, "prover", "P0.load", "phase", "plain",
                       peak_bytes=100, start_bytes=100, **mem),
                 _cost("memory", run, t, "prover", "P0.update", "phase", "plain",
                       peak_bytes=300, start_bytes=300, **mem),
                 _cost("memory", run, t, "prover", "P0", "component", "plain",
                       peak_bytes=700, start_bytes=100, **mem)]
    flops = {"load": 0, "forward": 40, "backward": 60, "update": 0}
    rows += [_cost("count", run, 1, "prover", f"P0.{x}", "phase", "plain", **_counts(f))
             for x, f in flops.items()]
    rows.append(_cost("count", run, 1, "prover", "P0", "component", "plain", **_counts(100)))
    return rows


def _honest(root):
    run = "honest-s1"
    _write_jsonl(root / run / "records.jsonl", [
        _env(run),
        _step(run, "honest", 1, judged=False, rho=(50.0, 0.5, 99.0)),
        _step(run, "honest", 2, rho=(INF, INF, INF)),
        _step(run, "honest", 3, check="5", rho=(3.0, 0.9, 0.3)),
        _step(run, "honest", 4, rho=(0.6, 0.5, 0.2)),
        *_honest_costs(run)])
    # fine classes q and k are both forward products in §5.7; S is an attention score
    _archive(root, run, "honest", 1, p_layer=[0], p_cls=["q"], p_spec_class=["forward"],
             rho_5=[[50.0, 40.0]], rho_kappa=[9.0], rho_6=[99.0])
    _archive(root, run, "honest", 2, p_layer=[0, 1], p_cls=["q", "S"],
             p_spec_class=["forward", "attention score"], rho_5=[[0.5, 2.0], [0.1, NAN]],
             rho_kappa=[0.3, INF], rho_6=[0.2, 1.5, NAN])
    _archive(root, run, "honest", 3, p_layer=[0, 1], p_cls=["q", "k"],
             p_spec_class=["forward", "forward"], rho_5=[[0.25, 0.75], [3.0, 0.0]],
             rho_kappa=[0.9, 0.4], rho_6=[0.1, 0.2, 0.3])

    run = "honest-s2"
    _write_jsonl(root / run / "records.jsonl", [
        _env(run, seed=2),
        _step(run, "honest", 1, judged=False, rho=(0.4, None, 0.3), seed=2),
        _step(run, "honest", 2, rho=(0.5, None, 0.25), seed=2)])
    _archive(root, run, "honest", 2, p_layer=[0], p_cls=["q"], rho_5=[[0.5, 0.125]],
             rho_kappa=[NAN], rho_6=[0.25])

    _write_jsonl(root / "plain_baseline" / "records.jsonl",
                 [{"record": "environment", "run": "plain_baseline", **_ENV},
                  *_plain_costs("plain_baseline")])

    _write_jsonl(root / "calibrate" / "k9" / "records.jsonl", [
        _env("calibrate"), _step("calibrate", "honest", 1, judged=False, rho=(70.0, 1, 1)),
        _cost("time", "calibrate", 1, "verifier", "5", "component", time_s=100.0)])
    _write_jsonl(root / "sweep" / "records.jsonl", [
        _env("sweep"), _step("sweep", "honest", 1, rho=(500.0, 1.0, 1.0)),
        _step("sweep", "flip", 1, check="5", rho=(600.0, 1.0, 1.0), bpr=(0, 0.0))])
    _archive(root, "sweep", "honest", 1, p_layer=[0], p_cls=["q"], rho_5=[[500.0, 1.0]],
             rho_kappa=[0.5], rho_6=[0.5])


def _cheats(root):
    def run(name, *steps):
        _write_jsonl(root / name / "records.jsonl", [_env(name, seed=steps[0]["seed"]), *steps])

    nf = lambda a, b, c: {"rho_5": a, "rho_kappa": b, "rho_6": c}  # noqa: E731
    not_run = {"status": "not run", "measured": nf(0, 0, 0)}
    a2_pass = {"status": "complete", "measured": nf(8, 4, 3)}
    run("A1-s1", _step("A1-s1", "A1", 1, check="4", cheat_step=1, bpr=BPR, record_pass=not_run))
    run("A2-s1", _step("A2-s1", "A2", 1, check="5", cheat_step=1, bpr=BPR,
                       rho=(100.0, 0.6, 0.5), n_failing=nf(10, 2, 0), record_pass=a2_pass))
    run("A2-s2",
        _step("A2-s2", "A2", 1, cheat_step=1, bpr=BPR, rho=(0.8, 0.5, 0.4), n_failing=nf(0, 0, 0)),
        _step("A2-s2", "A2", 2, check="5", cheat_step=1, bpr=BPR, rho=(30.0, 0.5, 2.0),
              n_failing=nf(20, 0, 2), record_pass=a2_pass))
    run("A2-s3",
        _step("A2-s3", "A2", 1, cheat_step=1, bpr=BPR, rho=(0.5, 0.5, 0.4), n_failing=nf(0, 0, 0)),
        _step("A2-s3", "A2", 2, cheat_step=1, bpr=BPR, rho=(0.4, 0.5, 0.4), n_failing=nf(0, 0, 0)))
    # layer maxima at t*: s1 4, 7 and no row in the output layer; s2 2, 9, 0.5; s3 0.2, 0.4
    _archive(root, "A2-s1", "A2", 1, p_layer=[0, 0, 1],
             rho_5=[[0.5, 4.0], [1.0, 2.0], [7.0, 0.5]])
    _archive(root, "A2-s2", "A2", 1, p_layer=[0, 1, 1, 2],
             rho_5=[[2.0, 1.0], [3.0, 5.0], [1.0, 9.0], [0.5, 0.25]])
    _archive(root, "A2-s2", "A2", 2, p_layer=[0], rho_5=[[30.0, 1.0]])
    _archive(root, "A2-s3", "A2", 1, p_layer=[0, 1], rho_5=[[0.2, 0.1], [0.3, 0.4]])
    run("A3-s1", _step("A3-s1", "A3", 1, check="6a", cheat_step=1, bpr=BPR,
                       rho=(None, None, 1000.0), n_failing=nf(None, None, 3),
                       record_pass={"status": "complete", "measured": nf(0, 0, 3)}))
    run("A3-s2", _step("A3-s2", "A3", 1, check="6a", cheat_step=1, bpr=BPR,
                       rho=(None, None, INF), n_failing=nf(None, None, 3), seed=2,
                       record_pass={"status": "complete", "measured": nf(0, 0, 3)}))
    run("hidden-s1",
        _step("hidden-s1", "hidden", 1, judged=False, cheat_step=2, bpr=BPR,
              rho=(0.3, 0.5, 0.5), n_failing=nf(0, 0, 0)),
        _step("hidden-s1", "hidden", 2, check="7", cheat_step=2, bpr=BPR, record_pass=not_run))


@pytest.fixture(scope="package")
def out(tmp_path_factory):
    root = tmp_path_factory.mktemp("results_out")
    _honest(root)
    _cheats(root)
    (root / "bands.json").write_text(json.dumps({"stats": {"z": 8.0}, "tau": 38.0}))
    return root
