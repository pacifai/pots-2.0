"""FS33d: the entry points that run the loop, pipelined and sequential, on the same numbers."""

import functools
import gc
import multiprocessing as mp
import pickle
import re
import weakref

import pytest
import torch

from setup.config import load_config
from tests.verification.runs import pipelined_helpers as P
from tests.verification.runs.test_run_verified import (  # noqa: F401 (tiny_main is a fixture)
    K,
    _band_file,
    band_ctx,
    tiny_main,
)
from verification.commitment.leaves import dataset_tree
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.instances.mlp import (
    MLPComputation,
    init_weights,
    synthetic_dataset,
)
from verification.runs import bring_up, cheats, metrics_overhead, mlp_smoke, prover_spec
from verification.runs import flipped_matmul_sweep as fs
from verification.runs import run_verified as rv
from verification.runs import store_crosscheck
from verification.runs.hidden_steps import run_hidden
from verification.runs.loop import run_loop
from verification.runs.poisoned_step import run_poisoned
from verification.runs.scenarios import ReusedModel, SpliceWNext
from verification.verifier.calibration import load_bands
from setup.data import schedule

QUIET = lambda s: None  # noqa: E731
SEED = 1


@pytest.fixture
def tiny_spec(monkeypatch):
    """``decoder_spec`` builds the tiny decoder in the spawned child too: the child does not see
    monkeypatches, so the builder is passed in the spec. The parent seeds as the child does."""
    real = prover_spec.decoder_spec
    wrapped = lambda cfg, **k: real(cfg, build_computation=P.tiny_computation, **k)  # noqa: E731
    monkeypatch.setattr(prover_spec, "decoder_spec", wrapped)
    monkeypatch.setattr(cheats, "decoder_spec", wrapped)
    monkeypatch.setattr(rv, "setup_determinism", lambda cfg: torch.manual_seed(cfg.seed))


# ---- the spec ----------------------------------------------------------------------------


def test_spec_env_holds_only_the_configs_variables():
    env = prover_spec.spec_env({"VERIF_K": "3", "POTS_NATIVE_CACHE": "/x", "HF_TOKEN": "secret",
                                "PATH": "/bin"})
    assert env == {"VERIF_K": "3", "POTS_NATIVE_CACHE": "/x"}


def test_spec_pickles_small_and_refuses_another_config(monkeypatch):
    monkeypatch.setenv("VERIF_ETA", "1e-3")
    cfg = load_config()
    spec = prover_spec.decoder_spec(cfg)
    assert len(pickle.loads(pickle.dumps(spec)).args[0]) >= 1
    assert len(pickle.dumps(spec)) < 4000
    with pytest.raises(ValueError):
        prover_spec.decoder_spec(cfg, environ={**prover_spec.spec_env(), "VERIF_SEED": "9"})


def test_spec_in_a_spawned_child_has_the_parents_w0(tiny_main, monkeypatch):  # noqa: F811
    torch.manual_seed(load_config().seed)
    monkeypatch.setattr(prover_spec, "setup_determinism", lambda cfg: None)
    cfg = load_config()
    spec = prover_spec.decoder_spec(cfg, build_computation=P.tiny_computation)
    spec = pickle.loads(pickle.dumps(spec))
    here = P.side_digest(spec)
    with mp.get_context("spawn").Pool(1) as pool:
        there = pool.apply(P.side_digest, (spec,))
    assert there == here and len(here[0]) > 1


@pytest.mark.slow
def test_spec_in_a_spawned_child_has_the_parents_w0_on_smollm2(monkeypatch):
    import os
    from pathlib import Path

    out = os.environ.get("VERIF_OUTPUT_DIR", "trainer_output/verification")
    data = Path(out) / "data" / "D.bin"
    if not data.exists():
        pytest.skip(f"no committed dataset at {data}")
    cfg = load_config()
    spec = prover_spec.decoder_spec(cfg)
    here = P.side_digest(spec)
    with mp.get_context("spawn").Pool(1) as pool:
        there = pool.apply(P.side_digest, (spec,))
    assert there == here


# ---- run_verified -------------------------------------------------------------------------

_KEEP = ("step ", "final weights: root", "band-file hash", "B7", "loop mode")


def _lines(text):
    out = []
    for ln in text.splitlines():
        if re.match(r"step \d+: prover", ln):
            continue  # the timing line of the end summary
        if ln.startswith(_KEEP) or ln.startswith("stored step"):
            ln = re.sub(r"(?<=[ (])/\S+", "<path>", ln)
            out.append(ln.split("  (")[0])
    return out


def _run_main(root, capsys, *args):
    assert rv.main(["--no-metrics", *args]) == 0
    return capsys.readouterr().out


def _stored(root):
    (d,) = list(root.glob("*/transcripts/step_*"))
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_run_verified_pipelined_equals_sequential(tiny_main, tiny_spec, monkeypatch, capsys):
    _, _, root = tiny_main
    monkeypatch.setenv("VERIF_PIPELINED", "0")
    seq = _run_main(root, capsys, "--keep-step", "2")
    stored_seq = _stored(root)
    final_seq = next(root.glob("*/final_weights.json")).read_bytes()
    for d in root.glob("*/transcripts"):
        import shutil

        shutil.rmtree(d)
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    pip = _run_main(root, capsys, "--keep-step", "2")
    assert seq.count("loop mode: sequential") == 1 and "loop mode: pipelined" not in seq
    assert pip.count("loop mode: pipelined") == 1 and "loop mode: sequential" not in pip
    assert pip.index("loop mode: pipelined") < pip.index("step  1")
    a, b = _lines(seq), _lines(pip)
    assert [x for x in a if x != "loop mode: sequential"] == [
        x for x in b if x != "loop mode: pipelined"]
    assert sum("root" in x for x in a if x.startswith("step ")) == 3
    assert any(x.startswith("stored step 2") for x in b)
    stored_pip = _stored(root)
    assert stored_pip.keys() == stored_seq.keys() and len(stored_pip) > 10
    assert stored_pip == stored_seq  # every leaf file byte for byte
    assert next(root.glob("*/final_weights.json")).read_bytes() == final_seq


def test_run_verified_drops_the_parents_model_and_counts_sequentially(
        tiny_main, tiny_spec, monkeypatch, capsys):
    """During the pipelined run no prover model is alive in the parent; the counting pass
    (``VERIF_METRICS=1``) runs sequential and builds its own."""
    from verification.runs import pipeline

    alive = []
    real = pipeline.run_pipelined
    gc.collect()  # models other tests left behind are not this run's
    before = weakref.WeakSet(o for o in gc.get_objects() if type(o).__name__ == "LlamaForCausalLM")

    def spy(*a, **k):
        gc.collect()
        objs = gc.get_objects()
        replay = {id(o._vmodel) for o in objs  # the verifier's own replay model is meant to stay
                  if isinstance(o, DecoderComputation) and o._vmodel is not None}
        alive.append(sum(type(o).__name__ == "LlamaForCausalLM" and id(o) not in replay
                         and o not in before
                         for o in objs))
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "run_pipelined", spy)
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    assert rv.main(["--metrics", "--pass-steps", "1"]) == 0
    out = capsys.readouterr().out
    assert alive and set(alive) == {0}, alive
    assert out.count("loop mode:") == 1



# ---- the cheats ---------------------------------------------------------------------------


def _cheat_env(tmp_path, *, pipelined):
    c = MLPComputation(P.WIDTHS, n_s=P.N_S, eta=1e-3)
    D = synthetic_dataset(c.widths, P.N_RECORDS, seed=0)
    w0 = init_weights(c.widths, seed=0)
    path = tmp_path / "bands.json"
    if not path.exists():
        _band_file(c, D, w0, path)
    bands = load_bands(path, k=K, context=band_ctx(c, D))
    b = [D[i] for i in schedule(1, c.n_s, len(D), SEED)]
    other = synthetic_dataset(c.widths, 4, seed=7)
    return cheats.CheatEnv.build(
        c, D, w0, bands, k=K, h_D=dataset_tree(c, D).root,
        b_tilde=[b[0], other[1], b[2], other[3]], poison_step=1, seed=SEED,
        prover=P.MLPSpec() if pipelined else None, pipelined=pipelined)


def _untimed(lines):
    """The oracle lines without their millisecond columns and timing lines."""
    return [x.split(" | ")[0] if " | " in x else x for x in lines
            if not x.lstrip().startswith("run checks")]


def _verdicts(rs):
    return [(r.scenario.name, r.passed, r.outcome, r.actual) for r in rs]


def test_cheats_pipelined_equal_sequential(tmp_path, monkeypatch):
    seq_env = _cheat_env(tmp_path, pipelined=False)
    pip_env = _cheat_env(tmp_path, pipelined=True)
    lines_s, lines_p = [], []
    seq = run_poisoned(seq_env, out=lines_s.append)
    pip_env.prepare(pip_env.poison_step)
    pip = run_poisoned(pip_env, out=lines_p.append)
    assert _verdicts(pip) == _verdicts(seq) and all(r.passed for r in pip)
    assert _untimed(lines_p) == _untimed(lines_s) and any("S6e" in x for x in lines_p)
    assert [(r.actual.step, r.actual.check_id) for r in pip] == [(1, "4"), (1, "5"), (1, "6a")]
    w1, w2 = pip[0].loop.w_final, pip[1].loop.w_final
    assert all(torch.equal(w1[n], w2[n]) for n in pip_env.c.weight_names)
    assert all(torch.equal(w1[n], seq[0].loop.w_final[n]) for n in pip_env.c.weight_names)
    hs, hp = [], []
    rs = run_hidden(seq_env, out=hs.append)
    rp = run_hidden(pip_env, out=hp.append)
    assert (rp.passed, rp.outcome, rp.actual) == (rs.passed, rs.outcome, rs.actual)
    assert _untimed(hp) == _untimed(hs)


def test_flip_pipelined_equals_sequential(tmp_path):
    seq_env = _cheat_env(tmp_path, pipelined=False)
    pip_env = _cheat_env(tmp_path, pipelined=True)
    ls, lp = [], []
    rs = fs.run_flip(seq_env, out=ls.append)
    pip_env.prepare(fs.SWEEP_STEP)
    rp = fs.run_flip(pip_env, out=lp.append)
    assert (rp.passed, rp.outcome, rp.actual) == (rs.passed, rs.outcome, rs.actual) and rp.passed
    assert _untimed(lp) == _untimed(ls)


def test_the_sweep_refuses_a_pipelined_env(tmp_path):
    env = _cheat_env(tmp_path, pipelined=True)
    with pytest.raises(ValueError, match="not run pipelined"):
        fs.run_sweep(env, out=QUIET)


@pytest.mark.parametrize("args", [[], ["--source", "in-process"]])
def test_the_sweep_main_refuses_pipelined(monkeypatch, args, capsys):
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    with pytest.raises(SystemExit) as e:
        fs.main(args)
    assert e.value.code != 0
    assert "VERIF_PIPELINED" in capsys.readouterr().err


def test_splice_w_next_crosses_as_host_tensors():
    w = {"a": torch.ones(2, 2, requires_grad=True)}
    got = pickle.loads(pickle.dumps(SpliceWNext(1, w)))
    assert got.w_next["a"].device.type == "cpu" and not got.w_next["a"].requires_grad
    assert torch.equal(got.w_next["a"], w["a"].detach())


def test_reused_model_rebuilds_after_a_drop():
    built = []
    m = ReusedModel(lambda *a, **k: built.append(1) or torch.nn.Linear(2, 2).eval())
    a = m()
    m.drop()
    assert m.model is None and len(built) == 1
    b = m()
    assert b is not a and len(built) == 2


# ---- the refusals of every other entry point ------------------------------------------------


@pytest.mark.parametrize("mod", [bring_up, store_crosscheck, mlp_smoke, metrics_overhead])
def test_other_entry_points_refuse_pipelined_with_one_line(mod, monkeypatch, capsys):
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    with pytest.raises(SystemExit) as e:
        mod.main([])
    msg = str(e.value.code)
    assert (msg.startswith(f"{mod.__name__.rsplit('.', 1)[1]}")
            or "VERIF_PIPELINED=1 is refused" in msg)
    assert "VERIF_PIPELINED=1 is refused" in msg and "\n" not in msg


def test_refuse_pipelined_is_silent_when_off(monkeypatch):
    monkeypatch.setenv("VERIF_PIPELINED", "0")
    prover_spec.refuse_pipelined(load_config(), "x", "why")


# ---- calibrate ----------------------------------------------------------------------------


def test_calibrate_pipelined_writes_the_same_band_file(tiny_main, tiny_spec, monkeypatch,
                                                       capsys):
    from verification.runs import calibrate as cal

    monkeypatch.setattr(cal, "setup_determinism", lambda cfg: torch.manual_seed(cfg.seed))
    _, _, root = tiny_main
    got = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VERIF_PIPELINED", mode)
        (root / "bands.json").unlink(missing_ok=True)
        cal.main(["--steps", "2", "--metrics", "--write-bands-despite-guard"])
        out = capsys.readouterr().out
        assert out.count("loop mode:") == 1
        got[mode] = ((root / "bands.json").read_bytes(), out)
    assert len(got["0"][0]) > 100 and got["1"][0] == got["0"][0]
    assert "loop mode: pipelined" in got["1"][1] and "loop mode: sequential" in got["0"][1]
