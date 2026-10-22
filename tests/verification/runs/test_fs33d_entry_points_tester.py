"""FS33d-1 tester: tries to break the entry points' pipelined mode (items 1-7)."""

import functools
import gc
import json
import pickle
import shutil

import pytest
import torch

from setup.config import load_config
from tests.verification.runs import pipelined_helpers as P
from tests.verification.runs import test_fs33d_entry_points as E
from tests.verification.runs.test_fs33d_entry_points import (  # noqa: F401 (fixtures)
    _cheat_env,
    _lines,
    _run_main,
    _stored,
    tiny_spec,
)
from tests.verification.runs.test_run_verified import (  # noqa: F401 (tiny_main is a fixture)
    band_ctx,
    tiny_main,
)
from verification.commitment.leaves import dataset_tree
from verification.computation.interface import snapshot_weights
from verification.runs import cheats, pipeline, prover_spec
from verification.runs import flipped_matmul_sweep as fs
from verification.runs import kept_step
from verification.runs import run_verified as rv
from verification.runs.hidden_steps import CHAIN_STEP, run_hidden
from verification.runs.pipeline import ProverProcessError
from verification.runs.poisoned_step import run_poisoned
from verification.runs.scenarios import HONEST, ReusedModel, run_scenario


# ---- the mode line ---------------------------------------------------------------------


def _count_pipelined_runs(monkeypatch):
    calls = []
    real = pipeline.run_pipelined
    monkeypatch.setattr(pipeline, "run_pipelined",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_pipelined_run_verified_runs_exactly_two_pipelined_passes(
        tiny_main, tiny_spec, monkeypatch, capsys):  # noqa: F811
    """The timed run and the memory pass are pipelined; the counting pass never is."""
    calls = _count_pipelined_runs(monkeypatch)
    seen = []
    real = rv.count_run
    monkeypatch.setattr(rv, "count_run", lambda *a, **k: seen.append(k) or real(*a, **k))
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    assert rv.main(["--metrics", "--pass-steps", "1"]) == 0
    out = capsys.readouterr().out
    assert len(calls) == 2 and len(seen) == 1 and seen[0]["pipelined"] is False
    assert out.count("loop mode:") == 1 and out.count("loop mode: pipelined") == 1


def test_sequential_run_verified_prints_one_sequential_line_with_metrics(
        tiny_main, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setenv("VERIF_PIPELINED", "0")
    assert rv.main(["--metrics", "--pass-steps", "1"]) == 0
    out = capsys.readouterr().out
    assert out.count("loop mode:") == 1 and out.count("loop mode: sequential") == 1


def test_sequential_no_metrics_run_computes_only_its_own_reference(
        tiny_main, monkeypatch, capsys):  # noqa: F811
    calls = []
    real = rv.honest_final
    monkeypatch.setattr(rv, "honest_final", lambda *a, **k: calls.append(a[3]) or real(*a, **k))
    monkeypatch.setenv("VERIF_PIPELINED", "0")
    assert rv.main(["--no-metrics", "--pass-steps", "1"]) == 0
    assert calls == [3]


# ---- bit equality ----------------------------------------------------------------------


@pytest.mark.parametrize("keep", ["1", "3"])
def test_run_verified_with_metrics_pipelined_equals_sequential(
        tiny_main, tiny_spec, monkeypatch, capsys, keep):  # noqa: F811
    _, _, root = tiny_main
    got = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VERIF_PIPELINED", mode)
        for d in root.glob("*/"):
            if d.name not in ("data",):
                shutil.rmtree(d) if d.is_dir() else None
        out = _run_metrics(rv, capsys, "--keep-step", keep)
        got[mode] = ([x for x in _lines(out) if " peak " not in x], _stored(root),
                     next(root.glob("*/final_weights.json")).read_bytes())
    s, p = got["0"], got["1"]
    assert [x for x in s[0] if not x.startswith("loop mode")] == \
        [x for x in p[0] if not x.startswith("loop mode")]
    assert s[1] == p[1] and len(s[1]) > 10 and s[2] == p[2]
    assert any(x.startswith(f"stored step {keep}") for x in p[0])


def _run_metrics(rv_mod, capsys, *args):
    assert rv_mod.main(["--metrics", "--pass-steps", "1", *args]) == 0
    return capsys.readouterr().out


def test_calibrate_band_file_bytes_with_metrics_off(tiny_main, tiny_spec, monkeypatch,
                                                    capsys):  # noqa: F811
    from verification.runs import calibrate as cal

    monkeypatch.setattr(cal, "setup_determinism", lambda cfg: torch.manual_seed(cfg.seed))
    _, _, root = tiny_main
    got = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VERIF_PIPELINED", mode)
        (root / "bands.json").unlink(missing_ok=True)
        cal.main(["--steps", "2", "--no-metrics", "--write-bands-despite-guard"])
        out = capsys.readouterr().out
        assert out.count("loop mode:") == 1
        got[mode] = (root / "bands.json").read_bytes()
    assert len(got["0"]) > 100 and got["0"] == got["1"]


def test_sequential_calibration_stopped_at_a_gate_computes_no_counting_reference(
        tiny_main, tiny_spec, monkeypatch):  # noqa: F811
    """The sequential path keeps its old order: the counting pass's reference is made after the
    gates, so a run that stops at one never pays for it (pipelined makes it before the drop)."""
    from verification.runs import calibrate as cal

    monkeypatch.setattr(cal, "setup_determinism", lambda cfg: torch.manual_seed(cfg.seed))
    steps = []
    real = cal.honest_final
    monkeypatch.setattr(cal, "honest_final", lambda *a, **k: steps.append(a[3]) or real(*a, **k))
    seen = {}
    monkeypatch.setattr(cal, "_run", lambda *a, **k: seen.update(k) or 2)  # a gate stops it
    monkeypatch.setenv("VERIF_PIPELINED", "0")
    assert cal.main(["--steps", "2", "--metrics"]) == 2
    assert steps == [2] and seen["count_final"] is None
    steps.clear()
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    assert cal.main(["--steps", "2", "--metrics"]) == 2
    assert steps == [2, cal.COUNT_STEPS] and seen["count_final"] is not None


def test_llama_step_pipelined_equals_sequential(
        tiny_main, tiny_spec, monkeypatch, capsys):  # noqa: F811
    from verification.runs import llama_step

    _, _, root = tiny_main
    monkeypatch.setattr(llama_step, "setup_determinism", prover_spec.setup_determinism)
    outs = {}
    for mode in ("0", "1"):
        monkeypatch.setenv("VERIF_PIPELINED", mode)
        assert llama_step.main(["--steps", "2", "--no-metrics", "--bands",
                                str(root / "bands.json")]) == 0
        out = capsys.readouterr().out
        assert out.count("loop mode:") == 1
        assert "loop mode" in out
        outs[mode] = [ln for ln in out.splitlines()
                      if "loop mode" not in ln and " s (" not in ln and "RSS" not in ln
                      and "run checks" not in ln and "metrics:" not in ln
                      and not (ln.startswith("   ") and "|" in ln)]  # the wall-clock step rows
    assert len(outs["0"]) >= 3 and outs["0"] == outs["1"]


def test_run_scenario_pipelined_step_roots_and_weights_equal(tmp_path):
    """The loop under the cheats' env, honest: every step's root and the final weights."""
    s = _cheat_env(tmp_path, pipelined=False)
    p = _cheat_env(tmp_path, pipelined=True)
    kw = dict(T=3, k=s.k, seed=s.seed, h_D=s.h_D)
    a = run_scenario(s.c, s.D, s.w0, HONEST, final=s.final(3), verifier=s.verifiers.for_steps(3),
                     **kw)
    b = run_scenario(p.c, p.D, p.w0, HONEST, final=p.final(3), verifier=p.verifiers.for_steps(3),
                     pipelined=True, prover=P.MLPSpec(), **kw)
    assert [x.root for x in a.loop.steps] == [x.root for x in b.loop.steps]
    assert all(torch.equal(a.loop.w_final[n], b.loop.w_final[n]) for n in s.c.weight_names)


# ---- the spec --------------------------------------------------------------------------


def test_spec_env_has_every_verif_variable_and_no_secret(monkeypatch):
    from verification.runs import fs11_session as fs11

    for n in fs11.SECRETS:
        monkeypatch.setenv(n, "s3cret")
    monkeypatch.setenv("VERIF_SOMETHING_NEW", "x")
    monkeypatch.setenv("POTS_NATIVE_CACHE", "/cache")
    monkeypatch.setenv("VERIF_ETA", "1e-3")
    spec = prover_spec.decoder_spec(load_config())
    env = spec.args[0]
    assert not set(env) & set(fs11.SECRETS) and "s3cret" not in pickle.dumps(spec).decode("latin1")
    assert {n for n in __import__("os").environ if n.startswith("VERIF_")} <= set(env)
    assert env["POTS_NATIVE_CACHE"] == "/cache"


def test_a_child_on_other_data_fails_at_the_handshake(
        tiny_main, monkeypatch, tmp_path):  # noqa: F811
    c, D, root = tiny_main
    other = tmp_path / "other"
    (other / "data").mkdir(parents=True)
    from setup.data import encode_records_file
    from tests.verification.decoder_helpers import make_records

    D2 = make_records([12, 9, 7, 12, 11, 8, 10, 6], seed=5)  # different records, same sizes
    (other / "data" / "D.bin").write_bytes(encode_records_file(D2))
    (other / "data" / "meta.json").write_text(
        json.dumps({"h_D": dataset_tree(P.tiny_computation(load_config()), D2).root.hex()}))
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    c = P.tiny_computation(cfg)
    w0 = snapshot_weights(c, c.build_model())
    env = {**prover_spec.spec_env(), "VERIF_OUTPUT_DIR": str(other)}
    spec = functools.partial(prover_spec.decoder_prover, env, P.tiny_computation)
    h_D = dataset_tree(c, D).root
    from verification.verifier.calibration import band_context, load_bands

    bands = load_bands(root / "bands.json", k=E.K,
                       context=band_context(cfg, c, h_D=h_D, k=E.K))
    fac = rv.JudgingVerifiers(c, len(D), w0, bands, k=E.K, h_D=h_D, seed=cfg.seed)
    with pytest.raises(ProverProcessError) as e:
        run_scenario(c, D, w0, HONEST, T=2, k=E.K, seed=cfg.seed, final=w0, h_D=h_D,
                     verifier=fac.for_steps(2), pipelined=True, prover=spec)
    assert e.value.kind == "handshake", str(e.value)[-600:]


# ---- the drop --------------------------------------------------------------------------


def _reused_env(tmp_path, *, pipelined):
    env = _cheat_env(tmp_path, pipelined=pipelined)
    models = ReusedModel(env.c.build_model)
    return cheats.dataclasses.replace(env, models=models), models


def _spy_model_alive(monkeypatch, models):
    alive, finals = [], []
    real = pipeline.run_pipelined
    monkeypatch.setattr(pipeline, "run_pipelined",
                        lambda *a, **k: alive.append(models.model is not None) or real(*a, **k))
    import verification.runs.scenarios as sc

    rf = sc.honest_final
    monkeypatch.setattr(sc, "honest_final", lambda *a, **k: finals.append(a[3]) or rf(*a, **k))
    return alive, finals


@pytest.mark.parametrize("which", ["poisoned", "hidden", "flip"])
def test_cheats_run_with_no_parent_model_and_no_rebuild(tmp_path, monkeypatch, which):
    env, models = _reused_env(tmp_path, pipelined=True)
    t = {"poisoned": env.poison_step, "hidden": 2, "flip": fs.SWEEP_STEP}[which]
    env.prepare(t)
    assert models.model is None
    alive, finals = _spy_model_alive(monkeypatch, models)
    {"poisoned": run_poisoned, "hidden": run_hidden, "flip": fs.run_flip}[which](
        env, out=lambda s: None)
    assert alive and not any(alive) and finals == []


def test_full_scales_hidden_stage_prepares_the_hidden_steps_reference(tmp_path, monkeypatch):
    env, models = _reused_env(tmp_path, pipelined=True)
    assert env.poison_step != 2
    env.prepare(CHAIN_STEP)  # what run_cheats(hidden=True) prepares
    alive, _ = _spy_model_alive(monkeypatch, models)
    run_hidden(env, out=lambda s: None)
    assert alive and not any(alive)


def test_the_flip_main_path_prepares_before_dropping(tmp_path, monkeypatch):
    env, models = _reused_env(tmp_path, pipelined=True)
    env.prepare(fs.SWEEP_STEP)
    env.prepare(fs.SWEEP_STEP)  # idempotent: no second model build
    assert models.model is None


# ---- the stored step -------------------------------------------------------------------


def test_stored_step_write_is_outside_the_steps_clocks(tiny_main, tiny_spec, monkeypatch,
                                                       capsys):  # noqa: F811
    import time

    delay = 1.5
    real = rv.KeptStepHandoff.write_from

    def slow(self, *a, **k):
        time.sleep(delay)
        return real(self, *a, **k)

    monkeypatch.setattr(rv.KeptStepHandoff, "write_from", slow)
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    steps = []
    real_scn = pipeline.run_pipelined

    def spy(*a, **k):
        r = real_scn(*a, **k)
        steps.extend(r.steps)
        return r

    monkeypatch.setattr(pipeline, "run_pipelined", spy)
    assert rv.main(["--no-metrics", "--keep-step", "2"]) == 0
    assert steps and all(s.handoff_s < delay * 0.8 for s in steps), [s.handoff_s for s in steps]


def test_a_stored_root_mismatch_raises_and_stops_the_child(tiny_main, tiny_spec, monkeypatch,
                                                            capsys):  # noqa: F811
    monkeypatch.setattr(kept_step, "transcript_root", lambda *a, **k: b"\0" * 32)
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    with pytest.raises(RuntimeError, match="root over the written files"):
        rv.main(["--no-metrics", "--keep-step", "2"])
    gc.collect()
    import multiprocessing as mp

    assert not [p for p in mp.active_children() if p.name == "pots-prover"]


def test_an_unsupported_handoff_is_still_refused(tmp_path):
    from verification.transcript.store import DiskHandoff

    env = _cheat_env(tmp_path, pipelined=True)
    with pytest.raises(ValueError, match="hand-?off"):
        run_scenario(env.c, env.D, env.w0, HONEST, T=1, k=env.k, seed=env.seed,
                     final=env.final(1), h_D=env.h_D, verifier=env.verifiers.for_steps(1),
                     pipelined=True, prover=P.MLPSpec(), handoff=DiskHandoff(tmp_path / "x"))


# ---- the refusals ----------------------------------------------------------------------


@pytest.mark.parametrize("mod", ["bring_up", "store_crosscheck", "mlp_smoke", "metrics_overhead"])
def test_refusals_happen_before_any_model_is_loaded(mod, monkeypatch, capsys):
    import importlib

    m = importlib.import_module(f"verification.runs.{mod}")
    from verification.computation.instances.decoder import DecoderComputation
    from verification.runs import scenarios

    def boom(*a, **k):
        raise AssertionError("a model was built before the refusal")

    monkeypatch.setattr(DecoderComputation, "from_config", classmethod(boom))
    monkeypatch.setattr(scenarios, "run_scenario", boom)
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    with pytest.raises(SystemExit) as e:
        m.main([])
    assert e.value.code not in (0, None) and mod in str(e.value.code)


def test_full_scales_sweep_stage_runs_sequential_with_pipelined_in_the_driver_env(
        monkeypatch, tmp_path):
    from verification.runs import full_scale

    monkeypatch.setenv("VERIF_PIPELINED", "1")
    envs = []
    from verification.runs.run_verified import KEEP_STEP, TRANSCRIPTS_DIR

    kept_dir = tmp_path / "h"
    (kept_dir / TRANSCRIPTS_DIR / f"step_{KEEP_STEP}").mkdir(parents=True)
    (kept_dir / TRANSCRIPTS_DIR / f"step_{KEEP_STEP}" / "meta.json").write_text("{}")

    class Cell:
        cell = tmp_path
        sweep_args, k_args, failures = [], [], []
        h1_seed = 1

        def honest(self, seed):
            return kept_dir

        def env(self, seed):
            return {**__import__("os").environ, "VERIF_SEED": str(seed)}

        def run_stage(self, stage, d, module, args, env, **kw):
            envs.append(env)
            return 0

    full_scale.stage_sweep_k(Cell())
    assert envs and all(e["VERIF_PIPELINED"] == "0" for e in envs)
    assert __import__("os").environ["VERIF_PIPELINED"] == "1"  # the driver's own is untouched
