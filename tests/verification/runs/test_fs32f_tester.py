"""FS32f tester's tests, beyond the coder's: the sharded run over fake devices (2, 3 and 4 of
them, every decoder switch, both dtypes, ``head_blocks`` 1 and 2, a forged leaf in every unit),
the callers (``honest_final``'s storage check, ``check6_trial``, ``run_flip``), and the
``fs32-pair`` session's comparison, plan and peak sampler."""

import contextlib
import functools
import io
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.verification.decoder_helpers import (
    SWITCHES,
    make_records,
    randomize_biases,
    switch_id,
    tiny_config,
)
from tests.verification.runs import test_sharded_run as sr
from verification.commitment.leaves import dataset_tree, leaf_hashes_of
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.runs import fs11_session as fs
from verification.runs import scenarios
from verification.runs.loop import run_loop
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

JOB = Path(fs.__file__).with_name("fs11_job.sh")
README = Path(fs.__file__).parents[1] / "README.md"
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
SEED, K, T = 1, 5, 2


# ---- the sharded run over fake devices ---------------------------------------------------


def fakes(n):
    return ("cpu",) * n


def build_run(switch, dtype, B, devices, edit=None):
    """A verified two-step run of the tiny decoder under provisional bands, as
    ``test_sharded_run.run`` but with every switch of the decoder: the same numbers the
    comparison needs."""
    family, tied = switch
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                           compute_dtype=DTYPES[dtype], device="cpu", head_blocks=B,
                           verifier_devices=devices)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)
    model = c.build_model()
    randomize_biases(dict(model.named_parameters()))
    w0 = snapshot_weights(c, model)
    w_T = scenarios.honest_final(c, D, w0, T, seed=SEED, build_model=lambda: model)
    final = leaf_hashes_of(c, ((c.w_t_index(n), w_T[n]) for n in c.weight_names))
    v = Verifier(c, h_D=dataset_tree(c, D).root, n_records=len(D), k=K, n_steps=T,
                 bands=Bands.provisional(), w0=w0, allow_provisional=True, seed=SEED)
    hand = sr.Recording(edit)
    res = run_loop(c, model, D, w0, v, final=final, handoff=hand)
    return c, res, v, hand


@functools.lru_cache(maxsize=None)
def unsharded(switch, dtype, B, spec=None):
    return build_run(switch, dtype, B, None, forge_unit(*spec) if spec else None)


def forge_unit(kind, unit):
    """An ``edit`` for ``Recording``: one leaf changed in replay unit ``unit``: its forward
    product's, a linear weight's ``W_{t+1}`` or a glue weight's."""
    def edit(c, leaves):
        if kind == "product":
            m = next((s.m for s in c.products
                      if s.kind.name == "FORWARD" and c.product_unit(s.m) == unit), None)
            i = None if m is None else c.product_index(m)
        else:
            names = c.linear_weights if kind == "linear" else c.glue_gradient_weights
            n = next((n for n in names if c.weight_unit(n) == unit), None)
            i = None if n is None else c.w_next_index(n)
        if i is None:
            pytest.skip(f"this decoder has no {kind} leaf in unit {unit}")
        return i, sr._bumped(leaves[i])
    return edit


LLAMA = ("llama", True)
CASES = [(s, d, b) for s in SWITCHES for d in DTYPES for b in (1, 2)]
IDS = [f"{switch_id(s)}-{d}-B{b}" for s, d, b in CASES]


@pytest.mark.parametrize("n_dev", [2, 3, 4])
@pytest.mark.parametrize("switch,dtype,B", CASES, ids=IDS)
def test_every_switch_dtype_and_block_count_decides_alike(switch, dtype, B, n_dev):
    one, many = unsharded(switch, dtype, B), build_run(switch, dtype, B, fakes(n_dev))
    assert one[1].verdict.accepted and many[1].verdict.accepted
    assert len(one[3].roots) == T
    sr._same(one, many)


FORGE = [("glue", 0), ("glue", 1), ("glue", 2), ("glue", 3), ("linear", 1), ("linear", 2),
         ("linear", 3), ("product", 1), ("product", 2), ("product", 3)]


@pytest.mark.parametrize("n_dev", [2, 3, 4])
@pytest.mark.parametrize("spec", FORGE, ids=[f"{k}-unit{u}" for k, u in FORGE])
@pytest.mark.parametrize("switch", [("llama", True), ("qwen2", False)], ids=switch_id)
def test_a_forged_leaf_in_any_unit_is_rejected_alike(switch, spec, n_dev):
    """The first unit (the embedding), the middle ones, the last (the head): the rejection,
    the verdict and every recorded number, ``record_rest``'s included, are the unsharded
    run's."""
    one = unsharded(switch, "fp32", 1, spec)
    many = build_run(switch, "fp32", 1, fakes(n_dev), forge_unit(*spec))
    rej = one[1].rejection
    assert rej is not None and rej.step == 1 and len(one[1].steps) == 1
    assert many[1].rejection == rej
    sr._same(one, many)


@pytest.mark.parametrize("dtype,B", [("bf16", 1), ("bf16", 2), ("fp32", 2)])
@pytest.mark.parametrize("spec", [("glue", 0), ("linear", 2), ("product", 3)])
def test_a_forged_leaf_with_the_other_dtypes_and_blocks(spec, dtype, B):
    one = unsharded(LLAMA, dtype, B, spec)
    many = build_run(LLAMA, dtype, B, fakes(3), forge_unit(*spec))
    assert one[1].rejection is not None and many[1].rejection == one[1].rejection
    sr._same(one, many)


def test_meta_weights_land_on_the_placements_devices():
    """``snapshot_weights`` in a sharded run: each weight on its placement's device, so a move
    that was skipped shows (a ``meta`` tensor has no values to read)."""
    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3,
                           verifier_devices=("meta", "cpu", "meta"))
    w = snapshot_weights(c, c.build_model())
    assert {t.device.type for t in w.values()} == {"meta", "cpu"}
    assert all(t.device == c.placement.weight_device(n) for n, t in w.items())


# ---- honest_final's storage check --------------------------------------------------------


def _final_with(monkeypatch, param, final):
    model = torch.nn.Module()
    model.p = torch.nn.Parameter(param)
    monkeypatch.setattr(scenarios, "run_plain", lambda *a, **k: SimpleNamespace(w_final=final))
    c = SimpleNamespace()
    return scenarios.honest_final(c, [], {}, 1, seed=0, build_model=lambda: model)


def test_equal_data_ptrs_on_two_devices_do_not_collide(monkeypatch):
    """An empty host tensor and a ``meta`` tensor both report ``data_ptr() == 0``: two tensors
    on different devices with the same pointer are not shared storage."""
    param = torch.empty(0)
    final = {"w": torch.empty(4, device="meta")}
    assert param.untyped_storage().data_ptr() == final["w"].untyped_storage().data_ptr() == 0
    assert _final_with(monkeypatch, param, final) is final


@pytest.mark.parametrize("alias", [lambda t: t, lambda t: t.view(2, 2), lambda t: t[:2],
                                   lambda t: t.reshape(-1)], ids=["same", "view", "slice", "flat"])
def test_real_sharing_on_one_device_is_still_caught(monkeypatch, alias):
    p = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(RuntimeError, match="alias"):
        _final_with(monkeypatch, p, {"w": alias(p.data)})


def test_a_copy_on_one_device_is_not_sharing(monkeypatch):
    param = torch.zeros(4)
    final = {"w": param.clone()}
    assert _final_with(monkeypatch, param, final) is final


# ---- check6_trial and run_flip (the MLP of test_cheats) ----------------------------------

from tests.verification.runs import test_cheats as tc  # noqa: E402
from tests.verification.runs.test_cheats import QUIET, _ctx, env, kept  # noqa: E402,F401
from verification.runs import cheats  # noqa: E402
from verification.runs import flipped_matmul_sweep as fsw  # noqa: E402


class SpyLeaf:
    """Stands for a stored product leaf: ``to`` notes its arguments, then converts."""

    def __init__(self, t, log):
        self.t, self.log = t, log

    def to(self, *a, **k):
        self.log.append((a, k))
        return self.t.to(*a, **k)

    def __getattr__(self, name):
        return getattr(self.t, name)


def _trial(env, store, name):
    return fsw.check6_trial(env.c, store, env.bands, _ctx(env), name,
                            gen=torch.Generator().manual_seed(0))


def _first_read_of_the_product(store, c, name, wrap):
    """Make the first ``store.leaf`` call on ``name``'s product (check6_trial's own) return
    ``wrap(leaf)``; every other read (check 2's included) sees the stored leaf."""
    (g_m,) = c.linear_weights[name]
    real, calls = store.leaf, []

    def leaf(i):
        if i == c.product_index(g_m) and not calls:
            calls.append(i)
            return wrap(real(i))
        return real(i)
    store.leaf = leaf
    return calls


def test_check6_trial_asks_for_the_product_on_the_weights_device_and_dtype(env, kept):
    store, _ = kept
    c, log = env.c, []
    name = fsw.check6_weight(c)
    _first_read_of_the_product(store, c, name, lambda t: SpyLeaf(t, log))
    try:
        rec = _trial(env, store, name)
    finally:
        del store.leaf
    w = store.leaf(c.w_t_index(name))
    assert log == [((w.device, c.weight_dtype), {})]
    assert rec["rejected"]


def test_check6_trial_numbers_do_not_depend_on_the_product_leafs_dtype(env, kept):
    """The product leaf read back as float64 (an exact upcast) gives the trial's numbers, the
    root included: it is moved to the weight's dtype before use."""
    store, _ = kept
    name = fsw.check6_weight(env.c)
    ref = _trial(env, store, name)
    _first_read_of_the_product(store, env.c, name, lambda t: t.to(torch.float64))
    try:
        other = _trial(env, store, name)
    finally:
        del store.leaf
    assert other == ref


def _flip_numbers(r):
    return (r.passed, r.outcome, r.actual, r.loop.rejection,
            {t: (repr(s.products), repr(s.tensors)) for t, s in r.verifier.stats.items()})


def test_run_flip_is_the_sweeps_flip_on_step_4(env):
    kw = dict(xs=(1e3,), shapes=("row2",), trials=1, deep_trials=0, flip_name="dX_3", out=QUIET)
    alone = fsw.run_flip(env, flip_name="dX_3", out=QUIET)
    swept = fsw.run_sweep(env, **kw).flip
    assert alone.passed and alone.outcome == "exact"
    assert _flip_numbers(alone) == _flip_numbers(swept)
    # the default product is the binding one, as run_sweep's
    assert _flip_numbers(fsw.run_flip(env, out=QUIET)) == _flip_numbers(
        fsw.run_sweep(env, **{**kw, "flip_name": None}).flip)
    assert alone.loop.rejection.step == fsw.SWEEP_STEP


def test_no_sweep_main_runs_only_the_flip(env, monkeypatch):
    """``--no-sweep``: ``main`` builds the env, runs ``run_flip`` once, prints the oracle
    table and returns 0 when it passes; it never reaches the stored step or the sweep."""
    cfg = SimpleNamespace(metrics=False, model="m", model_revision="r" * 12, output_dir=Path("."),
                          seed=1, attack="a")
    monkeypatch.setattr(fsw, "load_config", lambda: cfg)
    monkeypatch.setattr(fsw, "load_protocol_config", lambda: SimpleNamespace(band_file="b"))
    monkeypatch.setattr(fsw, "setup_determinism", lambda cfg: None)
    monkeypatch.setattr(fsw, "load_env", lambda cfg, pc: env)
    monkeypatch.setattr(fsw, "band_sources", lambda *a, **k: env.bands.source)
    monkeypatch.setattr(fsw, "load_kept_step", lambda *a, **k: pytest.fail("loaded a step"))
    monkeypatch.setattr(fsw, "run_sweep", lambda *a, **k: pytest.fail("ran the sweep"))
    calls = []
    real = fsw.run_flip
    monkeypatch.setattr(fsw, "run_flip", lambda e, **k: calls.append(1) or real(e, out=QUIET))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fsw.main(["--no-sweep"])
    assert rc == 0 and calls == [1]
    assert any(l.startswith("flip ") and " exact " in l for l in buf.getvalue().splitlines())


def test_the_oracle_table_is_what_the_sessions_cheat_reader_takes(env):
    """The real ``report_oracle`` lines of A1-A3, the hidden step and the flip, read by the
    session's ``_cheat_extras``: every cheat is found, with its outcome."""
    from verification.runs.hidden_steps import run_hidden
    from verification.runs.poisoned_step import run_poisoned
    results = [*run_poisoned(env, out=QUIET), run_hidden(env, out=QUIET),
               fsw.run_flip(env, out=QUIET)]
    lines = []
    cheats.report_oracle(results, env.bands, out=lines.append)
    log = Path(env_dir := tc.__file__).parent / "unused"  # noqa: F841 (names only)
    got = fs._cheat_extras(_write(lines))["outcomes"]
    assert got == {r.scenario.name: r.outcome for r in results}
    assert set(got) == set(fs.PAIR_CHEATS) and set(got.values()) == {"exact"}


def _write(lines, _n=[0]):
    import tempfile
    d = Path(tempfile.mkdtemp())
    p = d / "log"
    p.write_text("\n".join(lines) + "\n")
    return p


# ---- the session: the plan -----------------------------------------------------------------


def _plan(tmp_path):
    return fs.fs32_pair_plan(tmp_path / "up", tmp_path / "out")


def test_plan_env_names_no_secret_and_unsets_only_the_setting(tmp_path):
    for p in _plan(tmp_path):
        assert not [k for k in p.env if k in fs.SECRETS or re.search(r"TOKEN|KEY|SECRET", k)], p
        assert all(v is None for k, v in p.env.items() if v is None
                   and k in ("VERIF_VERIFIER_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
        assert p.name == "models" or p.env.get("HF_HUB_OFFLINE", "1") != "0"


def test_plan_every_part_has_a_timeout_and_the_two_arms_differ_only_in_the_setting(tmp_path):
    plan = {p.name: p for p in _plan(tmp_path)}
    assert all(p.timeout_s and p.timeout_s > 0 for p in plan.values())
    for a, b in (("card_unsharded", "card_sharded"), ("run_unsharded", "run_sharded")):
        ea, eb = plan[a].env, plan[b].env
        assert ea["VERIF_VERIFIER_DEVICES"] is None and eb["VERIF_VERIFIER_DEVICES"]
        strip = lambda e: {k: v for k, v in e.items()  # noqa: E731
                           if k not in ("VERIF_VERIFIER_DEVICES", "VERIF_OUTPUT_DIR")}
        assert strip(ea) == strip(eb)
        assert plan[a].argv[:3] == plan[b].argv[:3]
    assert {p.env["VERIF_VERIFIER_DEVICES"] for p in plan.values()
            if p.kind == "cheat"} == {fs.PAIR_DEVICES}
    assert [p.name for p in plan.values() if p.kind == "cheat"] == [
        "poisoned_step", "hidden_steps", "flipped_matmul"]


def test_the_default_deadline_is_about_twice_the_estimates_and_inside_the_guard(tmp_path):
    est = sum(fs.PAIR_ESTIMATE_MIN.values())
    assert 70 <= est <= 90
    text = JOB.read_text()
    m = re.search(r"\[ \"\$SESSION\" = fs32-pair \] && DEFAULT_DEADLINE_S=(\d+)", text)
    assert m and int(m[1]) >= 2 * est * 60 * 0.95
    assert int(m[1]) < 3.5 * 3600
    guard = re.search(r"sleep (\d+)", README.read_text())
    assert guard and int(guard[1]) > int(m[1])


def test_with_the_native_build_first_and_no_gpu_peak_on_it(tmp_path):
    plan = fs.choose_plan(fs.fs32_pair_plan, tmp_path / "up", tmp_path / "out", plan_file=None,
                          selftest=False)
    assert plan[0].name == fs.NATIVE_BUILD_NAME and not plan[0].info.get("gpu_peak")
    assert all(p.info.get("gpu_peak") for p in plan[1:])


def test_the_other_sessions_listings_are_what_they_were(tmp_path):
    """Every older session's ``SESSIONS`` entry is still there with its plan function."""
    assert [k for k in fs.SESSIONS][:4] == ["fs11", "recheck", "profile-check", "nvlink-probe"]
    assert fs.SESSIONS["fs32-pair"][1] is fs.fs32_pair_plan


def _env_doc(gpus, cuda=True):
    return {"gpus": gpus, "cuda_available": cuda}


@pytest.mark.parametrize("gpus,cuda,refused", [
    ([], False, True), (["A40"], True, True), (["A40", "A40"], False, True),
    (["A40", "A40"], True, False), (["H100"] * 4, True, False)])
def test_main_refuses_fewer_than_two_gpus_and_not_by_type(tmp_path, monkeypatch, gpus, cuda,
                                                          refused):
    real = fs.environment(tmp_path / "e")
    monkeypatch.setattr(fs, "environment", lambda out: {**real, **_env_doc(gpus, cuda)})
    monkeypatch.setattr(fs, "pin_to_gpu_numa", lambda *a, **k: {"pinned": False, "message": "-"})
    monkeypatch.setattr(fs, "choose_plan", lambda *a, **k: [])
    monkeypatch.delenv("FS11_ALLOW_ANY_GPU", raising=False)
    monkeypatch.delenv("FS31_CARDS", raising=False)
    out = tmp_path / "out"
    rc = fs.main(["run", "--upload", str(tmp_path / "up"), "--out", str(out),
                  "--deadline", str(time.time() + 60), "--session", "fs32-pair"])
    summary = json.loads((out / "summary.json").read_text())
    assert (rc == fs.EXIT_REFUSED) == refused
    assert ("refused" in summary) == refused


def test_the_job_script_switches_on_the_session(tmp_path):
    text = JOB.read_text()
    assert re.search(
        r"fs11\|recheck\|profile-check\|nvlink-probe\|kernel-speed\|fs32-pair\|device-leaves\)",
        text)
    assert subprocess.run(["bash", "-n", str(JOB)]).returncode == 0


# ---- the session: the comparison fails closed ------------------------------------------


def _p(name, kind, extras=None, status="ok", **info):
    return {"name": name, "kind": kind, "status": status, "info": info, "extras": extras}


def _good(**over):
    def run(arm, **kw):
        ex = {"verdicts": ["accepted"] * 3, "kept_root": "k" * 16, "final_root": "f" * 8,
              "losses": [3.0, 2.5, 2.0]}
        ex = {**ex, **kw}
        ex.setdefault("step_lines", [f"{v}; max residual 1.00" for v in ex["verdicts"]])
        ex.setdefault("step_roots", ["r1", "r2", "r3"])
        return _p(f"run_{arm}", "pair_run", ex, arm=arm)

    def card(arm, **kw):
        ex = {"root": "r", "w_next_root": "w", "plain_w_next_root": "w", "loss": 1.5}
        return _p(f"card_{arm}", "pair_card", {**ex, **kw}, arm=arm)

    parts = [_p("models", "models"), _p("native_build", "other"),
             _p("fast_suite", "pytest", {"tests": 9, "failures": 0, "errors": 0, "skipped": 1,
                                         "slowest_failing": None}),
             _p("two_gpu_slow", "pytest", {"tests": 2, "failures": 0, "errors": 0, "skipped": 0,
                                           "slowest_failing": None}),
             _p("smollm2_calibrate", "other"),
             card("unsharded"), card("sharded"), run("unsharded"), run("sharded")]
    parts += [_p(n, "cheat", {"outcomes": o}) for n, o in (
        ("poisoned_step", {k: "exact" for k in ("A1", "A2", "A3")}),
        ("hidden_steps", {"hidden": "exact"}), ("flipped_matmul", {"flip": "exact"}))]
    return [over.get(p["name"], p) for p in parts]


def test_the_good_fixture_passes():
    assert fs.fs32_pair(_good())["pass"]


@pytest.mark.parametrize("bad", [
    {"card_sharded": _p("card_sharded", "pair_card", {"root": "X", "w_next_root": "w", "loss": 1.5},
                        arm="sharded")},
    {"card_sharded": _p("card_sharded", "pair_card", {"root": "r", "w_next_root": "X", "loss": 1.5},
                        arm="sharded")},
    {"card_sharded": _p("card_sharded", "pair_card", {"root": "r", "w_next_root": "w", "loss": 9.0},
                        arm="sharded")},
    {"card_sharded": _p("card_sharded", "pair_card", None, arm="sharded")},
    {"run_sharded": _p("run_sharded", "pair_run", {"verdicts": ["accepted"] * 3,
                                                   "kept_root": "k" * 16, "final_root": "X",
                                                   "losses": [3.0, 2.5, 2.0]}, arm="sharded")},
    {"run_sharded": _p("run_sharded", "pair_run", {"verdicts": ["accepted"] * 3,
                                                   "kept_root": "X", "final_root": "f" * 8,
                                                   "losses": [3.0, 2.5, 2.0]}, arm="sharded")},
    {"run_sharded": _p("run_sharded", "pair_run", {"verdicts": ["accepted"] * 3,
                                                   "kept_root": "k" * 16, "final_root": "f" * 8,
                                                   "losses": [3.0, 2.5, 2.1]}, arm="sharded")},
    {"run_sharded": _p("run_sharded", "pair_run", {"verdicts": ["accepted", "REJECTED",
                                                                "accepted"],
                                                   "kept_root": "k" * 16, "final_root": "f" * 8,
                                                   "losses": [3.0, 2.5, 2.0]}, arm="sharded")},
    {"run_sharded": _p("run_sharded", "pair_run", {"verdicts": [], "kept_root": None,
                                                   "final_root": None, "losses": None},
                       arm="sharded")},
    {"run_sharded": _p("run_sharded", "pair_run", {"verdicts": ["accepted"] * 2,
                                                   "kept_root": "k" * 16, "final_root": "f" * 8,
                                                   "losses": [3.0, 2.5, 2.0]}, arm="sharded")},
], ids=["root", "w_next_root", "loss", "no-card-report", "final_root", "kept_root",
        "run-losses", "verdict-differs", "run-never-wrote", "fewer-verdicts"])
def test_a_difference_between_the_arms_is_no_pass(bad):
    pair = fs.fs32_pair(_good(**bad))
    assert not pair["pass"]
    assert "FS32 roots equal: NO" in "\n".join(fs._pair_lines(pair)) or not pair["verdicts_equal"]


@pytest.mark.parametrize("outcome", ["accepted", "rejected", "early", "overrun", "wrong-check",
                                     "wrong-kind", "wrong-product", None])
@pytest.mark.parametrize("name", ["poisoned_step", "hidden_steps", "flipped_matmul"])
def test_a_cheat_that_is_not_exact_is_no_pass(name, outcome):
    ex = {"poisoned_step": {k: "exact" for k in ("A1", "A2", "A3")},
          "hidden_steps": {"hidden": "exact"}, "flipped_matmul": {"flip": "exact"}}[name]
    ex[next(iter(ex))] = outcome
    if outcome is None:
        del ex[next(iter(ex))]
    pair = fs.fs32_pair(_good(**{name: _p(name, "cheat", {"outcomes": ex})}))
    assert not pair["pass"]
    assert any("NOT exact" in l for l in fs._pair_lines(pair))


def test_a_cheat_part_that_wrote_nothing_is_no_pass():
    assert not fs.fs32_pair(_good(hidden_steps=_p("hidden_steps", "cheat", {"outcomes": {}})))[
        "pass"]
    assert not fs.fs32_pair(_good(hidden_steps=_p("hidden_steps", "cheat", None)))["pass"]


@pytest.mark.parametrize("name", ["fast_suite", "two_gpu_slow"])
@pytest.mark.parametrize("extras", [{"tests": 5, "failures": 1, "errors": 0, "skipped": 0,
                                     "slowest_failing": {"name": "t", "time_s": 1.0}},
                                    {"tests": 5, "failures": 0, "errors": 1, "skipped": 0,
                                     "slowest_failing": {"name": "t", "time_s": 1.0}}])
def test_a_failing_test_is_no_pass(name, extras):
    pair = fs.fs32_pair(_good(**{name: _p(name, "pytest", extras)}))
    assert not pair["pass"] and pair["slowest_failing"]["part"] == name


@pytest.mark.parametrize("name", ["fast_suite", "two_gpu_slow"])
@pytest.mark.parametrize("extras", [{"junit": None}, None],
                         ids=["no-junit", "no-extras"])
def test_a_test_part_that_never_ran_is_no_pass(name, extras):
    assert not fs.fs32_pair(_good(**{name: _p(name, "pytest", extras, status="not_started")}))[
        "pass"]


@pytest.mark.parametrize("name", ["models", "native_build", "smollm2_calibrate", "card_sharded",
                                  "card_unsharded", "run_sharded", "run_unsharded",
                                  "poisoned_step", "hidden_steps", "flipped_matmul"])
@pytest.mark.parametrize("status", ["failed", "killed_timeout", "killed_deadline",
                                    "not_started"])
def test_a_part_that_failed_or_never_ran_is_no_pass(name, status):
    parts = _good()
    for p in parts:
        if p["name"] == name:
            p["status"] = status
    assert not fs.fs32_pair(parts)["pass"]


def test_two_arms_that_both_reject_the_honest_run_are_no_pass():
    rej = {"verdicts": ["REJECTED"] * 3, "kept_root": "k" * 16, "final_root": "f" * 8,
           "losses": [3.0, 2.5, 2.0]}
    pair = fs.fs32_pair(_good(run_unsharded=_p("run_unsharded", "pair_run", rej,
                                               arm="unsharded"),
                              run_sharded=_p("run_sharded", "pair_run", rej, arm="sharded")))
    assert not pair["pass"]


def test_a_two_gpu_part_that_skipped_everything_is_no_pass():
    ex = {"tests": 2, "failures": 0, "errors": 0, "skipped": 2, "slowest_failing": None}
    assert not fs.fs32_pair(_good(two_gpu_slow=_p("two_gpu_slow", "pytest", ex)))["pass"]


def test_a_comparison_that_finds_no_run_parts_is_none_not_a_pass():
    assert fs.fs32_pair([_p("fast_suite", "pytest")]) is None


# ---- the session end to end, with stand-in parts ------------------------------------------

PY = ["{python}", "-c"]


def _stand_ins(tmp_path, *, bad_cheat=False, sharded_root="aa"):
    """Parts of the real kinds whose commands write what the real ones do."""
    d = tmp_path / "w"
    d.mkdir()
    card = ("import json,sys;json.dump({'card':{'name':'A40','uuid':'u','pci_bus_id':'p'},"
            "'plain':{'w_next_root':'w','loss':1.0},'proved':{'root':ROOT,'w_next_root':'w',"
            "'loss':1.0},'n_leaves':3,'wall_s':{}},open(sys.argv[1],'w'))")
    run = ("import json,sys\nfor t in (1, 2, 3):\n"
           "    print(f'step  {t} (in-sample): accepted; x')\n"
           "    print(f'step  {t} root ' + str(t) * 64)\n"
           "print('stored step 3: p, 3 files, 0.1 GiB, root '+'ab'*8+'… (r)')\n"
           "json.dump({'root':ROOT,'losses':[1.0,2.0]},open(sys.argv[1],'w'))")
    cheat = "print('A1 d a OUT 1 2 3');print('flip d a OUT 1 2 3')"
    junit = ("import sys;open(sys.argv[1],'w').write('<testsuites><testsuite><testcase "
             "classname=\"a\" name=\"b\" time=\"1\"/></testsuite></testsuites>')")
    parts = []
    for name, kind, src, args, info, root in [
            ("card_unsharded", "pair_card", card, [str(d / "cu.json")], {"arm": "unsharded"}, "aa"),
            ("card_sharded", "pair_card", card, [str(d / "cs.json")], {"arm": "sharded"}, "aa"),
            ("run_unsharded", "pair_run", run, [str(d / "wu.json")], {"arm": "unsharded"}, "aa"),
            ("run_sharded", "pair_run", run, [str(d / "ws.json")], {"arm": "sharded"},
             sharded_root)]:
        report = args[0]
        info = {**info, "gpu_peak": True,
                **({"report": report} if kind == "pair_card" else {"weights": report})}
        parts.append(fs.Part(name, [*PY, src.replace("ROOT", repr(root))], timeout_s=30,
                             kind=kind, info=info) if False else
                     fs.Part(name, ["{python}", "-c", src.replace("ROOT", repr(root)), *args],
                             timeout_s=30, kind=kind, info=info))
    parts.append(fs.Part("poisoned_step", ["{python}", "-c",
                                           cheat.replace("OUT", "wrong-check" if bad_cheat
                                                         else "exact")],
                         timeout_s=30, kind="cheat", info={"gpu_peak": True}))
    for name in ("fast_suite", "two_gpu_slow"):
        j = d / f"{name}.xml"
        parts.append(fs.Part(name, ["{python}", "-c", junit, str(j)], timeout_s=30, kind="pytest",
                             info={"junit": str(j), "gpu_peak": True}))
    return parts


def _session(tmp_path, monkeypatch, parts):
    monkeypatch.setattr(fs, "_gpu_used_mib", lambda: {"0": 1, "1": 2})
    lines = []
    s = fs.run_session(parts, upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=time.time() + 60, cwd=Path.cwd(), say=lines.append,
                       session="fs32-pair")
    return s, lines


def test_the_job_log_ends_with_one_line_per_acceptance_item(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _stand_ins(tmp_path))
    fs32 = [l for l in lines if l.startswith("FS32")]
    assert lines[-len(fs32):] == fs32
    assert [l.split(":")[0] for l in fs32] == [
        "FS32 roots equal", "FS32 verdicts equal",
        "FS32 unsharded plain and proved W_{t+1}", "FS32 sharded plain and proved W_{t+1}",
        "FS32 cheat A1", "FS32 cheat A2",
        "FS32 cheat A3", "FS32 cheat hidden", "FS32 cheat flip", "FS32 fast_suite",
        "FS32 two_gpu_slow", *["FS32 FAIL"] * 5, "FS32 slowest failing test"]
    assert {l for l in fs32 if l.startswith("FS32 FAIL")} == {
        f"FS32 FAIL: {n} did not run" for n in (
            "models", "native_build", "smollm2_calibrate", "hidden_steps", "flipped_matmul")}
    assert s["fs32_pair"]["roots_equal"] and s["fs32_pair"]["verdicts_equal"]
    assert s["fs32_pair"]["cheats"]["A1"] == "exact" and s["fs32_pair"]["cheats"]["A2"] is None
    assert not s["fs32_pair"]["pass"]  # A2, A3, hidden and flip did not run


def test_a_differing_root_in_a_session_reads_no(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _stand_ins(tmp_path, sharded_root="bb"))
    assert not s["fs32_pair"]["roots_equal"]
    assert any(l.startswith("FS32 roots equal: NO") and "final_root !=" in l for l in lines)


def test_a_wrong_cheat_outcome_in_a_session_reads_not_exact(tmp_path, monkeypatch):
    s, lines = _session(tmp_path, monkeypatch, _stand_ins(tmp_path, bad_cheat=True))
    assert "FS32 cheat A1: wrong-check (NOT exact)" in lines and not s["fs32_pair"]["pass"]


# ---- the peak sampler -------------------------------------------------------------------


@pytest.mark.parametrize("src,status", [("import sys; sys.exit(3)", "failed"),
                                        ("import time; time.sleep(30)", "killed_timeout")])
def test_the_sampler_stops_and_writes_even_when_the_part_fails(tmp_path, monkeypatch, src, status):
    monkeypatch.setattr(fs, "KILL_GRACE_S", 1)
    orig = fs._GpuPeaks
    monkeypatch.setattr(fs, "_GpuPeaks",
                        lambda path: orig(path, every=0.05, sample=lambda: {"0": 9, "1": 3}))
    before = threading.active_count()
    part = fs.Part("p", ["{python}", "-c", src], timeout_s=1.5 if "sleep" in src else None,
                   info={"gpu_peak": True})
    s = fs.run_session([part], upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=time.time() + 60, cwd=Path.cwd(), say=lambda l: None)
    rec = s["parts"][0]
    assert rec["status"] == status
    assert json.loads((tmp_path / "out/parts/p/gpu_peak.json").read_text())["peak_mib"] == {
        "0": 9, "1": 3}
    assert rec["extras"]["gpu_peak_mib"] == {"0": 9, "1": 3}
    time.sleep(0.2)
    assert threading.active_count() == before  # no sampler thread outlives its part


def test_a_part_that_never_started_has_no_peak_file(tmp_path):
    blocked = fs.Part("p", ["{python}", "-c", "pass"], info={"gpu_peak": True})
    s = fs.run_session([blocked], upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=time.time() - 1, cwd=Path.cwd(), say=lambda l: None)
    assert s["parts"][0]["status"] == "not_started"
    assert s["parts"][0]["extras"]["gpu_peak_mib"] is None
