"""FS53: the cache release's auto rule (HOST_SIDE_PLAN_DRAFT §4).

``VERIF_RELEASE_CACHE`` unset = auto: release iff the run uses CUDA and prover and verifier
may share a card (unsharded, or the prover's device is a verifier device). ``0`` and ``1``
force it. The rule is pure device arithmetic, so the table runs with fake devices."""

import pytest
import torch

from verification.computation.instances.mlp import MLPComputation
from verification.computation.placement import placement_for
from verification.runs import loop as loop_mod
from verification.runs import metrics
from verification.runs.loop import resolve_release_cache

ONE = None  # unsharded
PAIR = ("cuda:0", "cuda:1")  # the prover's card is also a verifier's
FOUR = ("cuda:1", "cuda:2", "cuda:3")  # GPU 0 trains, GPUs 1-3 verify
CPU3 = ("cpu", "cpu", "cpu")


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("VERIF_RELEASE_CACHE", raising=False)


# (prover, verifier devices, auto answer)
TABLE = [
    ("cuda:0", ONE, True),  # one card, unsharded
    ("cuda", ONE, True),
    ("cuda:0", PAIR, True),  # the pair layout
    ("cuda", PAIR, True),  # "cuda" is cuda:0 (no CUDA here: index 0)
    ("cuda:0", ("cuda:0",), True),
    ("cuda:0", FOUR, False),  # the four-GPU layout
    ("cuda:0", ("cuda:1",), False),
    ("cuda:0", CPU3, False),  # CUDA prover, verifiers elsewhere
    ("cpu", ONE, False),  # CPU: nothing to release
    ("cpu", CPU3, False),
    ("cpu", FOUR, False),  # a CPU prover (pipelined parent's view) never shares a card
]


@pytest.mark.parametrize("prover, devices, auto", TABLE)
def test_auto(prover, devices, auto):
    assert resolve_release_cache(None, prover, devices) == (auto, "auto")


@pytest.mark.parametrize("prover, devices, auto", TABLE)
@pytest.mark.parametrize("on", [True, False])
def test_the_argument_wins(prover, devices, auto, on, monkeypatch):
    monkeypatch.setenv("VERIF_RELEASE_CACHE", "0" if on else "1")
    assert resolve_release_cache(on, prover, devices) == (on, "argument")


@pytest.mark.parametrize("prover, devices, auto", TABLE)
@pytest.mark.parametrize("raw", ["0", "1"])
def test_the_env_beats_auto(prover, devices, auto, raw, monkeypatch):
    monkeypatch.setenv("VERIF_RELEASE_CACHE", raw)
    assert resolve_release_cache(None, prover, devices) == (raw == "1", "env")


def test_a_bad_setting_is_still_an_error(monkeypatch):
    monkeypatch.setenv("VERIF_RELEASE_CACHE", "yes")
    with pytest.raises(ValueError, match="VERIF_RELEASE_CACHE"):
        resolve_release_cache(None, "cuda:0", FOUR)
    monkeypatch.setenv("VERIF_RELEASE_CACHE", "")
    with pytest.raises(ValueError, match="VERIF_RELEASE_CACHE"):
        resolve_release_cache(None, "cuda:0", FOUR)


def test_torch_devices_are_accepted():
    assert resolve_release_cache(None, torch.device("cuda", 0),
                                 (torch.device("cuda", 0),)) == (True, "auto")
    assert resolve_release_cache(None, torch.device("cuda", 0),
                                 (torch.device("cuda", 1),)) == (False, "auto")


def test_release_cache_setting_reads_the_computations_placement():
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=1e-3)
    assert loop_mod.release_cache_setting(None, c, "cuda:0") is True  # unsharded
    assert loop_mod.release_cache_setting(None, c, "cpu") is False
    assert loop_mod.release_cache_setting(False, c, "cuda:0") is False


# ---- the loop calls the release exactly when the rule says ----------------------------------


@pytest.fixture
def spy(monkeypatch):
    """The loop with CUDA faked (no kernels, no allocator): the cache releases it makes."""
    import tests.verification.runs.test_loop as tl
    calls = []

    class Warm:
        def kernels(self, device):
            pass

    monkeypatch.setattr(loop_mod, "GpuBlake3", Warm)
    monkeypatch.setattr(loop_mod.measure_kernel, "kernels", lambda device, k=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    c = MLPComputation((16, 32, 32, 8), n_s=4, eta=tl.ETA)
    D = tl.synthetic_dataset(c.widths, tl.N_RECORDS, seed=0)
    w0 = tl.init_weights(c.widths, seed=0)
    final = tl._plain_final(c, D, w0, tl.SEED)
    return tl, c, D, w0, final, calls


def _loop(spy, monkeypatch, *, prover, devices=None, **kw):
    tl, c, D, w0, final, calls = spy
    monkeypatch.setattr(loop_mod, "model_device", lambda model: torch.device(prover))
    if devices is not None:  # the placement says where the verifiers are (CPU here: the fake CUDA has no card)
        monkeypatch.setattr(type(c), "placement",
                            property(lambda self: placement_for(self, prover, devices)))
    res, _ = tl._run(c, D, w0, final, **kw)
    assert res.verdict.accepted
    return len(calls), tl.T


@pytest.mark.parametrize("prover, devices, kw, released", [
    ("cuda:0", None, {}, True),  # unsharded CUDA: as before
    ("cpu", None, {}, False),  # CPU: no release
    ("cuda:0", CPU3, {}, False),  # the prover alone on its card, as the four-GPU layout
    ("cuda:0", CPU3, {"release_cache": True}, True),  # explicit 1 on it
    ("cuda:0", None, {"release_cache": False}, False),  # explicit 0
])
def test_the_loop_releases_exactly_when_the_rule_says(spy, monkeypatch, prover, devices, kw,
                                                      released):
    n, T = _loop(spy, monkeypatch, prover=prover, devices=devices, **kw)
    assert n == (T if released else 0)


def test_the_loop_follows_the_env_with_the_prover_alone(spy, monkeypatch):
    monkeypatch.setenv("VERIF_RELEASE_CACHE", "1")
    n, T = _loop(spy, monkeypatch, prover="cuda:0", devices=CPU3)
    assert n == T


# ---- the environment record -----------------------------------------------------------------


def _env(monkeypatch, device, devices, **kw):
    probe = metrics.memory_probe("cpu")
    cfg = None if devices is None else {"verifier_devices": devices}
    return metrics.environment_record("r", device=device, probe=probe, config=cfg, **kw)


def test_the_environment_record_holds_the_resolved_value_and_its_rule(monkeypatch):
    env = _env(monkeypatch, "cpu", None)
    assert (env["release_cache"], env["release_cache_rule"]) == (False, "auto")
    env = _env(monkeypatch, "cpu", None, release_cache=True)
    assert (env["release_cache"], env["release_cache_rule"]) == (True, "argument")
    monkeypatch.setenv("VERIF_RELEASE_CACHE", "0")
    env = _env(monkeypatch, "cpu", None)
    assert (env["release_cache"], env["release_cache_rule"]) == (False, "env")
    monkeypatch.setenv("VERIF_RELEASE_CACHE", "1")
    env = _env(monkeypatch, "cpu", None)
    assert (env["release_cache"], env["release_cache_rule"]) == (True, "env")


@pytest.mark.parametrize("prover, devices, auto", [x for x in TABLE if x[0] != "cuda"])
def test_allocator_settings_resolve_like_the_loop(prover, devices, auto):
    s = metrics.allocator_settings(None, prover, devices)
    assert (s["release_cache"], s["release_cache_rule"]) == (auto, "auto")
