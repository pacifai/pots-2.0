"""FS33 fix round 1: recorder device lists for the parent, no deadline on ``bye``, one-line
sweep refusal, normalized device lists."""

import json
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from setup.config import load_config
from setup.data import encode_records_file
from tests.verification.decoder_helpers import make_records, tiny_config
from tests.verification.runs.test_run_verified import K, _band_file, tiny_main  # noqa: F401
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.runs import calibrate as cal
from verification.runs import cheats, llama_step, pipeline, scenarios
from verification.runs import flipped_matmul_sweep as fs
from verification.runs import run_verified as rv
from verification.runs.loop import run_devices
from verification.runs.metrics import MetricsWriter, normalize_devices
from verification.verifier.calibration import band_context


def _c(devices, prover="cuda"):
    sharded = devices is not None
    pl = SimpleNamespace(sharded=sharded, devices=None if devices is None else tuple(
        torch.device(d) for d in devices), prover=torch.device(prover))
    return SimpleNamespace(placement=pl)


# ---- item 4: normalize_devices --------------------------------------------------------------

def test_normalize_devices_merges_repeats_and_keeps_order():
    assert normalize_devices(["cpu", "cpu", torch.device("cpu")]) == (torch.device("cpu"),)
    got = normalize_devices(["cuda:1", "cuda:0", "cuda:1"])
    assert got == (torch.device("cuda:1"), torch.device("cuda:0"))


@pytest.mark.skipif(torch.cuda.is_available(), reason="the index of a bare 'cuda' is the current")
def test_normalize_devices_gives_a_bare_cuda_index_zero_without_cuda():
    assert normalize_devices(["cuda", "cuda:0", "cuda:1"]) == (
        torch.device("cuda:0"), torch.device("cuda:1"))


# ---- item 1: run_devices --------------------------------------------------------------------

def test_run_devices_unsharded_is_none():
    assert run_devices(_c(None), pipelined=False) is None
    assert run_devices(_c(None), pipelined=True) is None


def test_run_devices_sharded_sequential_puts_the_prover_first():
    c = _c(["cuda:0", "cuda:1"], prover="cuda:0")
    assert run_devices(c, pipelined=False) == (torch.device("cuda:0"), torch.device("cuda:1"))
    c = _c(["cuda:1", "cuda:2"], prover="cuda:0")
    assert run_devices(c, pipelined=False) == tuple(
        torch.device(f"cuda:{i}") for i in (0, 1, 2))


def test_run_devices_sharded_pipelined_is_the_verifier_devices():
    c = _c(["cuda:1", "cuda:2"], prover="cuda:0")
    assert run_devices(c, pipelined=True) == (torch.device("cuda:1"), torch.device("cuda:2"))


def test_run_devices_cpu_placement_normalizes_to_one_device():
    c = _c(["cpu", "cpu", "cpu"], prover="cpu")
    assert run_devices(c, pipelined=True) == (torch.device("cpu"),)
    assert run_devices(c, pipelined=False) == (torch.device("cpu"),)


def test_a_writer_given_devices_builds_a_recorder_of_them(tmp_path):
    devs = (torch.device("cuda:0"), torch.device("cuda:1"))
    with MetricsWriter(tmp_path, "r", device="cpu", model="m", corpus="c", attack="a", seed=1,
                       config={}, band_file_hash=None, h_D=b"\0" * 32) as mw:
        assert mw.recorder("s", devices=devs)._devices == devs
        assert mw.recorder("s")._devices == (torch.device("cpu"),)


def test_memory_run_probes_the_first_device(monkeypatch):
    seen = {}
    monkeypatch.setattr(scenarios, "memory_probe", lambda d: seen.setdefault("probe", d) or d)
    monkeypatch.setattr(scenarios, "memory_pass",
                        lambda run, name, fn, probe, devices=None: seen.update(devices=devices)
                        or [])
    devs = (torch.device("cuda:1"), torch.device("cuda:0"))
    scenarios.memory_run(None, [], {}, run="r", T=1, k=1, seed=1, final={}, devices=devs)
    assert seen == {"probe": devs[0], "devices": devs}


def test_run_cheat_passes_run_devices_to_the_writer(monkeypatch):
    calls, got = [], {}
    monkeypatch.setattr(cheats, "run_devices", lambda c, *, pipelined: calls.append(pipelined)
                        or ("sentinel",))

    class W:
        run = "r"
        def recorder(self, name, **kw):
            got.update(kw)
            return None
        def write(self, rows):
            pass

    def boom(*a, **k):
        raise RuntimeError("stop")

    monkeypatch.setattr(cheats, "run_scenario", boom)
    env = SimpleNamespace(c=object(), pipelined=True, D=[], w0={}, k=1, seed=1, h_D=b"",
                          models=None, prover=None, final=lambda T: {},
                          verifiers=SimpleNamespace(for_steps=lambda T: None))
    with pytest.raises(RuntimeError, match="stop"):
        cheats.run_cheat(env, SimpleNamespace(name="x"), T=1, writer=W())
    assert calls == [True] and got["devices"] == ("sentinel",)


SENTINEL = ("cpu",)  # a valid device list the real recorders accept


def _spy(monkeypatch, mod, modes):
    """``mod.run_devices`` returns ``SENTINEL``; returns the devices that reached each
    ``MetricsWriter.recorder`` (by scenario name) and each ``memory_run``."""
    seen = SimpleNamespace(recorders={}, memory=[])
    real_rec, real_mem = MetricsWriter.recorder, mod.memory_run

    def rec(self, name, **kw):
        seen.recorders.setdefault(name, []).append(kw.get("devices"))
        return real_rec(self, name, **kw)

    def mem(*a, **kw):
        seen.memory.append(kw.get("devices"))
        return real_mem(*a, **kw)

    monkeypatch.setattr(mod, "run_devices", lambda c, *, pipelined: modes.append(pipelined)
                        or SENTINEL)
    monkeypatch.setattr(MetricsWriter, "recorder", rec)
    monkeypatch.setattr(mod, "memory_run", mem)
    return seen


def test_run_verified_passes_run_devices(tiny_main, monkeypatch, capsys):
    modes = []
    seen = _spy(monkeypatch, rv, modes)
    assert rv.main(["--pass-steps", "1", "--metrics"]) == 0
    assert modes == [False]
    assert seen.recorders == {"honest": [SENTINEL]} and seen.memory == [SENTINEL]


def test_calibrate_passes_run_devices(tiny_main, monkeypatch, capsys):
    modes = []
    seen = _spy(monkeypatch, cal, modes)
    monkeypatch.setattr(cal, "setup_determinism", lambda cfg: torch.manual_seed(cfg.seed))
    cal.main(["--steps", "2", "--metrics", "--write-bands-despite-guard"])
    assert modes and set(modes) == {False}
    assert seen.recorders == {n: [SENTINEL] for n in ("calibrate", "frozen", "frozen_no_kappa")}
    assert seen.memory == [SENTINEL]


def test_llama_step_passes_run_devices(monkeypatch, tmp_path, capsys):
    n = 12
    c = DecoderComputation(tiny_config(), n_s=2, n=n, eta=1e-3)
    D = make_records([n, n - 3, n - 5, n, n - 1, n - 4, n - 2, n], seed=1)
    monkeypatch.setattr(DecoderComputation, "from_config",
                        classmethod(lambda cls, cfg: DecoderComputation(
                            tiny_config(), n_s=2, n=n, eta=cfg.require_eta())))
    monkeypatch.setattr(llama_step, "setup_determinism", lambda cfg: None)
    for var, value in (("VERIF_OUTPUT_DIR", str(tmp_path)), ("VERIF_ETA", "1e-3"),
                       ("VERIF_K", str(K)), ("VERIF_SEED", "1")):
        monkeypatch.setenv(var, value)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "D.bin").write_bytes(encode_records_file(D))
    h_D = dataset_tree(c, D).root
    (tmp_path / "data" / "meta.json").write_text(json.dumps({"h_D": h_D.hex()}))
    path = tmp_path / "bands.json"
    _band_file(c, D, snapshot_weights(c, c.build_model()), path,
               context=band_context(load_config(), c, h_D=h_D, k=K))
    modes = []
    seen = _spy(monkeypatch, llama_step, modes)
    assert llama_step.main(["--steps", "1", "--metrics", "--bands", str(path)]) == 0
    assert modes == [False]
    assert list(seen.recorders.values()) == [[SENTINEL]] and seen.memory == [SENTINEL]


# ---- item 2: _await_bye has no deadline -----------------------------------------------------

def test_await_bye_has_no_timeout_parameter():
    import inspect
    assert list(inspect.signature(pipeline._await_bye).parameters) == ["conn"]


def test_await_bye_waits_through_silence_and_returns_on_bye():
    a, b = mp.Pipe()
    done = threading.Event()
    t = threading.Thread(target=lambda: (pipeline._await_bye(b), done.set()))
    t.start()
    a.send(("release", 1))  # not a bye: ignored
    time.sleep(3.0)
    assert not done.is_set()
    a.send(("bye",))
    t.join(10)
    assert done.is_set()


def test_await_bye_returns_at_eof():
    a, b = mp.Pipe()
    done = threading.Event()
    t = threading.Thread(target=lambda: (pipeline._await_bye(b), done.set()))
    t.start()
    time.sleep(0.3)
    a.close()
    t.join(10)
    assert done.is_set()


# ---- item 3: the sweep's refusal is one line ------------------------------------------------

@pytest.mark.parametrize("args", [[], ["--source", "in-process"]])
def test_the_sweep_main_refuses_pipelined_in_one_line(monkeypatch, args):
    monkeypatch.setenv("VERIF_PIPELINED", "1")
    with pytest.raises(SystemExit) as e:
        fs.main(args)
    msg = str(e.value.code)
    assert msg.startswith("flipped_matmul_sweep: VERIF_PIPELINED=1 is refused") and "\n" not in msg
