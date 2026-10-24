"""O2's check-5 profile: its counters on a toy function, and one profile of the MLP's stored
step 4."""

import json
from types import SimpleNamespace

import torch

from tests.verification.runs.test_cheats import K, QUIET, SEED, env, kept, stored  # noqa: F401
from verification.runs import check5_profile as cp
from verification.runs import flipped_matmul_sweep as fs
from verification.runs import run_verified as rv


def test_count_ops_counts_ops_and_host_reads_per_section():
    """``item`` is one read (its ``_local_scalar_dense``); ``tolist`` is one read, whatever it
    dispatches; ops outside a section go to ``other``. The wrappers come off afterwards."""
    tr = cp.SectionTracker()
    x = torch.arange(4.0)
    tolist = torch.Tensor.tolist
    with cp.count_ops(tr) as n:
        with tr("5.glue"):
            y = x * 2 + 1
            y.sum().item()
        with tr("5.measure"):
            (y @ y).tolist()
            x.tolist()
        _ = x + 1
    assert torch.Tensor.tolist is tolist
    assert n.ops["5.glue"] == 4 and n.reads["5.glue"] == 1
    assert n.reads["5.measure"] == 2 and n.ops["5.measure"] == 1
    assert n.ops[cp.OTHER] == 1 and n.reads[cp.OTHER] == 0
    assert n.kinds[("5.glue", "_local_scalar_dense")] == 1
    assert n.kinds[("5.measure", "Tensor.tolist")] == 2
    assert tr.calls == {"5.glue": 1, "5.measure": 1}


def test_cuda_counts_places_runtime_calls_in_the_innermost_section():
    def ev(name, start, end, cuda=False):
        dt = torch.autograd.DeviceType.CUDA if cuda else torch.autograd.DeviceType.CPU
        return SimpleNamespace(name=name, time_range=SimpleNamespace(start=start, end=end),
                               device_type=dt)
    events = [ev("section:5.measure", 0, 100), ev("section:5.glue", 10, 20),
              ev("cudaLaunchKernel", 15, 16), ev("cudaLaunchKernel", 50, 51),
              ev("cudaStreamSynchronize", 60, 70), ev("cudaMemcpyAsync", 80, 81),
              ev("cuLaunchKernel", 200, 201), ev("void gemv_kernel", 30, 40, cuda=True)]
    r = cp.cuda_counts(events)
    assert r["5.glue"] == {"launches": 1}
    assert r["5.measure"] == {"launches": 1, "syncs": 1, "async_copies": 1}
    assert r[cp.OTHER] == {"launches": 1}
    assert r["total"] == {"kernels": 1, "kernel_us": 10}


def test_profile_of_the_stored_step(env, stored):  # noqa: F811
    """Accepted, equal to the archive, every product's ops counted and some host reads seen;
    two timed passes and no CUDA section on CPU."""
    c = env.c
    ks = rv.load_kept_step(c, stored, step=fs.SWEEP_STEP)
    p = cp.profile(c, ks, env.bands, seed=SEED, h_D=env.h_D, n_records=len(env.D), k=K,
                   repeats=2, out=QUIET)
    assert p.accepted and p.archive_equal and p.cuda is None
    assert p.M == c.M == p.dense + p.member and len(p.check5_s) == 2
    assert p.ops["5.measure"] > 0 and sum(p.reads.values()) > 0
    assert set(p.section_s) == set(cp.SECTIONS)
    lines: list[str] = []
    cp.report(p, out=lines.append)
    assert lines[0].startswith(f"check 5: {c.M} products") and "equal to the archive" in lines[0]


def test_out_appends_one_line_per_call_and_stdout_is_unchanged(env, stored, monkeypatch, capsys,
                                                               tmp_path):  # noqa: F811
    c = env.c
    ks = rv.load_kept_step(c, stored, step=fs.SWEEP_STEP)
    prof = cp.profile(c, ks, env.bands, seed=SEED, h_D=env.h_D, n_records=len(env.D), k=K,
                      repeats=2, out=QUIET)
    cfg = SimpleNamespace(model="m", model_revision="r" * 12, device="cpu", seed=SEED,
                          attack="a", data_dir=tmp_path, output_dir=tmp_path)
    monkeypatch.setattr(cp, "load_config", lambda: cfg)
    monkeypatch.setattr(cp, "load_protocol_config", lambda: SimpleNamespace(band_file=None, k=K))
    monkeypatch.setattr(cp, "setup_determinism", lambda cfg: None)
    monkeypatch.setattr(cp.DecoderComputation, "from_config", lambda cfg: c)
    monkeypatch.setattr(cp, "load_committed_dataset", lambda c, d: (env.D, env.h_D))
    monkeypatch.setattr(cp, "band_context", lambda *a, **k: {})
    monkeypatch.setattr(cp, "load_bands", lambda *a, **k: env.bands)
    monkeypatch.setattr(cp, "load_kept_step", lambda *a, **k: SimpleNamespace(path="p"))
    monkeypatch.setattr(cp, "profile", lambda *a, **k: prof)
    monkeypatch.setattr(cp, "report", lambda p, out=print: out("report"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    base = ["--run-dir", str(run_dir), "--repeats", "2", "--json"]

    cp.main(base)
    first = capsys.readouterr().out
    cp.main([*base, "--out", str(tmp_path / "x.jsonl")])
    assert capsys.readouterr().out == first
    default = run_dir / "check5_profile.jsonl"
    lines = [json.loads(x) for x in default.read_text().splitlines()]
    assert len(lines) == 1
    cp.main(base)
    capsys.readouterr()
    lines = [json.loads(x) for x in default.read_text().splitlines()]
    assert len(lines) == 2
    printed = json.loads(first.splitlines()[-1])
    for line in lines:
        assert set(line) == set(printed) | {"machine_facts", "section_s_per_pass"}
        assert set(line["section_s_per_pass"]) == set(cp.SECTIONS)
        assert all(len(v) == 2 for v in line["section_s_per_pass"].values())
        assert line["machine_facts"]["ram_total_bytes"] > 0
    assert len((tmp_path / "x.jsonl").read_text().splitlines()) == 1


def _ev(name, start, end, cuda=False, index=None):
    dt = torch.autograd.DeviceType.CUDA if cuda else torch.autograd.DeviceType.CPU
    return SimpleNamespace(name=name, time_range=SimpleNamespace(start=start, end=end),
                           device_type=dt, device_index=index)


def test_cuda_counts_splits_kernels_and_device_time_per_device():
    events = [_ev("section:5.measure", 0, 100), _ev("cudaLaunchKernel", 5, 6),
              _ev("k0", 10, 40, cuda=True, index=0), _ev("k0b", 50, 60, cuda=True, index=0),
              _ev("k1", 10, 25, cuda=True, index=1), _ev("k?", 70, 71, cuda=True)]
    r = cp.cuda_counts(events)
    assert r["total"] == {"kernels": 4, "kernel_us": 56}
    assert r["by_device"] == {"cuda:0": {"kernels": 2, "kernel_us": 40},
                              "cuda:1": {"kernels": 1, "kernel_us": 15}}
    assert r["5.measure"] == {"launches": 1}
    assert cp.cuda_counts([])["by_device"] == {}


def test_every_device_of_the_run_is_synced_around_each_pass(monkeypatch):
    synced = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: synced.append(str(d)))
    devs = tuple(torch.device("cuda", i) for i in range(3))
    cp._sync(devs)
    cp._sync(torch.device("cuda", 1))
    cp._sync((torch.device("cpu"),))
    assert synced == ["cuda:0", "cuda:1", "cuda:2", "cuda:1"]


def test_run_devices_are_the_runs_own_then_the_verifier_devices():
    plc = SimpleNamespace(devices=(torch.device("cuda", 1), torch.device("cuda", 2),
                                   torch.device("cuda", 0)))
    c = SimpleNamespace(placement=plc)
    assert cp.run_devices_of(c, "cuda:0") == tuple(torch.device("cuda", i) for i in (0, 1, 2))
    assert cp.run_devices_of(SimpleNamespace(placement=SimpleNamespace(devices=None)),
                             "cpu") == (torch.device("cpu"),)
    assert cp.run_devices_of(SimpleNamespace(), "cpu") == (torch.device("cpu"),)


def test_profile_of_a_fake_sharded_cpu_run(env, stored):  # noqa: F811
    """A computation sharded over ``cpu,cpu`` (one device, twice) profiles as an unsharded one:
    accepted, equal to the archive, and the devices it names are ``["cpu"]``."""
    import copy
    c = copy.copy(env.c)
    from verification.computation.placement import placement_for
    cpu = torch.device("cpu")
    c.__dict__["placement"] = placement_for(c, cpu, (cpu, cpu))  # the MLP has no verifier_devices
    assert c.placement.sharded
    ks = rv.load_kept_step(c, stored, step=fs.SWEEP_STEP)
    p = cp.profile(c, ks, env.bands, seed=SEED, h_D=env.h_D, n_records=len(env.D), k=K,
                   repeats=1, out=QUIET)
    assert p.accepted and p.archive_equal and p.cuda is None and p.devices == ["cpu"]
    assert "devices" not in p.as_record()  # the record keeps its keys
