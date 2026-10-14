"""O2's check-5 profile: its counters on a toy function, and one profile of the MLP's stored
step 4."""

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
