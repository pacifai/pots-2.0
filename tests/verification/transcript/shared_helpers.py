"""The producer side of the cross-process store tests (FS33a), as a module-level function a
``spawn`` child can import. The test files start :func:`producer` in a real second process.

The producer is the prover's process: it builds the tiny decoder as ``test_sharded_run._tiny``
does, runs ``prove_step`` and ``InMemoryStore.commit_step`` for ``steps`` steps (the path
``run_loop`` takes), shares every step before releasing any, and sends
``(package, digests, root, n_leaves)`` per step. It then serves commands on the pipe:
``("release", t)`` -> ``"ok"`` or the error text; ``("mutate", t, i)`` writes leaf ``i`` of
step ``t`` in place; ``("poke", t, i)`` writes leaf ``i``'s first element through ``.data``
(a write ``_version`` does not see) and returns ``(old, new)``; ``("mem",)`` returns
``(memory_allocated before the first counted step (taken after one warm-up step), now, bytes
of one weight set)`` on the first device (CUDA only); ``("quit",)``. ``strategy`` is the CPU sharing
strategy and ``alloc`` the value of ``PYTORCH_ALLOC_CONF`` (``None`` unsets it), both set
before anything is shared or allocated.
"""

import gc
import os

import torch

from tests.verification.decoder_helpers import randomize_biases
from tests.verification.runs.test_sharded_run import SEED, _tiny, forge
from verification.commitment.leaves import dataset_tree, hash_step_leaves, leaf_hash
from verification.commitment.merkle import MerkleTree
from verification.computation.interface import snapshot_weights
from verification.prover.step import prove_step
from verification.runs.loop import _default_schedule
from verification.transcript.errors import StoreMutationError
from verification.transcript.shared import share_step
from verification.transcript.store import InMemoryStore


def reap(proc, wait=30.0):
    """Wait for a child to exit by polling, kill it after ``wait`` seconds. ``Process.join()``
    without a timeout can block long after the child is gone: on macOS the child's
    ``torch_shm_manager`` (started by the first shared CPU storage) inherits the join pipe."""
    import time

    end = time.monotonic() + wait
    while proc.is_alive() and time.monotonic() < end:
        time.sleep(0.1)
    if proc.is_alive():
        proc.kill()
        while proc.is_alive():
            time.sleep(0.1)


def setup(devices, device="cpu", layers=None):
    """``(c, D, model, w0)``, the same in the parent and the child (seeded)."""
    c, D, build = _tiny(devices, device, layers)
    model = build()
    randomize_biases(dict(model.named_parameters()))
    return c, D, model, snapshot_weights(c, model)


def configure(strategy=None, alloc=None):
    """Set the sharing strategy and the allocator setting in this process (before CUDA starts).
    ``alloc=None`` unsets both allocator variables, as ``fs11_session.alloc_env`` does."""
    if strategy is not None:
        torch.multiprocessing.set_sharing_strategy(strategy)
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if alloc is None:
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_ALLOC_CONF"] = alloc


def _warm_up(c, model, w, D, pi, dtree):
    """One whole step, committed, hashed, shared and released, before the baseline of ``mem``.

    A first step makes what a process keeps for good: the cuBLAS workspace (with
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` 8 x 4 MiB per handle and stream, 64 MiB for the
    default and the side stream, allocated through the caching allocator and so counted by
    ``memory_allocated``), the hashing kernel's per-device buffers and the loaded kernels.
    None of that is a step's transcript; the baseline measured before it made the test report
    64 MiB "kept" on the pair session. The step is the schedule's first, from the same ``w``,
    and leaves nothing that the steps after it do not rebuild."""
    idx = list(pi(1))
    out = prove_step(c, model, w, [D[i] for i in idx])
    leaves, tree = InMemoryStore.commit_step(c, out)
    hash_step_leaves(c, leaves)
    _, hold = share_step(c, 1, leaves, tree, [dtree.path(i) for i in idx])
    del out, leaves, tree
    hold.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def producer(conn, devices, device, steps, forged=None, layers=None, strategy=None, alloc=None):
    import faulthandler

    faulthandler.enable()
    configure(strategy, alloc)
    cuda = torch.device(device).type == "cuda"
    c, D, model, w = setup(devices, device, layers)
    pi = _default_schedule(c, D, SEED)
    dtree = dataset_tree(c, D)
    if cuda:
        _warm_up(c, model, w, D, pi, dtree)
    before = torch.cuda.memory_allocated(torch.device(device)) if cuda else 0
    w_bytes = sum(x.numel() * x.element_size() for x in w.values()
                  if isinstance(x, torch.Tensor)) if hasattr(w, "values") else 0
    holds, leaves_of = {}, {}
    for t in range(1, steps + 1):
        idx = list(pi(t))
        out = prove_step(c, model, w, [D[i] for i in idx])
        paths = [dtree.path(i) for i in idx]
        leaves, tree = InMemoryStore.commit_step(c, out)
        if forged and t == 1:
            i, obj = forge(forged)(c, leaves)
            tree.update_leaf(i, leaf_hash(c, i, obj))
            leaves[i] = obj
        w = out.w_next
        digests = hash_step_leaves(c, leaves)
        assert MerkleTree(digests).root == tree.root
        package, hold = share_step(c, t, leaves, tree, paths)
        holds[t], leaves_of[t] = hold, leaves
        conn.send((package, digests, tree.root, len(leaves)))
        del package, out
    while True:
        msg = conn.recv()
        if msg[0] == "quit":
            break
        if msg[0] == "release":
            try:
                holds[msg[1]].release()
                conn.send("ok")
            except StoreMutationError as e:
                conn.send(str(e))
            finally:
                leaves_of.pop(msg[1], None)  # the producer's own references go with the step
                if cuda:
                    torch.cuda.empty_cache()
        elif msg[0] == "poke":
            x = leaves_of[msg[1]][msg[2]].data
            at = (0,) * x.ndim
            old = x[at].item()
            x[at] = 96.0 if old != 96.0 else 64.0
            conn.send((old, x[at].item()))
        elif msg[0] == "mem":
            conn.send((before, torch.cuda.memory_allocated(torch.device(device)), w_bytes))
        elif msg[0] == "mutate":
            x = leaves_of[msg[1]][msg[2]]
            x.add_(0) if isinstance(x, torch.Tensor) else None
            conn.send("done")


def synthetic_leaves(device="cpu"):
    """Leaves with the shapes a step does not give a tiny decoder for sure: a non-contiguous
    view, an offset slice of one storage, a bf16 and an int tensor, an empty tensor, a record."""
    from setup.records import Record

    g = torch.Generator().manual_seed(3)
    base = torch.randn(4, 6, generator=g).to(device)
    leaves = [
        base,
        base.t(),  # non-contiguous, same storage
        base[1:3, 2:5],  # offset into the same storage
        torch.randn(5, generator=g).to(device).to(torch.bfloat16),
        torch.arange(7, device=device),
        torch.empty(0, 3, device=device),
        Record(torch.arange(4, dtype=torch.int32), torch.arange(4, dtype=torch.int32) + 1,
               torch.tensor([0, 0, 1, 1], dtype=torch.int32)),
        torch.randn(2, 2, generator=g).to(device),
    ]
    return leaves


def synthetic_tree(n):
    return MerkleTree([bytes([i]) * 32 for i in range(n)])


def producer_synthetic(conn, device, strategy=None, alloc=None):
    """Shares :func:`synthetic_leaves`; sends the package and value copies for comparison."""
    from types import SimpleNamespace

    from setup.records import Record

    configure(strategy, alloc)
    leaves = synthetic_leaves(device)
    pkg, hold = share_step(SimpleNamespace(n_s=0), 1, leaves, synthetic_tree(len(leaves)))
    copies = [x if isinstance(x, Record) else x.detach().cpu().clone() for x in leaves]
    conn.send((pkg, copies, device))
    del pkg
    assert conn.recv() == "release"
    try:
        hold.release()
        conn.send("ok")
    except StoreMutationError as e:
        conn.send(str(e))


def producer_cuda_release(conn, alloc=None):
    """Shares CUDA synthetic leaves, drops every name of its own, sends the package, and after
    ``"release"`` replies ``("ok", memory_allocated before the leaves, memory_allocated after
    release and empty_cache)`` or the error text."""
    from types import SimpleNamespace

    configure(None, alloc)
    dev = torch.device("cuda:0")
    torch.zeros(1, device=dev)  # the context and its fixed allocations exist before the baseline
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(dev)
    leaves = synthetic_leaves("cuda:0")
    pkg, hold = share_step(SimpleNamespace(n_s=0), 1, leaves, synthetic_tree(len(leaves)))
    from setup.records import Record

    copies = [x if isinstance(x, Record) else x.detach().cpu().clone() for x in leaves]
    conn.send((pkg, copies, before))
    del pkg, leaves
    assert conn.recv() == "release"
    try:
        hold.release()
        torch.cuda.empty_cache()
        conn.send(("ok", before, torch.cuda.memory_allocated(dev)))
    except StoreMutationError as e:
        conn.send(str(e))
    conn.recv()
