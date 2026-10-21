"""The producer side of the cross-process store tests (FS33a), as a module-level function a
``spawn`` child can import. The test files start :func:`producer` in a real second process.

The producer is the prover's process: it builds the tiny decoder as ``test_sharded_run._tiny``
does, runs ``prove_step`` and ``InMemoryStore.commit_step`` for ``steps`` steps (the path
``run_loop`` takes), shares every step before releasing any, and sends
``(package, digests, root, n_leaves)`` per step. It then serves commands on the pipe:
``("release", t)`` -> ``"ok"`` or the error text; ``("mutate", t, i)`` writes leaf ``i`` of
step ``t`` in place; ``("quit",)``.
"""

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


def producer(conn, devices, device, steps, forged=None, layers=None):
    import faulthandler

    faulthandler.enable()
    c, D, model, w = setup(devices, device, layers)
    pi = _default_schedule(c, D, SEED)
    dtree = dataset_tree(c, D)
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


def producer_synthetic(conn, device):
    """Shares :func:`synthetic_leaves`; sends the package and value copies for comparison."""
    from types import SimpleNamespace

    from setup.records import Record

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
