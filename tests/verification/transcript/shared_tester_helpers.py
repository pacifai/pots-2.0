"""Producer functions for ``test_shared_tester.py`` (FS33a tester): spawn-importable."""

import os
from types import SimpleNamespace

import torch

from tests.verification.transcript import shared_helpers as H
from verification.transcript.shared import share_step


def producer_drop(conn):
    """Shares, drops its own names for the leaves, allocates same-size tensors with other
    bytes, then sends. Returns the original bytes with the package."""
    leaves = H.synthetic_leaves("cpu")
    pkg, hold = share_step(SimpleNamespace(n_s=0), 1, leaves, H.synthetic_tree(len(leaves)))
    copies = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in leaves]
    del leaves
    junk = [torch.full((4, 6), 7.0), torch.full((5,), 9.0, dtype=torch.bfloat16),
            torch.full((7,), 5, dtype=torch.int64), torch.full((2, 2), 3.0)]
    conn.send((pkg, copies))
    del pkg
    assert conn.recv() == "release"
    hold.release()
    conn.send("ok" + str(len(junk)))


def producer_dies(conn, mode, strategy=None):
    """``mode`` 'after_send': send then exit at once. 'while_held': send, wait for the
    consumer's 'opened', then exit without releasing."""
    H.configure(strategy)
    leaves = H.synthetic_leaves("cpu")
    pkg, hold = share_step(SimpleNamespace(n_s=0), 1, leaves, H.synthetic_tree(len(leaves)))
    copies = [x.detach().clone() if isinstance(x, torch.Tensor) else x for x in leaves]
    conn.send((pkg, copies))
    if mode == "while_held":
        conn.recv()
    os._exit(0)
