"""Decision 4's per-shape check of the GPU BLAKE3 kernel against the C library (FS34).

Once the verifier hashes with the kernel too, no step recomputes a kernel digest by another
implementation, so each run checks the first digest of each device and leaf header: the leaf's
host copy is hashed by the C library (``native``, not the kernel and not the ``blake3``
package) and the two digests are compared.

A *key* is ``(path, device, tag, dtype, shape)``. ``path`` is ``"arrival"`` for a digest a
caller gave :func:`verification.commitment.leaves.hash_step_leaves` as known (hashed on
arrival), and ``"table"`` for a digest from the hashing call's own kernel tables. The set of
checked keys is process-wide, and a key joins it only after its comparison passes;
:func:`new_run` clears it, so each run checks again. A call outside a run uses the set as it
stands, so every key is checked at least once per process.

:class:`KernelMismatchError` is a ``RuntimeError``, not a ``ValueError`` or ``LookupError``: the
verifier's prover-data guard never turns it into a rejection, because a wrong kernel decides
finiteness as well and no verdict may rest on it. The run stops with no verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verification.commitment import native
from verification.commitment.encoding import tensor_payload_span

__all__ = ["KernelMismatchError", "new_run", "is_checked", "mark_checked", "key_of",
           "c_digest", "compare"]


class KernelMismatchError(RuntimeError):
    """The GPU kernel and the C library gave different digests for the same leaf bytes."""


_checked: set[tuple[Any, ...]] = set()


def new_run() -> None:
    """Forget every checked key: the next run checks each again."""
    _checked.clear()


def key_of(path: str, device: torch.device, tag: int, dtype: torch.dtype,
           shape: tuple[int, ...]) -> tuple[Any, ...]:
    return (path, device, tag, dtype, tuple(shape))


def is_checked(key: tuple[Any, ...]) -> bool:
    return key in _checked


def mark_checked(key: tuple[Any, ...]) -> None:
    _checked.add(key)


def c_digest(header: bytes, host: torch.Tensor) -> bytes | None:
    """The leaf digest the C library gives ``header`` and the bytes of the host tensor ``host``,
    or ``None`` when it holds a NaN or Inf."""
    ptr, nbytes = tensor_payload_span(host)
    digests, bad = native.get().hash_leaves(
        header, np.zeros(1, np.uint64), np.array([len(header)], np.uint32),
        np.zeros(1, np.uint32), np.array([ptr], np.uint64), np.array([nbytes], np.uint64),
        np.array([native.FINITE_KIND[host.dtype]], np.uint8), threads=torch.get_num_threads())
    return None if bad == 0 else digests[0].tobytes()


def compare(key: tuple[Any, ...], index: int, kernel: bytes | None,
            c: bytes | None) -> bool:
    """Compare the kernel's digest of leaf ``index`` with the C library's (``None``: non-finite).

    Equal digests, or both non-finite, agree; a pair that disagrees raises
    :class:`KernelMismatchError`. Returns whether a digest was compared (and so the key
    joins the checked set): two non-finite verdicts leave the key unchecked.
    """
    if kernel is None and c is None:
        return False
    if kernel != c:
        path, device, tag, dtype, shape = key
        show = lambda d: "non-finite" if d is None else d.hex()  # noqa: E731
        raise KernelMismatchError(
            f"the GPU kernel and the C library disagree on {device}, leaf header "
            f"(tag {tag:#04x}, {dtype}, shape {shape}), leaf {index} ({path} path): "
            f"kernel {show(kernel)}, C library {show(c)}")
    mark_checked(key)
    return True
