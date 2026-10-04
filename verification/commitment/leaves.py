"""Leaf encoding and hashing for the step commitment ``h`` and the dataset root ``h_D`` (P9b).

The prover commits with these functions and the verifier recomputes ``h`` with them (check 2),
so both run one encoding path. A leaf is encoded by its position in the computation's layout
(ref block §6). Records go through ``computation.encode_record``, byte-identical to their
``h_D`` leaf (P9b); weights carry tag ``0x02`` and products tag ``0x03``, streamed zero-copy.

Bulk hashing (:func:`leaf_hashes_of` and the functions built on it) validates and encodes each
leaf in order on the calling thread and hashes on worker threads (``merkle.hash_leaves``), so
the digests and the first error raised are those of a leaf-by-leaf loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from setup.records import Record
from verification.commitment.encoding import (
    TAG_PRODUCT,
    TAG_WEIGHT,
    tensor_leaf_header,
    tensor_leaf_payload,
)
from verification.commitment.merkle import (
    MerkleTree,
    hash_leaf,
    hash_leaves,
    hash_record_leaf,
    merkle_root,
)
from verification.transcript.errors import LeafDtypeError, LeafShapeError

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation
    from verification.transcript.reader import LeafReader

__all__ = [
    "leaf_parts",
    "leaf_hash",
    "leaf_hashes_of",
    "leaf_hashes",
    "transcript_root",
    "commit_leaves",
    "dataset_tree",
    "dataset_root",
]


def _tensor_slot(c: DeclaredComputation,
                 index: int) -> tuple[int, tuple[int, ...], torch.dtype, str]:
    """``(tag, shape, dtype, slot name)`` of a non-record leaf, from the ref block §6 layout."""
    n_s, n_w, M = c.n_s, c.n_w, c.M
    if n_s <= index < n_s + n_w:
        name = c.weight_names[index - n_s]
        return TAG_WEIGHT, tuple(c.weight_shapes[name]), c.weight_dtype, f"W_t[{name}]"
    if n_s + n_w <= index < n_s + n_w + M:
        spec = c.product(index - n_s - n_w + 1)
        return TAG_PRODUCT, spec.p_shape, c.product_dtype, f"P_{spec.m} ({spec.name})"
    if n_s + n_w + M <= index < c.n_leaves:
        name = c.weight_names[index - n_s - n_w - M]
        return TAG_WEIGHT, tuple(c.weight_shapes[name]), c.weight_dtype, f"W_t+1[{name}]"
    raise IndexError(f"leaf index {index} outside 0..{c.n_leaves - 1}")


def leaf_parts(c: DeclaredComputation, index: int, obj: Any) -> tuple[bytes | memoryview, ...]:
    """Canonical bytes of leaf ``index`` holding ``obj``, as parts to stream into ``hash_leaf``.

    A tensor payload is a zero-copy view of ``obj``, so ``obj`` must not change until hashed.
    """
    if 0 <= index < c.n_s:
        return (c.encode_record(obj),)
    tag, shape, dtype, slot = _tensor_slot(c, index)
    if not isinstance(obj, torch.Tensor):
        raise LeafShapeError(f"leaf {index} {slot}: expected a tensor, got {type(obj).__name__}")
    if tuple(obj.shape) != shape:
        raise LeafShapeError(f"leaf {index} {slot}: shape {tuple(obj.shape)}, declared {shape}")
    if obj.dtype != dtype:
        raise LeafDtypeError(f"leaf {index} {slot}: dtype {obj.dtype}, declared {dtype}")
    return tensor_leaf_header(tag, obj), tensor_leaf_payload(obj)


def leaf_hash(c: DeclaredComputation, index: int, obj: Any) -> bytes:
    """``H_leaf`` of leaf ``index`` holding ``obj`` (P9)."""
    return hash_leaf(*leaf_parts(c, index, obj))


def leaf_hashes_of(c: DeclaredComputation, items: Iterable[tuple[int, Any]]) -> list[bytes]:
    """:func:`leaf_hash` of each ``(index, obj)``, in order, hashed in parallel.

    ``items`` is consumed and each leaf validated on the calling thread, in order, so an error
    from either is the one a leaf-by-leaf loop raises first. The objects must not change until
    this returns.
    """
    return hash_leaves(leaf_parts(c, i, obj) for i, obj in items)


def leaf_hashes(c: DeclaredComputation, reader: LeafReader) -> list[bytes]:
    """Every leaf hash, recomputed from the reader's leaves over the declared ``n_leaves``."""
    return leaf_hashes_of(c, ((i, reader.leaf(i)) for i in range(c.n_leaves)))


def transcript_root(c: DeclaredComputation, reader: LeafReader) -> bytes:
    """Check 2's recomputation: the root over the reader's leaves, never over supplied hashes."""
    return merkle_root(leaf_hashes(c, reader))


def commit_leaves(c: DeclaredComputation, leaves: Sequence[Any]) -> MerkleTree:
    """The Merkle tree over a full transcript's leaves, in layout order; ``.root`` is ``h``."""
    if len(leaves) != c.n_leaves:
        raise ValueError(f"transcript has {len(leaves)} leaves, the computation declares "
                         f"{c.n_leaves}")
    return MerkleTree(leaf_hashes_of(c, enumerate(leaves)))


def dataset_tree(c: DeclaredComputation, dataset: Sequence[Any]) -> MerkleTree:
    """The ``h_D`` tree over ``D``'s record leaves, for the batch's dataset paths (P9b)."""
    return MerkleTree([hash_leaf(c.encode_record(r)) for r in dataset])


def dataset_root(records: Sequence[Record]) -> bytes:
    """`h_D`: the Merkle root over the record leaves in record order (P9b)."""
    return merkle_root([hash_record_leaf(r) for r in records])
