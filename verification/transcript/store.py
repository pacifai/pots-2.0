"""The step transcript store: what the verifier may hold of one step (spec §4, S3, S6f).

Two layers, kept apart because they sit on different sides of the S3 boundary:

- **The verifier's interface.** :class:`TranscriptStore` is everything the verifier may hold of
  a step (S3, invariant 1): the leaf objects, the claimed root, and the claimed audit paths into
  ``h`` and ``h_D``. All of it is data under test. The store hands out no leaf hashes and no leaf
  count (invariant 7); the verifier hashes leaves itself (``commitment/leaves.py``) and takes
  ``n_leaves`` from ``C``.
- **The prover and harness side.** :class:`InMemoryStore` is the test-scale handoff, and
  :func:`perturb_leaf` is the S6f single-leaf re-root. The verifier never calls these.

Handoff without copying (§8.A.8). :meth:`InMemoryStore.from_step` keeps the step's own tensors,
not clones: §8.A.8's one-process peak of about 4 GB assumes it, and a clone would add the whole
transcript again (about 2.62 GB at test scale, 1.55 GB of it products). The store keeps no
reference to the ``StepOutput`` or the prover's model. It records every leaf tensor's
``_version`` at handoff and checks it on each read, so an in-place write through any alias, the
prover's or the verifier's, raises :class:`StoreMutationError` rather than serving changed bytes
(invariant 6). Writes that bypass the version counter (``.data``, a NumPy view) are not caught
there; check 2 still sees any change made before it rehashes. ``copy=True`` clones instead, for
full isolation at that memory cost.

A ``DiskStore`` (A14) implements the same :class:`TranscriptStore`: it reads leaves from its
per-step directory and returns the stored root and paths, which is all the interface asks for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from setup.records import Record
from verification.commitment.leaves import commit_leaves, leaf_hash
from verification.commitment.merkle import MerkleTree
from verification.transcript.errors import StoreMutationError
from verification.transcript.reader import LeafReader

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation
    from verification.prover.step import StepOutput

__all__ = ["TranscriptStore", "InMemoryStore", "perturb_leaf"]


class TranscriptStore(LeafReader, ABC):
    """One step's transcript as the verifier sees it (S3, invariant 1).

    Everything here comes from the prover and is under test: the verifier recomputes ``h`` from
    :meth:`leaf` (check 2) before trusting :attr:`root`, and checks any path with
    ``verify_path`` and its own ``n_leaves``. Leaves must not be mutated.

    Reading or hashing prover data can raise:

    - :class:`TranscriptFormatError` and its subclasses :class:`LeafShapeError`,
      :class:`LeafDtypeError` (from :func:`leaf_hash`) and :class:`StoreMutationError` (from a
      read of a leaf changed since handoff);
    - ``EncodingError`` and ``NonFiniteError`` from ``commitment.encoding`` (an unencodable
      leaf, NaN or Inf, S9d), and whatever ``c.encode_record`` raises on a malformed record
      (a ``ValueError``, such as ``setup.records.RecordError``);
    - ``IndexError`` (a leaf or record index the store doesn't hold) and ``LookupError`` (no
      dataset paths supplied).

    The rule for A5: each of these is a rejection at the check that read the leaf, reported as
    ``(step, check_id, detail)``, never a crash.
    """

    @abstractmethod
    def leaf(self, index: int) -> Any:
        """Leaf ``index`` (0-based, ref block §6 order): a record object or a tensor."""

    @property
    @abstractmethod
    def root(self) -> bytes:
        """The claimed step commitment ``h``."""

    @abstractmethod
    def path(self, index: int) -> list[bytes]:
        """The claimed audit path of leaf ``index`` into :attr:`root`, nearest sibling first."""

    @abstractmethod
    def dataset_path(self, i: int) -> list[bytes]:
        """The claimed audit path of batch record ``i`` into ``h_D`` (spec §6, prover step 3).

        Check 4 verifies it at index ``π(t)_i`` with ``n_leaves = |D|`` from the manifest.
        """


# ---- prover and harness side --------------------------------------------------------------


def _tensors(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, Record):
        return [obj.ids, obj.targets, obj.mask]
    raise TypeError(f"unsupported leaf object {type(obj).__name__}")


def _versions(obj: Any) -> tuple[int, ...]:
    return tuple(t._version for t in _tensors(obj))


def _clone(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, Record):
        return Record(obj.ids.clone(), obj.targets.clone(), obj.mask.clone())
    raise TypeError(f"unsupported leaf object {type(obj).__name__}")


def _detached(leaves: Sequence[Any]) -> list[Any]:
    return [x.detach() if isinstance(x, torch.Tensor) else x for x in leaves]


class InMemoryStore(TranscriptStore):
    """Test-scale handoff of one committed step (S3, §8.A.8). Build it with :meth:`from_step`."""

    def __init__(self, leaves: list[Any], tree: MerkleTree,
                 dataset_paths: Sequence[Sequence[bytes]] | None) -> None:
        self._leaves = leaves
        self._tree = tree
        self._versions = [_versions(x) for x in leaves]
        self._dataset_paths = (None if dataset_paths is None
                               else tuple(tuple(bytes(h) for h in p) for p in dataset_paths))

    @classmethod
    def from_step(cls, c: DeclaredComputation, step: StepOutput, *,
                  dataset_paths: Sequence[Sequence[bytes]] | None = None,
                  copy: bool = False) -> InMemoryStore:
        """Commit ``step`` and hold its leaves; the ``StepOutput`` itself is not kept.

        ``copy=False`` keeps the step's tensors (zero-copy, version-guarded); ``copy=True``
        clones every leaf before hashing, so the store is isolated from the prover's tensors.
        ``dataset_paths[i]`` is record ``i``'s audit path into ``h_D``.
        """
        if dataset_paths is not None and len(dataset_paths) != c.n_s:
            raise ValueError(f"{len(dataset_paths)} dataset paths for {c.n_s} records")
        leaves = step.leaves()
        leaves = [_clone(x) for x in leaves] if copy else _detached(leaves)
        tree = commit_leaves(c, leaves)
        step.assert_unmodified()  # invariant 6, as in prover.step.commit()
        return cls(leaves, tree, dataset_paths)

    def leaf(self, index: int) -> Any:
        if not 0 <= index < len(self._leaves):
            raise IndexError(f"leaf index {index} out of range")
        obj = self._leaves[index]
        if _versions(obj) != self._versions[index]:
            raise StoreMutationError(f"leaf {index} was mutated in place after handoff")
        return obj.detach() if isinstance(obj, torch.Tensor) else obj

    @property
    def root(self) -> bytes:
        return self._tree.root

    def path(self, index: int) -> list[bytes]:
        return self._tree.path(index)

    def dataset_path(self, i: int) -> list[bytes]:
        if self._dataset_paths is None:
            raise LookupError("this store was built without dataset paths")
        return list(self._dataset_paths[i])


def perturb_leaf(c: DeclaredComputation, store: InMemoryStore, index: int, obj: Any) -> bytes:
    """S6f: replace leaf ``index`` with ``obj`` and re-root in O(log n); returns the new ``h``.

    Harness-side only. The store then holds ``obj`` as given (not cloned). To restore, perturb
    again with the original leaf, read beforehand. Every validation runs before the tree is
    touched, so a failure leaves the store unchanged.
    """
    if not 0 <= index < len(store._leaves):
        raise IndexError(f"leaf index {index} out of range")
    new_hash = leaf_hash(c, index, obj)
    if isinstance(obj, torch.Tensor):
        obj = obj.detach()
    versions = _versions(obj)
    root = store._tree.update_leaf(index, new_hash)
    store._leaves[index] = obj
    store._versions[index] = versions
    return root
