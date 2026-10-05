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

**On disk (A14, C3).** :class:`DiskStore` implements the same :class:`TranscriptStore` from a
per-step directory, S3's storage layout: one ``torch.save`` file per leaf, the claimed root and
the ``h_D`` paths in ``meta.json``, and the tree's leaf hashes, from which
:meth:`DiskStore.path` rebuilds an audit path. This storage is non-cryptographic. The commitment
is still computed over the canonical encoding (S9), and the verifier rehashes what it reads.
Every :meth:`DiskStore.leaf` call reads its file again and returns new tensors that share memory
with nothing: no cache and no ``mmap``. So the only copy of a leaf the verifier keeps is the one
check 2 hashed (``CommittedLeaves``). A file changed between two reads shows up as a hash
mismatch at check 2, as a changed in-memory leaf would.

**The hand-off.** A :class:`StoreHandoff` is how the run loop turns a committed step into the
store it gives the verifier (``P5.write``), and how it discards that store afterwards.
:data:`IN_MEMORY` holds references, as before. :class:`DiskHandoff` writes the step under a
directory and deletes it once the step is verified. C3 (``runs/store_crosscheck.py``) runs the
same scenario through both and asserts the verifier decides identically.
"""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from setup.records import Record
from verification.commitment.leaves import commit_leaves, leaf_hash
from verification.commitment.merkle import DIGEST_SIZE, MerkleTree
from verification.transcript.errors import LeafReadError, StoreMutationError
from verification.transcript.reader import LeafReader

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation
    from verification.prover.step import StepOutput

__all__ = ["TranscriptStore", "InMemoryStore", "perturb_leaf", "DiskStore", "StoreHandoff",
           "InMemoryHandoff", "IN_MEMORY", "DiskHandoff", "LEAF_FILE", "META_FILE", "HASHES_FILE",
           "DISK_FORMAT"]


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

        It is :meth:`commit_step` then :meth:`hold`; the run loop calls the two apart, so the
        metrics see hashing (P3) and the hand-off to the store (P5) as separate rows.
        """
        if dataset_paths is not None and len(dataset_paths) != c.n_s:
            raise ValueError(f"{len(dataset_paths)} dataset paths for {c.n_s} records")
        leaves, tree = cls.commit_step(c, step, copy=copy)
        return cls.hold(c, leaves, tree, dataset_paths)

    @staticmethod
    def commit_step(c: DeclaredComputation, step: StepOutput, *,
                    copy: bool = False) -> tuple[list[Any], MerkleTree]:
        """``(leaves, tree)``: the step's leaves (cloned if ``copy``) and the Merkle tree over
        them, with invariant 6 rechecked after hashing."""
        leaves = step.leaves()
        leaves = [_clone(x) for x in leaves] if copy else _detached(leaves)
        tree = commit_leaves(c, leaves)
        step.assert_unmodified()  # invariant 6, as in prover.step.commit()
        return leaves, tree

    @classmethod
    def hold(cls, c: DeclaredComputation, leaves: list[Any], tree: MerkleTree,
             dataset_paths: Sequence[Sequence[bytes]] | None = None) -> InMemoryStore:
        """The store over leaves :meth:`commit_step` committed. At test scale it keeps
        references; A14's ``DiskStore`` writes the leaves here."""
        if dataset_paths is not None and len(dataset_paths) != c.n_s:
            raise ValueError(f"{len(dataset_paths)} dataset paths for {c.n_s} records")
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


# ---- on disk (A14, C3) --------------------------------------------------------------------

LEAF_FILE = "leaf_{:05d}.pt"
META_FILE = "meta.json"
HASHES_FILE = "leaf_hashes.bin"
DISK_FORMAT = 1
_RECORD_KEYS = ("ids", "mask", "targets")


def _owned(t: torch.Tensor) -> torch.Tensor:
    """``t`` itself when it owns its whole storage, else a copy that does. ``torch.save``
    writes a tensor's whole storage, so a view (an attention member of a ``bmm`` output, a
    record sliced out of ``D``) would write its base with it."""
    t = t.detach()
    if t.storage_offset() == 0 and t.untyped_storage().nbytes() == t.numel() * t.element_size():
        return t
    return t.clone()


def _to_disk(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return _owned(obj)
    if isinstance(obj, Record):
        return {k: _owned(getattr(obj, k)) for k in _RECORD_KEYS}
    raise TypeError(f"unsupported leaf object {type(obj).__name__}")


def _from_disk(index: int, obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict) and tuple(sorted(obj)) == _RECORD_KEYS:
        return Record(**obj)  # a malformed record raises RecordError, a ValueError
    raise LeafReadError(f"leaf {index}: the stored object is a {type(obj).__name__}, not a leaf")


def _digests(hexes: Any, what: str) -> list[bytes]:
    if not isinstance(hexes, list):
        raise LeafReadError(f"{what} is not a list")
    try:
        return [bytes.fromhex(h) for h in hexes]
    except (TypeError, ValueError) as e:
        raise LeafReadError(f"{what} is not a list of hex digests") from e


class DiskStore(TranscriptStore):
    """One committed step read back from its directory (S3's on-disk mode, C3).

    Build it with :meth:`write`, or open a written directory with ``DiskStore(directory)``.
    Opening reads nothing. :meth:`leaf`, :attr:`root` and the paths read their files on every
    call, so a missing or corrupt file is a prover-data error at the check that reads it
    (:class:`LeafReadError`, or ``IndexError`` for a leaf file that isn't there), never a crash.
    Leaves load with ``weights_only=True``, which unpickles only tensors and plain containers.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @classmethod
    def write(cls, directory: str | Path, c: DeclaredComputation, leaves: Sequence[Any],
              tree: MerkleTree, dataset_paths: Sequence[Sequence[bytes]] | None = None
              ) -> DiskStore:
        """Write the committed ``leaves``, their ``tree`` (both from ``commit_step``) and the
        records' ``h_D`` paths into ``directory``, replacing what is there, and open it.

        ``meta.json`` is written last, so a directory without it is an incomplete write. A
        failed write deletes the directory."""
        if dataset_paths is not None and len(dataset_paths) != c.n_s:
            raise ValueError(f"{len(dataset_paths)} dataset paths for {c.n_s} records")
        if len(leaves) != tree.n_leaves:
            raise ValueError(f"{len(leaves)} leaves for a tree of {tree.n_leaves}")
        d = Path(directory)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        try:
            for i, obj in enumerate(leaves):
                torch.save(_to_disk(obj), d / LEAF_FILE.format(i))
            (d / HASHES_FILE).write_bytes(b"".join(tree.leaf(i) for i in range(tree.n_leaves)))
            meta = {"format": DISK_FORMAT, "root": tree.root.hex(),
                    "dataset_paths": (None if dataset_paths is None
                                      else [[bytes(h).hex() for h in p] for p in dataset_paths])}
            (d / META_FILE).write_text(json.dumps(meta))
        except BaseException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        return cls(d)

    def _meta(self) -> dict[str, Any]:
        try:
            meta = json.loads((self.directory / META_FILE).read_text())
        except (OSError, ValueError) as e:
            raise LeafReadError(f"store metadata unreadable: {type(e).__name__}: {e}") from e
        if not isinstance(meta, dict) or meta.get("format") != DISK_FORMAT:
            raise LeafReadError("store metadata has an unknown format")
        return meta

    def leaf(self, index: int) -> Any:
        if not isinstance(index, int) or index < 0:
            raise IndexError(f"leaf index {index!r} out of range")
        path = self.directory / LEAF_FILE.format(index)
        if not path.is_file():
            raise IndexError(f"leaf index {index} out of range (no file {path.name})")
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # a file that won't load is bad prover data, not a verifier bug
            raise LeafReadError(f"leaf {index} unreadable: {type(e).__name__}: {e}") from e
        return _from_disk(index, obj)

    @property
    def root(self) -> bytes:
        (root,) = _digests([self._meta().get("root")], "the root")
        return root

    def path(self, index: int) -> list[bytes]:
        try:
            raw = (self.directory / HASHES_FILE).read_bytes()
        except OSError as e:
            raise LeafReadError(f"leaf hashes unreadable: {e}") from e
        if not raw or len(raw) % DIGEST_SIZE:
            raise LeafReadError(f"a leaf-hash file of {len(raw)} bytes")
        hashes = [raw[i:i + DIGEST_SIZE] for i in range(0, len(raw), DIGEST_SIZE)]
        if not 0 <= index < len(hashes):
            raise IndexError(f"leaf index {index} out of range")
        return MerkleTree(hashes).path(index)

    def dataset_path(self, i: int) -> list[bytes]:
        paths = self._meta().get("dataset_paths")
        if paths is None:
            raise LookupError("this store was written without dataset paths")
        if not isinstance(paths, list):
            raise LeafReadError("the dataset paths are not a list")
        return _digests(paths[i], f"dataset path {i}")

    def remove(self) -> None:
        """Delete the step's directory."""
        shutil.rmtree(self.directory, ignore_errors=True)


# ---- the hand-off: how the loop gives a committed step to the verifier --------------------


class StoreHandoff(ABC):
    """How the run loop hands step ``t``'s committed leaves to the verifier (S3, ``P5.write``),
    and how it discards the store once the step is verified. Harness side only."""

    @abstractmethod
    def hold(self, c: DeclaredComputation, t: int, leaves: list[Any], tree: MerkleTree,
             dataset_paths: Sequence[Sequence[bytes]] | None = None) -> TranscriptStore:
        """The store over the leaves ``commit_step`` committed."""

    def release(self, store: TranscriptStore) -> None:  # noqa: B027 (in memory: nothing to do)
        """Discard ``store``. The loop calls it once the step is verified (S3)."""


class InMemoryHandoff(StoreHandoff):
    """The test-scale default: :meth:`InMemoryStore.hold`, which keeps references only."""

    def hold(self, c: DeclaredComputation, t: int, leaves: list[Any], tree: MerkleTree,
             dataset_paths: Sequence[Sequence[bytes]] | None = None) -> InMemoryStore:
        return InMemoryStore.hold(c, leaves, tree, dataset_paths)


IN_MEMORY = InMemoryHandoff()


class DiskHandoff(StoreHandoff):
    """Write each step to ``directory/step_<t>/`` and serve it as a :class:`DiskStore` (C3).

    :meth:`release` deletes the step's directory unless ``keep``. So a run holds one step on
    disk at a time, as the in-memory loop holds one step in memory."""

    def __init__(self, directory: str | Path, *, keep: bool = False) -> None:
        self.directory = Path(directory)
        self.keep = keep

    def hold(self, c: DeclaredComputation, t: int, leaves: list[Any], tree: MerkleTree,
             dataset_paths: Sequence[Sequence[bytes]] | None = None) -> DiskStore:
        return DiskStore.write(self.directory / f"step_{t}", c, leaves, tree, dataset_paths)

    def release(self, store: TranscriptStore) -> None:
        if not self.keep and isinstance(store, DiskStore):
            store.remove()
