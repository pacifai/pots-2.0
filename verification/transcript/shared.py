"""The cross-process store: one committed step from the prover's process to the verifier's (FS33a).

Harness side, like :class:`~verification.transcript.store.InMemoryStore`; the verifier never
imports this module. The pipelined loop (F22, FS33's process model) trains in a child process
and verifies in the parent, so step ``t``'s transcript must cross while the child trains
``t + 1``. This moves it without copying device memory and without pickling 394k tensors one
by one (a Llama-3.2-1B step).

The flow, one call each side::

    package, hold = share_step(c, t, leaves, tree, dataset_paths)   # producer, after the commit
    conn.send(package)                                              # any multiprocessing pipe
    store = SharedStore.open(c, conn.recv())                        # consumer: a TranscriptStore
    ...                                                             # verifier.verify_step(t, store)
    store.close()                                                   # consumer drops every view
    hold.release()                                                  # producer, after the consumer

**One transfer per storage, a table per leaf.** The step's tensor leaves are grouped by the
storage they view (``W_{t+1}`` leaves of one step are often the next step's ``W_t`` leaves, and
several leaves can view one storage). Each distinct storage crosses once. A leaf is described by
(storage index, storage offset, dtype, shape, stride); the consumer rebuilds it with ``set_`` over
the received storage, so bytes, dtype, shape and strides are the producer's exactly. The table is
a few flat ``array`` columns, which pickle as bytes. The consumer builds each leaf tensor when
``leaf`` first reads it (the version it checks afterwards is that tensor's), so ``open`` costs
the table, not 394k ``set_`` calls.

**The transport, by device.** This is the only CPU/GPU difference; both go through the same
functions and the same table.

- *CUDA storages* cross through PyTorch's CUDA inter-process handles: ``torch.multiprocessing``
  pickles a CUDA *tensor* (not a storage) as a handle on the whole ``cudaMalloc`` block plus an
  offset, so the package holds, per distinct storage, one ``uint8`` tensor over the whole
  storage. Nothing is copied: the consumer reads the producer's memory. PyTorch's rules:
  the producer must keep the memory alive while the consumer uses it, and the consumer must free
  its views for the producer to reuse it (``torch.cuda.ipc_collect`` then reclaims the
  reference-counting files). Here the *hold* keeps the leaves, and so their storages, alive
  until :meth:`StepHold.release`; the package's tensors are the objects that get pickled and
  :meth:`StepHold.release` empties that list too, so the caller need not delete the package for
  the memory to go. The consumer's :meth:`SharedStore.close` drops its tensors (and the
  storages they view). ``release`` then calls ``ipc_collect``. The producer must call
  ``release`` only after the consumer's ``close``, and drop its own references to ``leaves``
  and the step first for the memory to be freed. Device indices are the same in both processes
  (same ``CUDA_VISIBLE_DEVICES``).
- *CPU storages* are copied once into **one** shared-memory buffer per step (each storage's whole
  bytes, 64-byte aligned), which the consumer maps; with Linux's default sharing strategy a
  storage per descriptor would mean thousands of descriptors per step. The copy is test scale's
  only extra cost. On the consumer side every CPU leaf views that one buffer's storage.
- *Records* (three small int32 CPU tensors) go **by value** inside the package (raw bytes), not
  in the CPU buffer, so they never become shared-memory objects.
- *The tree* is the producer's :class:`MerkleTree`, pickled whole (its node buffer is one
  ``bytearray``), with the dataset paths, so ``root``, ``path`` and ``dataset_path`` answer as
  ``InMemoryStore``'s do.

**Invariant 6 on both sides, each on its own tensors** (the parent cannot see the child's
writes through ``_version``). The consumer's :meth:`SharedStore.leaf` checks each rebuilt
tensor's ``_version`` as ``InMemoryStore.leaf`` does and raises :class:`StoreMutationError`. The
producer's :class:`StepHold` records every leaf tensor's ``_version`` in ``share_step`` and
:meth:`StepHold.release` checks them, raising :class:`StoreMutationError` naming the first changed
leaf, after dropping every reference it holds. On CUDA the consumer reads the producer's memory,
so a producer write during verification is seen only there: the caller (FS33b) must treat a
failed ``release`` as a run error and publish no verdict for that step. ``close`` and ``release``
are idempotent (a second ``release`` after a failed one does nothing).

**Two steps alive.** Step ``t + 1`` may be shared before step ``t`` is released, and one storage
can sit in both packages. Each package and hold owns its own references; on the consumer PyTorch
opens one handle per block and each store holds its own views, so releasing either step leaves
the other readable.

**A process note for FS33b.** On macOS, a child that shared CPU storages (``torch_shm_manager``
starts on the first one) can leave ``Process.join()`` without a timeout blocked after the child
has exited; poll ``is_alive()`` or pass a timeout (``tests/.../shared_helpers.reap``).

**Refusals.** A leaf that is neither a tensor nor a ``Record``, a tensor on a device other than
CPU or CUDA, and a sparse, non-strided, quantized, conjugate or negated tensor raise
:class:`TypeError` in ``share_step``.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.multiprocessing  # noqa: F401  registers the tensor reducers on the pickler

from setup.records import Record
from verification.commitment.merkle import MerkleTree
from verification.transcript.errors import StoreMutationError
from verification.transcript.store import TranscriptStore, _versions

if TYPE_CHECKING:
    from verification.computation.interface import DeclaredComputation

__all__ = ["StepPackage", "StepHold", "SharedStore", "share_step"]

_ALIGN = 64
_TENSOR, _EMPTY, _RECORD = 0, 1, 2  # leaf kinds in the table


@dataclass
class StepPackage:
    """What crosses the pipe for one step. Build it with :func:`share_step`.

    ``kind[i]`` says what leaf ``i`` is: a tensor over storage ``store[i]``, an empty tensor
    with no storage, or record ``store[i]``. ``dims`` and ``strides`` hold every leaf's shape and
    stride back to back, ``ndim[i]`` long each. ``s_cuda[j]`` and ``s_ref[j]`` place distinct
    storage ``j``: an index into :attr:`cuda`, or a byte offset into :attr:`cpu`.
    """

    t: int
    kind: array
    store: array
    offset: array
    dtype: array  # index into dtypes
    device: array  # index into devices
    ndim: array
    dims: array
    strides: array
    dtypes: list[torch.dtype]
    devices: list[torch.device]
    s_cuda: array
    s_ref: array
    cuda: list[torch.Tensor]  # one uint8 tensor over each CUDA storage (crosses by handle)
    cpu: torch.Tensor | None  # the step's one shared-memory buffer, uint8
    records: list[tuple[bytes, bytes, bytes]]
    tree: MerkleTree
    dataset_paths: tuple[tuple[bytes, ...], ...] | None
    n_leaves: int = 0
    n_storages: int = 0
    cpu_bytes: int = 0
    stats: dict = field(default_factory=dict)


def _refuse(i: int, t: torch.Tensor) -> None:
    if t.layout != torch.strided:
        raise TypeError(f"leaf {i}: a {t.layout} tensor cannot be shared")
    if t.device.type not in ("cpu", "cuda"):
        raise TypeError(f"leaf {i}: a tensor on device {t.device.type} cannot be shared")
    if t.is_quantized or t.is_conj() or t.is_neg():
        raise TypeError(f"leaf {i}: a quantized, conjugate or negated tensor cannot be shared")


class StepHold:
    """The producer's half of one shared step: keeps the leaves (and so the storages) alive,
    and checks invariant 6 on :meth:`release`. See the module docstring."""

    def __init__(self, t: int, leaves: list[Any], package: StepPackage) -> None:
        self.t = t
        self._leaves: list[Any] | None = leaves
        self._versions = [_versions(x) for x in leaves]
        self._package: StepPackage | None = package
        self._cuda = bool(package.cuda)

    @property
    def released(self) -> bool:
        return self._leaves is None

    def release(self) -> None:
        """Check every leaf's ``_version``, drop every reference, reclaim IPC memory.

        Raises :class:`StoreMutationError` naming the first changed leaf, after dropping the
        references. Call it after the consumer's ``close``. Idempotent."""
        leaves, versions, pkg = self._leaves, self._versions, self._package
        if leaves is None:
            return
        self._leaves, self._versions, self._package = None, [], None
        bad = next((i for i, x in enumerate(leaves) if _versions(x) != versions[i]), None)
        if pkg is not None:
            pkg.cuda.clear()
            pkg.cpu = None
        del leaves, pkg
        if self._cuda:
            torch.cuda.ipc_collect()
        if bad is not None:
            raise StoreMutationError(f"leaf {bad} was mutated in place while shared (step "
                                     f"{self.t})")


def share_step(c: DeclaredComputation, t: int, leaves: Sequence[Any], tree: MerkleTree,
               dataset_paths: Sequence[Sequence[bytes]] | None = None
               ) -> tuple[StepPackage, StepHold]:
    """``(package, hold)`` for step ``t``'s committed leaves (``InMemoryStore.commit_step``).

    ``package`` is what to send; ``hold`` is the producer's, to release once the consumer has
    closed its store. Raises :class:`TypeError` on a leaf that cannot be shared and
    ``ValueError`` on a count that does not match the tree or ``c.n_s``.
    """
    leaves = list(leaves)
    if tree.n_leaves != len(leaves):
        raise ValueError(f"{len(leaves)} leaves for a tree of {tree.n_leaves}")
    if dataset_paths is not None and len(dataset_paths) != c.n_s:
        raise ValueError(f"{len(dataset_paths)} dataset paths for {c.n_s} records")
    kind, store, offset, dcode, vcode = (array("b"), array("q"), array("q"), array("b"),
                                         array("b"))
    ndim, dims, strides = array("b"), array("q"), array("q")
    dtypes: list[torch.dtype] = []
    devices: list[torch.device] = []
    storages: dict[tuple[torch.device, int], int] = {}
    s_list: list[torch.UntypedStorage] = []

    def code(table: list, x: Any) -> int:
        if x not in table:
            table.append(x)
        return table.index(x)

    records: list[tuple[bytes, bytes, bytes]] = []
    for i, x in enumerate(leaves):
        if isinstance(x, Record):
            for q in (x.ids, x.targets, x.mask):
                _refuse(i, q)
            records.append(tuple(q.detach().cpu().contiguous().numpy().tobytes()
                                 for q in (x.ids, x.targets, x.mask)))
            kind.append(_RECORD); store.append(len(records) - 1); offset.append(0)  # noqa: E702
            dcode.append(0); vcode.append(0); ndim.append(0)  # noqa: E702
            continue
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"leaf {i}: unsupported leaf object {type(x).__name__}")
        _refuse(i, x)
        dcode.append(code(dtypes, x.dtype)); vcode.append(code(devices, x.device))  # noqa: E702
        ndim.append(x.dim())
        dims.extend(x.shape); strides.extend(x.stride())  # noqa: E702
        st = x.untyped_storage()
        if x.numel() == 0 or st.nbytes() == 0:
            kind.append(_EMPTY); store.append(0); offset.append(0)  # noqa: E702
            continue
        key = (x.device, st.data_ptr())
        if key not in storages:
            storages[key] = len(s_list)
            s_list.append(st)
        kind.append(_TENSOR)
        store.append(storages[key])
        offset.append(x.storage_offset())
    s_cuda, s_ref = array("b"), array("q")
    cuda: list[torch.Tensor] = []
    cpu_at, total = [], 0
    for st in s_list:
        if st.device.type == "cpu":
            cpu_at.append(total)
            total += -(-st.nbytes() // _ALIGN) * _ALIGN
    buf = None
    if cpu_at:
        shared = torch.UntypedStorage._new_shared(total)  # noqa: SLF001
        buf = torch.empty(0, dtype=torch.uint8).set_(shared, 0, (total,))
    cpu_iter = iter(cpu_at)
    for st in s_list:
        n = st.nbytes()
        view = torch.empty(0, dtype=torch.uint8, device=st.device).set_(st, 0, (n,))
        if st.device.type == "cpu":
            at = next(cpu_iter)
            buf[at:at + n].copy_(view)
            s_cuda.append(0); s_ref.append(at)  # noqa: E702
        else:
            s_cuda.append(1); s_ref.append(len(cuda))  # noqa: E702
            cuda.append(view)
    paths = (None if dataset_paths is None
             else tuple(tuple(bytes(h) for h in p) for p in dataset_paths))
    pkg = StepPackage(t, kind, store, offset, dcode, vcode, ndim, dims, strides, dtypes, devices,
                      s_cuda, s_ref, cuda, buf, records, tree, paths, n_leaves=len(leaves),
                      n_storages=len(s_list), cpu_bytes=total,
                      stats={"cuda_storages": len(cuda), "cpu_storages": len(cpu_at)})
    return pkg, StepHold(t, leaves, pkg)


class SharedStore(TranscriptStore):
    """The consumer's :class:`TranscriptStore` over a received :class:`StepPackage`.

    Build it with :meth:`open`. Leaves are views over the producer's storages (CUDA) or over the
    one mapped buffer (CPU), built on first read; each read afterwards checks ``_version``.
    """

    def __init__(self, pkg: StepPackage) -> None:
        self._pkg: StepPackage | None = pkg
        n = pkg.n_leaves
        self._n = n
        self._built: list[Any] = [None] * n
        self._versions: list[tuple[int, ...] | None] = [None] * n
        # where leaf i's shape and stride start in dims and strides
        starts, s = [], 0
        for k in pkg.ndim:
            starts.append(s)
            s += k
        self._starts = starts
        self._tree = pkg.tree
        self._dataset_paths = pkg.dataset_paths
        self._storages: dict[int, torch.UntypedStorage] = {}
        self._buf_storage = pkg.cpu.untyped_storage() if pkg.cpu is not None else None
        self._cuda = list(pkg.cuda)  # keeps the received CUDA tensors, hence the handles

    @classmethod
    def open(cls, c: DeclaredComputation, package: StepPackage) -> SharedStore:
        """The store over ``package`` as received from the producer."""
        if package.dataset_paths is not None and len(package.dataset_paths) != c.n_s:
            raise ValueError(f"{len(package.dataset_paths)} dataset paths for {c.n_s} records")
        return cls(package)

    def _storage(self, j: int) -> tuple[torch.UntypedStorage, int]:
        """``(storage, byte offset of storage j in it)``."""
        pkg = self._pkg
        if pkg.s_cuda[j]:
            return self._cuda[pkg.s_ref[j]].untyped_storage(), 0
        return self._buf_storage, pkg.s_ref[j]

    def _build(self, i: int) -> Any:
        pkg = self._pkg
        k = pkg.kind[i]
        if k == _RECORD:
            ids, tg, mk = (torch.frombuffer(bytearray(b), dtype=torch.int32) if b
                           else torch.empty(0, dtype=torch.int32)
                           for b in pkg.records[pkg.store[i]])
            return Record(ids, tg, mk)
        nd, s0 = pkg.ndim[i], self._starts[i]
        shape = tuple(pkg.dims[s0:s0 + nd])
        stride = tuple(pkg.strides[s0:s0 + nd])
        dt, dev = pkg.dtypes[pkg.dtype[i]], pkg.devices[pkg.device[i]]
        if k == _EMPTY:
            return torch.empty_strided(shape, stride, dtype=dt, device=dev)
        st, base = self._storage(pkg.store[i])
        isz = torch.empty(0, dtype=dt).element_size()
        return torch.empty(0, dtype=dt, device=dev).set_(
            st, (base + pkg.offset[i] * isz) // isz, shape, stride)

    def leaf(self, index: int) -> Any:
        if self._pkg is None:
            raise RuntimeError("the store is closed")
        if not 0 <= index < self._n:
            raise IndexError(f"leaf index {index} out of range")
        obj = self._built[index]
        if obj is None:
            obj = self._built[index] = self._build(index)
            self._versions[index] = _versions(obj)
        elif _versions(obj) != self._versions[index]:
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

    def close(self) -> None:
        """Drop every view and the received storages; later ``leaf`` calls raise. Idempotent.
        Call it before the producer's :meth:`StepHold.release`."""
        self._pkg = None
        self._built, self._storages, self._buf_storage, self._cuda = [], {}, None, []
        self._n = 0
