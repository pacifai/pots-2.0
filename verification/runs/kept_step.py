"""The hand-off that keeps one step's transcript on disk (EV2), in a module no one runs.

:class:`KeptStepHandoff` lives here, not in ``run_verified``, because ``run_verified`` is an
entry point (``python -m verification.runs.run_verified``): run that way its module is
``__main__``, and a class defined in it is a different class from the one ``import
verification.runs.run_verified`` gives, so ``pipeline.run_pipelined``'s ``isinstance`` test (and
a pickle sent to the prover's child) would not see it. ``run_verified`` re-exports the name.
The rule for this package: a class that another module checks or pickles is defined in a module
that is never an entry point.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verification.commitment.leaves import transcript_root
from verification.commitment.merkle import MerkleTree
from verification.computation.interface import DeclaredComputation
from verification.transcript.store import (
    IN_MEMORY,
    DiskHandoff,
    DiskStore,
    StoreHandoff,
    TranscriptStore,
)

__all__ = ["KeptStepHandoff"]


class KeptStepHandoff(StoreHandoff):
    """The in-memory hand-off, which also keeps one step's transcript on disk (EV2).

    Every step is held in memory and verified from there, as in every other run. :meth:`hold`
    keeps a reference to step ``step``'s committed leaves and tree; :meth:`write`, called once
    the step has closed, writes them to ``directory/step_<t>/`` with ``DiskHandoff(keep=True)``
    and checks that the root recomputed from the written files is the step's ``h``. The write
    falls outside every timed section and the loop's step clock, so the stored step costs what
    an in-memory step costs."""

    def __init__(self, directory: str | Path, step: int) -> None:
        self.disk = DiskHandoff(directory, keep=True)
        self.step = step
        self._held: tuple[DeclaredComputation, list[Any], MerkleTree,
                          Sequence[Sequence[bytes]] | None] | None = None
        self.kept: DiskStore | None = None

    @property
    def directory(self) -> Path:
        """The stored step's directory, once written or not."""
        return self.disk.directory / f"step_{self.step}"

    def hold(self, c: DeclaredComputation, t: int, leaves: list[Any], tree: MerkleTree,
             dataset_paths: Sequence[Sequence[bytes]] | None = None) -> TranscriptStore:
        if t == self.step:
            self._held = (c, leaves, tree, dataset_paths)
        return IN_MEMORY.hold(c, t, leaves, tree, dataset_paths)

    def release(self, store: TranscriptStore) -> None:
        IN_MEMORY.release(store)

    def write(self) -> DiskStore | None:
        """Write the held step, or return ``None`` when the run never reached it. Raises
        ``RuntimeError`` if the files' root isn't the step's ``h``."""
        if self._held is None:
            return self.kept  # written already by a pipelined run, or never reached
        c, leaves, tree, paths = self._held
        self._held = None
        return self.write_from(c, leaves, tree, paths)

    def write_from(self, c: DeclaredComputation, leaves: list[Any], tree: MerkleTree,
                   paths: Sequence[Sequence[bytes]] | None) -> DiskStore:
        """Write step ``step`` from its leaves, tree and dataset paths, and check the root.
        The pipelined loop calls it with the parent's opened store's leaves."""
        store = self.disk.hold(c, self.step, leaves, tree, paths)
        root = transcript_root(c, store)
        if root != tree.root or store.root != tree.root:
            raise RuntimeError(f"stored step {self.step}: root over the written files "
                               f"{root.hex()} is not the step's h {tree.root.hex()}")
        self.kept = store
        return store
