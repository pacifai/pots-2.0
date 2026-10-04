"""Errors raised by prover data that doesn't fit the declared transcript format (S3).

The verifier maps each one to a rejection at the check that read the leaf, never a crash.
"""

from __future__ import annotations

__all__ = ["TranscriptFormatError", "LeafShapeError", "LeafDtypeError", "StoreMutationError"]


class TranscriptFormatError(ValueError):
    """Prover data that doesn't fit the declared transcript format."""


class LeafShapeError(TranscriptFormatError):
    """A weight or product leaf doesn't have the shape the computation declares for its slot."""


class LeafDtypeError(TranscriptFormatError):
    """A weight or product leaf doesn't have the dtype the computation declares for its slot."""


class StoreMutationError(TranscriptFormatError):
    """A stored leaf was mutated in place after handoff (invariant 6)."""
