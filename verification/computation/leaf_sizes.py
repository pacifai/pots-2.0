"""The leaves of one decoder step's transcript, with each one's slot, shape, dtype and size.

For sizing the step (the hashing benchmark under ``benchmarks/hashing/``, and the memory and
transcript-size estimates). Nothing here runs a model or allocates a leaf. The layout comes
from the two places that define it: :class:`DecoderComputation` declares the weights and the
products, and ``commitment.leaves`` maps each index to its slot. So the list matches the leaves
the prover commits, in the order it commits them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from setup.records import Record
from verification.commitment.encoding import tensor_leaf_header
from verification.commitment.leaves import _tensor_slot
from verification.computation.instances.decoder import DecoderComputation

__all__ = ["LeafSize", "leaf_sizes"]

# Sections of the layout (ref block §6), in leaf order.
SECTIONS = ("record", "w_t", "product", "w_next")


@dataclass(frozen=True)
class LeafSize:
    """One leaf of the step transcript.

    ``nbytes`` is the length of the leaf's canonical encoding (header and payload, what
    ``leaf_parts`` yields); ``H_leaf`` hashes one byte more, the ``0x00`` prefix. A record is
    sized at its longest, ``ℓ = n``: three int32 arrays of length ``n`` after a 6-byte header,
    so its ``shape`` is ``(3, n)``. ``role`` is a product's κ class (``"Y_q"``, ``"S"``,
    ``"Lambda"``, …) and ``None`` for records and weights.
    """

    index: int
    section: str
    name: str
    role: str | None
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


def leaf_sizes(config: Any, *, n_s: int, n: int, weight_dtype: torch.dtype | None = None,
               product_dtype: torch.dtype | None = None) -> list[LeafSize]:
    """Every leaf of one step for a Llama or Qwen2 ``config`` at batch ``n_s × n``, in leaf order.

    ``weight_dtype`` and ``product_dtype`` replace the declared dtypes (fp32 today), for sizing
    a mixed-precision step: full scale commits fp32 weights and bf16 products (F1a). Left as
    ``None``, each is the computation's own.
    """
    c = DecoderComputation(config, n_s=n_s, n=n, eta=1.0)
    w_dtype = weight_dtype or c.weight_dtype
    p_dtype = product_dtype or c.product_dtype
    zeros = torch.zeros(n, dtype=torch.int32)
    record_bytes = len(c.encode_record(Record(ids=zeros, targets=zeros, mask=zeros)))
    out = [LeafSize(i, "record", f"record {i}", None, (3, n), torch.int32, record_bytes)
           for i in range(n_s)]
    header_bytes: dict[tuple[int, int], int] = {}
    n_w, M = c.n_w, c.M
    for i in range(n_s, c.n_leaves):
        tag, shape, _, slot = _tensor_slot(c, i)
        role = None
        if i < n_s + n_w:
            section, dtype, name = "w_t", w_dtype, c.weight_names[i - n_s]
        elif i < n_s + n_w + M:
            spec = c.product(i - n_s - n_w + 1)
            section, dtype, name, role = "product", p_dtype, spec.name, c.product_class(spec)
        else:
            section, dtype, name = "w_next", w_dtype, c.weight_names[i - n_s - n_w - M]
        assert name in slot, (name, slot)
        key = (tag, len(shape))
        if key not in header_bytes:  # the header's length depends only on ndim
            header_bytes[key] = len(tensor_leaf_header(tag, torch.empty(shape, device="meta")))
        numel = 1
        for s in shape:
            numel *= s
        nbytes = header_bytes[key] + numel * dtype.itemsize
        out.append(LeafSize(i, section, name, role, shape, dtype, nbytes))
    return out
