"""The Python side of check 5's measuring core (decision O2): descriptors and packing.

``measure_core.h`` computes a product's row from strided views. This module mirrors its two
descriptor structs as ctypes ``Structure``s and packs torch tensors into a table of them, for
the host build now (``measure_core_host.c``) and the fused kernel later. The tensors are never
copied: a view is its base address, dtype and strides, so a transposed view or a slice of a
bigger tensor goes in as it is. Nothing calls this yet besides the tests.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

DT_FP32 = 1
DT_BF16 = 2
"""The leaf dtype codes of ``commitment/encoding.py``, as ``measure_core.h`` has them."""

M_GUARD = 1
M_BIASED = 2

ERR_OK, ERR_DTYPE, ERR_K, ERR_SHAPE, ERR_NOMEM = 0, 1, 2, 3, 4

_DTYPE_CODE = {torch.float32: DT_FP32, torch.bfloat16: DT_BF16}


class MView(ctypes.Structure):
    """``pots_mview_t``: a strided matrix, strides in elements."""

    _fields_ = [("addr", ctypes.c_uint64), ("rows", ctypes.c_int64), ("cols", ctypes.c_int64),
                ("row_stride", ctypes.c_int64), ("col_stride", ctypes.c_int64),
                ("dtype", ctypes.c_uint32), ("pad", ctypes.c_uint32)]


class MProd(ctypes.Structure):
    """``pots_mprod_t``: one product ``A·B`` (+ bias), its challenge matrix and flags."""

    _fields_ = [("a", MView), ("b", MView), ("p", MView), ("bias", MView),
                ("r_addr", ctypes.c_uint64), ("k", ctypes.c_uint32), ("flags", ctypes.c_uint32)]


MVIEW_FIELDS = tuple(f for f, _ in MView._fields_)
MPROD_FIELDS = tuple(f for f, _ in MProd._fields_)


@dataclass
class MeasureInput:
    """One product to measure: ``a`` ``[n, q]``, ``b`` ``[q, w]``, ``p`` ``[n, w]`` (bf16 or
    fp32, any strides), ``r`` fp32 contiguous ``[w, k]``, ``bias`` ``[w]`` or ``None``."""

    a: torch.Tensor
    b: torch.Tensor
    p: torch.Tensor
    r: torch.Tensor
    bias: torch.Tensor | None = None
    guard: bool = True


@dataclass
class Table:
    """The packed products. ``keepalive`` holds every tensor the descriptors point into, so the
    caller keeps this object alive for as long as any build reads ``array``."""

    array: ctypes.Array
    n: int
    k: int
    device: torch.device
    keepalive: list = field(repr=False)

    @property
    def ptr(self) -> int:
        return ctypes.addressof(self.array)


def _view(t: torch.Tensor, what: str) -> MView:
    if t.dtype not in _DTYPE_CODE:
        raise ValueError(f"{what} has dtype {t.dtype}, not float32 or bfloat16")
    if t.dim() == 1:
        rows, cols, rs, cs = 1, t.shape[0], 0, t.stride(0)
    else:
        rows, cols, rs, cs = t.shape[0], t.shape[1], t.stride(0), t.stride(1)
    return MView(t.data_ptr(), rows, cols, rs, cs, _DTYPE_CODE[t.dtype], 0)


def pack(products: Sequence[MeasureInput]) -> Table:
    """The :class:`Table` of ``products``; ``ValueError`` names the product and the reason."""
    if len(products) == 0:
        raise ValueError("the table is empty")
    arr = (MProd * len(products))()
    keep: list = []
    k = device = None
    for i, mi in enumerate(products):
        def bad(msg: str) -> ValueError:
            return ValueError(f"product {i}: {msg}")

        ts = {"a": mi.a, "b": mi.b, "p": mi.p}
        if mi.bias is not None:
            ts["bias"] = mi.bias
        for name, t in ts.items():
            if not isinstance(t, torch.Tensor):
                raise bad(f"{name} is not a tensor")
            if t.dtype not in _DTYPE_CODE:
                raise bad(f"{name} has dtype {t.dtype}, not float32 or bfloat16")
        for name in ("a", "b", "p"):
            if ts[name].dim() != 2:
                raise bad(f"{name} is not 2-D (shape {tuple(ts[name].shape)})")
        n, q = mi.a.shape
        w = mi.b.shape[1]
        if mi.b.shape[0] != q or tuple(mi.p.shape) != (n, w):
            raise bad(f"shapes a {tuple(mi.a.shape)}, b {tuple(mi.b.shape)}, "
                      f"p {tuple(mi.p.shape)} do not make a product")
        if mi.bias is not None and (mi.bias.dim() != 1 or mi.bias.shape[0] != w):
            raise bad(f"bias has shape {tuple(mi.bias.shape)}, not ({w},)")
        if n < 1 or q < 1 or w < 1:
            raise bad(f"a dimension is below 1 (n = {n}, q = {q}, w = {w})")
        r = mi.r
        if not isinstance(r, torch.Tensor) or r.dtype != torch.float32:
            raise bad("r is not a float32 tensor")
        if r.dim() != 2 or r.shape[0] != w or r.shape[1] < 1:
            raise bad(f"r has shape {tuple(r.shape)}, not ({w}, k) with k >= 1")
        if not r.is_contiguous():
            raise bad("r is not contiguous")
        devs = {t.device for t in (*ts.values(), r)}
        if len(devs) != 1:
            raise bad(f"tensors on several devices {sorted(map(str, devs))}")
        dev = devs.pop()
        if device is None:
            device = dev
        elif dev != device:
            raise bad(f"device {dev} differs from the table's {device}")
        if k is None:
            k = r.shape[1]
        elif r.shape[1] != k:
            raise bad(f"k = {r.shape[1]} differs from the table's {k}")
        e = arr[i]
        e.a, e.b, e.p = _view(mi.a, "a"), _view(mi.b, "b"), _view(mi.p, "p")
        if mi.bias is not None:
            e.bias = _view(mi.bias, "bias")
        e.r_addr = r.data_ptr()
        e.k = k
        e.flags = (M_GUARD if mi.guard else 0) | (M_BIASED if mi.bias is not None else 0)
        keep.extend(ts.values())
        keep.append(r)
    return Table(arr, len(products), k, device, keep)


def row_width(k: int) -> int:
    """The length of a measured row (``freivalds.ROW`` with ``k`` residuals)."""
    return 5 + k


def empty_rows(table: Table) -> np.ndarray:
    """A float64 ``[n, 5 + k]`` array for a build to fill."""
    return np.empty((table.n, row_width(table.k)), dtype=np.float64)
