"""FS29: how fast the GPU BLAKE3 kernel hashes a real full-scale step's leaves on one H100.

    .venv/bin/python -m verification.runs.kernel_speed sizes --out FILE            (on the Mac)
    .venv/bin/python -m verification.runs.kernel_speed run --sizes FILE --out REPORT  (on the GPU)

``sizes`` loads each full-scale model's config (:data:`~verification.runs.fs11_session.RUN_PINS`)
from the local Hugging Face cache, offline, config only, and writes
``leaf_sizes(config, n_s, n, compute_dtype)`` for it (``n_s``, ``n`` and the compute dtype from
:data:`~verification.runs.fs11_session.FULL_SCALE_ENV`) as JSON: the leaves in commit order, each
a row ``[section, dtype, shape, nbytes]`` (:func:`write_sizes`). The pod reads the file, so it
needs no model files and no token.

``run`` hashes each model's step on the GPU from that list. One device **pool** is allocated once
(:func:`pool_size`: the smaller of the largest step and the free GPU memory less
:data:`MARGIN_BYTES` and :data:`NODE_FRACTION` of the pool, which the kernel's node buffers take)
and filled with random bf16 values; a copy of it is made on the host. The step's leaves, in
commit order, are laid into consecutive tables whose payloads fit the pool (:func:`layout`), each
leaf at its own 256-byte-aligned offset with its real header: ``tensor_leaf_header`` for a weight
or a product, the record header for a record. The bf16 values are finite as bf16 and, read as
fp32 (the high half of each word is a finite bf16, so its exponent is not all ones), as fp32
too, so every table passes the finiteness check whatever the leaf's dtype; an int32 leaf takes
the same bytes as random integers. A table's bytes don't depend on its leaf, so the speed
doesn't either. Each table is hashed

- by the whole call (``hash_leaves``, the host's table building included), best of ``--repeats``
  after a warm-up call, and
- by the kernels alone (``prepare`` once, then CUDA events around ``launch``), best of
  ``--repeats``,

and every digest and ``first_bad`` is compared with the C library's (``native``) on the host
copy of the pool. A step's rate is its bytes (headers and payloads) over the sum of its tables'
times; GB/s are 10⁹ bytes per second. A model's four leaf-size bands (:data:`BANDS`) are then
hashed as one set of tables each, from that band's leaves alone, for the speed by size (not
checked against the C library again: the same code and pool, checked above).

The **gate** (:func:`gate`), on the slowest model's whole-call rate: ``pass`` at 300 GB/s or
more, ``review`` from 150 to under 300 (the user decides), ``fail`` below 150 (the choice returns
to the user, with cuPQC's SHA-256 as the alternative). Any digest mismatch, a model that errored
or a missing rate makes it ``fail``. The line is printed and in the report. A leaf larger than
the pool is an error (:func:`layout` raises ``ValueError``); the pool is always larger than the
largest leaf (622 MB) on an H100. Exit codes: 0 when the run finished, whatever the gate says;
1 on a digest mismatch or an error; 2 without CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from setup.records import INT32_CODE, TAG_RECORD
from verification.commitment import gpu_blake3 as gb
from verification.commitment import native
from verification.commitment.encoding import TAG_PRODUCT, TAG_WEIGHT, tensor_leaf_header
from verification.runs.fs11_session import FULL_SCALE_ENV, RUN_PINS

__all__ = ["SIZES_NAME", "BANDS", "MARGIN_BYTES", "NODE_FRACTION", "PASS_GBPS", "FAIL_GBPS",
           "Row", "ModelSizes", "write_sizes", "read_sizes", "make_sizes", "pool_size", "layout",
           "gate", "gate_line", "run_models", "main"]

SIZES_NAME = "kernel_speed_sizes.json"
ALIGN = 256  # a leaf's payload offset in the pool, as a torch allocation's alignment
MARGIN_BYTES = 4 << 30  # free GPU memory kept back from the pool, for CUDA's own use
NODE_FRACTION = 0.08  # the kernel's node buffers, per byte of a table, with room to spare
PASS_GBPS = 300.0
FAIL_GBPS = 150.0
BANDS = (("under 64 KiB", 0, 64 << 10), ("64 KiB to 1 MiB", 64 << 10, 1 << 20),
         ("1 to 16 MiB", 1 << 20, 16 << 20), ("over 16 MiB", 16 << 20, None))
SECTIONS = ("record", "w_t", "product", "w_next")
FORMAT = 1


# ---- the sizes file --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One leaf: its section (``leaf_sizes.SECTIONS``), dtype, shape, and encoded length."""

    section: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class ModelSizes:
    repo: str
    revision: str
    rows: list[Row]

    @property
    def nbytes(self) -> int:
        return sum(r.nbytes for r in self.rows)


def _dtype_name(d: torch.dtype) -> str:
    return str(d).removeprefix("torch.")


def write_sizes(path: Path, models: Sequence[ModelSizes], *, n_s: int, n: int,
                compute_dtype: str) -> None:
    """The sizes JSON: ``{"format", "n_s", "n", "compute_dtype", "sections", "dtypes",
    "models": [{"repo", "revision", "leaves", "bytes", "rows"}]}``, a row ``[section index,
    dtype index, shape, nbytes]`` in commit order."""
    dtypes = sorted({_dtype_name(r.dtype) for m in models for r in m.rows})
    doc = {"format": FORMAT, "n_s": n_s, "n": n, "compute_dtype": compute_dtype,
           "sections": list(SECTIONS), "dtypes": dtypes,
           "models": [{"repo": m.repo, "revision": m.revision, "leaves": len(m.rows),
                       "bytes": m.nbytes,
                       "rows": [[SECTIONS.index(r.section), dtypes.index(_dtype_name(r.dtype)),
                                 list(r.shape), r.nbytes] for r in m.rows]} for m in models]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))


def read_sizes(path: Path) -> list[ModelSizes]:
    """The models of a sizes file, with every row's length checked against its header and
    shape (:func:`leaf_headers`)."""
    doc = json.loads(path.read_text())
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: format {doc.get('format')!r}, expected {FORMAT}")
    dtypes = [getattr(torch, d) for d in doc["dtypes"]]
    out = []
    for m in doc["models"]:
        rows = [Row(SECTIONS[s], dtypes[d], tuple(shape), nbytes)
                for s, d, shape, nbytes in m["rows"]]
        if len(rows) != m["leaves"] or sum(r.nbytes for r in rows) != m["bytes"]:
            raise ValueError(f"{path}: {m['repo']}'s rows disagree with its leaf and byte counts")
        leaf_headers(rows)
        out.append(ModelSizes(m["repo"], m["revision"], rows))
    return out


def make_sizes(repo: str, revision: str, config: Any, *, n_s: int, n: int,
               compute_dtype: torch.dtype) -> ModelSizes:
    from verification.computation.leaf_sizes import leaf_sizes

    return ModelSizes(repo, revision, [
        Row(l.section, l.dtype, tuple(l.shape), l.nbytes)
        for l in leaf_sizes(config, n_s=n_s, n=n, compute_dtype=compute_dtype)])


def sizes_command(out: Path) -> int:
    """Each run model's leaf list at the full-scale batch, from the local cache (offline)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from setup.model import load_model_config

    n_s, n = int(FULL_SCALE_ENV["VERIF_BATCH"]), int(FULL_SCALE_ENV["VERIF_SEQ_LEN"])
    name = FULL_SCALE_ENV["VERIF_COMPUTE_DTYPE"]
    models = []
    for repo, rev in RUN_PINS:
        m = make_sizes(repo, rev, load_model_config(repo, rev), n_s=n_s, n=n,
                       compute_dtype=getattr(torch, name))
        print(f"{repo}: {len(m.rows)} leaves, {m.nbytes / 1e9:.2f} GB")
        models.append(m)
    write_sizes(out, models, n_s=n_s, n=n, compute_dtype=name)
    print(f"wrote {out}")
    return 0


# ---- headers, tables ---------------------------------------------------------------------------


def leaf_headers(rows: Sequence[Row]) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
    """``(hdr_buf, hdr_off, hdr_len, hdr_idx, payload_len, kind)`` for every row: the header
    a leaf of that section, dtype and shape has in the commitment (a record's is
    ``0x01 ‖ 0x10 ‖ ℓ``, a weight's and a product's ``tensor_leaf_header``), one copy per
    distinct header, and the payload's length and finiteness kind. Raises ``ValueError`` where a
    row's ``nbytes`` isn't its header plus its values."""
    uniq: dict[tuple[Any, ...], int] = {}
    heads: list[bytes] = []
    idx = np.empty(len(rows), dtype=np.uint32)
    payload = np.empty(len(rows), dtype=np.uint64)
    kind = np.empty(len(rows), dtype=np.uint8)
    for i, r in enumerate(rows):
        key = (r.section, r.dtype, r.shape)
        u = uniq.get(key)
        if u is None:
            if r.section == "record":
                if r.dtype != torch.int32 or len(r.shape) != 2 or r.shape[0] != 3:
                    raise ValueError(f"leaf {i}: a record is int32 of shape (3, n), not "
                                     f"{r.dtype} {r.shape}")
                head = struct.pack(">BBI", TAG_RECORD, INT32_CODE, r.shape[1])
            else:
                tag = TAG_PRODUCT if r.section == "product" else TAG_WEIGHT
                head = tensor_leaf_header(tag, torch.empty(r.shape, dtype=r.dtype, device="meta"))
            u = uniq[key] = len(heads)
            heads.append(head)
        numel = math.prod(r.shape)
        if len(heads[u]) + numel * r.dtype.itemsize != r.nbytes:
            raise ValueError(f"leaf {i}: {r.nbytes} bytes, but its header and {numel} "
                             f"{r.dtype} values are {len(heads[u]) + numel * r.dtype.itemsize}")
        idx[i] = u
        payload[i] = r.nbytes - len(heads[u])
        kind[i] = native.FINITE_KIND[r.dtype]
    hdr_len = np.array([len(h) for h in heads], dtype=np.uint32)
    hdr_off = np.zeros(len(heads), dtype=np.uint64)
    if heads:
        np.cumsum(hdr_len[:-1], out=hdr_off[1:])
    return b"".join(heads), hdr_off, hdr_len, idx, payload, kind


def _aligned(payload: np.ndarray) -> np.ndarray:
    return (payload.astype(np.uint64) + np.uint64(ALIGN - 1)) // np.uint64(ALIGN) * np.uint64(ALIGN)


def pool_size(step_payload: np.ndarray | int, free_bytes: int) -> int:
    """The pool's bytes: the smaller of the step's (aligned) payloads and the free GPU memory
    less :data:`MARGIN_BYTES`, scaled down by :data:`NODE_FRACTION` for the kernel's buffers,
    rounded down to a multiple of :data:`ALIGN`."""
    step = int(_aligned(step_payload).sum()) if isinstance(step_payload, np.ndarray) \
        else int(step_payload)
    room = int((free_bytes - MARGIN_BYTES) / (1 + NODE_FRACTION)) // ALIGN * ALIGN
    if room <= 0:
        raise ValueError(f"{free_bytes} bytes of free GPU memory leave no pool after the "
                         f"{MARGIN_BYTES}-byte margin")
    return min(step, room)


@dataclass(frozen=True)
class TableSpec:
    start: int  # the table holds leaves ``start:end`` of the sequence laid out
    end: int
    offsets: np.ndarray  # each leaf's payload offset in the pool, uint64
    used: int  # bytes of the pool the table touches


def layout(payload_len: np.ndarray, pool_bytes: int) -> list[TableSpec]:
    """Consecutive tables over ``payload_len``, in order, each with its leaves at aligned
    offsets from 0 and its total within ``pool_bytes``. Every leaf is in exactly one table. A
    leaf whose aligned payload exceeds the pool raises ``ValueError`` naming it (it gets no
    table of its own: the pool is the most the device holds)."""
    size = _aligned(payload_len).astype(np.int64)
    end_cum = np.cumsum(size)
    n = len(size)
    out: list[TableSpec] = []
    start = 0
    while start < n:
        base = int(end_cum[start - 1]) if start else 0
        stop = int(np.searchsorted(end_cum, base + pool_bytes, side="right"))
        if stop <= start:
            raise ValueError(f"leaf {start} needs {int(size[start])} bytes, more than the "
                             f"{pool_bytes}-byte pool")
        starts = end_cum[start:stop] - size[start:stop] - base
        out.append(TableSpec(start, stop, starts.astype(np.uint64), int(end_cum[stop - 1]) - base))
        start = stop
    return out


# ---- the gate ---------------------------------------------------------------------------------


def gate(models: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The verdict on the slowest model's whole-call rate (module docstring). Each model needs
    ``repo``, ``call_gbps``, ``kernel_gbps`` and ``mismatches``; a model with an ``error``, a
    rate that isn't a finite number, or a mismatch makes the verdict ``fail`` whatever the
    speed."""
    def rate(m: Mapping[str, Any], key: str) -> float | None:
        v = m.get(key)
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None

    reasons = []
    for m in models:
        if m.get("error"):
            reasons.append(f"{m['repo']}: {m['error']}")
        if m.get("mismatches") != 0:
            reasons.append(f"{m['repo']}: {m.get('mismatches')} digest mismatches")
        if rate(m, "call_gbps") is None:
            reasons.append(f"{m['repo']}: no whole-call rate")
    if not models:
        reasons.append("no model was measured")
    slowest = min((m for m in models if rate(m, "call_gbps") is not None),
                  key=lambda m: rate(m, "call_gbps"), default=None)
    call = rate(slowest, "call_gbps") if slowest else None
    kernel = rate(slowest, "kernel_gbps") if slowest else None
    if reasons or call is None:
        verdict = "fail"
    elif call >= PASS_GBPS:
        verdict = "pass"
    elif call >= FAIL_GBPS:
        verdict = "review"
    else:
        verdict = "fail"
    return {"verdict": verdict, "slowest": slowest["repo"] if slowest else None,
            "call_gbps": call, "kernel_gbps": kernel, "reasons": reasons}


def gate_line(g: Mapping[str, Any]) -> str:
    speed = ("" if g["call_gbps"] is None else
             f" Slowest model {g['slowest']}: whole call {g['call_gbps']:.0f} GB/s, kernels "
             f"alone {g['kernel_gbps']:.0f} GB/s." if g["kernel_gbps"] is not None else
             f" Slowest model {g['slowest']}: whole call {g['call_gbps']:.0f} GB/s.")
    why = f" Because: {'; '.join(g['reasons'])}." if g["reasons"] else ""
    return (f"FS29 gate: {g['verdict']}.{speed}{why} (pass at {PASS_GBPS:.0f} GB/s or more, "
            f"review from {FAIL_GBPS:.0f} to under {PASS_GBPS:.0f}, fail below {FAIL_GBPS:.0f}.)")


# ---- the measurement --------------------------------------------------------------------------


def _best_kernel_seconds(g: Any, prep: Any, device: torch.device, repeats: int) -> float:
    best = math.inf
    for _ in range(repeats):
        if device.type == "cuda":
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.launch(prep, device)
            b.record()
            b.synchronize()
            best = min(best, a.elapsed_time(b) / 1e3)
        else:
            t0 = time.perf_counter()
            g.launch(prep, device)
            best = min(best, time.perf_counter() - t0)
    return best


@dataclass
class Group:
    """The leaves of one model or band: arrays over them (header index, payload length, kind)
    and the shared headers."""

    hdr_buf: bytes
    hdr_off: np.ndarray
    hdr_len: np.ndarray
    hdr_idx: np.ndarray
    payload_len: np.ndarray
    kind: np.ndarray
    nbytes: np.ndarray  # each leaf's encoded length

    def take(self, sel: np.ndarray) -> Group:
        return Group(self.hdr_buf, self.hdr_off, self.hdr_len, self.hdr_idx[sel],
                     self.payload_len[sel], self.kind[sel], self.nbytes[sel])


def _table_args(grp: Group, spec: TableSpec, base: int) -> tuple[Any, ...]:
    s = slice(spec.start, spec.end)
    return (grp.hdr_buf, grp.hdr_off, grp.hdr_len, grp.hdr_idx[s], spec.offsets + np.uint64(base),
            grp.payload_len[s], grp.kind[s])


def measure(g: Any, grp: Group, pool: torch.Tensor, host: torch.Tensor, repeats: int, *,
            threads: int, check: bool, say: Callable[[str], None]) -> dict[str, Any]:
    """Hash ``grp``'s leaves in tables over ``pool`` (module docstring): the leaves, bytes,
    tables, summed best kernel-only and whole-call seconds and their GB/s, and, when ``check``,
    ``mismatches`` (leaves whose digest differs from the C library's, plus one when
    ``first_bad`` isn't the table's leaf count)."""
    device = pool.device
    tables = layout(grp.payload_len, pool.numel())
    lib = native.get() if check else None
    kernel_s = call_s = 0.0
    mismatches = 0
    for k, spec in enumerate(tables):
        args = _table_args(grp, spec, pool.data_ptr())
        n = spec.end - spec.start
        g.hash_leaves(*args, device=device)  # warm up (and compile)
        best = math.inf
        got = got_bad = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            got, got_bad = g.hash_leaves(*args, device=device)
            best = min(best, time.perf_counter() - t0)
        call_s += best
        prep = g.prepare(*args, device=device)
        kernel_s += _best_kernel_seconds(g, prep, device, repeats)
        del prep
        if check:
            ref, ref_bad = lib.hash_leaves(*_table_args(grp, spec, host.data_ptr()),
                                           threads=threads)
            have = got.cpu().numpy()
            wrong = int((have != ref).any(axis=1).sum()) if n else 0
            mismatches += wrong + (got_bad != n) + (ref_bad != n)
            if wrong or got_bad != n or ref_bad != n:
                say(f"  table {k}: {wrong} digests differ; first_bad kernel {got_bad}, C "
                    f"library {ref_bad}, leaves {n}")
        del got
    total = int(grp.nbytes.sum())
    out = {"leaves": len(grp.payload_len), "bytes": total, "tables": len(tables),
           "kernel_s": kernel_s, "kernel_gbps": total / kernel_s / 1e9 if kernel_s else None,
           "call_s": call_s, "call_gbps": total / call_s / 1e9 if call_s else None}
    if check:
        out["mismatches"] = mismatches
    return out


def run_models(models: Sequence[ModelSizes], pool: torch.Tensor, host: torch.Tensor,
               g: Any, repeats: int, *, threads: int,
               say: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """One result per model (:func:`measure` on its whole step, checked, then each band of
    :data:`BANDS` alone). A model that raises is recorded with ``error`` and the run goes on."""
    out = []
    for m in models:
        say(f"{m.repo}: {len(m.rows)} leaves, {m.nbytes / 1e9:.2f} GB")
        try:
            grp = Group(*leaf_headers(m.rows), np.array([r.nbytes for r in m.rows], np.int64))
            res = measure(g, grp, pool, host, repeats, threads=threads, check=True, say=say)
            say(f"  {res['tables']} tables: kernels alone {res['kernel_gbps']:.0f} GB/s, whole "
                f"call {res['call_gbps']:.0f} GB/s, {res['mismatches']} mismatches")
            bands = []
            for name, lo, hi in BANDS:
                sel = np.flatnonzero((grp.nbytes >= lo) & ((grp.nbytes < hi) if hi else True))
                row: dict[str, Any] = {"band": name, "leaves": len(sel),
                                       "bytes": int(grp.nbytes[sel].sum())}
                if len(sel):
                    row.update({k: v for k, v in measure(
                        g, grp.take(sel), pool, host, repeats, threads=threads, check=False,
                        say=say).items() if k not in ("leaves", "bytes")})
                    say(f"  {name}: {row['leaves']} leaves, {row['bytes'] / 1e9:.2f} GB, kernels "
                        f"alone {row['kernel_gbps']:.0f} GB/s, whole call "
                        f"{row['call_gbps']:.0f} GB/s")
                bands.append(row)
            res["bands"] = bands
            out.append({"repo": m.repo, "revision": m.revision, **res})
        except Exception as e:  # recorded, so the other models still run and the gate fails
            say(f"  {m.repo} failed: {type(e).__name__}: {e}")
            out.append({"repo": m.repo, "revision": m.revision, "leaves": len(m.rows),
                        "bytes": m.nbytes, "mismatches": None,
                        "error": f"{type(e).__name__}: {e}"})
    return out


NEEDS_CUDA = "kernel_speed run: needs CUDA, and torch.cuda.is_available() is False here"


def run_command(sizes: Path, out: Path, repeats: int) -> int:
    if not torch.cuda.is_available():
        print(NEEDS_CUDA, file=sys.stderr)
        return 2
    try:
        models = read_sizes(sizes)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"kernel_speed run: cannot use {sizes}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    dev = torch.device("cuda")
    free = torch.cuda.mem_get_info()[0]
    step = max(int(_aligned(np.array([r.nbytes for r in m.rows], np.int64)).sum())
               for m in models)
    pool_bytes = pool_size(step, free)
    print(f"free GPU memory {free / 2**30:.1f} GiB; pool {pool_bytes / 2**30:.1f} GiB "
          f"(largest step {step / 2**30:.1f} GiB; margin {MARGIN_BYTES / 2**30:.0f} GiB plus "
          f"{NODE_FRACTION:.0%} of the pool for the kernel's buffers)", flush=True)
    pool = torch.empty(pool_bytes, dtype=torch.uint8, device=dev)
    pool.view(torch.bfloat16).normal_()
    host = pool.cpu()
    threads = max(1, int(os.environ.get("VERIF_THREADS", "8")))
    g = gb.GpuBlake3()
    results = run_models(models, pool, host, g, repeats, threads=threads,
                         say=lambda s: print(s, flush=True))
    verdict = gate(results)
    line = gate_line(verdict)
    print(line, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "describe": gb.describe(dev), "pool_bytes": pool_bytes, "free_bytes": free,
        "margin_bytes": MARGIN_BYTES, "node_fraction": NODE_FRACTION, "repeats": repeats,
        "threads": threads, "sizes": str(sizes), "gate": verdict, "gate_line": line,
        "models": results}, indent=2))
    return 1 if any(m.get("error") or m.get("mismatches") for m in results) else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verification.runs.kernel_speed",
                                 description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sizes", help="the four models' leaf lists, from the local cache")
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="hash each step on the GPU, check it, print the gate")
    r.add_argument("--sizes", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args(argv)
    if a.cmd == "sizes":
        return sizes_command(a.out)
    return run_command(a.sizes, a.out, max(1, a.repeats))


if __name__ == "__main__":
    sys.exit(main())
