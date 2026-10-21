"""F22's peer-copy byte check: does a GPU-to-GPU copy land the right bytes?

    .venv/bin/python -m verification.runs.peer_check --out peer_check.json

On the two A40s of the FS32 session, ``nvidia-smi topo -p2p r`` said OK and
``torch.cuda.can_device_access_peer`` was True, yet every direct copy between the cards left wrong
bytes at the destination, while the same copy staged through host memory was right. So before any
session on two or more GPUs counts a rate or a result, this check compares, for every ordered pair
``(i, j)`` of the visible CUDA devices, each copy against the source's bytes read back to host
memory:

* ``host_staged``: source to host to destination (the control). If it fails, the card itself is
  suspect and the pair's direct copies are not tried, so the report never blames the peer path for
  a card fault;
* ``to_*`` and ``copy_*`` at about 4 KB, 4 MB and 256 MB: ``src.to(dst)`` and ``dst.copy_(src)``;
* ``strided``: ``dst.copy_(src.t())`` of a non-square source, which needs a kernel, not a memcpy,
  and whose bytes are compared in the source's logical order;
* what the transcript copies, each at 4 MB (:data:`EXTRA_CASES`): ``bf16_4MB`` and ``fp32_4MB``
  (the product and weight dtypes), ``offset_4MB`` (a view with a non-zero storage offset, a bmm
  member's slice, by ``copy_``) and ``3d_4MB`` (a 3-D source copied whole by ``to``, a bmm's
  output);
* ``side_stream``: the capture's own route (``verification/prover/capture.py``, ``_route``): one
  side stream per source device for the whole check (as ``HostBuffers.stream`` keeps one), which
  waits on the current stream, where an op has just produced the source with no synchronization
  before the copy, so the ``wait_stream`` ordering is exercised; an ``empty`` on the destination
  and ``copy_(..., non_blocking=True)`` run under the side stream, then an event recorded on it
  is synchronized. What still differs from ``_route``: the event is synchronized at once, not at
  ``_drain``, and the source is released after the comparison, not held until the event completes.

Every case draws its own random source, and every destination ``cuda_copy`` allocates (``copy_``,
``side_stream``) is first filled with ``0xFF`` bytes, so a copy that writes nothing cannot pass on
what the caching allocator left in the block (a freed ``to`` result of the same size holds the
right bytes). The result must be on the destination with the source's shape and dtype, else the
case fails with that reason. The sources are random 32-bit integers, viewed as the case's dtype,
and the comparison is of bytes (``uint8`` views), so a NaN pattern can neither hide a difference
nor fake one. Both devices are synchronized before each comparison. A case that raises is false
with its error in ``errors``, and the report says it raised, not that the bytes were wrong.

The report (``--out``) holds, per pair ``"i->j"``, ``{"ok", "cases": {case: bool}, "gbps",
"errors", "card_suspect"}`` (``gbps``: the direct 256 MB ``copy_`` rate, one number;
``nvlink_probe`` owns the real bandwidth measurement), ``all_ok``, the device names, the driver,
torch and CUDA versions, and the seconds taken. ``--out`` is deleted first, so a killed rerun
leaves no old report. Exit 0 when every pair is right, 1 otherwise (and on fewer than two CUDA
devices), with one line per wrong pair naming its cases. The comparison logic
(:func:`check_pair`, :func:`check`) takes the copy as an argument, so a CPU test can inject a
copy that corrupts a byte.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

__all__ = ["SIZES", "CONTIGUOUS_CASES", "EXTRA_CASES", "case_name", "cuda_copy", "host_staged",
           "check_pair", "check", "pair_lines", "report_lines", "main"]

# int32 elements: 4 KB, 4 MB, 256 MB
SIZES = {"4KB": 1 << 10, "4MB": 1 << 20, "256MB": 64 << 20}
CONTIGUOUS_KINDS = ("to", "copy_")


def case_name(kind: str, label: str) -> str:
    """``to_4KB``, ``copy_4MB``: the kind and the size, one underscore between."""
    return f"{kind}{'' if kind.endswith('_') else '_'}{label}"


CONTIGUOUS_CASES = tuple(case_name(k, s) for s in SIZES for k in CONTIGUOUS_KINDS)
# What the transcript copies, at 4 MB (the cases above are the contiguous int32 ones).
EXTRA_CASES = ("bf16_4MB", "fp32_4MB", "offset_4MB", "3d_4MB")
STRIDED_SHAPE = (512, 2048)  # int32, 4 MB; the copy is of the transpose, shape (2048, 512)
OFFSET = 1536  # elements: the storage offset of the view case
SHAPE_3D = (16, 256, 256)  # fp32, 4 MB
FILL = 0xFF  # the byte every destination holds before a copy
RATE_REPS = 3
HOST_STAGED = "host_staged"
BAD_RESULT = "BadResult"  # the error type of a result on the wrong device, shape or dtype

Copy = Callable[[str, torch.Tensor, Any], torch.Tensor]  # (kind, source, destination device)
Control = Callable[[torch.Tensor, Any], torch.Tensor]  # (source, destination device)


class BadResult(Exception):
    """A copy's result is not on the destination, or has not the source's shape or dtype."""


def _device(x: int | str | torch.device) -> torch.device:
    return torch.device(f"cuda:{x}") if isinstance(x, int) else torch.device(x)


def host_staged(src: torch.Tensor, dst_device: Any) -> torch.Tensor:
    """The control: the source read back to host memory, then sent to the destination."""
    return src.cpu().to(dst_device)


_SIDE: dict[torch.device, Any] = {}


def _side_stream(device: torch.device) -> Any:
    """The one side stream of ``device`` for the whole check (``HostBuffers.stream``'s way)."""
    s = _SIDE.get(device)
    if s is None:
        s = _SIDE[device] = torch.cuda.Stream(device)
    return s


def _filled(shape: Sequence[int], dtype: torch.dtype, device: Any) -> torch.Tensor:
    """A fresh destination holding ``FILL`` bytes, unlike the random source."""
    dst = torch.empty(tuple(shape), dtype=dtype, device=device)
    dst.view(torch.uint8).fill_(FILL)
    return dst


def cuda_copy(kind: str, src: torch.Tensor, dst_device: Any) -> torch.Tensor:
    """One direct copy of ``src`` to ``dst_device``. ``kind``: ``to`` (``src.to``), ``copy_``
    (``dst.copy_(src)`` into a fresh contiguous tensor filled with ``0xFF`` bytes, which a strided
    ``src`` makes a kernel), or ``side_stream`` (capture's ``_route``, into a filled tensor too)."""
    if kind == "to":
        return src.to(dst_device)
    if kind == "copy_":
        dst = _filled(src.shape, src.dtype, dst_device)
        dst.copy_(src)
        return dst
    if kind == "side_stream":
        side = _side_stream(src.device)
        side.wait_stream(torch.cuda.current_stream(src.device))
        with torch.cuda.stream(side):
            moved = _filled(src.shape, src.dtype, dst_device)
            moved.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
        done.synchronize()
        return moved
    raise ValueError(f"unknown copy kind {kind!r}")


def _bytes(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().contiguous().view(torch.uint8)  # contiguous: the logical order


def _random_bits(shape: Sequence[int], device: torch.device,
                 dtype: torch.dtype = torch.int32) -> torch.Tensor:
    """Random 32-bit integers of ``shape``, viewed as ``dtype`` (a 16-bit dtype doubles the last
    dimension, so ``shape`` holds the 4-byte words)."""
    x = torch.randint(-(1 << 31), (1 << 31) - 1, tuple(shape), dtype=torch.int32, device=device)
    return x if dtype == torch.int32 else x.view(dtype)


def _sync(devices: Sequence[torch.device]) -> None:
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)


def _located(got: torch.device, want: torch.device) -> bool:
    return got.type == want.type and (want.index is None or got.index == want.index)


def _expect(got: torch.Tensor, want: torch.Tensor, dst: torch.device) -> None:
    """Raise :class:`BadResult` unless ``got`` is on ``dst`` with ``want``'s shape and dtype."""
    if not _located(got.device, dst):
        raise BadResult(f"the result is on {got.device}, not the destination {dst}")
    if got.dtype != want.dtype:
        raise BadResult(f"the result is {got.dtype}, the source {want.dtype}")
    if got.shape != want.shape:
        raise BadResult(f"the result has shape {tuple(got.shape)}, the source "
                        f"{tuple(want.shape)}")


def check_pair(src_device: Any, dst_device: Any, copy: Copy = cuda_copy,
               control: Control = host_staged, sync: Callable[[], None] | None = None
               ) -> dict[str, Any]:
    """Every case for one ordered pair: ``{"ok", "cases", "gbps", "errors", "card_suspect"}``. A
    case is true when the result is on the destination with the source's shape and dtype and its
    bytes equal the source's, read back to host memory (in the source's logical order). A case
    that raises, or whose result is on the wrong device or of the wrong shape or dtype, is false,
    its error (``"Type: message"``) in ``errors``. When the control fails, ``card_suspect`` is
    true and the direct cases are not run. ``gbps`` is the rate of repeated 256 MB ``copy_``
    (``None`` when a case failed, or the rate is not timed). Each case draws its own source."""
    src_d, dst_d = _device(src_device), _device(dst_device)
    sync = sync or (lambda: _sync([src_d, dst_d]))
    cases: dict[str, bool] = {}
    errors: dict[str, str] = {}

    def run(name: str, make: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> bool:
        try:
            got, want = make()
            sync()
            _expect(got, want, dst_d)
            ok = bool(torch.equal(_bytes(got), _bytes(want)))
        except Exception as e:  # noqa: BLE001 - one case's failure must not lose the pair
            errors[name] = f"{type(e).__name__}: {e}"
            ok = False
        cases[name] = ok
        return ok

    def source(shape: Sequence[int], dtype: torch.dtype = torch.int32) -> torch.Tensor:
        x = _random_bits(shape, src_d, dtype)
        sync()
        return x

    src = source((SIZES["4KB"],))
    control_ok = run(HOST_STAGED, lambda: (control(src, dst_d), src))
    if not control_ok:
        return {"ok": False, "cases": cases, "gbps": None, "errors": errors,
                "card_suspect": True}
    big = None
    for label, n in SIZES.items():
        for kind in CONTIGUOUS_KINDS:
            x = source((n,))
            run(case_name(kind, label), lambda kind=kind, x=x: (copy(kind, x, dst_d), x))
            if kind == "copy_":
                big = x
    n4 = SIZES["4MB"]
    x = source((n4,), torch.bfloat16)  # n4 words of 4 bytes: 4 MB as 2 * n4 bf16
    run("bf16_4MB", lambda: (copy("copy_", x, dst_d), x))
    x = source((n4,), torch.float32)
    run("fp32_4MB", lambda: (copy("copy_", x, dst_d), x))
    base = source((n4 + OFFSET,))
    view = base[OFFSET:]
    run("offset_4MB", lambda: (copy("copy_", view, dst_d), view))
    x = source(SHAPE_3D, torch.float32)
    run("3d_4MB", lambda: (copy("to", x, dst_d), x))
    m = source(STRIDED_SHAPE)
    run("strided", lambda: (copy("copy_", m.t(), dst_d), m.t()))
    a, b = source((n4,)), source((n4,))

    def side() -> tuple[torch.Tensor, torch.Tensor]:
        made = a ^ b  # enqueued on the current stream, not synchronized before the copy
        return copy("side_stream", made, dst_d), made

    run("side_stream", side)
    gbps = None
    if big is not None and all(cases.values()):
        try:
            copy("copy_", big, dst_d)  # warm up
            sync()
            t0 = time.perf_counter()
            for _ in range(RATE_REPS):
                copy("copy_", big, dst_d)
            sync()
            gbps = RATE_REPS * big.numel() * big.element_size() / (time.perf_counter() - t0) / 1e9
        except Exception as e:  # noqa: BLE001
            errors["rate"] = f"{type(e).__name__}: {e}"
    return {"ok": all(cases.values()), "cases": cases, "gbps": gbps, "errors": errors,
            "card_suspect": False}


def check(devices: Sequence[Any], copy: Copy = cuda_copy, control: Control = host_staged,
          sync: Callable[[], None] | None = None) -> dict[str, Any]:
    """:func:`check_pair` for every ordered pair of ``devices`` (``"i->j"``, ``i`` and ``j`` the
    positions in ``devices``, which for CUDA are the device indices). ``all_ok`` is true only for
    at least one pair and every pair right."""
    t0 = time.perf_counter()
    pairs: dict[str, Any] = {}
    _SIDE.clear()  # the side streams are the check's own: one per source device, for all pairs
    try:
        for i, a in enumerate(devices):
            for j, b in enumerate(devices):
                if i != j:
                    pairs[f"{i}->{j}"] = check_pair(a, b, copy, control, sync)
    finally:
        _SIDE.clear()
    return {"all_ok": bool(pairs) and all(p["ok"] for p in pairs.values()), "pairs": pairs,
            "seconds": round(time.perf_counter() - t0, 2)}


def _error_text(case: str, err: str) -> str:
    """``<case> raised <Type>: <msg>``; a bad result says what was wrong with it."""
    if err.startswith(f"{BAD_RESULT}: "):
        return f"{case} gave a bad result ({err.split(': ', 1)[1]})"
    return f"{case} raised {err}"


def pair_lines(key: str, p: Mapping[str, Any]) -> list[str]:
    """The account of one pair that is not ``ok`` (empty when it is). A control that raised, or
    every direct case raising, says the check could not copy on that pair and names the error,
    since nothing is known of the peer path; a control that compared wrong says the card is
    suspect; otherwise the cases that compared wrong are named as wrong bytes and the cases that
    raised as raised."""
    if p.get("ok"):
        return []
    errors = p.get("errors") or {}
    cases = p.get("cases") or {}
    head = f"peer check FAIL: pair {key}: "
    if HOST_STAGED in errors:
        return [head + "the check could not copy on this pair, the host-staged control failed: "
                + _error_text(HOST_STAGED, errors[HOST_STAGED])]
    if p.get("card_suspect"):
        return [head + "the host-staged control is wrong, so the card itself is suspect, not "
                "the peer path"]
    direct = [c for c in cases if c != HOST_STAGED]
    raised = [c for c in direct if c in errors]
    if direct and len(raised) == len(direct):
        return [head + f"the check could not copy on this pair, every case raised "
                f"(first: {_error_text(raised[0], errors[raised[0]])})"]
    wrong = [c for c in direct if not cases[c] and c not in errors]
    bits = []
    if wrong:
        bits.append(f"wrong bytes in {', '.join(wrong)}")
    bits += [_error_text(c, errors[c]) for c in raised]
    return [head + ("; ".join(bits) or "no case ran")]


def report_lines(report: dict[str, Any]) -> list[str]:
    """One line for each wrong pair (:func:`pair_lines`); a report with no wrong pair gets one
    line saying so (or its ``reason``)."""
    if report.get("reason"):
        return [f"peer check FAIL: {report['reason']}"]
    lines = []
    for key, p in report["pairs"].items():
        lines += pair_lines(key, p)
    if not lines and not report["all_ok"]:
        lines.append("peer check FAIL: no pair was checked")
    if not lines:
        lines.append(f"peer check pass: every one of {len(report['pairs'])} ordered pairs "
                     f"landed the right bytes ({report['seconds']:.0f} s)")
    return lines


def _driver() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return r.stdout.splitlines()[0].strip() if r.returncode == 0 and r.stdout else None
    except (OSError, subprocess.SubprocessError):
        return None


def build_report() -> dict[str, Any]:
    """The check on every visible CUDA device, with the versions; a ``reason`` and no pairs when
    there are fewer than two."""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    head = {"version": 1, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__, "cuda": torch.version.cuda, "driver": _driver(),
            "devices": [torch.cuda.get_device_name(i) for i in range(n)]}
    if n < 2:
        return {**head, "all_ok": False, "pairs": {}, "seconds": 0.0,
                "reason": f"{n} CUDA device{'s' * (n != 1)} visible; a peer copy needs two"}
    return {**head, **check([torch.device(f"cuda:{i}") for i in range(n)]), "reason": None}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", type=Path, required=True, help="the JSON report")
    args = p.parse_args(argv)
    args.out.unlink(missing_ok=True)  # a killed rerun must not leave the last run's report
    try:
        report = build_report()
    except Exception as e:  # noqa: BLE001 - a broken CUDA context is a failed check, not a crash
        report = {"version": 1, "all_ok": False, "pairs": {}, "seconds": 0.0,
                  "reason": f"the check raised {type(e).__name__}: {e}"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for line in report_lines(report):
        print(line)
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
