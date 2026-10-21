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
* ``strided``: ``dst.copy_(src.t())``, which needs a kernel, not a memcpy;
* ``side_stream``: the capture's own route (``verification/prover/capture.py``, ``_route``): the
  source device's side stream waits on its current stream, an ``empty`` on the destination and
  ``copy_(..., non_blocking=True)`` run under that stream, then an event recorded on it is
  synchronized.

The sources are random 32-bit integers and the comparison is of bytes (``uint8`` views), so a NaN
pattern can neither hide a difference nor fake one. Both devices are synchronized before each
comparison. The report (``--out``) holds, per pair ``"i->j"``, ``{"ok", "cases": {case: bool},
"gbps"}`` (the direct 256 MB ``copy_`` rate, one number; ``nvlink_probe`` owns the real bandwidth
measurement), ``all_ok``, the device names, the driver, torch and CUDA versions, and the seconds
taken. Exit 0 when every pair is right, 1 otherwise (and on fewer than two CUDA devices), with one
line per wrong pair naming its cases. The comparison logic (:func:`check_pair`, :func:`check`) takes
the copy as an argument, so a CPU test can inject a copy that corrupts a byte.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

__all__ = ["SIZES", "CONTIGUOUS_CASES", "case_name", "cuda_copy", "host_staged", "check_pair", "check",
           "report_lines", "main"]

# int32 elements: 4 KB, 4 MB, 256 MB
SIZES = {"4KB": 1 << 10, "4MB": 1 << 20, "256MB": 64 << 20}
CONTIGUOUS_KINDS = ("to", "copy_")


def case_name(kind: str, label: str) -> str:
    """``to_4KB``, ``copy_4MB``: the kind and the size, one underscore between."""
    return f"{kind}{'' if kind.endswith('_') else '_'}{label}"


CONTIGUOUS_CASES = tuple(case_name(k, s) for s in SIZES for k in CONTIGUOUS_KINDS)
STRIDED_SHAPE = (1024, 1024)  # int32, 4 MB; the copy is of the transpose
RATE_REPS = 3
HOST_STAGED = "host_staged"

Copy = Callable[[str, torch.Tensor, Any], torch.Tensor]  # (kind, source, destination device)
Control = Callable[[torch.Tensor, Any], torch.Tensor]  # (source, destination device)


def _device(x: int | str | torch.device) -> torch.device:
    return torch.device(f"cuda:{x}") if isinstance(x, int) else torch.device(x)


def host_staged(src: torch.Tensor, dst_device: Any) -> torch.Tensor:
    """The control: the source read back to host memory, then sent to the destination."""
    return src.cpu().to(dst_device)


def cuda_copy(kind: str, src: torch.Tensor, dst_device: Any) -> torch.Tensor:
    """One direct copy of ``src`` to ``dst_device``. ``kind``: ``to`` (``src.to``), ``copy_``
    (``dst.copy_(src)`` into a fresh contiguous tensor, which a strided ``src`` makes a kernel),
    or ``side_stream`` (capture's ``_route``)."""
    if kind == "to":
        return src.to(dst_device)
    if kind == "copy_":
        dst = torch.empty(src.shape, dtype=src.dtype, device=dst_device)
        dst.copy_(src)
        return dst
    if kind == "side_stream":
        side = torch.cuda.Stream(src.device)
        side.wait_stream(torch.cuda.current_stream(src.device))
        with torch.cuda.stream(side):
            moved = torch.empty(src.shape, dtype=src.dtype, device=dst_device)
            moved.copy_(src, non_blocking=True)
            done = torch.cuda.Event()
            done.record(side)
        done.synchronize()
        return moved
    raise ValueError(f"unknown copy kind {kind!r}")


def _bytes(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().contiguous().view(torch.uint8)


def _random_bits(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.randint(-(1 << 31), (1 << 31) - 1, tuple(shape), dtype=torch.int32,
                         device=device)


def _sync(devices: Sequence[torch.device]) -> None:
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)


def check_pair(src_device: Any, dst_device: Any, copy: Copy = cuda_copy,
               control: Control = host_staged, sync: Callable[[], None] | None = None
               ) -> dict[str, Any]:
    """Every case for one ordered pair: ``{"ok", "cases", "gbps", "errors"}``. A case is true when
    the bytes at the destination equal the source's, read back to host memory. A case that raises
    is false, its error in ``errors``. When the control fails, ``card_suspect`` is true and the
    direct cases are not run. ``gbps`` is the rate of repeated 256 MB ``copy_`` (``None`` when a
    case failed, or the rate is not timed)."""
    src_d, dst_d = _device(src_device), _device(dst_device)
    sync = sync or (lambda: _sync([src_d, dst_d]))
    cases: dict[str, bool] = {}
    errors: dict[str, str] = {}

    def run(name: str, make: Callable[[], tuple[torch.Tensor, torch.Tensor]]) -> bool:
        try:
            got, want = make()
            sync()
            ok = bool(torch.equal(_bytes(got), _bytes(want)))
        except Exception as e:  # noqa: BLE001 - one case's failure must not lose the pair
            errors[name] = f"{type(e).__name__}: {e}"
            ok = False
        cases[name] = ok
        return ok

    src = _random_bits((SIZES["4KB"],), src_d)
    sync()
    control_ok = run(HOST_STAGED, lambda: (control(src, dst_d), src))
    if not control_ok:
        return {"ok": False, "cases": cases, "gbps": None, "errors": errors,
                "card_suspect": True}
    big = None
    for label, n in SIZES.items():
        x = _random_bits((n,), src_d)
        sync()
        for kind in CONTIGUOUS_KINDS:
            run(case_name(kind, label), lambda kind=kind, x=x: (copy(kind, x, dst_d), x))
        big = x
    m = _random_bits(STRIDED_SHAPE, src_d)
    sync()
    run("strided", lambda: (copy("copy_", m.t(), dst_d), m.t()))
    x = _random_bits((SIZES["4MB"],), src_d)
    sync()
    run("side_stream", lambda: (copy("side_stream", x, dst_d), x))
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
    for i, a in enumerate(devices):
        for j, b in enumerate(devices):
            if i != j:
                pairs[f"{i}->{j}"] = check_pair(a, b, copy, control, sync)
    return {"all_ok": bool(pairs) and all(p["ok"] for p in pairs.values()), "pairs": pairs,
            "seconds": round(time.perf_counter() - t0, 2)}


def report_lines(report: dict[str, Any]) -> list[str]:
    """One line for each wrong pair, naming its wrong cases; a corrupt control says the card is
    suspect. A report with no wrong pair gets one line saying so (or its ``reason``)."""
    if report.get("reason"):
        return [f"peer check FAIL: {report['reason']}"]
    lines = []
    for key, p in report["pairs"].items():
        if p["ok"]:
            continue
        if p.get("card_suspect"):
            lines.append(f"peer check FAIL: pair {key}: the host-staged control is wrong, so "
                         f"the card itself is suspect, not the peer path")
        else:
            bad = [c for c, ok in p["cases"].items() if not ok]
            lines.append(f"peer check FAIL: pair {key}: wrong bytes in {', '.join(bad)}")
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
