"""FS31, part A: is NVLink exposed between the cards (F22's check HW2)?

    .venv/bin/python -m verification.runs.nvlink_probe --out nvlink.json

Writes one JSON report with the machine's own account of its links (``nvidia-smi topo -m``,
``topo -p2p r`` and ``-p2p w``, ``nvlink -s`` and the card list, each with its raw text), torch's
view (device count, names, memory, ``can_device_access_peer`` per ordered pair), the measured
bandwidth of a 1 GiB device-to-device copy for every ordered pair, the GPU 0 fan-out to the
others, pinned host memory to each card for contrast, and ``nvbandwidth``'s device-to-device
tests when that tool is already on the PATH (never downloaded or built). A command that fails,
times out or is missing is recorded, not raised, and the measures that follow still run.

The verdict ``hw2`` is four checks: every pair of cards is joined by ``NV<n>`` links
(``nv_links``), every pair by exactly ``NV18`` (``nv18_everywhere``, the H100 SXM's link count),
peer reads are ``OK`` for every pair (``p2p_read_ok``), and the slowest pair's copy rate reaches
:data:`PEER_COPY_MIN_GBPS`. It exits 0 once the report is written, whatever the verdict; a
non-zero exit means the probe itself broke.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["PEER_COPY_MIN_GBPS", "COMMANDS", "run_command", "parse_gpu_matrix", "hw2_verdict",
           "peer_copy_bandwidth", "fanout_bandwidth", "pinned_bandwidth", "torch_view",
           "build_report", "main"]

VERSION = 1
GIB = 1 << 30
COPY_BYTES = GIB
WARMUPS = 2
REPS = 10
MIN_FREE = 4 * GIB
# Hand-picked to separate NVLink from PCIe: a PCIe Gen5 x16 link tops out near 64 GB/s, and an
# H100's NVLink near 300-400 GB/s one way, so 200 GB/s sits well clear of both.
PEER_COPY_MIN_GBPS = 200.0

COMMAND_TIMEOUT_S = 60.0
NVBANDWIDTH_TIMEOUT_S = 300.0
COMMANDS: dict[str, list[str]] = {
    "topo": ["nvidia-smi", "topo", "-m"],
    "p2p_read": ["nvidia-smi", "topo", "-p2p", "r"],
    "p2p_write": ["nvidia-smi", "topo", "-p2p", "w"],
    "nvlink": ["nvidia-smi", "nvlink", "-s"],
    "gpus": ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version",
             "--format=csv"],
}
NVBANDWIDTH_TESTS = ("device_to_device_memcpy_read_ce", "device_to_device_memcpy_write_ce")

Run = Callable[..., Any]


# ---- commands and their text --------------------------------------------------------------


def run_command(argv: Sequence[str], timeout_s: float, run: Run = subprocess.run
                ) -> dict[str, Any]:
    """``argv``'s outcome as a record: ``ok``, ``returncode``, the raw ``text`` (stdout, then
    stderr) and ``error`` (a missing tool, a timeout, or a non-zero exit). Never raises."""
    rec: dict[str, Any] = {"argv": list(argv), "ok": False, "returncode": None, "text": "",
                           "error": None}
    try:
        r = run(list(argv), capture_output=True, text=True, timeout=timeout_s, check=False)
    except FileNotFoundError:
        rec["error"] = f"{argv[0]} is not installed"
        return rec
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        rec.update(text=out, error=f"timed out after {timeout_s:g} s")
        return rec
    except Exception as e:  # a probe records what it can't run
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec.update(returncode=r.returncode, text=(r.stdout or "") + (r.stderr or ""))
    rec["ok"] = r.returncode == 0
    if r.returncode != 0:
        rec["error"] = f"exit code {r.returncode}"
    return rec


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_GPU = re.compile(r"GPU\d+$")


def parse_gpu_matrix(text: str) -> dict[str, dict[str, str]] | None:
    """``matrix[row][col]`` of the GPU-by-GPU block of ``nvidia-smi topo -m`` or ``topo -p2p``
    output (``"X"`` on the diagonal), or ``None`` when no such block is found. The header's
    leading ``GPU<n>`` columns define the block; NIC columns, the affinity columns and the
    legend are ignored."""
    cols: list[str] | None = None
    matrix: dict[str, dict[str, str]] = {}
    for line in _ANSI.sub("", text).splitlines():
        tok = line.split()
        if not tok:
            continue
        if cols is None:
            if len(tok) > 1 and tok[0] == "GPU0" and _GPU.match(tok[1]):
                cols = []
                for t in tok:
                    if not _GPU.match(t):
                        break
                    cols.append(t)
            continue
        if _GPU.match(tok[0]) and len(tok) > len(cols):
            matrix[tok[0]] = dict(zip(cols, tok[1:1 + len(cols)], strict=True))
    return matrix if cols is not None and len(matrix) == len(cols) else None


def _pairs(names: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in names for b in names if a != b]


# ---- the verdict --------------------------------------------------------------------------


def hw2_verdict(topo: Mapping[str, Mapping[str, str]] | None,
                p2p_read: Mapping[str, Mapping[str, str]] | None,
                pair_gbps: Mapping[str, float | None] | None) -> dict[str, Any]:
    """F22's HW2: ``nv_links``, ``nv18_everywhere``, ``p2p_read_ok``, ``peer_copy_min_gbps``
    (the slowest ordered pair, ``None`` when none was measured) against
    :data:`PEER_COPY_MIN_GBPS`. ``pass`` needs all four; ``reasons`` names each failed one."""
    reasons: list[str] = []
    nv_links = nv18 = p2p_ok = False
    if topo is None:
        reasons.append("nvidia-smi topo -m gave no GPU matrix")
    else:
        pairs = _pairs(list(topo))
        labels = {(a, b): topo[a].get(b) for a, b in pairs}
        nv_links = bool(pairs) and all(re.fullmatch(r"NV\d+", str(v)) for v in labels.values())
        nv18 = bool(pairs) and all(v == "NV18" for v in labels.values())
        odd = [f"{a}-{b} is {v}" for (a, b), v in labels.items() if a < b
               and not re.fullmatch(r"NV\d+", str(v))]
        if not nv_links:
            reasons.append("not every GPU pair is joined by NVLink: "
                           + (", ".join(odd[:6]) or "fewer than two GPUs"))
        elif not nv18:
            few = [f"{a}-{b} is {v}" for (a, b), v in labels.items() if a < b and v != "NV18"]
            reasons.append("not every pair has 18 NVLink links: " + ", ".join(few[:6]))
    if p2p_read is None:
        reasons.append("nvidia-smi topo -p2p r gave no GPU matrix")
    else:
        pairs = _pairs(list(p2p_read))
        p2p_ok = bool(pairs) and all(p2p_read[a].get(b) == "OK" for a, b in pairs)
        if not p2p_ok:
            bad = [f"{a}->{b} is {p2p_read[a].get(b)}" for a, b in pairs
                   if p2p_read[a].get(b) != "OK"]
            reasons.append("peer reads are not OK for every pair: " + ", ".join(bad[:6]))
    rates = [v for v in (pair_gbps or {}).values() if v is not None]
    low = min(rates) if rates else None
    if low is None:
        reasons.append("no peer-copy bandwidth was measured")
    elif low < PEER_COPY_MIN_GBPS:
        slow = min((v, k) for k, v in (pair_gbps or {}).items() if v is not None)[1]
        reasons.append(f"the slowest pair ({slow}) copied at {low:.0f} GB/s, under "
                       f"{PEER_COPY_MIN_GBPS:g} GB/s")
    elif pair_gbps is not None and len(rates) < len(pair_gbps):
        reasons.append("some pairs were not measured")
    return {"nv_links": nv_links, "nv18_everywhere": nv18, "p2p_read_ok": p2p_ok,
            "peer_copy_min_gbps": low, "threshold_gbps": PEER_COPY_MIN_GBPS,
            "pass": not reasons and nv_links and nv18 and p2p_ok, "reasons": reasons}


# ---- torch's view and the copies ----------------------------------------------------------


def _cuda_ok() -> tuple[Any, str | None]:
    try:
        import torch
    except ImportError:
        return None, "torch is not installed"
    if not torch.cuda.is_available():
        return torch, "torch.cuda.is_available() is false"
    return torch, None


def torch_view() -> dict[str, Any]:
    """torch's device count, each card's name and total memory, and ``can_device_access_peer``
    for every ordered pair (``"i->j"``)."""
    torch, why = _cuda_ok()
    if why:
        return {"skipped": why, "device_count": 0}
    n = torch.cuda.device_count()
    return {"device_count": n,
            "devices": [{"index": i, "name": torch.cuda.get_device_name(i),
                         "total_memory": torch.cuda.get_device_properties(i).total_memory}
                        for i in range(n)],
            "can_access_peer": {f"{i}->{j}": bool(torch.cuda.can_device_access_peer(i, j))
                                for i in range(n) for j in range(n) if i != j}}


def _sync_all(torch: Any, n: int) -> None:
    for d in range(n):
        torch.cuda.synchronize(d)


def _free(torch: Any, n: int) -> None:
    for d in range(n):
        with torch.cuda.device(d):
            torch.cuda.empty_cache()


def _short(torch: Any, devices: Sequence[int]) -> str | None:
    for d in devices:
        free = torch.cuda.mem_get_info(d)[0]
        if free < MIN_FREE:
            return f"cuda:{d} has {free / GIB:.1f} GiB free, under {MIN_FREE // GIB} GiB"
    return None


def peer_copy_bandwidth(nbytes: int = COPY_BYTES, reps: int = REPS) -> dict[str, Any]:
    """GB/s (1 GB = 1e9 bytes) of ``reps`` timed ``cuda:j.copy_(cuda:i)`` of ``nbytes``, for each
    ordered pair ``"i->j"``, after warm-ups, with both devices synchronized before and after."""
    torch, why = _cuda_ok()
    if why:
        return {"skipped": why, "pairs": {}}
    n = torch.cuda.device_count()
    pairs: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            key = f"{i}->{j}"
            short = _short(torch, (i, j))
            if short:
                pairs[key], notes[key] = None, f"skipped: {short}"
                continue
            try:
                src = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{i}")
                dst = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{j}")
                for _ in range(WARMUPS):
                    dst.copy_(src)
                _sync_all(torch, n)
                t0 = time.perf_counter()
                for _ in range(reps):
                    dst.copy_(src)
                _sync_all(torch, n)
                pairs[key] = reps * nbytes / (time.perf_counter() - t0) / 1e9
            except Exception as e:
                pairs[key], notes[key] = None, f"{type(e).__name__}: {e}"
            finally:
                src = dst = None
                _free(torch, n)
    return {"bytes": nbytes, "reps": reps, "pairs": pairs, "notes": notes}


def fanout_bandwidth(nbytes: int = COPY_BYTES, reps: int = REPS) -> dict[str, Any]:
    """GPU 0 to every other card at once, one stream per target: the aggregate GB/s and each
    target's (its bytes over the shared timed window)."""
    torch, why = _cuda_ok()
    if why:
        return {"skipped": why}
    n = torch.cuda.device_count()
    if n < 2:
        return {"skipped": f"{n} CUDA device"}
    targets = list(range(1, n))
    short = _short(torch, range(n))
    if short:
        return {"skipped": short}
    try:
        src = torch.empty(nbytes, dtype=torch.uint8, device="cuda:0")
        dsts = [torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{j}") for j in targets]
        streams = [torch.cuda.Stream(device=0) for _ in targets]

        def go(count: int) -> None:
            for s, d in zip(streams, dsts, strict=True):
                with torch.cuda.stream(s):
                    for _ in range(count):
                        d.copy_(src, non_blocking=True)

        go(WARMUPS)
        _sync_all(torch, n)
        t0 = time.perf_counter()
        go(reps)
        _sync_all(torch, n)
        dt = time.perf_counter() - t0
        per = reps * nbytes / dt / 1e9
        return {"bytes": nbytes, "reps": reps, "targets": targets,
                "aggregate_gbps": per * len(targets),
                "per_target_gbps": {f"0->{j}": per for j in targets}}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        src = dsts = streams = None
        _free(torch, n)


def pinned_bandwidth(nbytes: int = COPY_BYTES, reps: int = REPS) -> dict[str, Any]:
    """Pinned host memory to each card (PCIe), GB/s, for contrast with the peer copies."""
    torch, why = _cuda_ok()
    if why:
        return {"skipped": why, "cards": {}}
    n = torch.cuda.device_count()
    cards: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    host = None
    try:
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        for j in range(n):
            short = _short(torch, (j,))
            if short:
                cards[f"host->{j}"], notes[f"host->{j}"] = None, f"skipped: {short}"
                continue
            try:
                dst = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{j}")
                for _ in range(WARMUPS):
                    dst.copy_(host, non_blocking=True)
                _sync_all(torch, n)
                t0 = time.perf_counter()
                for _ in range(reps):
                    dst.copy_(host, non_blocking=True)
                _sync_all(torch, n)
                cards[f"host->{j}"] = reps * nbytes / (time.perf_counter() - t0) / 1e9
            except Exception as e:
                cards[f"host->{j}"], notes[f"host->{j}"] = None, f"{type(e).__name__}: {e}"
            finally:
                dst = None
                _free(torch, n)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "cards": cards}
    finally:
        host = None
    return {"bytes": nbytes, "reps": reps, "cards": cards, "notes": notes}


def nvbandwidth(run: Run = subprocess.run, which: Callable[[str], str | None] = shutil.which
                ) -> dict[str, Any] | str:
    """The raw text of ``nvbandwidth``'s two device-to-device tests, or ``"unavailable"`` when
    the tool isn't on the PATH."""
    if which("nvbandwidth") is None:
        return "unavailable"
    return {t: run_command(["nvbandwidth", "-t", t], NVBANDWIDTH_TIMEOUT_S, run)
            for t in NVBANDWIDTH_TESTS}


# ---- the report ---------------------------------------------------------------------------


def build_report(run: Run = subprocess.run) -> dict[str, Any]:
    """Run everything and assemble the report (module docstring)."""
    commands = {k: run_command(v, COMMAND_TIMEOUT_S, run) for k, v in COMMANDS.items()}
    topo = parse_gpu_matrix(commands["topo"]["text"])
    p2p_read = parse_gpu_matrix(commands["p2p_read"]["text"])
    p2p_write = parse_gpu_matrix(commands["p2p_write"]["text"])
    torch_doc = torch_view()
    peer = peer_copy_bandwidth()
    report = {
        "version": VERSION, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commands": commands,
        "topo": topo, "p2p_read": p2p_read, "p2p_write": p2p_write,
        "torch": torch_doc,
        "peer_copy": peer,
        "fanout": fanout_bandwidth(),
        "pinned_host": pinned_bandwidth(),
        "nvbandwidth": nvbandwidth(run),
    }
    report["hw2"] = hw2_verdict(topo, p2p_read, peer.get("pairs") or None)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", type=Path, required=True, help="the JSON report")
    args = p.parse_args(argv)
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    hw2 = report["hw2"]
    print(f"HW2 {'pass' if hw2['pass'] else 'FAIL'}: " + ("; ".join(hw2["reasons"]) or "ok")
          + f" (slowest pair {hw2['peer_copy_min_gbps']} GB/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
