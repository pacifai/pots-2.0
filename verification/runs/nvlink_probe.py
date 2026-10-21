"""FS31, part A: is NVLink exposed between the cards (F22's check HW2)?

    .venv/bin/python -m verification.runs.nvlink_probe --out nvlink.json

Writes one JSON report with the machine's own account of its links (``nvidia-smi topo -m``,
``topo -p2p r`` and ``-p2p w``, ``nvlink -s`` and the card list, each with its raw text), torch's
view (device count, names, memory, ``can_device_access_peer`` per ordered pair), the measured
bandwidth of a 1 GiB device-to-device copy for every ordered pair, the GPU 0 fan-out to the
others, pinned host memory to each card for contrast, and ``nvbandwidth``'s device-to-device
tests when that tool is already on the PATH (never downloaded or built). A command that fails,
times out or is missing is recorded, not raised, and the measures that follow still run.

The verdict ``hw2`` (:func:`hw2_verdict`) checks that the topology matrix shows ``FS31_CARDS``
cards (default 4), that every pair is joined by exactly ``NV18`` (the H100 SXM's link count), that
peer reads are ``OK`` for every pair, that every ordered pair has a finite, positive copy rate,
and that every ordered pair's direct copies land the right bytes (``peer_bytes_ok``: F22's
``peer_check``, run here on the same cards; on the FS32 two-A40 session every direct copy left wrong
bytes though the peer reads said ``OK``, so a rate over a wrong copy is no rate). A pair with no
byte result counts as wrong.
The slowest pair's rate, by the PyTorch copy timing, sets its ``verdict``: ``pass`` at
:data:`PEER_COPY_PASS_GBPS` or more, ``review`` from :data:`PEER_COPY_FAIL_GBPS` up to that (the
number returns to the user before the port relies on it), ``fail`` below it or on any other
failed check. ``nvbandwidth``'s text is recorded beside it and never enters the verdict. The
probe exits 0 once the report is written, whatever the verdict; a non-zero exit means the probe
itself broke. A measure that raises is recorded under its key, and the report is still written.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["PEER_COPY_PASS_GBPS", "PEER_COPY_FAIL_GBPS", "GATE_CARDS", "COMMANDS", "expected_cards",
           "hw2_line", "run_command", "parse_gpu_matrix", "hw2_verdict",
           "peer_copy_bandwidth", "peer_bytes_by_pair", "fanout_bandwidth", "pinned_bandwidth",
           "torch_view", "build_report", "main"]

VERSION = 1
GIB = 1 << 30
COPY_BYTES = GIB
WARMUPS = 2
REPS = 10
MIN_FREE = 4 * GIB
# The user's three bands for the slowest pair (2026-10-20): a PCIe Gen5 x16 link tops out near
# 64 GB/s and an H100's NVLink near 300-400 GB/s one way. At 300 or more the pair passes; from
# 150 up to 300 the number returns to the user; under 150 it fails.
PEER_COPY_PASS_GBPS = 300.0
PEER_COPY_FAIL_GBPS = 150.0
# F22 specifies four cards; ``FS31_CARDS`` below that is a rehearsal, never the gate.
GATE_CARDS = 4
CARDS_ENV = "FS31_CARDS"

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
    """``argv``'s outcome as a record: ``ok``, ``returncode``, ``stdout`` and ``stderr`` apart,
    the raw ``text`` (both, for reading) and ``error`` (a missing tool, a timeout, or a non-zero
    exit). Only ``stdout`` is parsed, and only when ``ok``. Never raises."""
    rec: dict[str, Any] = {"argv": list(argv), "ok": False, "returncode": None, "text": "",
                           "stdout": "", "stderr": "", "error": None}
    try:
        r = run(list(argv), capture_output=True, text=True, timeout=timeout_s, check=False)
    except FileNotFoundError:
        rec["error"] = f"{argv[0]} is not installed"
        return rec
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        rec.update(text=out + err, stdout=out, stderr=err,
                   error=f"timed out after {timeout_s:g} s")
        return rec
    except Exception as e:  # a probe records what it can't run
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    out, err = r.stdout or "", r.stderr or ""
    rec.update(returncode=r.returncode, text=out + err, stdout=out, stderr=err)
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


def expected_cards(environ: Mapping[str, str] | None = None) -> int:
    """``FS31_CARDS`` (default 4); a value the session would have refused counts as 4."""
    raw = (os.environ if environ is None else environ).get(CARDS_ENV, "").strip()
    try:
        n = int(raw) if raw else GATE_CARDS
    except ValueError:
        return GATE_CARDS
    return n if n >= 1 else GATE_CARDS


def _gpu_index(name: str) -> str:
    return name[3:]


def _command_failure(rec: Mapping[str, Any]) -> str | None:
    if rec.get("ok"):
        return None
    return f"{' '.join(rec['argv'])} failed: {rec.get('error') or 'no result'}"


def hw2_verdict(topo: Mapping[str, Mapping[str, str]] | None,
                p2p_read: Mapping[str, Mapping[str, str]] | None,
                pair_gbps: Mapping[str, float | None] | None, n_expected: int | None = None,
                command_failures: Mapping[str, str | None] | None = None,
                peer_bytes: Mapping[str, bool | None] | None = None) -> dict[str, Any]:
    """F22's HW2. The checks: the topology matrix shows ``n_expected`` GPUs (default
    ``FS31_CARDS``; ``n_cards`` is what it shows), ``nv_links`` and ``nv18_everywhere`` for every
    pair, ``p2p_read_ok`` for every pair of the same GPUs, a finite positive rate for every
    ordered pair of those GPUs, and ``peer_bytes_ok`` (``"i->j"`` to true, from ``peer_bytes``: the
    direct copies of that pair landed the right bytes; a pair that is false or has no entry is a
    reason, and the verdict ``fail``). With all of them, the slowest pair's rate gives ``verdict``:
    ``pass`` at :data:`PEER_COPY_PASS_GBPS` or more, ``review`` from :data:`PEER_COPY_FAIL_GBPS`
    up to that, ``fail`` below it; any other failed check is ``fail``. ``gate`` is true only when
    ``n_expected`` is :data:`GATE_CARDS`; ``pass`` only for ``"pass"`` with ``gate`` (a rehearsal
    keeps its measured ``verdict`` and never has ``pass``). ``command_failures`` names a
    ``topo`` or ``p2p_read`` command that failed; its matrix is not to be passed in."""
    expected = expected_cards() if n_expected is None else n_expected
    fails = dict(command_failures or {})
    reasons: list[str] = []
    nv_links = nv18 = p2p_ok = False
    n_cards: int | None = None
    names: list[str] = []
    if topo is None:
        reasons.append(fails.get("topo") or "nvidia-smi topo -m gave no GPU matrix")
    else:
        names = list(topo)
        n_cards = len(names)
        if n_cards != expected:
            reasons.append(f"the topology matrix shows {n_cards} GPUs, {CARDS_ENV} is {expected}")
        pairs = _pairs(names)
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
        reasons.append(fails.get("p2p_read") or "nvidia-smi topo -p2p r gave no GPU matrix")
    else:
        pairs = _pairs(list(p2p_read))
        p2p_ok = bool(pairs) and all(p2p_read[a].get(b) == "OK" for a, b in pairs)
        if not p2p_ok:
            bad = [f"{a}->{b} is {p2p_read[a].get(b)}" for a, b in pairs
                   if p2p_read[a].get(b) != "OK"]
            reasons.append("peer reads are not OK for every pair: "
                           + (", ".join(bad[:6]) or "fewer than two GPUs"))
        if topo is not None and set(p2p_read) != set(topo):
            p2p_ok = False
            reasons.append(f"the peer-read matrix shows {sorted(p2p_read)}, the topology "
                           f"matrix {sorted(topo)}")
    rates = pair_gbps or {}
    wanted = [f"{_gpu_index(a)}->{_gpu_index(b)}" for a, b in _pairs(names)]
    good = {k: v for k, v in rates.items()
            if isinstance(v, (int, float)) and math.isfinite(v) and v > 0}
    bytes_by_pair = {k: v is True for k, v in (peer_bytes or {}).items()}
    if topo is not None:
        for k in wanted:
            bytes_by_pair.setdefault(k, False)
    wrong = sorted(k for k, v in bytes_by_pair.items() if not v)
    if not peer_bytes:
        reasons.append("no peer-copy byte check result (peer copies unverified)")
    elif wrong:
        reasons.append("peer copies landed wrong bytes, or were not checked, for pair "
                       + ", ".join(wrong[:6]))
    complete = False
    if topo is not None:
        missing = [k for k in wanted if k not in rates or rates[k] is None]
        invalid = [k for k in wanted if k in rates and rates[k] is not None and k not in good]
        if missing:
            reasons.append("no copy rate for pair " + ", ".join(missing[:6]))
        if invalid:
            reasons.append("the copy rate is not a finite positive number for pair "
                           + ", ".join(f"{k} ({rates[k]!r})" for k in invalid[:6]))
        complete = not missing and not invalid and bool(wanted)
    if not good and not rates:
        reasons.append("no peer-copy bandwidth was measured")
    scored = {k: good[k] for k in wanted if k in good} if topo is not None else good
    slow = min(scored.items(), key=lambda kv: kv[1]) if scored else None
    low = slow[1] if slow else None
    slowest = slow[0] if slow else None
    review_note = None
    if slow and complete:
        if low < PEER_COPY_FAIL_GBPS:
            reasons.append(f"the slowest pair ({slowest}) copied at {low:.0f} GB/s, under "
                           f"{PEER_COPY_FAIL_GBPS:g} GB/s")
        elif low < PEER_COPY_PASS_GBPS:
            review_note = (f"the slowest pair ({slowest}) copied at {low:.0f} GB/s, under the "
                           f"{PEER_COPY_PASS_GBPS:g} GB/s target (150-300 returns to the user)")
    ok = (not reasons and nv_links and nv18 and p2p_ok and complete and bool(peer_bytes)
          and not wrong)
    review = ok and review_note is not None
    if review:
        reasons.append(review_note)
    verdict = "fail" if not ok else ("review" if review else "pass")
    return {"verdict": verdict, "pass": verdict == "pass" and expected == GATE_CARDS,
            "n_cards": n_cards, "expected_cards": expected, "gate": expected == GATE_CARDS,
            "nv_links": nv_links, "nv18_everywhere": nv18, "p2p_read_ok": p2p_ok,
            "peer_bytes_ok": dict(sorted(bytes_by_pair.items())),
            "peer_copy_min_gbps": low, "slowest_pair": slowest,
            "pass_gbps": PEER_COPY_PASS_GBPS, "fail_gbps": PEER_COPY_FAIL_GBPS,
            "reasons": reasons}


def _floor(x: float) -> str:
    return f"{math.floor(x)}"


def hw2_line(hw2: Mapping[str, Any]) -> str:
    """The one-line account of an HW2 verdict: the band, the slowest pair and its rate, and the
    300 and 150 GB/s lines. A rehearsal's pass or review says so instead of passing."""
    verdict = hw2.get("verdict") or ("pass" if hw2.get("pass") else "fail")
    pair, low = hw2.get("slowest_pair"), hw2.get("peer_copy_min_gbps")
    hi, lo = hw2.get("pass_gbps", PEER_COPY_PASS_GBPS), hw2.get("fail_gbps", PEER_COPY_FAIL_GBPS)
    if pair is None or low is None:
        speed = f"no peer-copy rate (target {hi:g} GB/s, fail under {lo:g})"
    elif low >= hi:
        speed = f"slowest pair {pair} at {math.floor(low)} GB/s, over the {hi:g} GB/s target"
    elif low >= lo:
        speed = (f"slowest pair {pair} at {math.floor(low)} GB/s, under the {hi:g} GB/s target "
                 f"({lo:g}-{hi:g} returns to the user)")
    else:
        speed = (f"slowest pair {pair} at {math.floor(low)} GB/s, under the {lo:g} GB/s fail "
                 f"line (target {hi:g})")
    if verdict != "fail" and hw2.get("gate") is False:
        n = hw2.get("expected_cards")
        return f"HW2 rehearsal ({n} of {GATE_CARDS} cards), not the gate: {speed}"
    head = {"pass": "HW2 pass", "review": "HW2 review", "fail": "HW2 FAIL"}[verdict]
    why = "; ".join(hw2.get("reasons") or []) if verdict == "fail" else ""
    wrong = sorted(k for k, v in (hw2.get("peer_bytes_ok") or {}).items() if not v)
    if wrong:  # said first and plainly: a rate over copies that landed wrong bytes means nothing
        speed = (f"peer copies were WRONG (or unchecked) on pair{'s' * (len(wrong) > 1)} "
                 f"{', '.join(wrong)}, so a rate is not a result; {speed}")
    return f"{head}: {speed}" + (f"; {why}" if why else "")


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
    """Release cached memory on every card; best effort, so a wedged card can't lose the rates
    already measured."""
    for d in range(n):
        try:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        except Exception:
            pass


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
            src = dst = None
            try:
                short = _short(torch, (i, j))
                if short:
                    pairs[key], notes[key] = None, f"skipped: {short}"
                    continue
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
    src = dsts = streams = None
    try:
        short = _short(torch, range(n))
        if short:
            return {"skipped": short}
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
            dst = None
            try:
                short = _short(torch, (j,))
                if short:
                    cards[f"host->{j}"], notes[f"host->{j}"] = None, f"skipped: {short}"
                    continue
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


def peer_bytes_by_pair() -> dict[str, bool]:
    """``peer_check.check`` on every visible CUDA device, as ``"i->j"`` to whether every case of
    that ordered pair landed the right bytes. Raises when there are fewer than two devices."""
    import torch

    from verification.runs.peer_check import check

    n = torch.cuda.device_count()
    if n < 2:
        raise RuntimeError(f"{n} CUDA devices visible; a peer copy needs two")
    return {k: bool(p["ok"]) for k, p in check([torch.device(f"cuda:{i}") for i in range(n)])
            ["pairs"].items()}


def _measure(fn: Callable[[], Any]) -> Any:
    """``fn()``'s result, or ``{"error": ...}`` when it raised, so one wedged card or a failed
    call can't lose the report."""
    try:
        return fn()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def build_report(run: Run = subprocess.run, n_expected: int | None = None) -> dict[str, Any]:
    """Run everything and assemble the report (module docstring). A matrix is parsed from a
    command's stdout only when the command succeeded."""
    commands = {k: run_command(v, COMMAND_TIMEOUT_S, run) for k, v in COMMANDS.items()}
    topo, p2p_read, p2p_write = (
        parse_gpu_matrix(commands[k]["stdout"]) if commands[k]["ok"] else None
        for k in ("topo", "p2p_read", "p2p_write"))
    peer = _measure(peer_copy_bandwidth)
    report = {
        "version": VERSION, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commands": commands,
        "topo": topo, "p2p_read": p2p_read, "p2p_write": p2p_write,
        "torch": _measure(torch_view),
        "peer_copy": peer,
        "fanout": _measure(fanout_bandwidth),
        "pinned_host": _measure(pinned_bandwidth),
        "nvbandwidth": _measure(lambda: nvbandwidth(run)),
    }
    pairs = peer.get("pairs") if isinstance(peer, dict) else None
    peer_bytes = _measure(peer_bytes_by_pair)
    report["peer_bytes"] = peer_bytes
    report["hw2"] = hw2_verdict(
        topo, p2p_read, pairs or None, n_expected,
        {"topo": _command_failure(commands["topo"]),
         "p2p_read": _command_failure(commands["p2p_read"])},
        None if "error" in peer_bytes else peer_bytes)
    if "error" in peer_bytes:
        report["hw2"]["reasons"].append(f"the peer-copy byte check raised: {peer_bytes['error']}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", type=Path, required=True, help="the JSON report")
    args = p.parse_args(argv)
    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(hw2_line(report["hw2"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
