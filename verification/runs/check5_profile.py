"""O2: check 5's host overhead per product, on the stored step 4 (PERFORMANCE_TASKS.md O2).

    .venv/bin/python -m verification.runs.check5_profile [--run-dir DIR] [--step N]
        [--repeats N] [--json] [--out PATH]

Set up as ``k_tunability`` is: the honest run's stored step read back (``load_kept_step``),
the full step run once at the configured ``k`` (it must accept), then the verifier's own
``checks.check_5_matmuls`` (``k_tunability.check5_pass``) on what check 2's read left. It
judges nothing and writes only the line below; it prints, per product and per section
(``5.glue``, ``5.measure``, and ``other``, the rest of check 5):

1. **Time.** ``--repeats`` timed passes (default 5), the median. On CUDA the device is
   synchronized before and after each pass, so the wall clock holds the GPU's work; the
   section times are host times, which on CUDA can run ahead of the device.
2. **Ops and host reads** (one untimed pass under a ``TorchDispatchMode``). Every aten op the
   dispatcher sees, on every thread (the replay's backward runs on autograd's workers). A host
   read is any call that makes the host wait for a value: ``aten._local_scalar_dense``
   (``.item()``, ``float(t)``, ``bool(t)``), ``Tensor.tolist`` and ``Tensor.numpy``, and a
   copy from a non-CPU tensor to a CPU one (``_to_copy``, ``copy_``). A read inside
   ``tolist`` or ``numpy`` counts once. On CPU none of them waits; they are counted because
   each is a synchronization on a GPU.
3. **CUDA launches and syncs** (CUDA only, one untimed pass under ``torch.profiler``): kernel
   launches (the runtime's ``cudaLaunchKernel``/``cuLaunchKernel`` calls), the kernels
   themselves and their summed device time (the GPU's busy time), and the runtime calls that
   block the host (``cudaStreamSynchronize``, ``cudaDeviceSynchronize``,
   ``cudaEventSynchronize``, ``cudaMemcpy``) and ``cudaMemcpyAsync`` apart. Each runtime call
   goes to the section whose time range holds its start, whatever its thread.

The step must be accepted, and its check-5 numbers are compared with the archive's (equal bit
for bit on the machine that wrote it; another device rounds differently, so a difference
there is reported, not an error). ``M`` splits into products measured alone (``dense``) and
members of a batched attention product (``member``); the counts are per product of all ``M``.

Each call appends one JSON line to ``--out`` (default ``<run dir>/check5_profile.jsonl``): the
``--json`` record, ``machine_facts`` and ``section_s_per_pass``, each section's time on every
timed pass.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import statistics
import struct
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from setup.config import load_config, setup_determinism
from setup.machine import machine_facts
from verification.computation.instances.decoder import DecoderComputation
from verification.computation.interface import DeclaredComputation
from verification.parameters import load_protocol_config
from verification.runs.k_tunability import base_context, check5_pass, full_step
from verification.runs.llama_step import load_committed_dataset
from verification.runs.run_verified import (
    KEEP_STEP,
    KeptStep,
    honest_run_name,
    load_kept_step,
    stored_band_hash,
)
from verification.transcript.store import TranscriptStore
from verification.verifier.bands import Bands
from verification.verifier.calibration import band_context, load_bands
from verification.verifier.context import StepStats

__all__ = ["OTHER", "Counts", "Profile", "SectionTracker", "count_ops", "cuda_counts",
           "main", "profile", "profile_step", "record_line", "stats_digest"]

DEFAULT_REPEATS = 5
OTHER = "other"  # check 5 outside its two named sections
SECTIONS = ("5.glue", "5.measure", OTHER)
_HOST_READS = ("tolist", "numpy")
_LAUNCHES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx")
_SYNCS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
          "cudaMemcpy")
_ASYNC_COPIES = ("cudaMemcpyAsync",)


class SectionTracker:
    """The verifier's metrics seam (``StepContext.section``): host time per section, and the
    section open now, which the counters read from any thread. On CUDA each section is also a
    ``record_function`` range, so a profile can place runtime calls in it."""

    def __init__(self, *, ranges: bool = False) -> None:
        self.current: str | None = None
        self.time_s: Counter[str] = Counter()
        self.calls: Counter[str] = Counter()
        self.ranges = ranges

    @contextlib.contextmanager
    def _section(self, name: str) -> Iterator[None]:
        prev, self.current = self.current, name
        rf = (torch.profiler.record_function(f"section:{name}") if self.ranges
              else contextlib.nullcontext())
        t0 = time.perf_counter()
        try:
            with rf:
                yield
        finally:
            self.time_s[name] += time.perf_counter() - t0
            self.calls[name] += 1
            self.current = prev

    def __call__(self, name: str) -> contextlib.AbstractContextManager[None]:
        return self._section(name)

    @property
    def section(self) -> str:
        return self.current or OTHER


@dataclass
class Counts:
    """Aten ops and host reads per section, and the ops by name."""

    ops: Counter[str] = field(default_factory=Counter)
    reads: Counter[str] = field(default_factory=Counter)
    kinds: Counter[tuple[str, str]] = field(default_factory=Counter)


def _is_device_to_host(func: Any, args: Sequence[Any], kwargs: dict[str, Any],
                       out: Any) -> bool:
    name = func.overloadpacket.__name__
    if name == "_to_copy":
        src = args[0] if args else None
        return (isinstance(src, torch.Tensor) and src.device.type != "cpu"
                and isinstance(out, torch.Tensor) and out.device.type == "cpu")
    if name == "copy_":
        dst, src = (args + (None, None))[:2]
        return (isinstance(dst, torch.Tensor) and dst.device.type == "cpu"
                and isinstance(src, torch.Tensor) and src.device.type != "cpu")
    return False


class _Count(TorchDispatchMode):
    def __init__(self, tracker: SectionTracker, counts: Counts, inside: threading.local
                 ) -> None:
        super().__init__()
        self.tracker, self.counts, self.inside = tracker, counts, inside

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        sec = self.tracker.section
        name = func.overloadpacket.__name__
        self.counts.ops[sec] += 1
        self.counts.kinds[(sec, name)] += 1
        if not getattr(self.inside, "read", False) and (
                name == "_local_scalar_dense" or _is_device_to_host(func, args, kwargs, out)):
            self.counts.reads[sec] += 1
        return out


@contextlib.contextmanager
def count_ops(tracker: SectionTracker) -> Iterator[Counts]:
    """Count aten ops and host reads per ``tracker`` section for the duration of the block.
    ``Tensor.tolist`` and ``Tensor.numpy`` are wrapped for the block and count as one read
    each, with whatever they dispatch inside them counted as ops only."""
    counts = Counts()
    inside = threading.local()
    orig = {n: getattr(torch.Tensor, n) for n in _HOST_READS}

    def wrap(n: str) -> Callable[..., Any]:
        f = orig[n]

        def g(self: torch.Tensor, *a: Any, **kw: Any) -> Any:
            sec = tracker.section
            counts.reads[sec] += 1
            counts.kinds[(sec, f"Tensor.{n}")] += 1
            inside.read = True
            try:
                return f(self, *a, **kw)
            finally:
                inside.read = False
        return g

    for n in _HOST_READS:
        setattr(torch.Tensor, n, wrap(n))
    try:
        with _Count(tracker, counts, inside):
            yield counts
    finally:
        for n, f in orig.items():
            setattr(torch.Tensor, n, f)


def _attribute(ranges: list[tuple[float, float, str]], t: float) -> str:
    """The innermost section whose time range holds ``t``, else ``OTHER``."""
    best: tuple[float, str] | None = None
    for start, end, name in ranges:
        if start <= t <= end and (best is None or end - start < best[0]):
            best = (end - start, name)
    return best[1] if best is not None else OTHER


def cuda_counts(events: Sequence[Any]) -> dict[str, Counter[str]]:
    """From a profile's events: per section, ``launches``, ``syncs`` and ``async_copies``
    (runtime calls placed by start time), and in all ``kernels`` and ``kernel_us``."""
    ranges = [(e.time_range.start, e.time_range.end, e.name.removeprefix("section:"))
              for e in events if e.name.startswith("section:")]
    out: dict[str, Counter[str]] = {s: Counter() for s in SECTIONS}
    total: Counter[str] = Counter()
    for e in events:
        if getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA:
            total["kernels"] += 1
            total["kernel_us"] += int(e.time_range.end - e.time_range.start)
            continue
        key = ("launches" if e.name in _LAUNCHES else "syncs" if e.name in _SYNCS
               else "async_copies" if e.name in _ASYNC_COPIES else None)
        if key is not None:
            out.setdefault(_attribute(ranges, e.time_range.start), Counter())[key] += 1
    out["total"] = total
    return out


def stats_digest(st: StepStats) -> str:
    """BLAKE2b-128 over every check-5 number of ``st``, product by product."""
    h = hashlib.blake2b(digest_size=16)
    for p in st.products:
        h.update(struct.pack("<id", p.m, p.kappa))
        h.update(struct.pack(f"<{len(p.normalized)}d", *p.normalized))
        h.update(struct.pack("<idddd", p.q, p.p_norm, p.nu, p.p_abs1, p.u))
    return h.hexdigest()


@dataclass
class Profile:
    M: int
    dense: int
    member: int
    accepted: bool
    digest: str
    archive_equal: bool | None
    check5_s: list[float]
    section_s: dict[str, float]  # medians
    section_calls: dict[str, int]
    ops: dict[str, int]
    reads: dict[str, int]
    top_ops: dict[str, list[tuple[str, int]]]
    cuda: dict[str, dict[str, int]] | None
    section_s_per_pass: dict[str, list[float]] = field(default_factory=dict)  # file only

    def as_record(self) -> dict[str, Any]:
        return {"record": "check5_profile",
                **{k: v for k, v in self.__dict__.items() if k != "section_s_per_pass"}}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile(c: DeclaredComputation, kept: KeptStep, bands: Bands, *, seed: int, h_D: bytes,
            n_records: int, k: int, repeats: int = DEFAULT_REPEATS, device: str = "cpu",
            out: Callable[[str], None] = print) -> Profile:
    """The three passes of the module docstring on ``kept``, after one full step."""
    if (bh := stored_band_hash(kept.run_dir)) != bands.source:
        raise ValueError(f"{kept.run_dir} was judged by band file {bh}, not this run's "
                         f"{bands.source}")
    return profile_step(c, kept.store, kept.step, kept.honest, bands, seed=seed, h_D=h_D,
                        n_records=n_records, k=k, repeats=repeats, device=device, out=out)


def profile_step(c: DeclaredComputation, store: TranscriptStore, step: int,
                 honest: StepStats | None, bands: Bands, *, seed: int, h_D: bytes,
                 n_records: int, k: int, repeats: int = DEFAULT_REPEATS, device: str = "cpu",
                 out: Callable[[str], None] = print) -> Profile:
    """:func:`profile` on a step held in memory, with no band-hash check: ``bands`` are the
    caller's own (``bring_up``'s in-sample fit has no stored hash). ``honest`` is the run's
    archive for the step, or ``None``."""
    dev = torch.device(device)
    base = base_context(c, store, step=step, seed=seed, h_D=h_D, n_records=n_records, k=k)
    t0 = time.perf_counter()
    rej, slots, ready = full_step(store, c, base, bands)
    if rej is not None:
        raise RuntimeError(f"the stored step {step} was rejected: {rej}")
    out(f"full step: accepted in {time.perf_counter() - t0:.2f} s; per slot "
        + ", ".join(f"{s} {v:.3f}" for s, v in slots.items()))

    times: list[float] = []
    per_section: dict[str, list[float]] = {s: [] for s in SECTIONS}
    accepted, st = True, StepStats()
    for _ in range(repeats):
        tr = SectionTracker()
        _sync(dev)
        t0 = time.perf_counter()
        rej, st = check5_pass(store, c, ready, bands, k, section=tr)
        _sync(dev)
        times.append(dt := time.perf_counter() - t0)
        accepted &= rej is None
        named = 0.0
        for s in ("5.glue", "5.measure"):
            per_section[s].append(tr.time_s[s])
            named += tr.time_s[s]
        per_section[OTHER].append(dt - named)
    digest = stats_digest(st)
    archive = None if honest is None else stats_digest(honest) == digest
    section_calls = dict(tr.calls)

    tr = SectionTracker()
    with count_ops(tr) as counts:
        rej, _ = check5_pass(store, c, ready, bands, k, section=tr)
    accepted &= rej is None

    cuda = None
    if dev.type == "cuda":
        tr = SectionTracker(ranges=True)
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        _sync(dev)
        with torch.profiler.profile(activities=acts) as prof:
            rej, _ = check5_pass(store, c, ready, bands, k, section=tr)
            _sync(dev)
        accepted &= rej is None
        cuda = {s: dict(v) for s, v in cuda_counts(prof.events()).items()}

    member = sum(1 for s in c.products if s.member is not None)
    top = {s: [(n, v) for (sec, n), v in counts.kinds.most_common() if sec == s][:15]
           for s in SECTIONS}
    return Profile(M=c.M, dense=c.M - member, member=member, accepted=accepted, digest=digest,
                   archive_equal=archive, check5_s=times,
                   section_s={s: statistics.median(v) for s, v in per_section.items()},
                   section_s_per_pass=per_section,
                   section_calls=section_calls, ops=dict(counts.ops), reads=dict(counts.reads),
                   top_ops=top, cuda=cuda)


def record_line(r: Profile, *, model: str, device: str, k: int, step: int) -> dict[str, Any]:
    """The line ``--out`` appends: the ``--json`` record, the run's identity, each section's
    time per pass and the machine's facts."""
    return {**r.as_record(), "model": model, "device": device, "k": k, "step": step,
            "section_s_per_pass": r.section_s_per_pass, "machine_facts": machine_facts()}


def report(p: Profile, out: Callable[[str], None] = print) -> None:
    M = p.M
    med = statistics.median(p.check5_s)
    out(f"check 5: {M} products ({p.dense} dense, {p.member} member), "
        f"{'accepted' if p.accepted else 'REJECTED'}, digest {p.digest}, "
        + {None: "no archive", True: "equal to the archive",
           False: "different from the archive"}[p.archive_equal])
    out(f"wall: median {med:.3f} s, {med / M * 1e6:.1f} µs/product "
        f"(all {[round(t, 3) for t in p.check5_s]})")
    rows = ("section", "time s", "µs/prod", "calls", "ops", "ops/prod", "reads", "reads/prod")
    out("{:<10} {:>8} {:>8} {:>7} {:>9} {:>8} {:>7} {:>10}".format(*rows))
    for s in (*SECTIONS, "total"):
        if s == "total":
            t, calls = med, sum(p.section_calls.values())
            ops, reads = sum(p.ops.values()), sum(p.reads.values())
        else:
            t, calls = p.section_s[s], p.section_calls.get(s, 0)
            ops, reads = p.ops.get(s, 0), p.reads.get(s, 0)
        out(f"{s:<10} {t:>8.3f} {t / M * 1e6:>8.1f} {calls:>7} {ops:>9} {ops / M:>8.2f} "
            f"{reads:>7} {reads / M:>10.3f}")
    if p.cuda is not None:
        tot = p.cuda["total"]
        out(f"CUDA: {tot.get('kernels', 0)} kernels ({tot.get('kernels', 0) / M:.2f}/product), "
            f"device busy {tot.get('kernel_us', 0) / 1e6:.3f} s")
        for s in SECTIONS:
            r = p.cuda.get(s, {})
            out(f"  {s:<10} launches {r.get('launches', 0):>8} "
                f"({r.get('launches', 0) / M:.2f}/product), syncs {r.get('syncs', 0):>7} "
                f"({r.get('syncs', 0) / M:.3f}/product), async copies "
                f"{r.get('async_copies', 0)}")
    for s in SECTIONS:
        out(f"top ops in {s}: {p.top_ops[s]}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run-dir", type=Path, default=None,
                   help="the honest run directory of the stored step (default "
                        "$VERIF_OUTPUT_DIR/honest-<model>-<attack>-bpr0-s<seed>)")
    p.add_argument("--step", type=int, default=KEEP_STEP, help="the stored step")
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                   help="timed passes; the median")
    p.add_argument("--json", action="store_true", help="also print one JSON line")
    p.add_argument("--out", type=Path, default=None,
                   help="append the JSON line, with the machine's facts and each section's "
                        "time per pass, to this file (default: <run dir>/check5_profile.jsonl)")
    args = p.parse_args(argv)
    if args.repeats < 1:
        p.error("need --repeats ≥ 1")
    cfg = load_config()
    pc = load_protocol_config()
    setup_determinism(cfg)
    c = DecoderComputation.from_config(cfg)
    D, h_D = load_committed_dataset(c, cfg.data_dir)
    bands = load_bands(pc.band_file, k=pc.k, context=band_context(cfg, c, h_D=h_D, k=pc.k))
    run_dir = args.run_dir or cfg.output_dir / honest_run_name(cfg.model, cfg.attack, cfg.seed)
    print(f"O2 check-5 profile: {cfg.model}@{cfg.model_revision[:12]}, device {cfg.device}, "
          f"k {pc.k}, step {args.step}, {torch.get_num_threads()} threads")
    t0 = time.perf_counter()
    kept = load_kept_step(c, run_dir, step=args.step)
    print(f"stored step {args.step}: {kept.path}, read and re-hashed in "
          f"{time.perf_counter() - t0:.1f} s")
    r = profile(c, kept, bands, seed=cfg.seed, h_D=h_D, n_records=len(D), k=pc.k,
                repeats=args.repeats, device=cfg.device)
    report(r)
    rec = {**r.as_record(), "model": cfg.model, "device": cfg.device, "k": pc.k,
           "step": args.step}
    if args.json:
        print(json.dumps(rec))
    out = args.out or run_dir / "check5_profile.jsonl"
    with open(out, "a") as f:
        f.write(json.dumps(record_line(r, model=cfg.model, device=cfg.device, k=pc.k,
                                       step=args.step)) + "\n")
    return 0 if r.accepted else 1


if __name__ == "__main__":
    sys.exit(main())
