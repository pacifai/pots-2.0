"""FS25: join the profiling files into ``profile.json`` and apply the bottleneck rule
(archive/PROFILING_PLAN §5.7, §6, §7).

Run as ``python -m evaluation.results.profile.analyze --run-dir D --session L [--markers M]
[--samples S] [--probe P] [--trace-summary T] [--plain D2] [--compute-dtype float32]
--out profile.json``. ``meta`` names the run (model, attack, scenario, device, compute dtype,
git commit, from the environment record) and the session ``L`` (archive/PROFILING_PLAN §5.7).
A missing markers, samples or probe file is a warning in ``profile.json``, not an error.
Beside what §5.7 lists, ``profile.json`` holds what the report (FS26) draws: ``probe`` and
``machine`` at the top, and a ``timeline`` per run (the untraced steps' spans and sampler
strips, at most 400 points per step).

**Units.** Marker times are µs from the step's ``t0_ns``, sample times are monotonic ns, and
time rows and every ``*_s`` value are seconds. :func:`_abs_ns` puts marker spans on the sample
clock and :data:`US` turns µs into s; nothing else converts.

**Choices the plan leaves open** (fixed by the orchestrator, 2026-10-19):

- GEMM floor rate: prover, plain and ``train.*`` sections use ``gemm_bf16_flops`` when the
  compute dtype is bfloat16 and ``gemm_fp32_flops`` otherwise; verifier sections use
  ``gemm_fp32_flops``; check 5's sections (``5``, ``5.*``) use ``skinny_bmm_fp32_flops``.
- The compute dtype is the environment record's ``compute_dtype``, else ``--compute-dtype``
  (default ``float32``); the fallback adds a warning.
- Usable GPU memory is ``gpu_memory_bytes`` of the machine block (``probe.json`` first, then
  the environment record); without it the memory mark is skipped, with a warning.
- ``trace_summary.json`` is ``{"version", "sections": {name: {...}}, "c5_slice": {class:
  {...}}, "warnings"}``. Sections join on the name string; the time rows' step total is the
  share's denominator.
- Trace work (launches, copy bytes, GEMM bytes) is the traced step's own values; check 5's
  sections (``5``, ``5.*``) scale by ``Σ scaled["run"] ÷ Σ phases_us["run"]`` over the classes
  of ``c5_slice`` (estimated total run time over traced run time).
- Copies cost at their own kind's rate (FS42; the 2026-10-19 rule priced every copy at the
  pageable rate, on the premise that the trace cannot tell pinned from pageable memory, but it
  can: it names each copy "Memcpy HtoD (Pinned -> Device)", "(Pageable -> Device)", "DtoH
  (Device -> Pinned)", "PtoP (Device -> Device)" for a GPU-to-GPU copy, "DtoD"). The trace
  summary's ``copy_kinds`` keeps the bytes by kind; each is priced at the probe's rate for it:
  ``h2d_pinned_bytes``, ``h2d_pageable_bytes``, ``d2h_pinned_bytes``, ``d2h_pageable_bytes``;
  peer copies (``nvlink`` floor) at the probe's ``peer`` rate of their device pair (the slowest
  pair's when the event names no pair), each pair on its own link, the busiest one setting the
  floor; ``hbm`` bytes are ``gemm_bytes`` plus the device-to-device copies (serving, before
  FS42 also the peer copies). A copy of unknown kind, or a trace summary without
  ``copy_kinds``, is priced pageable, with a warning. The PCIe floor is the larger of the two
  directions' times (the link is full duplex), each direction summing its pinned and pageable
  bytes. Section ``5`` on a CUDA run adds the measuring
  kernel's reads (``bytes.hbm_measure``, ``_measure_reads``): per layout entry A, B and P as
  stored (``operand_bytes`` is fp32, so it is rescaled to the product dtype), times
  once per step (no sweep factor) and not scaled by the trace slice; the section's
  ``floor_detail.hbm`` gives ``serving_s`` and ``measuring_reads_s`` apart. ``hash`` bytes are
  ``hash_in_bytes`` plus ``hash_out_bytes`` of the ``count`` row; ``leaves`` come from the
  ``storage`` record, for every section whose ``hash_in_bytes`` is at least half the step's
  ``transcript_bytes`` (the commit and check 2).
- ``floor_partial`` (FS42): on a CUDA run, a section with a floor but no trace entry has no
  trace-derived terms (memory traffic, PCIe, peer, launch): they are missing, not zero. Its
  ``floor_partial`` lists them (``missing``: ``hbm`` or, for check 5, whose floor holds the
  measuring reads, ``hbm serving``, then ``pcie``, ``nvlink``, ``launch``), ``floor_detail``
  shows them as null, and no bottleneck or utilization mark is set from the floor: its ratio
  and its predicted use are not comparable with a traced run's.
- Syncs are 2 per top-level section call on CUDA. The hook bound's events are the ``gpu_us``
  spans that hold events (with the folded calls), none for a stamp or a host-timed span
  (``seam.host``, null ``gpu_us`` entries: no events); both are 0 off CUDA.
- The launch floor needs positive evidence: it exists only for a section whose trace counts
  launches (then launches × ``launch_s`` + syncs × ``sync_s``). Without trace launches the
  syncs alone are no evidence of launch-bound work, so there is no launch floor, and a section
  with no floor at all has ``bound`` ``"unknown"`` (not ``"launch"``, and not null).
- Warm-up (2026-10-20): the lowest untraced step ≥ 1 is the warm-up when at least two untraced
  steps exist. It pays one-off costs (the pinned host-buffer pool, the first backward's
  kernels), so it is left out of every steady-state mean and share, and reported on its own as
  the view's ``warmup`` block (``step``, ``step_s`` and ``sections``, each section with the
  same fields as a steady one). ``warmup_excluded`` is false, and ``warmup`` null, when only
  one untraced step exists (it is then the steady state). The plain comparison drops each run's
  own warm-up the same way and shows both warm-up times beside the steady ones.
- Steady steps (FS42): ``--steady-steps A-B`` (or ``2,3``) names the steady set; each must be
  an untraced step ≥ 1 with rows (else a warning and it is left out), the warm-up is the lowest
  untraced step below the set, and every steady mean uses the set, the plain comparison too
  (a plain run lacking one keeps its own rule, with a warning). The view's ``steady_steps``
  lists the steps used. Not given, the rule above applies, with warnings when a pipelined
  run's set includes its last step (verified with the prover idle) or a steady step follows a
  traced one (its interval holds the trace's export).
- A section's hashed leaves are its ``count`` row's ``hash_leaves`` when the row has one (the
  CUDA commit hashes only the records, the weights and the dK leaves), else every leaf when it
  hashes at least half the transcript's bytes, else none.
- ``meta`` also holds ``n_s`` (the batch's sequences), ``M`` (products), ``n_leaves`` and
  ``us_per_leaf`` (``label`` for ``P1.label`` and ``commit`` for ``P3.commit``: steady mean
  seconds ÷ ``n_leaves`` × 10⁶). ``n_s`` and ``M`` come from the environment record, ``M``
  else from the largest ``capture`` ``n_products``; ``n_leaves`` from the environment record,
  else from the ``storage`` or ``step`` records. Missing values are null.
- Clock times are sums of a section's marker spans over the untraced steps; the trace time is
  compared with the host time of the traced step. ``agree`` is false when either differs by
  more than the rule's tolerance, and ``null`` when no second clock exists.
- Measured utilization on the bound's lane: ``gemm`` is ``g_tensor`` when the floor used the bf16
  rate, else ``g_sm_active`` (both falling back to ``g_util``; a measured 0 is a reading),
  ``hbm`` is ``g_dram_bw`` (else ``g_mem_util``), ``pcie`` is ``g_pcie_tx + g_pcie_rx``
  over the datasheet's ``pcie_bytes``, ``hash`` is ``c_tree`` over the probe's hash threads;
  ``launch`` has no lane, so it never gets the mark.
- Sampler fields are as FS21 writes them: utilization fields are fractions 0-1, so no
  scaling; a ``c_other`` below 0 is kept. Trace sections are inclusive and join one-to-one
  with the time rows (FS24), and no code here sums over them.
- A section's time is the time rows' ``time_s`` over untraced steps ≥ 1, else the markers'
  host time. The sections are those of the markers and of the time rows; one with no marker
  spans has null clocks and utilization. A time-row-only component on a side with phase rows
  (``P2`` of ``P2.w_t``, ``train_captured`` of ``train.*``) is their sum: it is marked ``rollup``
  and left out of ``share_total``. Clocks, sampler spans and the hook bound
  use the same steps as the time rows.
- Step time (FS42c, EQ14): a run's wall time per step is the interval between consecutive
  verified steps, the ``step_interval`` time row (``side`` ``run``, level ``total``). ``step_s``
  (run view and warm-up block) is its mean over the same steps; without the row it is the
  ``step`` total row's mean, else the sum of the top-level sections (``step_source``:
  ``"step_interval"``, ``"step"`` or ``"sections"``). ``work_s`` is the mean of the ``step``
  total row (the summed work of both sides; null without it). ``share`` stays each section's
  mean over ``step_s``, and ``share_by_side`` sums ``share`` per side (``prover``, ``verifier``,
  ``run``) over the depth-0, non-rollup sections: in a pipelined run the two sides overlap, so
  each side's sum is at most about 1, not the two together. A pipelined run (a ``step_wall``
  time row, or a markers step line with ``merged``) has ``pipelined`` true and ``latency_s``,
  the mean ``step_wall`` (a step's first prover section to its last verifier section); else
  false and null. ``step_interval`` and ``step_wall`` are not sections.
- Headline utilization by role (FS42c): for a section whose side (its time rows') is ``prover``
  or ``verifier``, ``util`` is the mean, field by field, of the means of its listed devices
  whose ``gpu_roles`` entry has that role (a field's mean over the devices with a value,
  ``n_samples`` the smallest), and ``util_devices`` names them; the ``utilization`` mark's lane
  is taken over them too. With no such device (no ``gpu_roles``, side ``run``, no time rows)
  the first listed device is the headline, as before, and ``util_devices`` is that device
  (null with none). ``util_by_device`` keeps every listed device. The check-5 view reads the
  same mean over check 5's verifier devices.
- The plain comparison's ``step`` also holds ``interval_s`` (the verified run's steady mean
  ``step_interval``), ``prover_side_s`` and ``verifier_side_s`` (its steady ``prover`` and
  ``verifier`` side totals); ``prover_s`` stays the ``step`` total.
- Per-GPU tools (FS35). Samples version 2 holds one value per GPU in each ``g_*`` field, and
  ``Sampler(samples, gpu=k)`` is GPU ``k``'s view (a version 1 file has only GPU 0).
- A device maps to a sampler GPU by UUID (the run's ``gpu_roles``, else the probe's
  ``per_device``, else the machine record's ``gpus``, against the sampler header); with no UUID
  on either side it maps to the same index, with a warning.
- A section's ``sec["devices"]`` names its devices. One device: ``util`` is that GPU's. Several:
  ``util`` follows the role rule below and ``util_by_device`` has each device's. No list: the run's
  device (``gpu_roles``' first, else the environment's), GPU 0 of a version 1 file.
- The bound-lane mark of a multi-device section uses the largest lane utilization over its
  role devices (the headline's ``util_devices``).
- With a version 2 probe a section's GPU rates and costs are its device's ``per_device``
  entry; the host-only rates are the top-level ones. A version 1 probe, or a device the probe
  didn't measure, uses the top level, with one warning naming the device.
- A multi-device section's GEMM, HBM, PCIe and launch floors use the slowest device's rate
  (the split replay runs its units one after another, so the work does not divide).
- The hash floor of a run whose transcript sits on the GPUs (its environment record's
  ``gpu_roles`` has a verifier-only device) uses each device's ``hash_kernel`` rate on an even
  share of the bytes and leaves, the slowest time. An even split is an estimate: the placement
  balances bytes per GPU, not leaves. Else ``hash``. Only the section's devices that hold
  transcript share it (``gpu_roles`` verifier), since a sequential run lists every device on
  every section; with none known to hold, all of its devices share it.
- Usable memory per device is the machine record's ``gpus[*].memory_bytes``, else
  ``gpu_memory_bytes``. Where memory rows hold
  ``device_peak_reserved_bytes`` the memory mark checks each device's peak against its own
  memory, and ``memory`` adds ``gpu_memory_bytes_by_device`` and each section's ``by_device``.
- The hook bound counts a section's syncs and events per device, each at that device's
  ``sync_s`` and ``event_pair_s``; stamps and host-timed spans have no events and cost none.
- The timeline has one strip per GPU for ``g_util``, ``g_pcie_tx`` and ``g_pcie_rx``
  (``strips["gpus"]``, labelled with the device and its role from ``gpu_roles``) when the
  samples hold several GPUs; ``c_busy`` stays one strip, the top-level ``g_*`` the run's device.
- Calls folded into a section's ``rest`` count as one span from ``first_us`` to ``last_us``,
  and ``util.truncated`` says so. ``g_throttle`` averages as the fraction of samples with any
  bit set. The check-5 view has no per-class floors: the counting pass has no per-class FLOPs.

**The hash breakdown.** Each run view has a ``hash`` entry for the commit (``P3.commit``) and
check 2 (``2``), built from the seconds the code adds inside their hashing section
(``P3.commit.hash``, ``2.hash``) as ``<section>.<part>`` time rows. These rows are not
sections: they are taken out of the section table, the shares and the trace join. An entry
holds the mean hashing wall time (``wall``, with its floor and ratio), the owner's other
parts (``siblings``: the commit's ``collect``, ``tree`` and ``check``, or the rest of check 2),
``leaves`` and ``hash_bytes``, and ``parts``. The caller thread's parts are ``read`` (reading
and validating each leaf), ``producer`` (batching and handing off) and ``wait`` (blocked on
the worker); they add up to the wall time, and ``unaccounted_s`` is what is left. The worker
thread's parts, which run beside them, are ``native`` (the batched C hashing), ``big`` (leaves
of 16 MiB or more) and ``other``. A part with no row is 0, since a zero is not written;
``has_stats`` is whether the ``read`` row exists, and a run from before the breakdown has null
parts. The one floor is the hash floor, bytes × ``hash.per_byte_s`` + leaves × ``hash.per_leaf_s``
of the probe; it goes beside the wall time and the ``native`` part. There is no tree floor: the
probe times whole ``leaf_hashes_of`` calls and has no rate for the tree's nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import warnings
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from evaluation.results.profile import io

__all__ = ["US", "THRESHOLDS", "Sampler", "weighted_means", "analyze", "main"]

US = 1e-6  # seconds per µs
THRESHOLDS = {"share": 0.10, "ratio": 3.0, "reserved": 0.95, "clock_rel": 0.05,
              "clock_abs_s": 0.002, "trace_rel": 0.10, "min_samples": 3}
CLOCK_MIN_S = 0.05  # §6 step 7: the clock mark needs a section of at least 50 ms
UTIL_FACTOR = 2.0  # §6 step 5
VAR_FIELDS = ("cpu_mhz", "c_other", "g_sm_mhz", "g_throttle", "g_other_procs")
PHASES = ("run", "glue", "host_stack", "copy", "measure", "judge")
STRIP_FIELDS = ("g_util", "c_busy", "g_pcie_tx", "g_pcie_rx")
STRIP_POINTS = 400  # the most points per step and field the timeline keeps
# The hash breakdown: the sections that hash a whole transcript, and the parts of their
# hashing the code adds as seconds (``<owner>.hash.<part>``). The caller's parts add up to the
# hashing wall time; the worker's run beside them on the background thread.
HASH_OWNERS = {"commit": "P3.commit", "check2": "2"}
HASH_LANES = {"read": "caller", "producer": "caller", "wait": "caller",
              "native": "worker", "big": "worker", "other": "worker"}
HASH_PARTS = tuple(HASH_LANES)


def _abs_ns(t0_ns: int, us: float) -> int:
    return t0_ns + round(us * 1000)


def _stats(xs: Sequence[float]) -> dict[str, Any]:
    if not xs:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "min": min(xs),
            "max": max(xs), "n": len(xs)}


def _mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


NO_GPU = 10 ** 6  # a GPU position no file has: every GPU field reads null


def _slot(v: Any, k: int) -> Any:
    """A version 2 row's GPU field for the ``k``-th GPU: the list's slot, ``None`` past it or
    for a field that is a bare null (no GPU reader)."""
    if isinstance(v, list):
        return v[k] if 0 <= k < len(v) else None
    return v if v is None or k == 0 else None


class Sampler:
    """The samples as intervals ``(previous t, t]``; the first starts at the header's ``mono_ns``.

    ``gpu=k`` is one GPU's view (FS35). A version 2 file's GPU fields (``g_*``) are lists, one
    value per header GPU; the view's rows hold the ``k``-th (``gpu`` unset is GPU 0). A version 1
    file has one GPU, ``k = 0``, and its rows are used as they are. ``gpus`` is the header's GPU
    list (version 1's ``gpu`` as a list of one) and ``n_gpus`` the number of GPUs the rows hold.
    :meth:`view` makes the view of another GPU, cached."""

    def __init__(self, samples: io.Samples, gpu: int | None = None):
        self.samples = samples
        head = samples.header
        self.version = int(head.get("version") or 1)
        raw = head.get("gpus") if self.version >= 2 else [head.get("gpu")]
        self.gpus: list[dict[str, Any]] = [g if isinstance(g, dict) else {} for g in raw] \
            if isinstance(raw, list) and any(isinstance(g, dict) for g in raw) else []
        self.n_gpus = len(self.gpus) or self._row_gpus()
        self.gpu = gpu
        self._views: dict[int, Sampler] = {}
        self.rows = samples.rows
        if self.version >= 2 or (gpu or 0) != 0:
            k = gpu or 0
            self.rows = [{f: (_slot(v, k) if f.startswith("g_") else v) for f, v in r.items()}
                         for r in samples.rows]
        self.ends = [int(r["t"]) for r in self.rows]
        self.starts = [int(samples.header.get("mono_ns", 0)), *self.ends[:-1]]
        fields = samples.header.get("fields") or (list(self.rows[0]) if self.rows else [])
        self.fields = [f for f in fields if f != "t"]

    def _row_gpus(self) -> int:
        if self.version < 2:
            return 1
        for r in self.samples.rows:
            for f, v in r.items():
                if f.startswith("g_") and isinstance(v, list):
                    return len(v)
        return 1

    def view(self, gpu: int | None) -> "Sampler":
        """The same samples as one GPU's view (``None``: the default, GPU 0)."""
        k = gpu or 0
        if k == (self.gpu or 0):
            return self
        if k not in self._views:
            self._views[k] = Sampler(self.samples, k)
        return self._views[k]

    def means(self, spans: Sequence[tuple[int, int]]) -> tuple[dict[str, float | None], int]:
        """Each field's mean over ``spans`` (ns), every sample weighted by the overlap of its
        interval with the spans and a null skipped; and the number of overlapping samples.
        ``g_throttle``, a bitmask, averages as the fraction of samples with any bit set."""
        w: dict[int, int] = {}
        for a, b in spans:
            i = bisect_right(self.ends, a)
            while i < len(self.rows) and self.starts[i] < b:
                ov = min(self.ends[i], b) - max(self.starts[i], a)
                if ov > 0:
                    w[i] = w.get(i, 0) + ov
                i += 1
        out: dict[str, float | None] = {}
        for f in self.fields:
            num = den = 0.0
            for i, wi in w.items():
                v = self.rows[i].get(f)
                if v is not None:
                    if f == "g_throttle":  # a bitmask: the share of samples with any bit set
                        v = 1.0 if int(v) else 0.0
                    num += wi * v
                    den += wi
            out[f] = num / den if den else None
        return out, len(w)


    def total(self, field: str, spans: Sequence[tuple[int, int]]) -> float | None:
        """The sum of a per-sample amount (a delta over the sample's interval, such as
        ``cg_throttled_us``) over ``spans``, each sample counted by the share of its interval
        inside them; None when no overlapping sample holds a value."""
        out = None
        for a, b in spans:
            i = bisect_right(self.ends, a)
            while i < len(self.rows) and self.starts[i] < b:
                v = self.rows[i].get(field)
                width = self.ends[i] - self.starts[i]
                ov = min(self.ends[i], b) - max(self.starts[i], a)
                if v is not None and ov > 0 and width > 0:
                    out = (out or 0.0) + v * ov / width
                i += 1
        return out


def _cpu_state(steps: list[dict[str, Any]], sampler: Sampler) -> dict[str, Any]:
    """Per step, what the host CPU was doing under us: ``throttled_frac`` (cgroup CFS throttled
    time over the step's wall time; above 0 the quota, not the work, set the pace),
    ``throttled_periods`` and ``mean_mhz`` (mean clock of our affinity CPUs, else the sampler's
    ``cpu_mhz``). Each is None when the samples lack it (macOS, no cgroup, no cpufreq)."""
    rows = []
    for s in steps:
        span = [(s["t0_ns"], s["t1_ns"])]
        wall = (s["t1_ns"] - s["t0_ns"]) * 1e-9
        means, _ = sampler.means(span)
        us = sampler.total("cg_throttled_us", span)
        nr = sampler.total("cg_nr_throttled", span)
        mhz = means.get("aff_mhz")
        rows.append({"step": s["step"], "wall_s": wall,
                     "throttled_frac": us * 1e-6 / wall if us is not None and wall > 0 else None,
                     "throttled_periods": nr,
                     "mean_mhz": mhz if mhz is not None else means.get("cpu_mhz")})
    fracs = [r["throttled_frac"] for r in rows if r["throttled_frac"] is not None]
    mhz = [r["mean_mhz"] for r in rows if r["mean_mhz"] is not None]
    return {"steps": rows, "mean_throttled_frac": _mean(fracs), "mean_mhz": _mean(mhz)}


def weighted_means(samples: io.Samples, spans: Sequence[tuple[int, int]]
                   ) -> tuple[dict[str, float | None], int]:
    return Sampler(samples).means(spans)


def _pearson(xs: Sequence[float | None], ys: Sequence[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = statistics.fmean(x for x, _ in pairs)
    my = statistics.fmean(y for _, y in pairs)
    sxx = sum((x - mx) ** 2 for x, _ in pairs)
    syy = sum((y - my) ** 2 for _, y in pairs)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(sxx * syy)


# ---- devices -----------------------------------------------------------------------------


def _norm_dev(dev: str) -> str:
    """A bare ``cuda`` is ``cuda:0`` (``metrics.normalize_devices`` resolves it to the current
    device, which is GPU 0 here)."""
    return "cuda:0" if dev == "cuda" else dev


def _dev_index(dev: str | None) -> int | None:
    dev = _norm_dev(dev or "")
    return int(dev[5:]) if dev.startswith("cuda:") and dev[5:].isdigit() else None


def _transcript_on_gpus(env: dict[str, Any]) -> bool:
    """Whether the run's transcript sits on GPUs: the environment record's ``gpu_roles`` has a
    device that is only a verifier (the record keeps ``config_hash`` only, no config)."""
    return any(isinstance(r, dict) and r.get("roles") == ["verifier"]
               for r in env.get("gpu_roles") or [])


class _Devices:
    """The run's devices (FS35): the sampler GPU, the probe's measurements and the memory of
    each, and the one the sections with no device list use (:attr:`primary`). A section names
    its devices with ``sec["devices"]``; ``named`` below is whether it did."""

    def __init__(self, sampler: Sampler, probe: dict[str, Any], records: list[dict[str, Any]],
                 header_device: Any, warns: set[str]):
        self.sampler, self.probe, self.warns = sampler, probe, warns
        self.env = next((r for r in records if r.get("record") == "environment"), {})
        roles = self.env.get("gpu_roles")
        self.roles = {_norm_dev(r["device"]): r for r in roles
                      if isinstance(r, dict) and r.get("device")} if isinstance(roles, list) else {}
        self.per_device = probe.get("per_device") if isinstance(probe.get("per_device"), dict) \
            else {}
        self.machine_gpus = next((m["gpus"] for m in [probe.get("machine"), self.env.get("machine")]
                                  if isinstance(m, dict) and isinstance(m.get("gpus"), list)), [])
        self.gpu_hash = _transcript_on_gpus(self.env)
        env_dev = str(self.env.get("device") or "")
        first = next(iter(self.roles), None)
        self.primary: str | None = first or (_norm_dev(env_dev) if env_dev.startswith("cuda")
                                              else None) \
            or ("cuda:0" if header_device == "cuda" else None)
        self._probes: dict[str, dict[str, Any]] = {}
        self._pos: dict[tuple[str, bool], int | None] = {}

    def holds_transcript(self, dev: str | None) -> bool:
        """Whether ``dev`` holds transcript: ``gpu_roles`` gives it the verifier role."""
        return dev is not None and "verifier" in ((self.roles.get(dev) or {}).get("roles") or [])

    def sections_devices(self, listed: list[str] | None) -> tuple[list[str | None], bool]:
        """A section's devices and whether it named them: the CUDA ones it lists, else the
        primary device alone (``None`` when the run has no CUDA device)."""
        cuda = [_norm_dev(d) for d in listed or [] if _dev_index(d) is not None]
        return (list(dict.fromkeys(cuda)), True) if cuda else ([self.primary], False)

    def uuid_of(self, dev: str) -> str | None:
        """``dev``'s UUID: the run's own ``gpu_roles`` entry first, then the probe's
        ``per_device`` one, then the machine record's."""
        idx = _dev_index(dev)
        for src in (self.roles.get(dev), self.per_device.get(dev),
                    next((g for g in self.machine_gpus
                          if isinstance(g, dict) and g.get("index") == idx), None)):
            if isinstance(src, dict) and src.get("uuid"):
                return str(src["uuid"])
        return None

    # -- the sampler's GPU ------------------------------------------------------------------

    def position(self, dev: str | None, named: bool) -> int | None:
        """The sampler GPU of ``dev``: matched by UUID (the sampler header's ``gpus`` against
        :meth:`uuid_of`), else the GPU with the same index, with a warning. ``None`` when no sampler
        GPU fits."""
        idx = _dev_index(dev)
        if dev is None or idx is None:
            return None
        if (dev, named) in self._pos:
            return self._pos[dev, named]
        sam = self.sampler
        loud = bool(sam.rows) and (named or sam.version >= 2)
        uuid, listed = self.uuid_of(dev), [g.get("uuid") for g in sam.gpus]
        pos: int | None = None
        if uuid and any(listed):
            pos = listed.index(uuid) if uuid in listed else None
            if pos is None and loud:
                self.warns.add(f"{dev} (UUID {uuid}) is not among the sampler's GPUs: matched by "
                               "index instead")
        elif loud:
            self.warns.add(f"no GPU UUIDs to match {dev} to the sampler's GPUs: used the same "
                           "index")
        if pos is None:
            pos = next((i for i, g in enumerate(sam.gpus) if g.get("index") == idx), None)
            if pos is None and not sam.gpus and idx < sam.n_gpus:
                pos = idx
        self._pos[dev, named] = pos
        return pos

    def view(self, dev: str | None, named: bool) -> Sampler:
        pos = self.position(dev, named)
        if pos is None and named and _dev_index(dev) is not None:
            if self.sampler.rows:
                self.warns.add(f"the samples hold no GPU for {dev}: its GPU fields are null")
            return self.sampler.view(NO_GPU)
        return self.sampler.view(pos)

    def gpu_of(self, pos: int) -> tuple[str | None, list[str]]:
        """The run's device and roles of sampler GPU ``pos``, or ``(None, [])``."""
        names = list(dict.fromkeys([*self.roles, *self.per_device, *(
            f"cuda:{g['index']}" for g in self.machine_gpus if isinstance(g, dict)
            and g.get("index") is not None)]))
        for d in names:
            if self.position(d, False) == pos:
                return d, list((self.roles.get(d) or {}).get("roles") or [])
        return None, []

    # -- the probe's measurements -----------------------------------------------------------

    def probe_of(self, dev: str | None, named: bool) -> dict[str, Any]:
        """The probe as one device sees it: the host-only rates (top level) beside the device's
        own ``per_device`` rates, costs and datasheet. A version 1 probe, or a device the probe
        didn't measure, gives the top level, with one warning naming the device."""
        if not named or dev is None or _dev_index(dev) is None:
            return self.probe
        if dev in self._probes:
            return self._probes[dev]
        d = self.per_device.get(dev)
        if isinstance(d, dict):
            out = {**self.probe,
                   "rates": {**(self.probe.get("rates") or {}), **(d.get("rates") or {})},
                   "costs": {**(self.probe.get("costs") or {}), **(d.get("costs") or {})},
                   "datasheet": d["datasheet"] if "datasheet" in d else self.probe.get("datasheet")}
        else:
            out = self.probe
            own = _norm_dev(str(self.probe.get("device") or ""))
            if self.probe.get("rates") and (self.per_device or dev != own):
                self.warns.add(f"the probe has no measurements of {dev}: the first device's "
                               "rates and costs are used for it")
        self._probes[dev] = out
        return out

    def memory_of(self, dev: str | None, fallback: int | None) -> int | None:
        """The usable memory of ``dev``: the machine record's ``gpus`` entry, else the machine
        block's one ``gpu_memory_bytes``."""
        idx = _dev_index(dev)
        for g in self.machine_gpus:
            if isinstance(g, dict) and g.get("index") == idx and g.get("memory_bytes"):
                return int(g["memory_bytes"])
        return fallback


# ---- sections ----------------------------------------------------------------------------


def _clock_s(sec: dict[str, Any], key: str, rest_key: str) -> float | None:
    spans = sec.get(key)
    if spans is None or spans and all(sp is None for sp in spans):
        return None  # off CUDA, or host-timed: no device clock
    s = sum(sp[1] - sp[0] for sp in spans if sp is not None) * US
    if sec.get("rest"):
        s += sec["rest"].get(rest_key) or 0.0
    return s


def _spans_ns(sec: dict[str, Any], t0_ns: int) -> list[tuple[int, int]]:
    """The kept spans, and for calls folded into ``rest`` one span from ``first_us`` to
    ``last_us``: an approximation that also covers the gaps between them."""
    spans = [(_abs_ns(t0_ns, a), _abs_ns(t0_ns, b)) for a, b in sec.get("host_us") or []]
    rest = sec.get("rest")
    if rest and rest.get("first_us") is not None and rest.get("last_us") is not None:
        spans.append((_abs_ns(t0_ns, rest["first_us"]), _abs_ns(t0_ns, rest["last_us"])))
    return spans


def _side(name: str, sides: dict[str, str]) -> str:
    if name in sides:
        return sides[name]
    return "prover" if name.startswith(("P", "train.")) else "verifier"


def _is_c5(name: str) -> bool:
    return name == "5" or name.startswith("5.")


def _rate_key(name: str, sides: dict[str, str], compute_bf16: bool) -> str:
    if _is_c5(name):
        return "skinny_bmm_fp32_flops"
    if _side(name, sides) == "prover":
        return "gemm_bf16_flops" if compute_bf16 else "gemm_fp32_flops"
    return "gemm_fp32_flops"


def _direction(d: dict[str, Any] | None, key: str) -> float:
    d = d or {}
    return float(d.get(key) or d.get("memcpy_" + key) or 0.0)


def _missing_terms(name: str, work: dict[str, Any]) -> list[str]:
    """The floor terms that come from the trace and so are missing (not zero) without one:
    memory traffic (``hbm serving`` for check 5, whose floor holds the measuring reads
    besides, else ``hbm``), PCIe, the GPU-to-GPU link and launches."""
    return ["hbm serving" if work["bytes"].get("hbm_measure") else "hbm", "pcie", "nvlink",
            "launch"]


def _work(name: str, rate_key: str, count: dict[str, Any] | None, leaves: int,
          trace: dict[str, Any] | None, c5_scale: float, syncs: int,
          measure_reads: float = 0.0, warns: set[str] | None = None) -> dict[str, Any]:
    """``measure_reads`` is check 5's measuring-kernel read bytes per step (section ``5`` only,
    not scaled by the trace slice); it is ``bytes.hbm_measure``, apart from the serving's
    ``bytes.hbm``.

    Copy bytes are kept by kind (the trace names each copy's host memory, FS42): ``h2d_pinned``,
    ``h2d_pageable``, ``d2h_pinned``, ``d2h_pageable`` (``h2d`` and ``d2h`` are the totals), the
    device-to-device ones in ``hbm`` with the GEMM bytes, and the GPU-to-GPU ones as ``peer``
    (their total) with ``work["peer_pairs"]`` (device pair -> bytes). A copy of unknown kind, or
    every copy of a trace summary without ``copy_kinds``, counts as pageable, with a warning."""
    count, trace = count or {}, trace or {}
    flops = float(count.get("flops") or 0.0)
    hash_bytes = float(count.get("hash_in_bytes") or 0.0) \
        + float(count.get("hash_out_bytes") or 0.0)
    scale = c5_scale if _is_c5(name) else 1.0
    copy = trace.get("copy_bytes")
    kinds = trace.get("copy_kinds")
    pairs: dict[str, float] = {}
    if kinds is None:
        if copy and warns is not None:
            warns.add("the trace summary has no copy kinds: its copies are priced at the "
                      "pageable rates")
        d2d = _direction(copy, "d2d")
        pinned = {"h2d": 0.0, "d2h": 0.0}
        pageable = {d: _direction(copy, d) for d in ("h2d", "d2h")}
    else:
        d2d = float(kinds.get("d2d") or 0.0)
        pairs = {k: scale * float(v or 0.0) for k, v in (kinds.get("peer") or {}).items()}
        pinned = {d: float(kinds.get(f"{d}_pinned") or 0.0) for d in ("h2d", "d2h")}
        unknown = {d: float(kinds.get(f"{d}_unknown") or 0.0) for d in ("h2d", "d2h")}
        if any(unknown.values()) and warns is not None:
            warns.add("the trace summary has copies of unknown kind (pinned or pageable): "
                      "priced at the pageable rates")
        pageable = {d: float(kinds.get(f"{d}_pageable") or 0.0) + unknown[d]
                    for d in ("h2d", "d2h")}
    nbytes = {"hbm": scale * (float(trace.get("gemm_bytes") or 0.0) + d2d),
              "h2d": scale * (pinned["h2d"] + pageable["h2d"]),
              "d2h": scale * (pinned["d2h"] + pageable["d2h"]),
              "h2d_pinned": scale * pinned["h2d"], "h2d_pageable": scale * pageable["h2d"],
              "d2h_pinned": scale * pinned["d2h"], "d2h_pageable": scale * pageable["d2h"],
              "peer": sum(pairs.values()), "hash": hash_bytes}
    if name == "5":
        nbytes["hbm_measure"] = measure_reads
    return {"flops": {rate_key: flops} if flops else {}, "bytes": nbytes, "peer_pairs": pairs,
            "leaves": leaves, "launches": scale * float(trace.get("launches") or 0.0),
            "syncs": syncs}


def _peer_floor(work: dict[str, Any], plist: Sequence[dict[str, Any]],
                warns: set[str]) -> float | None:
    """The GPU-to-GPU copies' floor: each device pair's bytes at the probe's ``peer`` rate for
    that pair, the slowest such time (the links are independent, so a lower bound is the busiest
    one). A pair the event did not name (``cuda:?``) or the probe did not measure runs at the
    slowest measured pair's rate (the probe's peer rates are the top level's, shared by every
    device)."""
    pairs = {k: v for k, v in (work.get("peer_pairs") or {}).items() if v}
    if not pairs:
        return None
    rates = {k: v for k, v in ((plist[0] if plist else {}).get("peer") or {}).items() if v}
    if not rates:
        warns.add("the probe has no peer rates: the nvlink floor is skipped")
        return None
    slowest = min(rates.values())
    times = []
    for pair, n in pairs.items():
        if pair not in rates and "?" not in pair:
            warns.add(f"the probe has no peer rate for {pair}: the slowest pair's is used")
        times.append(n / rates.get(pair, slowest))
    return max(times)


def _floors(work: dict[str, Any], probes: dict[str, Any] | Sequence[dict[str, Any]],
            warns: set[str], *, devices: Sequence[str | None] = (), gpu_hash: bool = False,
            holders: Sequence[bool] = ()) -> dict[str, float]:
    """§6 step 2: one floor per resource that has work. A term with a zero count adds 0
    whatever its rate; a non-zero count with a null rate drops the resource, with a warning.
    PCIe is the larger of the two directions' times (the link is full duplex), each direction
    summing its pinned and pageable bytes at their own rates; ``nvlink`` is the GPU-to-GPU
    copies (:func:`_peer_floor`).

    ``probes`` is the probe as each of the section's devices sees it (FS35; one dict is one
    device). The work does not divide over devices (the split replay runs its units one after
    another), so the GEMM, HBM, PCIe and launch floors use the slowest device's rate or cost.
    The hash floor, with ``gpu_hash`` (the transcript sits on the GPUs), is each device's
    ``hash_kernel`` rate on an even share of the bytes and leaves (each owner hashes its own
    leaves; an even split is an estimate), the slowest device's time; else the host ``hash``.
    ``holders`` marks the section's devices that hold transcript: a sequential sharded run lists
    every device on every section, so the share is over the holders only (all the devices when
    none is known to hold)."""
    plist = [probes] if isinstance(probes, dict) else list(probes)
    n_dev = len(plist)
    names = [d or "the device" for d in (list(devices) + [None] * n_dev)[:n_dev]]
    out: dict[str, float] = {}

    def rate_s(key: str) -> float | None:
        """Seconds per unit at the slowest device; ``None`` when any device lacks the rate."""
        rs = [(p.get("rates") or {}).get(key) for p in plist]
        return None if any(r is None for r in rs) else max(1.0 / r for r in rs)

    def cost_s(key: str) -> float | None:
        cs = [(p.get("costs") or {}).get(key) for p in plist]
        return None if any(c is None for c in cs) else max(cs)

    def total(resource: str, terms: list[tuple[str, float, Any]], combine=sum) -> None:
        """``terms`` are (label, count, seconds per count); skipped when every count is 0."""
        live = [(label, n, r) for label, n, r in terms if n]
        if not live:
            return
        missing = [label for label, _, r in live if r is None]
        if missing:
            warns.add(f"the probe has no {', '.join(missing)}: the {resource} floor is skipped")
            return
        out[resource] = combine(n * r for _, n, r in live)

    for key, fl in work["flops"].items():
        total("gemm", [(key, fl, rate_s(key))])
    b = work["bytes"]
    total("hbm", [("hbm_copy_bytes", b["hbm"] + b.get("hbm_measure", 0.0),
                   rate_s("hbm_copy_bytes"))])
    link = [(d, kind, f"{d}_{kind}_bytes") for d in ("h2d", "d2h")
            for kind in ("pinned", "pageable")]
    live = [(d, key, b[f"{d}_{kind}"], rate_s(key)) for d, kind, key in link
            if b.get(f"{d}_{kind}")]
    missing = [key for _, key, _, r in live if r is None]
    if missing:
        warns.add(f"the probe has no {', '.join(missing)}: the pcie floor is skipped")
    elif live:  # each direction's pinned and pageable bytes share the link; full duplex
        out["pcie"] = max(sum(n * r for dd, _, n, r in live if dd == d) for d in ("h2d", "d2h"))
    peer = _peer_floor(work, plist, warns)
    if peer is not None:
        out["nvlink"] = peer
    kernels = [((p.get("rates") or {}).get("hash_kernel") or {}) for p in plist]
    keep = [i for i, h in enumerate(list(holders)[:n_dev]) if h] or list(range(n_dev))
    kernels, hnames = [kernels[i] for i in keep], [names[i] for i in keep]
    n_hold = len(keep)
    if gpu_hash and (b["hash"] or work["leaves"]) and all(k for k in kernels):
        times = []
        for k, name in zip(kernels, hnames, strict=True):
            n_bytes, n_leaves = b["hash"] / n_hold, work["leaves"] / n_hold
            terms = [("hash_kernel.per_byte_s", n_bytes, k.get("per_byte_s")),
                     ("hash_kernel.per_leaf_s", n_leaves, k.get("per_leaf_s"))]
            missing = [label for label, c, r in terms if c and r is None]
            if missing:
                warns.add(f"the probe has no {', '.join(missing)} for {name}: the hash floor is "
                          "skipped")
                times = []
                break
            times.append(sum(c * r for _, c, r in terms if c))
        if times:
            out["hash"] = max(times)
    else:
        if gpu_hash and (b["hash"] or work["leaves"]):
            warns.add("the probe has no hash_kernel rate for "
                      f"{', '.join(n for n, k in zip(hnames, kernels, strict=True) if not k)}: "
                      "the host hash rate is used")
        hr = [(p.get("rates") or {}).get("hash") or {} for p in plist][0]
        total("hash", [("hash.per_byte_s", b["hash"], hr.get("per_byte_s")),
                       ("hash.per_leaf_s", work["leaves"], hr.get("per_leaf_s"))])
    if work["launches"]:  # syncs alone are no evidence of launch-bound work
        total("launch", [("launch_s", work["launches"], cost_s("launch_s")),
                         ("sync_s", work["syncs"], cost_s("sync_s"))])
    return out


def _lane(bound: str | None, means: dict[str, float | None], probe: dict[str, Any],
          rate_key: str | None = None, nan_missing: bool = False) -> float | None:
    """Measured utilization (a fraction) on the bound's lane, or ``None``. A measured 0 is a
    reading. The GEMM lane is the tensor pipe for a bf16 rate and the SMs for an fp32 one.
    A NaN primary reading is returned as NaN, or with ``nan_missing`` read as absent (the
    fallback field is used)."""
    def frac(primary: str, fallback: str) -> float | None:
        v = means.get(primary)
        if v is not None and not (nan_missing and math.isnan(v)):
            return v
        return means.get(fallback)

    if bound == "gemm":
        return frac("g_tensor" if rate_key == "gemm_bf16_flops" else "g_sm_active", "g_util")
    if bound == "hbm":
        return frac("g_dram_bw", "g_mem_util")
    if bound == "pcie":
        sheet = (probe.get("datasheet") or {}).get("pcie_bytes")
        tx, rx = means.get("g_pcie_tx"), means.get("g_pcie_rx")
        return None if not sheet or tx is None or rx is None else (tx + rx) / sheet
    if bound == "hash":
        threads = ((probe.get("rates") or {}).get("hash") or {}).get("threads")
        c = means.get("c_tree")
        return None if not threads or c is None else c / threads
    return None


def _lane_max(bound: str | None, means: Sequence[dict[str, float | None]],
              probes: Sequence[dict[str, Any]], rate_key: str | None) -> float | None:
    """The largest lane utilization over the devices' readings, NaN-aware: a device reporting
    NaN does not hide the others whatever the order. With no finite reading but a NaN one the
    lane is read again with NaN as absent (the fallback field), as for a missing field."""
    for nan_missing in (False, True):
        lanes = [m for dm, p in zip(means, probes, strict=True)
                 if (m := _lane(bound, dm, p, rate_key, nan_missing)) is not None
                 and not math.isnan(m)]
        if lanes:
            return max(lanes)
    return None


def _differ(a: float, b: float) -> bool:
    lo, hi = sorted((a, b))
    return hi > 0 if lo <= 0 else hi / lo > UTIL_FACTOR


def _times(rows: list[dict[str, Any]]
           ) -> tuple[dict[str, dict[int, float]], dict[str, str], dict[str, str]]:
    """Seconds per component per untraced step ≥ 1 from the time rows, and each side and
    level."""
    out: dict[str, dict[int, float]] = {}
    sides: dict[str, str] = {}
    levels: dict[str, str] = {}
    for r in rows:
        if r.get("record") != "time" or r.get("traced") or r.get("derived") or r["step"] < 1:
            continue
        out.setdefault(r["component"], {}).setdefault(r["step"], r["time_s"])
        sides.setdefault(r["component"], r["side"])
        levels.setdefault(r["component"], r.get("level", "component"))
    return out, sides, levels


def _warm_step(steps: Sequence[int]) -> int | None:
    """The warm-up step: the lowest of the untraced steps ≥ 1, when there are at least two."""
    ordered = sorted(set(steps))
    return ordered[0] if len(ordered) >= 2 else None


def parse_steps(text: str) -> list[int]:
    """``--steady-steps``: a range ``A-B`` or a comma list of steps and ranges (``2-3,5``)."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        lo, hi = int(a), int(b) if sep else int(a)
        if hi < lo:
            raise ValueError(f"steady steps: {part!r} runs backwards")
        out += range(lo, hi + 1)
    return sorted(set(out))


def _pick_steady(known: Sequence[int], requested: Sequence[int] | None, warns: set[str],
                 traced: Sequence[int] = ()) -> tuple[int | None, set[int] | None]:
    """The warm-up step and the steady set. ``known`` are the untraced steps ≥ 1 with rows.
    With ``requested`` steps: those present and untraced are the steady set (``None`` returned
    for it when none is, with a warning, so the default rule applies); the others are warned
    about and left out; the warm-up is the lowest known step below the set, when there is one.
    Without: the warm-up is the lowest known step (when at least two) and the set is ``None``
    (every known step but the warm-up)."""
    if requested:
        have = set(known)
        good = sorted(t for t in set(requested) if t in have)
        for t in sorted(set(requested) - have):
            warns.add(f"steady step {t} is " + ("traced" if t in set(traced) else
                      "missing from the markers and time rows") + ": left out")
        if good:
            below = [t for t in sorted(have) if t < good[0]]
            return (below[0] if below else None), set(good)
        warns.add("no requested steady step is usable: the default steady rule is used")
    return _warm_step(known), None


def _only(times: dict[str, dict[int, float]], keep) -> dict[str, dict[int, float]]:
    """``times`` restricted to the steps ``keep(step)`` accepts; a component left with none
    is dropped."""
    out = {n: {t: s for t, s in v.items() if keep(t)} for n, v in times.items()}
    return {n: v for n, v in out.items() if v}


def _first_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each component's ``count`` row at its lowest step ≥ 1 (the counts repeat every step)."""
    out: dict[str, dict[str, Any]] = {}
    for r in sorted((r for r in rows if r.get("record") == "count" and r["step"] >= 1),
                    key=lambda r: r["step"], reverse=True):
        out[r["component"]] = r
    return out


# ---- the timeline ------------------------------------------------------------------------


def _bin_strips(t_s: list[float], cols: dict[str, list[float | None]]
                ) -> tuple[list[float], dict[str, list[float | None]]]:
    """At most :data:`STRIP_POINTS` equal-width time bins from 0 to the last ``t_s``: each
    field the mean of its non-null values in the bin, ``t_s`` the bin's end. A bin with no
    row is left out."""
    if len(t_s) <= STRIP_POINTS:
        return t_s, cols
    width = max(t_s) / STRIP_POINTS
    bins: dict[int, list[int]] = {}
    for i, t in enumerate(t_s):
        bins.setdefault(min(max(math.ceil(t / width) - 1, 0), STRIP_POINTS - 1), []).append(i)
    order = sorted(bins)
    out = {f: [_mean([v for i in bins[b] if (v := col[i]) is not None]) for b in order]
           for f, col in cols.items()}
    return [(b + 1) * width for b in order], out


def _gpu_strips(sampler: Sampler, dv: _Devices | None) -> list[dict[str, Any]]:
    """One entry per sampler GPU when there are several (FS35): its view, and the device and
    roles the run gives it (``gpu_roles``, by UUID), the label the report draws."""
    out = []
    for pos in range(sampler.n_gpus if sampler.n_gpus > 1 else 0):
        dev, roles = dv.gpu_of(pos) if dv else (None, [])
        label = f"{dev} ({', '.join(roles)})" if dev and roles else dev or f"GPU {pos}"
        out.append({"gpu": pos, "device": dev, "roles": roles, "label": label,
                    "view": sampler.view(pos)})
    return out


def _strips(sampler: Sampler, t0_ns: int, end_ns: int, gpus: Sequence[dict[str, Any]] = ()
            ) -> dict[str, list[Any]] | None:
    """The sampler rows whose interval overlaps the step's spans, as ``t_s`` (row end in s from
    ``t0_ns``) and the fields of :data:`STRIP_FIELDS` that hold a value; ``None`` when no row
    overlaps or no field has data. With several GPUs (``gpus``, :func:`_gpu_strips`) the strips
    also hold ``gpus``: one entry per GPU with its ``gpu`` position, ``device``, ``roles``,
    ``label`` and its own ``g_*`` fields on the same ``t_s``; the ``g_*`` fields at the top are
    the run's own device's."""
    idx = [i for i, (a, b) in enumerate(zip(sampler.starts, sampler.ends))
           if a < end_ns and b > t0_ns]
    rows = [sampler.rows[i] for i in idx]
    cols: dict[Any, list[Any]] = {f: [r.get(f) for r in rows] for f in STRIP_FIELDS}
    for k, g in enumerate(gpus):
        for f in STRIP_FIELDS:
            if f.startswith("g_"):
                cols[k, f] = [g["view"].rows[i].get(f) for i in idx]
    cols = {f: v for f, v in cols.items() if any(x is not None for x in v)}
    if not cols:
        return None
    t_s, cols = _bin_strips([(r["t"] - t0_ns) * 1e-9 for r in rows], cols)
    out = {"t_s": t_s, **{f: v for f, v in cols.items() if isinstance(f, str)}}
    per_gpu = [{"gpu": g["gpu"], "device": g["device"], "roles": g["roles"],
                "label": g["label"],
                **{f: cols[k, f] for f in STRIP_FIELDS if (k, f) in cols}}
               for k, g in enumerate(gpus)]
    if gpus:
        out["gpus"] = per_gpu
    return out


def _timeline(untraced: list[dict[str, Any]], sampler: Sampler, dv: _Devices | None = None
              ) -> dict[str, Any]:
    """One entry per untraced timed step: each section's spans in s from the step's ``t0_ns``
    (a folded ``rest`` as its one span), ``end_s`` the latest span end, and the sampler strips."""
    steps = []
    gpus = _gpu_strips(sampler, dv)
    for s in untraced:
        t0 = s["t0_ns"]
        order: dict[str, list[tuple[int, int]]] = {}
        depth: dict[str, int] = {}
        for sec in s["sections"]:
            order.setdefault(sec["name"], []).extend(_spans_ns(sec, t0))
            depth.setdefault(sec["name"], sec["depth"])
        end_ns = max((b for sp in order.values() for _, b in sp), default=t0)
        steps.append({
            "step": s["step"], "end_s": (end_ns - t0) * 1e-9,
            "sections": [{"name": n, "depth": depth[n],
                          "spans_s": [[(a - t0) * 1e-9, (b - t0) * 1e-9] for a, b in sp]}
                         for n, sp in order.items() if sp],
            "strips": _strips(sampler, t0, end_ns, gpus)})
    return {"steps": steps}


# ---- check 5 -----------------------------------------------------------------------------


def _measure_reads(layout: dict[str, Any] | None, device: str | None) -> float:
    """HBM bytes the check-5 measuring needs to read in one step (GPU path only), else 0.

    Every product's A, B and P are read once, as stored: the operands are served as views in
    the replay's dtype, which for the decoder is the product's dtype (``operand_dtype ==
    product_dtype``). The layout counts A and B as fp32 (``operand_bytes``), so they are
    rescaled to the product dtype's item size. Counted once per step: the layout covers every
    run, not the trace's slice.

    This is a floor, so it leaves out what the kernel's design adds: the kernel sweeps each
    operand once per chunk of 16 challenge columns (``K_CHUNK``), which is part of its ratio
    over this floor. Products that share one stored view (``share=True``) read it once, so the
    kernel can read less than this count: 0.8 of 6.3 GB per layer in the kernel plan's
    arithmetic (FS38b; not a measurement), about 10-11% a step. That overstatement is known, and the
    layout cannot remove it (it records no addresses)."""
    if not layout or not str(device or "").startswith("cuda"):
        return 0.0
    return float(sum(io.entry_bytes(e) for e in layout["entries"]))


def _layout_maps(layout: dict[str, Any], warns: set[str]
                 ) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """The layout's entries by group and by run. A group whose entries carry different classes
    is spread over them by bytes, with a warning."""
    by_group: dict[int, list[dict[str, Any]]] = {}
    by_run: dict[int, list[dict[str, Any]]] = {}
    for e in layout["entries"]:
        g = by_group.setdefault(e["group"], [])
        if g and g[0]["class"] != e["class"]:
            warns.add("c5_layout: members of one group carry different classes; the group's "
                      "time is spread over them by bytes")
        g.append(e)
        by_run.setdefault(e["run"], []).append(e)
    return by_group, by_run


def _stamp_rows(step: dict[str, Any], maps: tuple[dict[int, list[dict[str, Any]]],
                                                  dict[int, list[dict[str, Any]]]]
                ) -> list[dict[str, Any]]:
    """The step's stamps as rows: name, class, window and host span in s from t0, and
    ``shares`` (class -> share of the row's time). A stamp covers the entries of its group;
    a run stamp, those of its run (the two classes of an attention run share it by bytes); a
    batched measure stamp (group -1), those of the runs it covers (:func:`io.batch_cover`)."""
    st, names = step["stamps"], step["stamp_names"]
    by_group, by_run = maps
    n = len(st["name"])
    parsed = [(st["h0"][i], names[st["name"][i]].removeprefix("c5."), st["run"][i],
               st["group"][i]) for i in range(n)]
    cover = io.batch_cover(parsed)
    rows = []
    for i in range(n):
        run, group, window = st["run"][i], st["group"][i], st["window"][i]
        own = by_group.get(group) if group >= 0 else by_run.get(run) if run >= 0 else None
        ents = own
        if i in cover:
            ents = [e for r in cover[i] for e in by_run.get(r, [])] or own
        if window < 0 and own:
            window = own[0]["window"]
        shares = io.class_shares(ents) if ents else {}
        rows.append({"name": names[st["name"][i]].removeprefix("c5."), "run": run,
                     "class": next(iter(shares), None), "shares": shares, "window": window,
                     "h0": st["h0"][i] * US, "h1": st["h1"][i] * US})
    return rows


def _phase_totals(rows: list[dict[str, Any]], cls: str | None = None) -> dict[str, float]:
    """Seconds per phase; ``host_stack`` is ``c5.serve - c5.copy`` (§5.2). With ``cls``, each
    row counts its ``shares`` of that class."""
    tot = dict.fromkeys(PHASES, 0.0)
    serve = 0.0
    for r in rows:
        d = r["h1"] - r["h0"]
        if cls is not None:
            d *= r["shares"].get(cls, 0.0)
        if r["name"] == "serve":
            serve += d
        elif r["name"] in tot:
            tot[r["name"]] += d
    tot["host_stack"] = serve - tot["copy"]
    return tot


def _extent_ns(rows: list[dict[str, Any]], t0_ns: int) -> list[tuple[int, int]]:
    return [(_abs_ns(t0_ns, min(r["h0"] for r in rows) / US),
             _abs_ns(t0_ns, max(r["h1"] for r in rows) / US))]


class _MeanView:
    """Several GPUs' views as one: :meth:`means` is, field by field, the mean of the views'
    means over the views that have a value, and the number of samples is the smallest of
    theirs (the same rule as a section's headline utilization)."""

    def __init__(self, views: Sequence[Sampler]):
        self.views = list(views)
        self.fields = self.views[0].fields

    def means(self, spans: Sequence[tuple[int, int]]) -> tuple[dict[str, float | None], int]:
        return _mean_means([v.means(spans) for v in self.views])


def _mean_means(per_dev: Sequence[tuple[dict[str, float | None], int]]
                ) -> tuple[dict[str, float | None], int]:
    """The headline of several devices: each field's mean over the devices with a non-null
    value, and the smallest sample count. One device is returned as it is."""
    if len(per_dev) == 1:
        return per_dev[0]
    out: dict[str, float | None] = {}
    for f in per_dev[0][0]:
        vals = [m[f] for m, _ in per_dev if m.get(f) is not None]
        out[f] = statistics.fmean(vals) if vals else None
    return out, min(n for _, n in per_dev)


def _role_devices(dv: "_Devices", devs: Sequence[str | None], side: str | None) -> list[str]:
    """The listed devices whose ``gpu_roles`` entry has the role of ``side`` (``prover`` or
    ``verifier``); empty for any other side or when no device has it."""
    if side not in ("prover", "verifier"):
        return []
    return [d for d in devs if d is not None
            and side in ((dv.roles.get(d) or {}).get("roles") or [])]


def _c5_view(steps: list[dict[str, Any]], layout: dict[str, Any], sampler: Sampler | _MeanView,
             warns: set[str]) -> dict[str, Any]:
    per_step: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    classes: dict[str, dict[str, list[float]]] = {}
    over_time: dict[str, Any] = {}
    maps = _layout_maps(layout, warns)
    for s in steps:
        if not s.get("stamps"):
            continue
        rows = _stamp_rows(s, maps)
        if not rows:
            continue
        t0 = s["t0_ns"]
        means, _ = sampler.means(_extent_ns(rows, t0))
        per_step.append({"step": s["step"], "phases_s": _phase_totals(rows),
                         "sampler": {f: means.get(f) for f in VAR_FIELDS}})
        for w in sorted({r["window"] for r in rows if r["window"] >= 0}):
            wr = [r for r in rows if r["window"] == w]
            wm, _ = sampler.means(_extent_ns(wr, t0))
            windows.append({"step": s["step"], "window": w, "phases_s": _phase_totals(wr),
                            "sampler": {f: wm.get(f) for f in VAR_FIELDS}})
        for c in sorted({c for r in rows for c in r["shares"]}):
            for ph, v in _phase_totals(rows, c).items():
                classes.setdefault(c, {}).setdefault(ph, []).append(v)
        runs = [r for r in rows if r["name"] == "run"]
        over_time[str(s["step"])] = {"run": [r["run"] for r in runs],
                                     "class": [r["class"] for r in runs],
                                     "start_s": [r["h0"] for r in runs],
                                     "dur_s": [r["h1"] - r["h0"] for r in runs]}
    out_classes = {}
    for c, phases in classes.items():
        es = [e for e in layout["entries"] if e["class"] == c]
        means = {ph: _mean(v) for ph, v in phases.items()}
        stored = sum(e["product_bytes"] for e in es)
        served = sum(e["served_bytes"] for e in es)
        copy = means["copy"]
        out_classes[c] = {"n_runs": len({e["run"] for e in es}), "n_groups": len(es),
                          "product_bytes": stored, "served_bytes": served, "phases_s": means,
                          "copy_bytes_per_s": {"stored": stored / copy if copy else None,
                                               "served": served / copy if copy else None}}

    def corr(table: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
        return {ph: {f: _pearson([t["phases_s"][ph] for t in table],
                                 [t["sampler"][f] for t in table]) for f in VAR_FIELDS}
                for ph in PHASES}

    return {"n_runs": layout["n_runs"], "n_groups": layout["n_groups"],
            "n_windows": layout["n_windows"], "classes": out_classes,
            "phases_s": {ph: _mean([p["phases_s"][ph] for p in per_step]) for ph in PHASES},
            "runs_over_time": over_time,
            "variance": {"steps": per_step, "windows": windows,
                         "correlation": {"across_steps": corr(per_step),
                                         "across_windows": corr(windows)}}}


# ---- the analysis ------------------------------------------------------------------------


def _compute_dtype(records: list[dict[str, Any]], cli: str, warns: list[str]) -> str:
    for r in records:
        if r.get("record") == "environment" and r.get("compute_dtype"):
            return str(r["compute_dtype"])
    warns.append(f"no environment record has compute_dtype: used {cli!r} from --compute-dtype")
    return cli


def _gpu_memory(probe: dict[str, Any], records: list[dict[str, Any]]) -> int | None:
    blocks = [probe.get("machine")] + [r.get("machine") for r in records
                                        if r.get("record") == "environment"]
    for m in blocks:
        if isinstance(m, dict) and m.get("gpu_memory_bytes"):
            return int(m["gpu_memory_bytes"])
    return None


def _hook_bound(device: Any, steps: list[dict[str, Any]], probe: dict[str, Any],
                dv: _Devices | None = None) -> float | None:
    """§7: ``(syncs × sync_s + events × event_pair_s) ÷ step time``, the mean over the timed
    steps; 0 off CUDA, where there are no syncs or events to add. With ``dv`` (FS35) the
    syncs and events of a section that names its devices are counted per device and cost that
    device's ``sync_s`` and ``event_pair_s`` (the probe's ``per_device``); the others cost the
    first device's. A device's events are its own ``gpu_us`` spans (``gpu_us_by_device``) that
    are not null, and a section's top-level syncs are 2 per call on each. Stamps record no
    events (plan §2.7), nor does a host-timed span (null entry; a section whose kept spans are
    all null counts none of its folded calls either)."""
    if not str(device).startswith("cuda"):
        return 0.0

    def costs_of(dev: str | None) -> dict[str, Any]:
        return (dv.probe_of(dev, dev is not None) if dv else probe).get("costs") or {}

    c0 = costs_of(None)
    if c0.get("sync_s") is None or c0.get("event_pair_s") is None:
        return None
    vals = []
    for s in steps:
        syncs: dict[str | None, int] = {}
        events: dict[str | None, int] = {}
        for sec in s["sections"]:
            devs: list[str | None] = [_norm_dev(d) for d in sec.get("devices") or []
                                      if _dev_index(d) is not None] if dv else []
            for i, d in enumerate(devs or [None]):
                if sec["depth"] == 0:
                    syncs[d] = syncs.get(d, 0) + 2 * sec["calls"]
                by_dev = sec.get("gpu_us_by_device")
                spans = by_dev[i] if devs and by_dev and i < len(by_dev) else sec.get("gpu_us")
                if spans is not None:
                    timed = sum(g is not None for g in spans)
                    if timed or not spans:
                        events[d] = events.get(d, 0) + timed \
                            + ((sec.get("rest") or {}).get("calls") or 0)
        step_s = (s["t1_ns"] - s["t0_ns"]) * 1e-9
        over = 0.0
        for d in {*syncs, *events}:
            c = costs_of(d)
            if c.get("sync_s") is None or c.get("event_pair_s") is None:
                return None
            over += syncs.get(d, 0) * c["sync_s"] + events.get(d, 0) * c["event_pair_s"]
        if step_s > 0:
            vals.append(over / step_s)
    return _mean(vals)


def _occurrences(steps: list[dict[str, Any]], name: str
                 ) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for s in steps:
        for sec in s["sections"]:
            if sec["name"] == name:
                yield s, sec


def _c5_scale(c5_slice: dict[str, Any], warns: set[str]) -> float:
    """Estimated total check-5 run time over the traced runs' time, from the classes'
    ``scaled`` and ``phases_us`` run sums (by members, as FS24 scales them). A class lacking
    either is left out of both sums."""
    est = traced = 0.0
    for name, c in c5_slice.items():
        phases, scaled = c.get("phases_us"), c.get("scaled")
        if not phases or not scaled or phases.get("run") is None or scaled.get("run") is None:
            warns.add(f"c5_slice class {name!r} lacks phases_us or scaled for run: left out")
            continue
        est, traced = est + scaled["run"], traced + phases["run"]
    if est <= 0 or traced <= 0:
        if c5_slice:
            warns.add("c5_slice gives no run time to scale by: check 5's trace work is unscaled")
        return 1.0
    return est / traced


def _hashes_transcript(count: dict[str, Any] | None, transcript_bytes: int) -> bool:
    """A section that hashes at least half of the step's transcript bytes hashes every leaf
    (the commit and check 2)."""
    return bool(count) and transcript_bytes > 0 \
        and (count.get("hash_in_bytes") or 0) >= transcript_bytes / 2


def _section_leaves(count: dict[str, Any] | None, leaves: int, transcript_bytes: int) -> int:
    """The leaves a section hashes: its ``count`` row's ``hash_leaves`` when it has one (the
    CUDA commit hashes only the records, the weights and the dK leaves; the rest arrive
    hashed), else every leaf for a section that hashes the transcript, else none."""
    if count and count.get("hash_leaves") is not None:
        return int(count["hash_leaves"])
    return leaves if _hashes_transcript(count, transcript_bytes) else 0


def _hash_part(name: str) -> bool:
    return any(name.startswith(f"{o}.hash.") for o in HASH_OWNERS.values())


def _hash_breakdown(parts: dict[str, dict[int, float]], times: dict[str, dict[int, float]],
                    sections: dict[str, Any]) -> dict[str, Any]:
    """The per-step hash breakdown of the commit and of check 2 (see the module docstring).

    ``parts`` holds the ``<owner>.hash.<part>`` rows, which are not sections. An owner whose
    ``<owner>.hash`` row is missing (a run from before the breakdown) has no entry; one with
    that row but no part rows has ``has_stats`` false and null parts."""
    out: dict[str, Any] = {}
    for key, owner in HASH_OWNERS.items():
        hname = f"{owner}.hash"
        if hname not in times:
            continue
        wall = statistics.fmean(times[hname].values())
        has_stats = f"{hname}.read" in parts
        sec = sections.get(hname) or sections.get(owner) or {}
        floor = (sec.get("floors_s") or {}).get("hash")
        work = sec.get("work") or {}
        section_s = statistics.fmean(times[owner].values()) if owner in times else None
        if key == "commit":
            siblings = {x: statistics.fmean(times[f"{owner}.{x}"].values())
                        for x in ("collect", "tree", "check") if f"{owner}.{x}" in times}
        else:
            siblings = {"rest": section_s - wall} if section_s is not None else {}
        out_parts: dict[str, Any] = {}
        caller = 0.0
        for p in HASH_PARTS:
            if not has_stats:
                out_parts[p] = None
                continue
            s = statistics.fmean(parts[f"{hname}.{p}"].values()) if f"{hname}.{p}" in parts \
                else 0.0
            lane = HASH_LANES[p]
            caller += s if lane == "caller" else 0.0
            out_parts[p] = {"s": s, "share": s / wall if wall else None, "lane": lane,
                            "floor_s": floor if p == "native" else None}
        out[key] = {
            "section": owner, "n_steps": len(times[hname]), "has_stats": has_stats,
            "section_s": section_s, "siblings": siblings,
            "wall": {"s": wall, "floor_s": floor,
                     "ratio": wall / floor if floor else None},
            "leaves": work.get("leaves"), "hash_bytes": (work.get("bytes") or {}).get("hash"),
            "unaccounted_s": wall - caller if has_stats else None, "parts": out_parts}
    return out


def _run_view(markers: io.Markers, records: list[dict[str, Any]], sampler: Sampler,
              probe: dict[str, Any], trace: dict[str, Any] | None,
              layout: dict[str, Any] | None, compute_bf16: bool, gpu_memory: int | None,
              warns: set[str], steady_steps: Sequence[int] | None = None) -> dict[str, Any]:
    dv = _Devices(sampler, probe, records, markers.header.get("device"), warns)
    all_times, sides, levels = _times(records)
    # the hash parts are added seconds, not sections: they feed "hash" below only
    all_parts = {n: all_times.pop(n) for n in [n for n in all_times if _hash_part(n)]}
    for n in all_parts:
        sides.pop(n, None)
        levels.pop(n, None)
    # the step interval and a pipelined step's latency are per-step rows, not sections
    special = {n: all_times.pop(n) for n in ("step_interval", "step_wall") if n in all_times}
    for n in special:
        sides.pop(n, None)
        levels.pop(n, None)
    pipelined = "step_wall" in special or any(s.get("merged") for s in markers.steps)
    all_untraced = [s for s in markers.steps if not s["traced"] and s["step"] >= 1]
    # the warm-up step: from the time rows' untraced steps, else from the markers'
    time_steps = sorted({t for v in all_times.values() for t in v})
    marker_steps = [s["step"] for s in all_untraced]
    traced = [s for s in markers.steps if s["traced"] and s["step"] >= 1]
    traced_steps = {s["step"] for s in traced}
    known = sorted(set(time_steps) | set(marker_steps)) if steady_steps \
        else (time_steps or marker_steps)
    warm, chosen = _pick_steady(known, steady_steps, warns, sorted(traced_steps))
    used = sorted(chosen) if chosen is not None else [t for t in known if t != warm]
    if chosen is None:  # the default rule: say what it can get wrong
        last = max([*known, *traced_steps], default=None)
        if pipelined and last in used:
            warns.add(f"the steady steps include the last step ({last}), verified with the "
                      "prover idle: pass --steady-steps")
        for t in used:
            if t - 1 in traced_steps:
                warns.add(f"steady step {t} follows traced step {t - 1}: its interval holds "
                          "the trace's export")

    def keep(t: int) -> bool:
        return t in chosen if chosen is not None else t != warm

    times = _only(all_times, keep)
    parts = _only(all_parts, keep)
    steady_special = _only(special, keep)
    counts = _first_counts(records)
    leaves = _leaves(records)
    peaks: dict[str, int] = {}
    dev_peaks: dict[str, dict[str, int]] = {}  # component -> device -> reserved peak
    for r in records:
        if r.get("record") == "memory" and r.get("peak_reserved_bytes") is not None:
            peaks[r["component"]] = max(peaks.get(r["component"], 0), r["peak_reserved_bytes"])
        if r.get("record") == "memory" and isinstance(r.get("devices"), list) \
                and isinstance(r.get("device_peak_reserved_bytes"), list):
            mine = dev_peaks.setdefault(r["component"], {})
            for d, v in zip(r["devices"], r["device_peak_reserved_bytes"]):
                if v is not None:
                    mine[_norm_dev(str(d))] = max(mine.get(_norm_dev(str(d)), 0), v)
    dev_memory = {d: m for d in {d for v in dev_peaks.values() for d in v}
                  if (m := dv.memory_of(d, gpu_memory))}
    tsec = (trace or {}).get("sections", {})
    c5_scale = _c5_scale((trace or {}).get("c5_slice") or {}, warns)
    measure_reads = _measure_reads(layout, markers.header.get("device"))
    transcript = max((r["transcript_bytes"] for r in records if r.get("record") == "storage"),
                     default=0)

    # the time rows' step set: steps >= 1, untraced (the traced step only serves the trace join)
    untraced = [s for s in all_untraced if keep(s["step"])]
    warm_marks = [s for s in all_untraced if s["step"] == warm]
    depths: dict[str, int] = {}
    for s in all_untraced + traced:
        for sec in s["sections"]:
            depths.setdefault(sec["name"], sec["depth"])
    for name in all_times:  # a section with time rows and no marker spans
        if levels[name] != "total":
            depths.setdefault(name, 1 if levels[name] == "sub" else 0)
    # a time-row-only component on a side that has phase rows is the sum of its phases
    # (P2 of P2.w_t and P2.w_next, train_captured of train.*), so it is in no share total
    spanned = {sec["name"] for s in all_untraced for sec in s["sections"]}
    phase_sides = {sides[n] for n in all_times if levels[n] == "phase"}
    rollups = {n for n in depths if n not in spanned and levels.get(n) == "component"
               and sides.get(n) in phase_sides}
    if trace is not None:
        # added-seconds rows (no host span, a rest aggregate) never reach a trace
        added = {sec["name"] for s in all_untraced + traced for sec in s["sections"]
                 if not sec.get("host_us") and sec.get("rest")}
        absent = sorted(n for n in depths if n not in rollups and n not in added
                        and n not in (trace.get("sections") or {}))
        if absent:
            warns.add(f"the trace summary has no section {', '.join(absent)}")

    def build(untraced: list[dict[str, Any]], times: dict[str, dict[int, float]],
              extra: dict[str, dict[int, float]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The section table and the step's numbers over ``untraced`` marker steps and the
        time rows ``times`` and ``extra`` (the interval and latency rows; one phase: the steady
        steps, or the warm-up)."""
        seconds: dict[str, list[float]] = {}
        for name in depths:
            host = [_clock_s(sec, "host_us", "host_s") or 0.0
                    for _, sec in _occurrences(untraced, name)]
            seconds[name] = list(times[name].values()) if name in times else host
        work_s = _mean(list(times["step"].values())) if "step" in times else None
        if "step_interval" in extra:
            step_s, source = _mean(list(extra["step_interval"].values())), "step_interval"
        elif work_s is not None:
            step_s, source = work_s, "step"
        else:
            step_s = sum(_mean(seconds[n]) or 0.0 for n, d in depths.items() if d == 0)
            source = "sections"

        sections: dict[str, Any] = {}
        for name, depth in depths.items():
            occ = list(_occurrences(untraced, name))
            host_m = _mean([_clock_s(sec, "host_us", "host_s") or 0.0 for _, sec in occ])
            truncated = any(sec.get("rest") for _, sec in occ)
            gpu_m = _mean([g for _, sec in occ
                           if (g := _clock_s(sec, "gpu_us", "gpu_s")) is not None])
            host_traced = _mean([_clock_s(sec, "host_us", "host_s") or 0.0
                                 for _, sec in _occurrences(traced, name)])
            tr = tsec.get(name)
            trace_s = tr["wall_us"] * US if tr and tr.get("wall_us") is not None else None
            agree: bool | None = None
            if gpu_m is not None and host_m is not None:
                tol = max(THRESHOLDS["clock_rel"] * host_m, THRESHOLDS["clock_abs_s"])
                agree = abs(host_m - gpu_m) <= tol
            if trace_s is not None and host_traced:
                ok = abs(trace_s - host_traced) <= THRESHOLDS["trace_rel"] * host_traced
                agree = ok if agree is None else agree and ok
            spans = [sp for s, sec in occ for sp in _spans_ns(sec, s["t0_ns"])]
            listed = next((sec.get("devices") for _, sec in occ if sec.get("devices")), None)
            devs, named = dv.sections_devices(listed)
            per_dev = [dv.view(d, named).means(spans) for d in devs]
            probes = [dv.probe_of(d, named) for d in devs]
            # the headline is the mean over the devices of the section's side (its role GPUs)
            role = _role_devices(dv, devs, sides.get(name))
            pick = [i for i, d in enumerate(devs) if d in role] or [0]
            means, n = _mean_means([per_dev[i] for i in pick])
            util_devices = [devs[i] for i in pick] if devs[pick[0]] is not None else None
            time_stats = _stats(seconds[name])
            mean_t = time_stats["mean"]
            share = mean_t / step_s if step_s and mean_t is not None else None
            calls = [sec["calls"] for _, sec in occ if depth == 0 and sec.get("gpu_us") is not None]
            work = _work(name, _rate_key(name, sides, compute_bf16), counts.get(name),
                         _section_leaves(counts.get(name), leaves, transcript), tr,
                         c5_scale,
                         2 * round(_mean(calls) or 0), measure_reads=measure_reads, warns=warns)
            floors = _floors(work, probes, warns, devices=devs, gpu_hash=dv.gpu_hash,
                             holders=[dv.holds_transcript(d) for d in devs])
            floor_s = max(floors.values()) if floors else None
            bound = max(floors, key=floors.__getitem__) if floors else "unknown"
            ratio = mean_t / floor_s if floor_s and mean_t is not None else None
            marks = []
            # on a CUDA run without the section's trace, the trace-derived terms (memory
            # traffic, PCIe, peer, launch) are missing, not zero: the floor is partial
            partial = _missing_terms(name, work) \
                if floor_s and str(markers.header.get("device") or "").startswith("cuda") \
                and not tr else []
            if share is not None and ratio is not None and share >= THRESHOLDS["share"] \
                    and ratio >= THRESHOLDS["ratio"] and not partial:
                marks.append("bottleneck")
            rate_key = next(iter(work["flops"]), None)
            measured = _lane_max(bound, [per_dev[i][0] for i in pick],
                                 [probes[i] for i in pick], rate_key)
            if floor_s and mean_t and measured is not None and n >= THRESHOLDS["min_samples"] \
                    and not partial and _differ(floor_s / mean_t, measured):
                marks.append("utilization")
            if gpu_memory and peaks.get(name) is not None \
                    and peaks[name] >= THRESHOLDS["reserved"] * gpu_memory \
                    and not dev_memory.keys() & dev_peaks.get(name, {}).keys():
                marks.append("memory")
            elif any(v >= THRESHOLDS["reserved"] * dev_memory[d]
                     for d, v in dev_peaks.get(name, {}).items() if d in dev_memory):
                marks.append("memory")
            if host_m is not None and host_m >= CLOCK_MIN_S and agree is False:
                marks.append("clock")
            util = {**{f: means.get(f) for f in sampler.fields}, "n_samples": n,
                    "few_samples": n < THRESHOLDS["min_samples"], "truncated": truncated}
            by_dev = {}
            if len(devs) > 1:
                by_dev = {"util_by_device": {d: {**{f: m.get(f) for f in sampler.fields},
                                                 "n_samples": k,
                                                 "few_samples": k < THRESHOLDS["min_samples"],
                                                 "truncated": truncated}
                                             for d, (m, k) in zip(devs, per_dev, strict=True)}}
            detail = {}
            if work["bytes"].get("hbm_measure") and floors.get("hbm"):
                rs = [(p.get("rates") or {}).get("hbm_copy_bytes") for p in probes]
                sec_s = max(1.0 / r for r in rs)
                detail = {"floor_detail": {"hbm": {
                    "serving_bytes": None if partial else work["bytes"]["hbm"],
                    "serving_s": None if partial else work["bytes"]["hbm"] * sec_s,
                    "measuring_reads_bytes": work["bytes"]["hbm_measure"],
                    "measuring_reads_s": work["bytes"]["hbm_measure"] * sec_s}}}
            if partial:  # the missing terms are null: unknown, not 0.0
                fd = detail.setdefault("floor_detail", {})
                for term in ("hbm", "pcie", "nvlink", "launch"):
                    fd.setdefault(term, None)
                detail["floor_partial"] = {"missing": partial}
            sections[name] = {
                **detail, "depth": depth, "time": time_stats, "share": share,
                "clocks": {"host_s": host_m, "gpu_s": gpu_m, "trace_s": trace_s, "agree": agree},
                "util": util, "util_devices": util_devices, **by_dev,
                "work": work, "floors_s": floors, "floor_s": floor_s, "bound": bound,
                "ratio": ratio, "marks": marks, **({"rollup": True} if name in rollups else {})}
        by_side: dict[str, float] = {}
        for name, sec in sections.items():
            if sec["depth"] == 0 and not sec.get("rollup"):
                side = sides.get(name) or "run"
                by_side[side] = by_side.get(side, 0.0) + (sec["share"] or 0.0)
        wall = extra.get("step_wall")
        return sections, {"step_s": step_s, "work_s": work_s, "step_source": source,
                          "share_by_side": by_side, "pipelined": pipelined,
                          "latency_s": _mean(list(wall.values())) if pipelined and wall else None}

    c5_listed = next((sec.get("devices") for s in untraced for sec in s["sections"]
                      if sec["name"] == "5" and sec.get("devices")), None)
    c5_devs, c5_named = dv.sections_devices(c5_listed)
    c5_role = _role_devices(dv, c5_devs, "verifier") or [c5_devs[0]]
    c5_sampler = _MeanView([dv.view(d, c5_named) for d in c5_role]) if len(c5_role) > 1 \
        else dv.view(c5_role[0], c5_named)
    sections, info = build(untraced, times, steady_special)
    warmup = None
    if warm is not None:
        w_sections, w_info = build(warm_marks, _only(all_times, lambda t: t == warm),
                                   _only(special, lambda t: t == warm))
        warmup = {"step": warm, "n_steps": 1, **w_info, "sections": w_sections}

    return {
        "device": markers.header.get("device"), "n_steps": len(untraced),
        "warmup_excluded": warm is not None, "warmup": warmup, "steady_steps": used, **info,
        "share_total": sum(s["share"] or 0.0 for s in sections.values()
                            if s["depth"] == 0 and not s.get("rollup")),
        "sections": sections, "hash": _hash_breakdown(parts, times, sections),
        "c5": _c5_view(untraced, layout, c5_sampler, warns)
        if layout else None,
        "cpu_state": _cpu_state(all_untraced, dv.view(dv.primary, False)),
        "timeline": _timeline(all_untraced, sampler, dv),  # the warm-up step is drawn too
        "hook_overhead_bound": _hook_bound(markers.header.get("device"), untraced, probe, dv),
        "memory": {"gpu_memory_bytes": gpu_memory,
                   "sections": {n: {"peak_reserved_bytes": p,
                                    "fraction": p / gpu_memory if gpu_memory else None,
                                    **({"by_device": {
                                        d: {"peak_reserved_bytes": v,
                                            "fraction": v / dev_memory[d] if d in dev_memory
                                            else None}
                                        for d, v in dev_peaks[n].items()}}
                                       if n in dev_peaks else {})}
                                for n, p in peaks.items()},
                   **({"gpu_memory_bytes_by_device": dev_memory} if dev_memory else {})}}


def _plain_comparison(records: list[dict[str, Any]], plain_dir: Path, warns: set[str],
                      steady_steps: Sequence[int] | None = None) -> dict[str, Any]:
    """The plain step's sections beside the prover's ``train.*``: the plain run names them
    ``train.*`` or ``P0.*``. Both runs use the steady steps when they are given (a plain run
    lacking one of them keeps its own steady rule, with a warning)."""
    prover, prover_warm, prover_step = _steady_warm(records, steady_steps, warns)
    plain_rows = io.read_records(plain_dir)
    plain_steps = steady_steps
    if steady_steps:
        have = {t for v in _times(plain_rows)[0].values() for t in v}
        if not set(steady_steps) <= have:
            warns.add("the plain run lacks some of the steady steps: its own steady rule is used")
            plain_steps = None
    plain, plain_warm, plain_step = _steady_warm(plain_rows, plain_steps, warns)
    out: dict[str, Any] = {}
    for name, t in prover.items():
        if not name.startswith("train."):
            continue
        other = next((o for o in (name, "P0." + name.removeprefix("train.")) if o in plain), None)
        if other:
            out[name] = {"plain_name": other, "prover_s": t, "plain_s": plain[other],
                         "ratio": t / plain[other] if plain[other] else None,
                         "warmup_prover_s": prover_warm.get(name),
                         "warmup_plain_s": plain_warm.get(other)}
    return {"plain_run_dir": str(plain_dir), "sections": out,
            "step": {"prover_s": prover.get("step"), "plain_s": plain.get("step"),
                     "interval_s": prover.get("step_interval"),
                     "prover_side_s": prover.get("prover"),
                     "verifier_side_s": prover.get("verifier"),
                     "warmup_prover_s": prover_warm.get("step"),
                     "warmup_plain_s": plain_warm.get("step")},
            "warmup_excluded": {"prover": prover_step is not None,
                                "plain": plain_step is not None},
            "warmup_step": {"prover": prover_step, "plain": plain_step}}


def _steady_warm(rows: list[dict[str, Any]], steady_steps: Sequence[int] | None = None,
                 warns: set[str] | None = None
                 ) -> tuple[dict[str, float], dict[str, float], int | None]:
    """Mean seconds per component over the steady untraced steps (``steady_steps`` when given,
    else all but the warm-up), the warm-up step's seconds per component, and the warm-up step
    (``None`` when only one step exists)."""
    times = _times(rows)[0]
    warm, chosen = _pick_steady(sorted({t for v in times.values() for t in v}), steady_steps,
                                set() if warns is None else warns)
    steady = {n: statistics.fmean(v.values()) for n, v in _only(
        times, lambda t: t in chosen if chosen is not None else t != warm).items()}
    first = {n: v[warm] for n, v in times.items() if warm in v}
    return steady, first, warm


def _leaves(records: list[dict[str, Any]]) -> int:
    """The step's leaf count: the largest of the ``storage`` rows' (and, for a run that wrote
    no ``storage`` rows, the ``step`` records')."""
    kinds = ("storage",) if any(r.get("record") == "storage" for r in records) \
        else ("step",)
    return max((r["leaves"] for r in records
                if r.get("record") in kinds and r.get("leaves") is not None), default=0)


def _work_meta(records: list[dict[str, Any]], view: dict[str, Any]) -> dict[str, Any]:
    """``n_s``, ``M``, ``n_leaves`` and the µs per leaf of the labelling and the commit."""
    env = next((r for r in records if r.get("record") == "environment"), {})
    caps = [r["n_products"] for r in records
            if r.get("record") == "capture" and r.get("n_products") is not None]
    n_leaves = env.get("n_leaves") or _leaves(records) or None
    secs = view["sections"]

    def per_leaf(name: str) -> float | None:
        mean = ((secs.get(name) or {}).get("time") or {}).get("mean")
        return None if mean is None or not n_leaves else mean / n_leaves * 1e6

    return {"n_s": env.get("n_s"), "M": env.get("M") or max(caps, default=None),
            "n_leaves": n_leaves,
            "us_per_leaf": {"label": per_leaf("P1.label"), "commit": per_leaf("P3.commit")}}


def _read(label: str, path: Path | None, reader, warns: list[str], default: Any) -> Any:
    """``reader(path)``, or ``default`` with one warning when the file is absent or empty.
    The reader's own warnings (a skipped line kind) join ``warns``."""
    if path is None:
        warns.append(f"no {label} file was given")
        return default
    if not path.exists():
        warns.append(f"{label}: {path} does not exist")
        return default
    if path.stat().st_size == 0:
        warns.append(f"{label}: {path} is empty")
        return default
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = reader(path)
    warns += [str(w.message) for w in caught]
    return out


def analyze(run_dir: Path, samples: Path | None = None, probe: Path | None = None,
            trace_summary: Path | None = None, plain: Path | None = None,
            compute_dtype: str = "float32", markers: Path | None = None,
            session: str | None = None, steady_steps: Sequence[int] | None = None
            ) -> dict[str, Any]:
    """The analysis. A missing or empty markers, samples or probe file, or a markers file
    with no steps, adds a warning instead of failing: the sections then come from the time
    rows alone, utilization is null and the probe-based floors are skipped."""
    warns: list[str] = []
    mk = _read("markers", markers or run_dir / io.MARKERS_FILE, io.read_markers, warns,
               io.Markers({}))
    if not mk.steps and mk.header:
        warns.append("markers: the file has no step lines")
    records = io.read_records(run_dir)
    sampler = Sampler(_read("samples", samples, io.read_samples, warns,
                            io.Samples({"mono_ns": 0}, [])))
    no_probe: dict[str, Any] = {"rates": {}, "costs": {}}
    pr = _read("probe", probe, io.read_probe, warns, no_probe)
    trace = io.read_trace_summary(trace_summary) if trace_summary else None
    layout_path = run_dir / io.C5_LAYOUT_FILE
    layout = io.read_c5_layout(layout_path) if layout_path.exists() else None
    dtype = _compute_dtype(records, compute_dtype, warns)
    gpu_memory = _gpu_memory(pr, records)
    if gpu_memory is None:
        warns.append("no machine block has gpu_memory_bytes: the memory mark is skipped")
    soft: set[str] = set()
    view = _run_view(mk, records, sampler, pr, trace, layout,
                     dtype.removeprefix("torch.") in ("bfloat16", "bf16"), gpu_memory, soft,
                     steady_steps)
    run = mk.header.get("run") or next((r["run"] for r in records if "run" in r), "unknown")
    env = next((r for r in records if r.get("record") == "environment"), {})
    meta = {"model": env.get("model"), "attack": env.get("attack"), "scenario": run,
            "device": mk.header.get("device") or env.get("device"), "compute_dtype": dtype,
            "git_commit": env.get("git_commit"), "session": session,
            **_work_meta(records, view)}
    if meta["n_s"] is None:
        warns.append("no environment record names n_s: meta.n_s and meta.M may be null")
    if session is None:
        warns.append("no --session was given: meta.session is null")
    machine = next((m for m in [pr.get("machine"), env.get("machine")]
                    if isinstance(m, dict) and m), None)
    comparisons = None if plain is None \
        else _plain_comparison(records, plain, soft, steady_steps)  # may add warnings
    return {"version": io.VERSION, "meta": meta,
            "probe": None if pr is no_probe else {k: v for k, v in pr.items() if k != "machine"},
            "machine": machine, "run": run, "compute_dtype": dtype,
            "thresholds": THRESHOLDS,
            "warnings": warns + sorted(soft) + list((trace or {}).get("warnings") or []),
            "runs": {run: view},
            "comparisons": comparisons}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Join the profiling files into profile.json.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--markers", type=Path, default=None,
                   help="default: markers.jsonl in the run directory")
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--probe", type=Path, default=None)
    p.add_argument("--trace-summary", type=Path, default=None)
    p.add_argument("--plain", type=Path, default=None, help="the plain run's directory")
    p.add_argument("--steady-steps", type=parse_steps, default=None,
                   help="the steady steps, e.g. 2-3 or 2,3 (default: every untraced step after "
                        "the warm-up); give them when the last pipelined step or a step after "
                        "a traced one would be in the default set")
    p.add_argument("--compute-dtype", default="float32",
                   help="used when no environment record names compute_dtype")
    p.add_argument("--session", required=True,
                   help="the session's label with its date, e.g. profile-check-2026-10-20")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    result = analyze(a.run_dir, a.samples, a.probe, a.trace_summary, a.plain, a.compute_dtype,
                     a.markers, a.session, a.steady_steps)
    a.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
