# Resource Profiling of the Full-Scale Runs — Plan

> **Frozen archive. Don't edit this file.** The resource-profiling plan, approved on
> 2026-10-19 and archived on 2026-10-20 once tasks FS18–FS27 were built and the report was
> reworked. The code it planned lives in `setup/machine.py`, `setup/sampler.py`,
> `verification/runs/` and `evaluation/results/profile/`. `../IMPLEMENTATION_PLAN.md` still
> points here for the second H100 session's profile (§11) and the report's choices (§12).

> **Status.** Approved by the user 2026-10-19, with FS27 on an H100. Session projects-cc
> (the O2 orchestrator) implements it; the plan's author session no longer edits it. Tasks
> FS18–FS27 and milestone FM8 in `IMPLEMENTATION_PLAN.md` point here.
>
> **Review 2026-10-19 (projects-cc, for check 5's speed item O2).** Added: the `c5.copy`
> stamp and allocator counters (§5.1, §5.2, FS19), the layout's stored and served bytes
> (§5.2), CPU affinity and the GPU's NUMA node (FS18), three copy and first-touch rates in
> the probe (§5.5, FS22). Decided by the user 2026-10-19: FS27 runs on an H100 and adds
> full-scale check-5 parts (§2 item 9, FS27).
>
> **Report choices 2026-10-19 (user, edited by session projects-82).** §12's questions are
> settled. For them, `profile.json` gains `meta` (§5.7), FS25's CLI gains `--session`, and
> FS26's brief is final.
>
> **Report rework 2026-10-20 (user, edited by session 2bef5eb0).** After the first H100 report
> the user asked for a page a reader can follow. §12 item 8 records the choices, items 2, 4, 5
> and 6 are amended to match, and FS26's brief describes the new panels.

## 1. Goal

On 2026-10-18 the user asked for full resource profiling of the full-scale runs:

- time and utilization for each section of the protocol, tied to the hardware parts that do
  the work (GPU compute, GPU memory bandwidth, PCIe, CPU threads, host RAM, and disk);
- bottlenecks recognized and marked;
- the measurements added to the resource logging the runs already write;
- an HTML diagram of the computation's architecture and components that presents the
  profile after the run. A live view isn't needed.

The profile serves two uses: investigating the second H100 session's costs (FS14), and
judging later changes to the code (`PERFORMANCE_TASKS.md`). It rides on the runs FS14
already plans. Before FS14 it adds one short H100 check (FS27), which also measures check 5
at full scale for the check-5 speed item (O2).

## 2. Settled choices

The user settled these choices in the clarification round of 2026-10-18 and 2026-10-19. Each
item gives the choice, then its reason.

1. **Two layers.** A light sampler runs on every part of every session. One heavy traced step
   per model runs under `torch.profiler`. The light layer gives every step's timeline at a
   cost below 1%; the heavy layer gives kernel-level detail for one step, whose times are
   never reported as the run's times.
2. **Close the gaps in what the runs record.** The CUDA clocks were never cross-checked, and
   the hook's overhead was measured on CPU only. `MatmulCapture.summary` and
   `check5_profile`'s output are never saved. The timed pass doesn't write transcript bytes,
   and the environment record lacks RAM total, power limit and PCIe link. The plan adds a
   three-clock cross-check (host clock after a sync, CUDA events, and the trace's kernel span),
   a capture record per step, `check5_profile` output in the run folder, transcript bytes in
   the timed pass, and machine facts in a shared module.
3. **Two dependencies, no privileged tools.** `nvidia-ml-py` (NVML, the driver's monitoring
   library) and `psutil` (process and host counters). No Nsight Compute, Nsight Systems or
   DCGM: rented pods often deny the GPU performance-counter permission they need, and DCGM
   needs a host service. GPU efficiency and memory traffic are estimated from op shapes in the
   trace, and from NVML's GPU Performance Monitoring (GPM) metrics where the pod allows them.
4. **The sampler is its own process.** It reads NVML and the run's process tree every 100 ms
   and writes its own file. The run writes section start and end times on the system
   monotonic clock, and the two files are joined after the run. A separate process can't slow
   the run's Python thread or change its numerics.
5. **Three levels of detail.** (1) Every metrics-hook section, about 25 rows, with every
   measure. (2) Drill-downs from the traced step: check 5 by its 30 product classes
   (`product_class`), the prover's capture by op kind, and the hashing against the probe's
   rate. (3) Every timed step stamps each check-5 run in the marker file, so check 5's
   progress shows over time on every step. The trace covers a slice of check 5: the first run
   of each class plus every 20th run (about 5%), scaled up per class by product count. No
   per-layer or per-product rows.
6. **Bottleneck rule.** A section's floor is the time its work needs at the hardware's
   measured rates: the max over its resources of work ÷ rate. A ~2-minute probe at the
   session's start measures the rates, and the datasheet gives a second line. A section is
   marked as a bottleneck when it takes at least 10% of the step and at least 3× its floor.
   The bound type is the resource whose floor is nearest the measured time, cross-checked
   against the sampler's utilization. Two more marks: reserved GPU memory at 95% or more of
   usable memory, and clocks that disagree. The prover's phases sit beside the plain step's.
   The thresholds are hand-picked and can change after FS27.
7. **Fold into FS14.** The sampler, the check-5 stamps, the capture record, transcript bytes
   and machine facts ride on every FS14 run. The traced step is H1's step 5 in each model's
   refusal cell (after calibration's steps 1–3 and the stored step 4), marked `traced` and
   skipped by the cost tables. A GPU hook on/off comparison is dropped, because check 5's 2–4×
   run-to-run variance (FS17) would swamp it; a bound replaces it: syncs × the probe's sync
   cost ÷ step time. Extra pod time is about 10–15 min per session.
8. **Process.** Questions first, then this plan, then implementation. The HTML report's
   questions don't block the measurement code and are discussed during implementation (§12).
9. **A short H100 check before FS14.** One H100 SXM pod (secure, $3.99/h), about 2–3.5 h,
   about $8–14, with the self-delete at about 6 h. It tests every CUDA part on SmolLM2 and
   measures check 5 on the four full-scale models (FS27). Changed 2026-10-19 from a 24–48 GB
   pod running SmolLM2 alone (about $1–2): check 5 takes about 1 s at SmolLM2 scale, so its
   full-scale cost (46–411 s per step in FS17) would first show in FS14, after the fix O2
   needs it for. The check also replaces O2's own planned check-5 session (about $6).

## 3. What is measured

The profile draws on eight sources. The table lists each one, the task that builds it, where
it runs and what it gives.

| Source | Task | Runs | Gives |
|---|---|---|---|
| Metrics hook (exists) | — | every timed step | host time per section (after a sync on CUDA), CUDA-event time for nested sections, peak allocated and reserved memory per section, FLOPs and hash bytes in the counting pass |
| Machine facts | FS18 | once per run | CPU model and counts, cgroup quotas, NUMA, RAM total, GPU name, power limit, max clocks, PCIe generation and width, driver |
| Markers and stamps | FS19 | every timed step | each section's start and end on the monotonic clock and on CUDA events; each check-5 run's glue, serve, measure and judge spans |
| Capture record, transcript bytes | FS20 | every timed step | `MatmulCapture.summary()` (calls and products per phase and op, copies, `q` range); bytes and leaves committed |
| `check5_profile` output | FS20 | per call | its per-section table, saved beside the run |
| Sampler | FS21 | every session part | 100 ms samples of GPU, CPU, RAM, disk and cgroup counters |
| Probe | FS22 | once per session | the machine's measured rates and fixed costs |
| Traced step | FS23 | one step per model | kernels, copies, launches and syncs per section; check 5 by class |

The profile ties each section to six hardware lanes and one overhead lane. The table names
each lane's measures and its floor's rate.

| Lane | Measured by | Floor's rate |
|---|---|---|
| GPU compute (SM, tensor cores) | NVML utilization; GPM SM-active, occupancy and tensor-pipe use; trace kernel time | probe GEMM rate by dtype |
| GPU memory (HBM) | NVML memory utilization; GPM DRAM bandwidth; trace byte estimate | probe copy and reduction rates |
| PCIe | NVML or GPM transmit and receive rates; trace copy bytes | probe pinned and pageable copy rates |
| CPU threads | process-tree CPU seconds, busy cores, other load, CPU frequency, cgroup throttling | probe hashing rate at the run's threads |
| Host RAM | tree RSS, available RAM, swap, page faults | probe host copy rate |
| Disk | system and tree read and write rates | none (the transcript stays in RAM, F1c) |
| Launch and sync overhead | trace launch and sync counts | probe launch and sync costs |

The sections are those the metrics hook already times. The prover's are `train.load`,
`train.forward`, `train.backward`, `train.update`, `P1.label`, `P2.w_t`, `P2.w_next`,
`P3.commit`, `P4.paths` and `P5.write`. The harness's are `handoff.*`. The verifier's are the
checks in `DEFAULT_ORDER` (`4`, `7`, `2`, `6a`, `5`, `6b`, `2.root`) with the nested `5.glue`,
`5.measure` and `6b.glue`. The run sections are `run:0`, `run:1`, `run:8`, `run:9` and
`run:P4.tree`.

## 4. Data flow

A profiled run writes its files in its run directory, and the session part's directory holds
the sampler's file. The flow is the following:

1. The probe part writes `probe.json` at the session's start.
2. The part launcher starts the sampler beside each part; it writes `samples.jsonl` in the
   part directory.
3. The run writes `records.jsonl` (with the capture record and the extra fields of §5.3),
   `markers.jsonl`, `c5_layout.json` and, on the traced step, `trace/`.
4. After the session, on the Mac: `evaluation.results.profile.trace` reads the trace and
   writes `trace_summary.json`; `evaluation.results.profile.analyze` joins everything into
   `profile.json`; `evaluation.results.profile.report` writes `profile.html`.

The analysis package sits under `evaluation/results/`, so it reads files only and imports no
torch, `setup` or `verification` (rule 6 of `tests/test_layering.py`).

## 5. File formats (version 1, frozen)

The formats below are frozen at version 1, so FS24 and FS25 can build against them in
parallel with the code that writes them. A change goes through the orchestrator, who edits
this section first.

### 5.1 `markers.jsonl`

The file holds one JSON object per line, in the run directory beside `records.jsonl`. Its
first line is the header:

```json
{"kind": "header", "version": 1, "run": "honest", "pid": 1234, "device": "cuda",
 "mono_ns": 123456789, "wall_unix_s": 1760000000.0}
```

`mono_ns` and `wall_unix_s` are read together, so a reader can map monotonic times to wall
time. Each timed step writes one step line:

```json
{"kind": "step", "scenario": "honest", "step": 5, "traced": false,
 "t0_ns": 123, "t1_ns": 456,
 "sections": [{"name": "5", "depth": 0, "calls": 1,
               "host_us": [[0.0, 98000000.0]], "gpu_us": [[3.1, 97999990.0]],
               "rest": null,
               "alloc": {"retries": 0, "device_alloc": 2, "device_free": 0, "ooms": 0}}],
 "stamps": {"name": [0, 1], "run": [0, 0], "group": [-1, 0], "window": [-1, -1],
            "h0": [10.0, 11.0], "h1": [50.0, 30.0], "g0": [10.5, 11.2], "g1": [49.8, 29.9]},
 "stamp_names": ["c5.run", "c5.glue", "c5.serve", "c5.copy", "c5.measure", "c5.judge"]}
```

Its fields mean the following:

- `t0_ns` is the step's first section start; `t1_ns` is the moment `on_step` runs. Both are
  `time.monotonic_ns()`.
- Every time in `host_us`, `gpu_us`, `h0`, `h1`, `g0` and `g1` is in microseconds from
  `t0_ns`.
- `host_us` lists a section's spans, at most 64 per section per step. Calls beyond 64 fold
  into `rest`: `{"calls", "host_s", "gpu_s", "first_us", "last_us"}`.
- `gpu_us` holds the CUDA-event spans, or `null` off CUDA (CPU and MPS).
- `alloc` holds, for a top-level section on CUDA, the change over the section of four
  `torch.cuda.memory_stats()` counters: `num_alloc_retries`, `num_device_alloc`,
  `num_device_free` and `num_ooms`. It's `null` for a nested section and off CUDA. The read
  is a host call with no sync. A retry means the caching allocator freed its cache and
  synced to fit a request, a cost that sits near a full GPU (FS17's 411 s check 5 on
  Qwen2.5-1.5B ran at 75–78 GB reserved).
- `stamps` is column-wise; `name` indexes `stamp_names`. An id that doesn't apply is `-1`.
  Off CUDA `g0` and `g1` are `null` (the column, not a list of nulls).
- The header's `device` is the device type (`cuda`, `cpu`, `mps`), not `cuda:0`.
- A run section (`run:*`) writes a `{"kind": "run", ...}` line with the same section fields
  and no stamps.

The GPU times come from anchors. A top-level section records an anchor event right after its
entry sync, with the host's `monotonic_ns`, and an end event at exit, before the sync. A
nested section or a stamp records its host `monotonic_ns` without a sync, plus a CUDA event
pair. A GPU time is `anchor_host_us + anchor.elapsed_time(e) × 1000`, resolved when the
top-level section exits (it syncs there already).

### 5.2 Check-5 stamps and `c5_layout.json`

Check 5's main loop writes six stamps:

| Stamp | Ids | Covers |
|---|---|---|
| `c5.run` | run | one run of `_member_runs`: one dense product, or one attention product's members in one layer |
| `c5.glue` | run, group | `replay.operand_group` or `replay.operands` |
| `c5.serve` | run, group | the host stack of the products and their copy to the device (`p_dev`) |
| `c5.copy` | run, group | inside `c5.serve`: the `.to(device, …)` call alone, so the host stack is `c5.serve − c5.copy` |
| `c5.measure` | run, group | `_measure_served`'s measuring, the span the `5.measure` section times |
| `c5.judge` | window | one `_judge_window` call, its `.tolist()` included |

`c5_layout.json` describes the runs, written once per run directory:

```json
{"version": 1, "run": "honest", "n_runs": 812, "n_groups": 830, "n_windows": 3,
 "entries": [{"run": 0, "group": 0, "window": 0, "layer": 1, "class": "Q.fwd",
              "kind": "dense", "first": 1, "members": 1,
              "a_shape": [4096, 576], "b_shape": [576, 576],
              "operand_bytes": 11796480, "product_bytes": 4718592,
              "product_dtype": "bf16", "served_bytes": 9437184}],
 "trace_slice": [0, 20, 33, 40]}
```

`product_bytes` counts the products as stored in the transcript, in `product_dtype`;
`served_bytes` counts them as check 5 measures them on the device (fp32). With
`c5.copy`'s span, the two give each run's effective copy rate on every step.

A blocking `Tensor.to(cuda, float32)` from a bf16 host tensor converts on the CPU first
(PyTorch's `copy_kernel_cuda`: "for blocking transfers conversions are performed on CPU"),
so it allocates a fresh fp32 host tensor and sends twice the stored bytes over PCIe. Check
5's serving does this today (`checks.py` `_served`, `_served_group`), so `c5.copy` and the
probe's `h2d_bf16_to_fp32_blocking_bytes` (§5.5) are what tell that cost apart.

`trace_slice` lists the traced runs: the first run of each class plus every run whose id is a
multiple of 20.

Amended 2026-10-19 (FS19 review):

- Group ids are global across the step's runs, numbered 0 to `n_groups − 1` in run order, in
  both the layout and every stamp. FS25 joins a stamp to its layout entry by `group` alone.
- `n_windows` equals the number of `c5.judge` stamps, also when the last run exactly fills a
  window.
- The first run of each class is taken over all entries, not group 0 alone; each run is
  listed once, sorted.

### 5.3 Additions to `records.jsonl`

The records file gains these fields and one record type:

- The step record gains `transcript_bytes`, `leaves` and `traced`.
- Every time row gains `traced`.
- A `capture` record per timed step: `{"record": "capture", "run", "scenario", "step",
  "traced", ...the keys of MatmulCapture.summary()..., "host_buffers": {"n_buffers",
  "nbytes"} | null}`.
- The environment record gains `machine`: the output of `setup.machine.machine_facts()`,
  and `compute_dtype` (amended 2026-10-19), which FS25 reads for the GEMM floor's rate.

### 5.4 `samples.jsonl`

The sampler writes one header line, then flat sample lines. The header:

```json
{"kind": "header", "version": 1, "pid": 1234, "interval_s": 0.1, "mono_ns": 1, "wall_unix_s": 1.0,
 "host": "pod-abc", "gpu": {"index": 0, "name": "NVIDIA H100 80GB HBM3", "uuid": "GPU-..."},
 "nvml": true, "gpm": true, "gpm_error": null, "pcie_source": "gpm",
 "cpu_logical": 32, "cpu_physical": 16, "cgroup": {"cpu_quota": 16.0, "mem_limit": null},
 "fields": ["t", "g_util", "..."]}
```

Each sample line holds `t` (monotonic ns) and the fields below; a field that can't be read is
`null`. A rate is the change over the interval from the previous sample's `t` to this one's.

| Group | Fields |
|---|---|
| GPU | `g_util`, `g_mem_util`, `g_mem_used`, `g_sm_mhz`, `g_mem_mhz`, `g_power_w`, `g_temp_c`, `g_throttle` (bitmask), `g_pcie_tx`, `g_pcie_rx` (bytes/s), `g_other_procs` |
| GPU (GPM) | `g_sm_active`, `g_occupancy`, `g_tensor`, `g_dram_bw` (fractions) |
| CPU | `c_tree` (cores used by the run's process tree), `c_busy` (cores busy machine-wide), `c_other` (`c_busy − c_tree`), `c_self` (the sampler's own), `threads` |
| Memory | `rss` (tree), `ram_used`, `ram_avail`, `swap_used` |
| I/O | `disk_r`, `disk_w` (machine, bytes/s), `io_r`, `io_w` (tree, bytes/s) |
| System | `cpu_mhz`, `load1`, `minflt`, `majflt` (tree, per second), `cg_throttled_us`, `cg_nr_throttled` (cgroup `cpu.stat`, per interval) |

Amended 2026-10-19 (FS21 review): a problem the sampler meets mid-run writes one
`{"kind": "error", ...}` line and the run goes on. A GPM metric that fails in a sample is
`null` there. If GPM fails for every sample of the first ~5 s, `g_pcie_tx` and `g_pcie_rx`
switch to NVML's PCIe counters, with one error line saying so (the header keeps
`pcie_source: "gpm"`). `g_other_procs` is `null`, with one error line, when NVML's process
ids don't resolve locally (a container's pid namespace). The cgroup is read from
`/proc/self/cgroup`, and the file is flushed after every line.

### 5.5 `probe.json`

The probe writes one JSON object:

```json
{"version": 1, "device": "cuda", "machine": {"...": "machine_facts()"},
 "threads": 8, "determinism": {"tf32": false, "deterministic": true},
 "rates": {"gemm_bf16_flops": 7.1e14, "gemm_fp32_flops": 5.0e13,
           "skinny_bmm_fp32_flops": 2.0e12, "hbm_copy_bytes": 2.9e12,
           "hbm_reduce_bytes": 3.0e12, "h2d_pinned_bytes": 5.5e10,
           "d2h_pinned_bytes": 5.5e10, "h2d_pageable_bytes": 2.0e10,
           "d2h_pageable_bytes": 1.5e10,
           "h2d_bf16_to_fp32_blocking_bytes": 6.0e9, "h2d_bf16_cast_on_gpu_bytes": 1.5e10,
           "host_first_touch_bytes": 5.0e9,
           "hash": {"per_byte_s": 1.0e-10, "per_leaf_s": 2.0e-6, "threads": 8},
           "host_memcpy_bytes": 1.2e10},
 "costs": {"launch_s": 5e-6, "sync_s": 1.5e-5, "event_pair_s": 4e-6,
           "tolist_s": 3e-5, "empty_cache_s": 0.002},
 "shapes": [{"name": "...", "dtype": "bf16", "m": 1, "n": 1, "k": 1, "flops_per_s": 1.0}],
 "datasheet": {"name": "H100 SXM", "bf16_flops": 9.894e14, "tf32_flops": 4.947e14,
               "fp32_flops": 6.7e13, "hbm_bytes": 3.35e12, "pcie_bytes": 6.4e10},
 "errors": {}, "wall_s": 118.0}
```

A rate the device can't measure is `null`, with its reason under `errors`. `datasheet` is
`null` for a GPU the table doesn't hold. The two `h2d_bf16_*` rates count stored (bf16)
bytes per second; `host_first_touch_bytes` is the rate of allocating and first writing
fresh host memory, the page-fault cost a new large host tensor pays.

### 5.6 The trace and `trace_summary.json`

The traced step writes `trace/step_00005.json.gz` (a Chrome trace from `torch.profiler`) and
`trace/meta.json`:

```json
{"version": 1, "step": 5, "slice": [0, 20, 33], "c5_slice": "slice", "torch": "2.x",
 "profiler": {"activities": ["cpu", "cuda"], "record_shapes": true, "with_flops": true},
 "start_ns": 1, "stop_ns": 2, "error": null}
```

`c5_slice` is `"slice"` or `"all"` (every check-5 run traced). FS24 writes
`trace_summary.json` with, per section:

- `wall_us`, `kernel_us` (sum of kernel durations), `kernel_span_us` (union of kernel
  intervals), `gpu_idle_us` (`wall_us − kernel_span_us`), `kernels`, `launches`;
- `categories_us` by kind: `gemm`, `attention`, `elementwise`, `reduction`, `memcpy_h2d`,
  `memcpy_d2h`, `memcpy_d2d`, `memset` and `other`;
- `copy_bytes` by direction, `gemm_flops`, `gemm_bytes` (estimated from shapes),
  `sync_wait_us`, and `top_cpu_ops` (the ten with the most self time).

Beside the sections it writes `c5_slice` by product class (`runs_traced`, `runs_total`,
`scale`, `phases_us`, `scaled`) and a `warnings` list.

Amended 2026-10-19 (FS24 review): sections nest and their numbers are inclusive, as the
time rows are, and each section carries `parent`, the enclosing section of its earliest
occurrence (`null` at the top). The file is `{"version", "sections", "c5_slice", "warnings"}`.

### 5.7 `profile.json`

FS25 writes one object per analysis. Its `meta` object names the run: `model`, `attack`,
`scenario` (the markers header's `run`), `device`, `compute_dtype` and `git_commit`, taken from
the environment record, and `session`, the label of the session that ran it, from FS25's
`--session`. The report (FS26) groups its inputs by `model` and `session` (§12 item 4).
`meta` was added on 2026-10-19, before FS25 merged, so the version stays 1.

Per run and section it holds:

- `time` (mean, median, min, max over the timed steps), `share` (of the step);
- `clocks`: `host_s`, `gpu_s`, `trace_s`, `agree`;
- `util`: each sampler field averaged with weights by overlap, `n_samples`, `few_samples`;
- `work`: FLOPs by dtype, bytes by lane, leaves, launches and syncs;
- `floors_s` by resource, `floor_s` (their max), `bound`, `ratio` (time ÷ floor), `marks`.

Beside the sections it holds a `c5` view (classes, phases, runs over time, variance), the
`hook_overhead_bound`, the memory marks, the comparisons (prover beside plain), and the
thresholds: `{"share": 0.10, "ratio": 3.0, "reserved": 0.95, "clock_rel": 0.05,
"clock_abs_s": 0.002, "trace_rel": 0.10, "min_samples": 3}`.

Amended 2026-10-19, additive, so the version stays 1:

- Top level, `probe` (the probe file without its `machine` block, or `null`) and `machine`
  (the probe's machine facts, else the environment record's, else `null`). They stay in the
  file for investigation; the report no longer draws them (§12 item 8).
- Per run, `timeline`: one entry per untraced timed step, with each section's spans in
  seconds from the step's start, the step's end, and sampler strips (`g_util`, `c_busy`,
  `g_pcie_tx`, `g_pcie_rx`) cut to the step and averaged into at most 400 bins. The report's
  timeline (§12 item 3) draws from it.
- A section that has time rows and no marker spans, on a side that has phase rows (`P2` of
  `P2.w_t` and `P2.w_next`, `train_captured` of `train.*`), is the sum of its phases: it's a
  rollup and is left out of `share_total`, which is otherwise counted twice.
- The choices this section leaves open (the GEMM rate per section, the compute dtype's and
  usable memory's sources, check 5's trace scale, the hash leaves rule, sync counts, the
  measured lanes) are listed in `analyze.py`'s module docstring.

## 6. The bottleneck rule

The analysis computes each section's floor and marks in these steps:

1. **Work.** FLOPs by dtype come from the counting pass (`count` records). Hash bytes and
   leaves come from the counting pass too. Copy bytes, launches and syncs come from the trace
   summary, scaled to the step. Memory bytes come from the trace's shape estimate.
2. **Floors.** One floor per resource:
   - `gemm`: FLOPs of each dtype ÷ the probe's GEMM rate of that dtype;
   - `hbm`: estimated bytes ÷ the probe's copy rate;
   - `pcie`: bytes of each direction ÷ the pageable rate (the trace doesn't tell pinned
     from pageable), and the floor is the larger direction's time, since the link is full
     duplex (amended 2026-10-19);
   - `hash`: `bytes × per_byte_s + leaves × per_leaf_s`;
   - `launch`: launches × `launch_s`, plus syncs × `sync_s`.
3. **Floor and bound.** `floor_s` is the max of the floors; `bound` names that resource;
   `ratio = time ÷ floor_s`.
4. **Bottleneck mark.** Set when `share ≥ 0.10` and `ratio ≥ 3`.
5. **Utilization mark.** Set when the floor-implied utilization (`floor_s ÷ time`) and the
   sampler's measured utilization on the bound's lane differ by more than 2×, with at least
   3 samples. It flags a floor or a counter that's wrong.
6. **Memory mark.** Set when a section's peak reserved memory reaches 95% of usable GPU
   memory.
7. **Clock mark.** Set for a section of at least 50 ms when host and GPU times differ by more
   than 5% or 2 ms, whichever is larger, or the trace time differs from host time by more than
   10% on the traced step.

The datasheet floors appear beside the probe floors in the report; only the probe floors
drive the marks.

## 7. Overhead

The profiling must not change what it measures. Each part's cost and its guard are as
follows:

- **Sampler.** A separate process, below 2% of one core (FS21 tests it). NVML queries run in
  the driver and add no GPU work.
- **Markers.** A host clock read per section entry and exit (about 25 sections per step) and,
  on CUDA, one event per section beyond the syncs that exist already.
- **Stamps.** About 4 per check-5 run, a few thousand per step: one host clock read and, on
  CUDA, one event pair each (a few µs). That's tens of ms against check 5's 45–400 s.
- **Transcript bytes.** One pass over the leaves' headers per step, in a harness section
  outside the protocol's time (FS20 measures it).
- **Traced step.** Slower by design, marked `traced` and excluded from the cost tables.
- **Hook overhead bound.** `profile.json` reports `(syncs × sync_s + events × event_pair_s) ÷
  step time`. FS14's acceptance needs it below 1%.

Bit identity is the hard guard: every task that touches the run's code tests that verdicts,
residuals, roots and final weights are identical with profiling on and off.

## 8. Working rules for the agents

The tasks run under these rules, which differ from the full-scale tasks' `opus` rule by the
user's preference of 2026-10-18:

- One Sonnet coder per task at medium effort, then a Sonnet tester and a Sonnet reviewer. At
  most 10 agents at once. The orchestrator (Opus) merges.
- Each coder works in its own worktree off `verification-impl` and commits there. Nothing is
  pushed. Before each wave the orchestrator lists the worktrees and checks that the base is
  current.
- Agents never edit `docs/verification/`. A format question or a brief that doesn't fit the
  code goes back to the orchestrator.
- Dependencies install into `.venv` only.
- Done means: the brief's tests pass, the fast suite passes, `tests/test_layering.py` passes,
  and the bit-identity tests pass.
- The tester runs the fast suite and the brief's tests, then tries the brief's edge cases
  (missing GPU, missing NVML, a failing reader, an empty file) and reports what broke.
- The reviewer reads the diff against the brief and its review focus, and checks the
  writing rules (`writing-tenets.md`): no comments beyond the surrounding code's density.
- Real-model runs on the Mac run one at a time.

## 9. Task briefs

Each brief gives the goal, the files, the steps, the tests with their expected results, when
it's done, and what the reviewer checks first.

### Wave 0 — dependencies (orchestrator)

Add `psutil` and `nvidia-ml-py` to `pyproject.toml`, each with a reason comment, install them
into `.venv`, and add both to `fs11_prep.PACKAGES` (`verification/runs/fs11_prep.py`), so the
pod installs them. `nvidia-ml-py` imports as `pynvml`.

### FS18 — machine facts in a shared module

**Goal.** One function gives the machine facts to the runs and to the hashing benchmark.

**Files.** `setup/machine.py` (created), `benchmarks/hashing/machine.py`,
`verification/runs/metrics.py` (`environment_record`), tests under `tests/setup/`.

**Steps.**
1. Move `_Record`, `_cpu`, `_frequency`, `_gpu`, `_numa` and the cgroup helpers from
   `benchmarks/hashing/machine.py` into `setup/machine.py`. The benchmark imports them back,
   so `machine_record()`'s output keys stay the same.
2. Add `machine_facts() -> dict`, with the moved facts plus `ram_total_bytes` (from
   `os.sysconf`, no psutil) and an `nvml` block: driver, power limit (W), max SM and memory
   clocks, PCIe max and current generation and width, and UUID. Add the process's CPU
   affinity (`os.sched_getaffinity`, Linux only) and the GPU's NUMA node (read from
   `/sys/bus/pci/devices/<bus id>/numa_node`), since a pageable copy's rate depends on
   where the host pages sit. `pynvml` is imported inside
   the function and every call is guarded, so the block is `null` without it or without a
   GPU, and one failed read gives one `null` field.
3. `environment_record` gains `"machine": machine_facts()`.

**Tests.**
- `machine_record()`'s key set equals the set before the move (write the set into the test
  first, from the code as it stands).
- `machine_facts()["ram_total_bytes"] > 0`; its `nvml` is `null` on the Mac.
- A fake `pynvml` that raises on one call gives `null` for that field only.
- `tests/test_layering.py` passes.

**Done when** the tests pass and the hashing benchmark's smoke test passes unchanged.

**Review focus.** No behavior change in the benchmark; no import of `pynvml` at module load.

### FS19 — markers and check-5 stamps

**Goal.** Every timed step writes its sections' spans on the monotonic clock and on CUDA
events, and check 5 stamps each run, in `markers.jsonl` (§5.1) and `c5_layout.json` (§5.2).

**Files.** `verification/runs/metrics.py`, `verification/verifier/context.py`,
`verification/verifier/checks.py`, `verification/runs/scenarios.py`,
`verification/runs/plain_baseline.py`, `verification/runs/run_verified.py`, tests.

**Steps.**
1. **The seam.** `_Recorder.seam()` returns one cached `Seam` object per recorder: calling it
   (`seam(name)`) is `recorder.section(name)`, and `seam.stamp(name, **ids)` is
   `recorder.stamp(name, **ids)`. The base `stamp` returns a null context. Pass
   `recorder.seam()` where `recorder.section` is passed: `scenarios.py:252`, and the two
   call sites in `plain_baseline.py` (grep `rec.section` and `r.section`). The identity check
   `v.section is not section` keeps working because the seam is cached.
2. **The context.** `StepContext.stamp(name, **ids)` returns
   `getattr(self.section, "stamp", None)(name, **ids)` when that exists, else `_NO_SECTION`.
3. **Spans.** `_Acc` gains host spans (at most 64 per step, then the `rest` aggregate). On
   CUDA, a top-level section records an anchor event after its entry sync and an end event
   before its exit sync; a nested section records an event pair, as `_CudaTimeSection` does
   already. Resolve GPU times in a pure function,
   `resolve_gpu_us(anchor_host_us, anchor, events) -> list[float]`, tested with fake events
   that have `elapsed_time`. On CUDA, a top-level section also reads the four allocator
   counters of §5.1 at entry and exit and writes their change as `alloc`.
4. **Stamps.** `TimeRecorder.stamp(name, **ids)` is a context manager that records host
   `monotonic_ns` at entry and exit and, on CUDA, an event pair. Store the step's stamps in
   column lists. Stamps open outside any check-5 section do nothing.
5. **Check 5.** In `check_5_matmuls`, number the runs (`for i, run in
   enumerate(_member_runs(...))`), wrap each in `ctx.stamp("c5.run", run=i)`, pass `run=i`
   to `_measure_served`, count windows, and wrap each main-loop `_judge_window` in
   `ctx.stamp("c5.judge", window=w)`. `_measure_served(..., run: int | None = None)` stamps
   nothing when `run` is `None`, so the record pass (the call near line 1004) stays unstamped.
   In `_served` and `_served_group`, `c5.glue` wraps the operands, `c5.serve` wraps the
   products' stack and device copy (bind it to `p_dev` inside the stamp), `c5.copy` wraps
   the `.to(device, …)` call alone inside `c5.serve`, and `c5.measure` wraps each group's
   measuring.
6. **Layout.** `check5_layout(c) -> list[dict]` in `checks.py` replays `_member_runs`, the
   group split by `_GROUP_BYTES` and the window rule by `_JUDGE_WINDOW`, exactly as
   `check_5_matmuls` applies them. `metrics.py` adds the layer through `layer_index` and the
   trace slice, and `run_verified` writes `c5_layout.json` once per run directory, where the
   computation is known.
7. **Writing.** `MetricsWriter.write_markers(obj)` opens `markers.jsonl` on first use and
   writes the header. `TimeRecorder._emit` writes the step line; a run section writes a run
   line. Add `read_markers(path)` that validates the header and every line.

**Tests.**
- On the fast suite's decoder run (CPU), each time row's section has a marker entry whose
  spans sum to the row's time within 1 ms + 1%.
- The `c5.run` count equals the layout's `n_runs`; `c5.glue`, `c5.serve`, `c5.copy` and
  `c5.measure` each equal `n_groups`; `c5.judge` equals `n_windows`.
- With a fake `torch.cuda.memory_stats`, a top-level section's `alloc` holds the counters'
  change; a nested section's is `null`.
- The record pass writes no stamps.
- `StepContext.stamp` with `section=None`, or a plain callable without `stamp`, returns a
  null context.
- `resolve_gpu_us` on fake events gives the expected microseconds.
- Verdicts, residuals, roots and final weights are identical with the recorder on and off.
- The overhead test (`metrics_overhead`) still passes its threshold with markers and stamps
  on.
- `read_markers` round-trips a written file and rejects a line without `kind`.

**Done when** the tests pass, `markers.jsonl` and `c5_layout.json` appear in a SmolLM2
test-scale run, and the bit-identity test passes.

**Review focus.** No sync added inside nested sections or stamps; the layout replays the
loop's rules exactly; the record pass unstamped; the seam's identity.

### FS20 — capture record, transcript bytes and `check5_profile` output

**Goal.** Every timed step records its capture summary and its committed bytes, and
`check5_profile` saves its output.

**Files.** `verification/prover/step.py`, `verification/runs/loop.py`,
`verification/commitment/leaves.py`, `verification/runs/metrics.py`,
`verification/runs/check5_profile.py`, `evaluation/records.py` (if it rejects unknown record
types), tests.

**Steps.**
1. `StepOutput` gains `capture: dict | None = field(default=None, compare=False,
   repr=False)`, set in `prove_step` from `cap.summary()` after `P1.label`. If FS9's pinned
   pool exposes its buffer count and bytes, add them as `host_buffers`; else `null`.
2. `leaf_totals(c, leaves) -> tuple[int, int]` in `leaves.py`: the leaf count, and the bytes
   `_counting_hash_leaf` would count: a record's `len(c.encode_record(r))`, a tensor's
   header length (`tensor_leaf_header`) plus its `nbytes`. It reads no tensor data.
3. In `loop.py`, read `out.capture` before `fault.emit`. After `t2`, compute `leaf_totals`
   inside a harness section `handoff.totals` (as `release_device_cache` times its work), then
   read `t2a`, and set `handoff_s = t2b − t2a`, so `commit_s` and `handoff_s` keep their
   meaning. `StepRecord` gains `capture`, `transcript_bytes` and `leaves`, each defaulting to
   `None`.
4. `TimeRecorder._emit` writes the step record's two fields and a `capture` record (§5.3).
   Add `"capture"` to `RECORD_TYPES` and update the record-kinds test (near
   `tests/.../test_metrics.py:970`).
5. `check5_profile` gains `--out PATH`, defaulting to `<run_dir>/check5_profile.jsonl`; each
   repeat appends the line `--json` prints, plus `machine_facts()`. Stdout stays the
   same.

**Tests.**
- On SmolLM2 test scale at fp32 and at `(fp32, bf16)`: `leaf_totals` equals the counting
  pass's `transcript_bytes` and `leaves` (the `storage` record).
- The capture record's keys equal `MatmulCapture.summary()`'s plus the four ids,
  `traced` and `host_buffers`.
- `commit_s` and `handoff_s` differ from a run without the change by no more than noise
  (the test checks that `handoff.totals` is a separate section).
- `check5_profile --out` writes one line per repeat; stdout matches the old output.
- Bit identity with and without the change.

**Done when** the tests pass and the cost of `handoff.totals` on SmolLM2 is reported.

**Review focus.** `leaf_totals` reads no payload; `handoff_s` keeps its meaning; the
capture summary is taken after the last capture change.

### FS21 — the sampler

**Goal.** A separate process samples the machine every 100 ms beside each session part
(§5.4).

**Files.** `setup/sampler.py` (created), `verification/runs/fs11_session.py` (`_run_part`),
tests.

**Steps.**
1. `python -m setup.sampler --pid PID --out PATH [--interval 0.1] [--gpu 0]`, or
   `--wrap -- CMD ...` to start a command and sample it. Imports `psutil` and `pynvml` lazily.
2. The loop sleeps to a deadline (`next += interval`), not a fixed sleep, flushes once a
   second, and exits on SIGTERM or when the root process is gone. It never raises: a failed
   read writes `null`, and a failed group logs once in the header's error list.
3. The process tree is refreshed once a second, not every sample. GPM metrics, when
   `nvmlGpmQueryDeviceSupport` allows them, come from consecutive GPM samples; else the GPM
   fields are `null` and the header says why. PCIe rates come from GPM when it's there, else
   from `nvmlDeviceGetPcieThroughput` (`pcie_source` names which). `g_other_procs` counts
   GPU compute processes outside the tree. On Linux, read the cgroup's `cpu.stat` for
   throttling.
4. `start_sampler(pid, out, interval) -> handle` with `handle.stop()` (SIGTERM, then SIGKILL
   after 5 s). The sampler starts in its own session, so the part's process-group kill
   doesn't reach it. `_run_part` starts it right after `Popen` and stops it in a `finally`,
   writing `<part dir>/samples.jsonl`. `POTS_SAMPLER=0` turns it off. A sampler failure
   never fails the part.

**Tests.**
- `--wrap` around a 2 s busy loop: the mean `c_tree` is above 0.5, and the file has a
  header and about 20 samples.
- SIGTERM ends the sampler within 2 s; it exits by itself within 2 s after the root dies.
- A fake `pynvml` fills the GPU fields; a `psutil` that raises on one call gives `null`s,
  not a crash.
- `c_self` stays below 0.02 on the Mac.
- The session self-test plan (`selftest_plan`) writes `samples.jsonl` for every part.

**Done when** the tests pass, and on the Mac a sampled SmolLM2 test-scale run shows
CPU, RSS and disk fields.

**Review focus.** No path where the sampler outlives the session or kills the part; no
busy loop; deadline pacing.

### FS22 — the probe

**Goal.** A ~2-minute part measures the machine's rates and fixed costs at the run's settings
(§5.5).

**Files.** `verification/runs/probe.py` (created), `verification/runs/fs11_session.py`
(`_probe_part`, `part_extras`), tests.

**Steps.**
1. `python -m verification.runs.probe --out probe.json [--seconds 120]`. Apply the run's
   `setup_determinism` and `VERIF_THREADS`, so the rates hold at the run's settings.
2. On CUDA, measure:
   - GEMM at the four models' linear shapes at full-scale batch, in bf16 and fp32;
   - the skinny batched matmul check 5 runs (a product times `k = VERIF_K` columns, fp32);
   - HBM copy and sum over 4 GiB;
   - 1 GiB host-to-device and device-to-host copies, pinned and pageable;
   - a 1 GiB bf16 host tensor sent as check 5 sends it, `.to("cuda", torch.float32)`, and
     as `.to("cuda").to(torch.float32)` (copy, then cast on the GPU);
   - the fixed costs: a kernel launch (10,000 tiny launches without a sync), an idle sync,
     an event pair, a 1,000-element `.tolist()` and `torch.cuda.empty_cache()`.
3. On every device, measure the host memory copy, the first-touch rate (allocate a fresh
   1 GiB host tensor and fill it, each repeat a new tensor) and the hashing rate. Hash synthetic leaves
   through `leaf_hashes_of` (the path `commit_step` uses) with the native hasher on: build a
   `DecoderComputation` from a small `transformers.LlamaConfig` (no weights), give its
   tensor slots random tensors, and time leaf mixes from many small leaves to few large
   ones. Fit `t = bytes × per_byte_s + leaves × per_leaf_s` by least squares over at least 4
   mixes.
4. Each measure runs 2 warm-ups, then repeats to its share of `--seconds`, and reports the
   median. A measure that fails records its error and leaves the others running.
5. A datasheet table in the module maps a GPU name to its peaks (H100 SXM: §2's numbers);
   an unknown name gives `null`.
6. `_probe_part(out)` returns a `Part` of kind `"probe"`; `part_extras` reads `probe.json`
   and adds a one-line summary to the session summary.

**Tests.**
- The least-squares fit on exact synthetic times recovers `per_byte_s` and `per_leaf_s`
  within 1%.
- The datasheet lookup matches "NVIDIA H100 80GB HBM3" and returns `null` for "Apple M3".
- A CPU run with `--seconds 5` writes a valid file whose CUDA rates are `null` with reasons.

**Done when** the tests pass and a Mac run's hashing rate is within 2× of the hashing
benchmark's Mac number.

**Review focus.** Syncs around every timed GPU region; the hashing goes through the
protocol's entry point; no measure sized past 4 GiB of GPU memory.

### FS23 — the traced step

**Goal.** One chosen step runs under `torch.profiler`, with every section and stamp
annotated and check 5 traced on its slice (§5.6).

**Files.** `setup/config.py`, `verification/runs/plain_baseline.py` (`_NOT_TRAINING`),
`verification/runs/run_verified.py` (config exclusion), `verification/runs/metrics.py`,
`verification/runs/full_scale.py`, `evaluation/records.py`, tests.

**Steps.**
1. `RunConfig.trace_step` from `VERIF_TRACE_STEP` (default 0, off) and `trace_c5` from
   `VERIF_TRACE_C5` (`slice` or `all`, default `slice`). Add both to `_NOT_TRAINING` and to
   `run_verified`'s exclusion tuple, so the config hash doesn't change.
2. The `TimeRecorder` counts steps through `on_step`. At the first non-run section of step N,
   it starts `torch.profiler.profile` with CPU (and CUDA on CUDA), `record_shapes=True` and
   `with_flops=True`.
3. On the traced step, each section and stamp also opens `torch.profiler.record_function`.
   A stamp's name carries its ids: `c5.glue|run=12|group=0`.
4. The slice: at section `5`'s entry, collection turns off
   (`toggle_collection_dynamic(False, ...)`); at a `c5.run` in the slice it syncs and turns
   on; at that run's exit it syncs and turns off; at `5`'s exit it turns on. With
   `trace_c5 = all`, collection stays on. If the installed torch lacks
   `toggle_collection_dynamic`, trace everything and set `c5_slice` to `"all"`.
5. At `on_step(N)` it stops the profiler and exports `trace/step_0000N.json.gz` and
   `trace/meta.json`. Any profiler error is caught: the trace is dropped, `meta.error` says
   why, and the run continues. A run with fewer than N steps writes `meta.error = "not
   reached"`.
6. The traced step's time rows, step record, capture record and marker line carry
   `traced: true`. `Records.table("time")` in `evaluation/records.py` leaves out traced rows
   and `Records.table("time_traced")` returns them; grep `evaluation/` for other readers of
   step times (`prove_s`, `verify_s`) and leave the traced step out of each.
7. `full_scale.py --trace-step N` puts `VERIF_TRACE_STEP=N` in H1's environment only.

**Tests.**
- A 3-step CPU decoder run with `VERIF_TRACE_STEP=2`: the trace holds a `user_annotation` for
  every section in step 2's time rows and for every stamp of the slice; its `meta.json` is
  valid.
- Verdicts, roots and final weights are identical with tracing on and off.
- The config hash is the same with `VERIF_TRACE_STEP` set and unset.
- A profiler that raises at start leaves the run passing, with `meta.error` set.
- `Records.table("time")` holds no traced row.

**Done when** the tests pass and the SmolLM2 test-scale traced step's trace opens in
Perfetto (`ui.perfetto.dev`) with its sections visible.

**Review focus.** The profiler can't outlive the step; collection toggles pair up; the
traced step is excluded wherever costs are read.

### FS24 — the trace summarizer

**Goal.** Turn the trace into per-section and per-class numbers (§5.6), with no torch.

**Files.** `evaluation/results/profile/__init__.py`, `evaluation/results/profile/trace.py`
(created), a fixture and tests.

**Steps.**
1. Read the gzip JSON with the standard library. Keep complete events (`ph == "X"`) of the
   categories `user_annotation`, `gpu_user_annotation`, `cpu_op`, `kernel`, `gpu_memcpy`,
   `gpu_memset`, `cuda_runtime` and `cuda_driver`.
2. Attribute each GPU event to a section through its correlation id: the launching runtime
   call lies inside the innermost CPU `user_annotation`. Cross-check against the innermost
   `gpu_user_annotation` and report the share that disagrees in `warnings`.
3. Sort kernels into categories by a name-pattern table kept in one place (GEMM:
   `gemm|cutlass|xmma|cublas`; attention: `flash|fmha|attention`; and so on). Read copy
   directions and bytes from the memcpy events' arguments.
4. Compute §5.6's fields per section. `gemm_bytes` comes from `cpu_op` input shapes and
   dtypes of `mm`, `addmm`, `bmm` and `matmul`; `gemm_flops` from the ops' `flops` argument.
   `sync_wait_us` sums `cudaDeviceSynchronize`, `cudaStreamSynchronize`,
   `cudaEventSynchronize` and blocking device-to-host copies.
5. For check 5, join the stamps' run ids with `c5_layout.json`, sum each class's phases over
   its traced runs, and scale by `products_total ÷ products_traced` of the class.

**Tests.**
- A hand-built fixture trace (two sections, nested stamps, kernels, copies and correlation
  ids) gives exact expected numbers.
- A fixture where one class has two traced runs of five scales by 2.5.
- `tests/test_layering.py` passes (no torch, `setup` or `verification`).

**Done when** the tests pass. FS27's real trace checks it on CUDA.

**Review focus.** Interval unions, not sums, for spans; no double counting of nested
sections; the scale's denominator.

### FS25 — the analysis

**Goal.** Join every input into `profile.json` (§5.7) and apply the bottleneck rule (§6).

**Files.** `evaluation/results/profile/analyze.py`, `evaluation/results/profile/io.py`
(readers), fixtures and tests.

**Steps.**
1. CLI: `python -m evaluation.results.profile.analyze --run-dir D --samples S --probe P
   --session L [--trace-summary T] [--plain D2] --out profile.json`. `L` names the session,
   with its date so a rerun of the same session script gets its own label (for example
   `profile-check-2026-10-20`); it goes to `meta.session` (§5.7).
2. Readers for every format of §5, each rejecting a wrong version.
3. Join samples to sections: map each sample's interval onto each section's spans and weight
   it by overlap; count `n_samples`; set `few_samples` below 3.
4. Compute work, floors, bound, ratio and marks by §6, and `clocks.agree` by its rule.
5. The check-5 view: per class phases; runs over time from the stamps; the variance view,
   which sets each step's check-5 phase totals beside the sampler's means over check 5
   (`cpu_mhz`, `c_other`, `g_sm_mhz`, `g_throttle`, `g_other_procs`) and reports their
   correlations across steps and across windows.
6. The hook-overhead bound (§7) and, with `--plain`, the plain step's `train.*` sections
   beside the prover's.

**Tests.**
- A synthetic run directory: a section at 20% share and 10× floor is marked; one at 20% and
  2× isn't; one at 5% and 10× isn't.
- Overlap weighting on a hand-built case gives the exact mean.
- `few_samples`, the memory mark and the clock mark each trigger on a crafted input.
- A wrong `version` is rejected.

**Done when** the tests pass, and on a SmolLM2 CPU run it writes a `profile.json` whose
step shares sum to within 2% of 1.

**Review focus.** Units (µs, s, ns) at every join; the floors' rates matched to dtype and
copy kind.

### FS26 — the HTML report

**Goal.** One self-contained `profile.html` that presents one or more `profile.json` files
(inline SVG, CSS and JavaScript, no library, no network), by §12's choices.

**Files.** `evaluation/results/profile/report.py`, tests.

**CLI.** `python -m evaluation.results.profile.report PROFILE [PROFILE ...] --out
profile.html`.

**Content.** Before drawing, the report reads each input's `meta` (§5.7) and lists the runs,
models and sessions it holds, with one line saying that every other measure stays in the
files. It's one long page with a fixed contents bar (§12 item 1), and a panel with nothing to
show is left out of the page and the bar. Every section carries a plain name and its group's
color; legends show the name alone, and the overview's table puts the code beside it (§12
item 8). The panels:

- per run, switched by the run picker when the report holds two or more runs (§12 item 7):
  - an overview: one sentence with the step time and the prover's, hand-off's and verifier's
    totals; a stacked bar of the depth-0 sections; a table of them grouped by part, each with
    its plain name, what it does, its mean time, its share of the step time and its marks; and
    a legend of the marks the run carries;
  - the step flow: one box per depth-0 section in run order, in prover, hand-off and verifier
    bands, with its time, share, GPU busy fraction and CPU cores, outlined when marked
    (§12 item 8, replacing item 2's lanes diagram);
  - the timeline: every timed step stacked, with sampler strips under each (§12 item 3);
  - check 5: its phases in plain words with their seconds, and the phase time by product
    class, each class labeled with the pass and the part of the model it belongs to;
  - prover beside plain, when the run's profile holds it: a table of the training parts and
    the whole step, the chosen run's alone;
- the comparisons across models or sessions that the inputs hold (§12 item 4), as stacked
  bars.

Hovering shows a section's exact numbers, the same strings the overview's table holds. Every
number has two significant digits (§12 item 8).

**Tests.**
- The file holds no `http://` or `https://` reference, and it stays below 5 MB.
- A synthetic `profile.json` renders.
- One run: no run picker and no comparison panel.
- Two runs of one model in one session: the picker, and no comparison panel.
- Two models in one session: the cross-model comparison, and no cross-session one.
- Two sessions sharing one of their models: the cross-session comparison holds that model's
  runs alone.
- A CPU profile renders with the GPU gauges and strips left out.
- The removed panels (section table, hashing breakdown, runs over time, appendix) don't render.
- The number format; check 5's classes in the order a step runs them, each with its pass and
  model part.

**Done when** the tests pass, and the report renders FS25's `profile.json` of a SmolLM2 CPU
run.

**Review focus.** Every panel and comparison drawn only from what the inputs hold; hover
numbers equal the table's.

### FS27 — the H100 check

**Goal.** Run every CUDA part once on an H100 and check the profile's numbers before FS14,
and measure check 5 on the four full-scale models, so O2's fix is chosen from full-scale
numbers (§2 item 9). The session runs the code with O2's first fix merged (check 5 sends
the stored bf16 products and casts on the GPU).

**Steps.**
1. Add a `profile-check` session: an entry in `SESSIONS` (`fs11_session.py`) and a case in
   `fs11_job.sh`'s session switch. Its parts, with the sampler on all of them:
   1. the probe;
   2. SmolLM2 with `VERIF_DEVICE=cuda` at `(fp32, bf16)`: calibration, then `run_verified`
      with 6 steps, `VERIF_TRACE_STEP=5` and `--metrics`;
   3. the same run with `VERIF_TRACE_C5=all`, to check the slice's scaling;
   4. the same run with tracing off, for bit identity;
   5. `plain_baseline` at the same settings;
   6. `check5_profile --out` on the stored step;
   7. per full-scale model, at the full-scale settings (`FULL_SCALE_ENV` with `RECHECK_ENV`):
      `bring_up --no-tamper --steps 2` with `--metrics` and `VERIF_TRACE_STEP=2`, then three
      repeats of check 5 on step 1 (untraced), as `check5_profile` measures them.
2. **Check 5's repeats at full scale.** `check5_profile` reads a kept step and a band file;
   a bring-up run has neither, and its in-sample fit is never written as a band file. So
   `bring_up` gains `--check5-repeats N`: after its last step it runs `check5_profile`'s
   measuring in-process on step 1's transcript (kept in host memory under that flag) with the
   fitted bands, and writes the same lines to `<run dir>/check5_profile.jsonl`. Off by
   default; it changes no verdict. If `check5_profile`'s functions can't take an in-memory
   step and bands, the coder stops and reports.
3. Rent one H100 SXM pod (secure, $3.99/h) under the pod rules: the Mac awake and on AC
   power, the hourly price stated before creation, the job script ends billing, and the boot
   guard deletes the pod at about 2× the estimated runtime (about 6 h). The estimate, from
   hand-picked inputs: setup and upload about 0.3 h, the SmolLM2 parts about 0.5 h, each model
   about 15–40 min (load, two steps at FS17's 3–10 min each, three check-5 repeats at 1–7 min
   each).
4. On the Mac, run FS24, FS25 and FS26 on the results.

**Acceptance.** The check passes when:
- host and GPU clocks agree within §6's tolerance on every section of at least 50 ms;
- the markers cover at least 95% of each step's wall time;
- each section of at least 300 ms has at least 3 samples;
- every section in the trace has its annotation and kernels, and the slice's run ids equal
  the layout's `trace_slice`;
- the stamp counts equal the layout's counts;
- the slice's scaled check-5 phases are within 20% of the `all` run's;
- verdicts, roots and final weights are identical with tracing on and off;
- the hook-overhead bound is reported (FS14 holds it below 1%);
- the probe's rates fall between 30% and 110% of the datasheet's where one exists;
- the header records whether GPM is available;
- for each full-scale model: check 5's time split into glue, serve, copy, measure and judge
  on both steps, each phase beside its floor; the three repeats' spread; and the allocator
  counters of section `5`.

The results go to the user → **Milestone FM8**.

## 10. Waves

The tasks run in this order:

1. **Wave 0** (orchestrator): the dependencies.
2. **Wave 1** (parallel): FS18, FS19, FS21 and FS22. FS22 codes against `machine_facts()`'s
   signature and merges after FS18.
3. **Wave 2**: FS20 after FS19 merges (both edit `metrics.py` and `loop.py`). FS24 and FS25
   run in parallel against the frozen formats and synthetic fixtures.
4. **Wave 3**: FS23 after FS20.
5. **FS27** → **FM8**: the user sees the H100 check's results, check 5's included.
6. **FM6**'s approval of the second H100 session, then **FS14** (**FM7**).

FS26 starts once FS25 merges, any time before FM7 (§12 was settled on 2026-10-19).

## 11. Additions to FS14

FS14 has no session script; its preparation writes one. For the profile, that script:

- runs the probe as its first part;
- keeps the sampler on for every part;
- passes `--trace-step 5` to each model's refusal cell.

The extra pod time is about 10–15 min: the probe (about 2 min) and four traced steps, each
slower than a timed step.

## 12. The report's choices (FS26)

The user settled these on 2026-10-19, taking the recommendations of that day, with item 4's
rule added. Each item gives the choice, its reason, what was rejected and its price.

1. **Layout: one long page.** One page per report, with a fixed contents bar that links to
   each panel. A panel with nothing to show is left out of the page and the bar. Search
   finds any number, the page saves to PDF in one piece, and no panel hides behind a tab.
   Rejected: tabs. Price: a long scroll when a report holds several runs.
2. **Diagram: nested blocks where the profile measures inside a section.** *Replaced on
   2026-10-20 by item 8's step flow; check 5's phases moved to the check-5 panel.* Check 5's block
   holds its phases (glue, host stack, copy, measure, judge), and check 6b's block holds its
   glue. Every other section is one block. Check 5 takes 46–411 s per step at full scale
   (FS17), and O2's fix is chosen from this split. Rejected: one block per section, which
   hides the split the stamps measure. Price: more drawing code and a busier check-5 block.
3. **Timeline: every timed step stacked.** One row per timed step on a shared axis from the
   step's start, each with thin sampler strips under it (GPU use, busy CPU cores, PCIe rate).
   A strip with no data is left out. The traced step isn't drawn, since tracing slows it by
   design. Check 5's 2–4× run-to-run spread is the open question, and stacked steps show
   whether one phase varies or all of them do. Rejected: one chosen step. Price: a taller
   panel, and crowded strips on long check-5 steps.
4. **Comparison: drawn from what the inputs hold.** The report takes any set of
   `profile.json` files. Before drawing, it lists the models and sessions they hold (from
   `meta`, §5.7) and draws only the comparisons that exist:
   - across models, within a session, when that session holds two or more models; each row
     is one run, labeled by model and attack;
   - across sessions, only when there are two or more sessions. It pairs runs of the same
     model, attack and scenario, and draws only the runs present in two or more sessions;
   - prover beside plain, when a profile holds that comparison (FS25's `--plain`); since
     2026-10-20 it's drawn per run (item 8).

   With one model in one session there's no comparison panel. The cross-session view serves
   the profile's second use, judging later code changes (§1). The rule that a comparison is
   drawn only when its models or sessions exist is the user's addition. Rejected: comparing
   the four models only. Price: the report reads several files and lines up sections that a
   code change renamed or added; a missing section has no segment in its bar.
5. **Test scale: yes.** The report renders a CPU run too, with the GPU gauges and strips left
   out. One
   code path serves both scales, and the report is built and tested on Mac runs before any
   pod. Price: nothing beyond handling missing GPU fields, which the CPU case needs anyway.
6. **Thresholds: kept for now.** The bottleneck mark's 10% share and 3× floor, and the memory
   mark's 95% reserved, stay as §5.7 lists them. They're reviewed at FM8 against FS27's
   numbers, for instance if every section or none is marked. Price: until then a mark may
   mislead; the overview shows its time and share beside it, and `profile.json` its floor and
   ratio (item 8).
7. **Interaction: hover, and a run picker.** Hovering shows exact numbers. When a report holds
   two or more runs, a picker at the top switches the per-run panels between them, each run
   labeled by model, attack, scenario and session. There's no deeper click-through. A static
   page would need every number printed on the figures. Rejected: static. Price: inline
   JavaScript, and a screenshot loses the hover text; the report is an internal diagnostic,
   not a paper figure. The picker chooses a run rather than a model, because FS14 profiles
   two attack cells per model and FS27 three SmolLM2 runs.
8. **A summary for the reader, not a dump (user, 2026-10-20).** The first H100 report carried
   every measure at full resolution, and the user couldn't read it: bare section codes, about
   twenty generated colors, a lanes diagram that placed most blocks in "Host" and a dozen
   tables. The report now shows:
   - **Plain names.** Every section has a short name and a one-line description beside its
     code (`train.forward` is "Forward pass: the training forward pass, every matmul
     captured"). Sections fall into seven groups, one color each: the prover's training,
     recording and commitment, the hand-off (hatched), and the verifier's reading and hashing,
     weight-update checks and matmul checks. A section is a shade of its group's color. The
     colors come from the dataviz skill's reference palette, checked for color-blind
     separation in light and dark mode. Legends show the plain name alone; the overview's
     table, the flow's boxes and check 5's rows keep the code beside it, so a reader can find
     the section in `profile.json`.
   - **Two tables only**, the overview's and prover beside plain. Removed: the section table,
     the hashing breakdown, the check-5 class table, runs over time, the variance and
     correlation tables, the comparison tables and the appendix (probe, machine facts,
     thresholds). All of it stays in `profile.json`, where an investigation reads it.
   - **Overview.** Rows grouped by prover, hand-off and verifier, with subtotals. The share
     column is "Share of step time": the section's mean time over the mean time of the whole
     verified step. A note says the `train.*` rows are the prover's own training with the
     capture on, and plain training appears only in prover beside plain. A legend explains the
     marks the run carries.
   - **Step flow** instead of item 2's lanes diagram: one box per depth-0 section in run
     order, in prover, hand-off and verifier bands, each with its time, its share, the
     sampler's GPU busy fraction and the run's CPU cores (scaled to the machine's CPU quota).
     A marked box gets a red outline.
   - **Timeline** as item 3, with a grouped legend of plain names and labeled strips (GPU
     busy, CPU cores, PCIe in GB/s; into the GPU solid, out of it dashed).
   - **Check 5.** The five phases in plain words with their seconds, and a phase-by-product
     chart whose rows say which pass and model part they belong to ("Backward · MLP"), in the
     order a step runs them, on a seconds axis.
   - **Comparisons** as stacked bars on a seconds axis with each run's total; no tables.
   - **Prover beside plain** is a per-run panel after check 5, before the comparisons, so the
     run picker shows the chosen run's table alone.
   - **Numbers** with two significant digits, in scientific form below 0.01 and from 10 000 up
     (0.008397 is 8.4e-03). Counts stay exact, and a share under 0.1% reads "<0.1%".

   Rejected: moving the removed tables into an appendix, since `profile.json` already holds
   them; one significant digit, which rounds ratios too hard (4.5 becomes 5). Price: a reader
   who wants a floor, a clock reading, a hashing part or check 5's run-to-run variance opens
   `profile.json`; the page no longer shows a mark beside the floor and ratio that set it.

## 13. Risks

The plan carries these risks, each with its fallback:

- **GPM unavailable on the pod.** The GPU lanes then rest on NVML's coarse utilization and
  the trace's estimates. FS27 records availability; the bound type still comes from the
  floors.
- **`toggle_collection_dynamic` missing or unreliable.** The traced step traces all of check
  5. FS27 measures the trace's size per run, to extrapolate to Qwen2.5-1.5B before FS14.
- **Large traces.** If FS27's trace size extrapolates past about 2 GB for the full-scale
  models, the slice drops to every 40th run.
- **Clock drift between CUDA events and the host clock.** The anchors reset at each
  top-level section, so drift stays within a section; FS27's clock check measures it.
- **Check 5's variance.** The sampler's signals might not explain it. The stamps still show
  where in check 5 the time goes, run by run, which narrows the next question.
