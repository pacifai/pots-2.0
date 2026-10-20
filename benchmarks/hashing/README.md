# Hashing benchmark

Standalone scripts for the H100 hashing benchmark of `docs/verification/HASHING_PLAN_DRAFT.md`
(Section 6). They measure what the full-scale run's hashing and challenge generation cost on
the CPU, how fast the GPU runs the training's largest matmuls, and how fast steps move between
host and GPU, and they record the machine. The GPU hashing slots (E2, E5b and E5c, and E3's
hashing half) say skipped: cuPQC sits behind a licence the user hasn't accepted (see
[cuPQC](#cupqc)), and the cited BLAKE3-on-GPU code needs a SYCL compiler.

The package imports `setup`, PyTorch, transformers, NumPy, `hashlib`, `blake3` and `certifi`
(for the native build's TLS only). From `verification` it imports only the leaf-size list
(`verification/computation/leaf_sizes.py`), so it times the protocol's costs without calling
the protocol's hashing or challenge code. The trees, the finiteness check and the challenge
path it times are its own copies, and the tests check each against the protocol's.
`tests/test_layering.py` enforces both rules (rule 8). One experiment, E5a_prime's native
batch, also builds a C library from BLAKE3's sources; see [Native batch](#native-batch).

## Usage

To run the H100 session, run the module from the repo root with the project's `.venv`:

```bash
.venv/bin/python -m benchmarks.hashing.run
```

The defaults are Section 6's sizes:

- A 4 GiB buffer, with threads 1, 8, 16 and all physical cores.
- Chunks of 8, 16, 64 and 256 KiB and 1 MiB.
- The four full-scale models at 128 × 128, plus the test-scale step.
- Llama-3.2-1B's whole challenge list at k = 21.
- E3: the largest matmuls of each full-scale model over 128 × 128 tokens, 20 timed runs each,
  and 5-second windows for the GEMM loop and the concurrent rows.
- E4: copy pieces of 1 and 8 GiB over a 64 GiB host buffer. The host needs that much free
  RAM, plus the 4 GiB hashing buffer and 4 GiB for the rest of the process; with less, E4
  halves the buffer until it fits.
- One SGD step per full-scale model on `cuda`.

The report goes to `$VERIF_OUTPUT_DIR/benchmarks/hashing_<UTC time>.json`, or to `--out`.

Before the session, prepare three things:

- E6 loads each model with `from_pretrained` at its pinned revision (`setup/model.py`), and
  E1, E3 and E4 read each model's config from the cache. Fill the Hugging Face cache first.
  Every pinned repo is public, so no token is needed. The exception is E6's Llama row: the
  unsloth mirror is pinned for its config only, and E6 times Meta's
  `meta-llama/Llama-3.2-1B-Instruct` at the verified run's commit (`step_time.WEIGHTS`), which
  is gated. That snapshot comes from the Mac's cache in FS11's upload; each row's params name
  the repo it loaded.
- `b3sum`, BLAKE3's command-line tool, gives BLAKE3's native ceiling. If it isn't on `PATH`,
  that result says skipped.
- E5a_prime's native batch needs a C compiler and, the first time, one download from
  github.com. To build it ahead of the session, run `.venv/bin/python -m
  benchmarks.hashing.native`; otherwise the runner builds it when E5 starts.

To try the runner on a laptop, use the smoke preset:

```bash
HF_HUB_OFFLINE=1 VERIF_OUTPUT_DIR=/tmp/out .venv/bin/python -m benchmarks.hashing.run --smoke
```

The smoke preset uses a 256 MiB buffer, one layer of the challenge list for the Python paths,
every layer for the native batch, and E6 at 2 × 16 on the CPU. E3 times Qwen2.5-0.5B's matmuls
over 2048 tokens, and E4 copies 64 and 256 MiB pieces of a 1 GiB buffer, in 1-second windows.
It took about 5.5 minutes on an Apple M5 Pro, 4.2 of them in E1's whole-object rows, which
hash about four tasks per thread at each leaf size and thread count. Any flag given explicitly
overrides the preset; see `--help`. To run E6 on a GPU at smoke size, add `--e6-device cuda`;
with it, the smoke run took 7.3 minutes on an A40 pod, 6 of them in E1.

The smoke sizes check that the code runs, and their rates are noisy: a 256 MiB buffer over 7
threads hashes in tens of milliseconds. On the A40 pod, E1's chunked BLAKE3 rows at 7 threads
and 1 MiB chunks read 7 to 17 GB/s on the smoke buffer and 33 GB/s on a 4 GiB one, which
matches E4's one-second hashing windows.

A step is named by its model (`llama-3.2-1b`), at `--n-s` × `--n` with bf16 products. To give
a step its own batch and product dtype, write `model:N_SxN:DTYPE`. For example,
`smollm2-135m:4x128:fp32` is the test-scale run's step. Weights are fp32 in both runs.

`--experiments E0,E1,E5,E6` picks experiments. The slots of the others say skipped. If an
experiment fails, its slots say skipped with the error, and the other experiments still run.
Inside E1, each part (chunked, whole objects, the tree, finiteness, the ceilings) fails on its
own the same way, and so do E5a_prime's native batch and each E6 model. The runner rewrites the
report after every experiment, so a crash late in the session keeps the finished rows; the
report's `complete` is true only once every selected experiment has run.

## Experiments

The report has one slot per experiment of Section 6.1:

E0, machine record
: CPU model, cores and core types, the SHA, AVX2, AVX-512F and AVX-512VL instruction flags,
  CPU 0's clock and frequency governor, RAM, OS, Python and package versions, and the OpenSSL
  versions of `hashlib` (asked of hashlib's own library), `ssl` and the CLI. It also holds
  GPU, driver, CUDA, PCIe link and NUMA fields. Fields that a machine lacks are `null` and are
  listed in `absent`. `cores_for_threads` is the thread cap: physical cores, capped by CPU
  affinity and cgroup quota, as `cores_for_threads_basis` says; rayon, and so BLAKE3's AUTO
  mode, sizes its pool by logical CPUs instead. The cgroup limits are read from cgroup v2
  (`cpu.max`, `memory.max`) or, failing that, v1 (`cpu.cfs_quota_us` over `cpu.cfs_period_us`,
  `memory.limit_in_bytes`): `cgroup_cpu_quota` in CPUs and `cgroup_memory_limit_bytes`. A
  container's `nproc` can show the host's CPUs while its quota allows far fewer; the A40 pod
  showed 96 logical CPUs under a 7.65-CPU quota, so `cores_for_threads` was 7. `notes` says
  that `pcie_gen_current` and the clock are read at idle.

E1, CPU hashing throughput
: SHA-256 (OpenSSL through `hashlib`) and BLAKE3 (`blake3`), in these measurements:
  - *Chunked*: the buffer cut into chunks hashed as `H(0x02 ‖ chunk)`, split over the threads.
    Each row also times the chunk tree's node level, `H(0x01 ‖ l ‖ r)` over the chunk hashes
    (`node_us_per_chunk`), and gives the end-to-end rate with it (`gb_per_s_end_to_end`).
  - *Whole objects*: one row per real leaf size of each step, hashed with the leaf encoding
    and `merkle.hash_leaves`'s threading policy. `step_from_sizes` weights each size by its
    count to estimate a step's time. `step_direct` hashes every leaf of a step and measures
    it. Each size gets an untimed warm-up and then `--reps` timed runs; `gb_per_s` is the
    median and `gb_per_s_best` the best. The sample holds at least 4 tasks of about 16 MiB
    per thread, so small leaves fill the pool (`tasks` in the params), up to `--max-objects`
    objects. A leaf larger than `--object-cap-mb` is timed on a smaller object and says
    estimated. BLAKE3 has two variants, and each row records `cpu_per_wall`, CPU time over
    wall time:
    - `as_built` is the protocol's policy. A leaf of 16 MiB or more gets BLAKE3's AUTO mode,
      rayon's global pool of every logical CPU, whatever `threads` says (`large_leaf_threads`
      in the params). Its `threads` bounds only the small leaves.
    - `capped` waits for the small leaves' tasks before a large leaf, then hashes it with
      `blake3(max_threads=threads)`, so at most `threads` CPUs hash at once and the per-core
      scaling is real.
  - *Outer tree*: the step's Merkle tree over its leaf hashes, as the prover builds it (every
    subtree cached) and as the verifier's check 2 does (the root only).
  - *Finiteness*: the encoding's check that every float leaf is finite, per leaf size and
    thread count, and summed over a step (`finiteness_per_step`). The protocol runs it on the
    calling thread while the pool hashes, so it adds to the step's time only where it outlasts
    the hashing.
  - *Native ceilings*: `openssl speed -multi` and `b3sum`.
  - *Under a copy*: the chunked measurement while a host-to-GPU copy runs. It's skipped
    without CUDA.

E2, GPU hashing throughput
: Two skipped rows. `cupqc_sha256` needs NVIDIA's cuPQC SDK (see [cuPQC](#cupqc)). Its note
  records one finding for when it's built: cuPQC's Merkle API takes only power-of-two leaf
  counts, so the protocol's rule for an odd node count (P9a) needs the node levels built from
  batched 65-byte SHA-256 calls. `blake3_gpu_proxy` is the plan's BLAKE3-on-GPU reference,
  github.com/itzmeanjan/blake3. It's written in SYCL, so on an NVIDIA GPU it needs Intel's
  open-source LLVM SYCL compiler built with its CUDA backend; nvcc can't build it.

E3, GPU hashing beside training's matmuls
: The GEMM half runs; the hashing half says skipped until E2 has a GPU hash.
  - `gemm`: each of a model's three largest matmuls in bf16, timed alone with CUDA events:
    the MLP's up and down projections and the LM head, each over `--gemm-tokens` rows. One
    event pair spans 10 back-to-back runs of the matmul, and the time is divided by 10, so the
    launch delay before the first run is a tenth of the measurement. `tflop_per_s_median` and
    `tflop_per_s_best` are over `--gemm-iters` such measurements.
  - `gemm_loop_alone`: the *GEMM loop*. A background thread enqueues the three matmuls round
    robin on their own CUDA stream, at most four ahead of the GPU. Its rate is the FLOPs of
    the matmuls that finished in the window over the window's wall time.
  - `gpu_hash_alone` and `gemm_loop_with_gpu_hash`: skipped, no GPU hash.

  Every row runs under `setup_determinism`'s matmul settings, as E6 does.

E4, host-GPU copies
: Copies GPU to host (`d2h`) and host to GPU (`h2d`), pinned and pageable, at each piece size.
  The host buffer is `--e4-total-mb` of pageable memory, every page written and cut into
  pieces, so each piece is distinct host memory, as a step's transcript is. For the pinned
  rows it's page-locked in place with `cudaHostRegister`. Each combination gives three rows:
  - `copy`: whole passes over the buffer, one piece at a time on a side stream, alone.
  - `copy_with_gemm_loop`: the same passes while E3's GEMM loop runs (on the first of
    `--gemm-models`). The loop runs one window alone just before the copies, and
    `gemm_slowdown` compares the two windows, so clock drift over a long session doesn't
    count as slowdown.
  - `copy_with_cpu_hashing`: the same passes while CPU threads hash a separate buffer as E1's
    chunked rows do (the largest `--chunk-kb`, all `--threads`), one row per algorithm. As
    with the loop, `hash_slowdown` compares the window during the copies with the one just
    before them.

  The `gemm_loop_alone` and `cpu_hashing_alone` rows are the rates at the start of E4, for
  reference.

  A slowdown under about 1% is counting noise: the loop counts a GEMM only once it sees it
  finish, so a window's count lags the GPU by up to four GEMMs. The hashing count lags by up
  to one 16 MiB task per thread, which can reach a few percent of a one-second window. Each
  row's note says so.

  A pageable copy blocks the thread that enqueues the GEMMs, so a pageable
  `copy_with_gemm_loop` row shows this process's slowdown, not the GPU's. On the A40 pod a
  pageable copy cut the loop's rate by 83–98%; the same copies from a second process left it
  unchanged. The row's note says so.

  Before it allocates, E4 checks the host buffer, the hashing buffer and a 4 GiB margin
  against the host's memory: the smaller of the cgroup's limit and `MemAvailable`. If they
  don't fit, it halves the buffer, keeping it a multiple of every piece, until they do, and
  says skipped if no such buffer fits. A container's out-of-memory kill would otherwise end
  the whole run. Each row records `total_bytes`, the buffer used, beside
  `total_bytes_requested` and the two memory readings.

E5a, challenge generation on the current Python path
: The protocol's challenge functions, reproduced call for call, including the fp32
  conversion. A test checks them bit for bit against `challenge_matrix`. When `--e5-layers`
  times only some layers, the per-step time is extrapolated and says estimated.

E5a_prime, batched challenge generation on the CPU
: The same calls in three forms:
  - `thread_pool` and `process_pool`: the Python path's calls spread over a pool. These are
    baselines that need no new code. The pool's threads start before the timer does.
  - `native_batch`: each of check 5's calls made once through BLAKE3's C library, its
    vectors spread over 1, 8, 16 and all cores in C. A test checks its output bit for bit
    against the protocol's `challenge_matrices` and `challenge_matrix`. `--e5-native-layers`
    sets how many layers it times (default all); `--no-native-batch` turns it off.

E5b and E5c: challenge generation on the GPU with cuPQC
: Skipped rows, `cupqc_shake256` and `cupqc_sha256_counter`: both need cuPQC (see
  [cuPQC](#cupqc)).

E6, step time
: One plain SGD step per model, eager and then sdpa attention, after warmup steps. It trains
  as FS3's step does: fp32 master weights, forward and loss under bf16 autocast, backward and
  the update outside it. Around the timing it sets `setup_determinism`'s two matmul precision
  settings, fp32 accumulation for cuBLAS's bf16 matmuls (F5a) and fp32 matmul precision
  "highest", and restores them after; each row records them and whether deterministic
  algorithms are on. The verified step isn't timed.

## cuPQC

E2's SHA-256, E5b and E5c need NVIDIA's cuPQC SDK. It has no pip or conda package; NVIDIA
ships it as a tarball, latest version 0.6.0 (189 MB). Its download page,
https://developer.nvidia.com/cupqc-download, says that downloading and using it accepts the
NVIDIA Software License Agreement, https://docs.nvidia.com/cuda/cupqc/additional/license.html.
Accepting that licence is the user's decision, so the SDK isn't downloaded here.

What the licence says, in the parts that bear on this repo:

- §1.1 and §1.2: you may distribute its binaries and headers only inside an application that
  adds material functionality.
- §2.3: no other redistribution or modification. The repo can't hold the SDK, so a build has
  to fetch it, as the native batch fetches BLAKE3.
- §2.6: nothing may put the SDK under an open-source licence.
- §2.1: NVIDIA GPUs only.
- It says nothing on publishing benchmark numbers.

If the user accepts the licence, these are the steps:

1. Download and extract the SDK outside the repo:

   ```bash
   wget https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-sdk-0.6.0-x86_64.tar.gz
   tar -xzf cupqc-sdk-0.6.0-x86_64.tar.gz -C <dir>
   ```

   The URL needs no login.
2. Write a small CUDA extension against cuPQC-Hash for E2 (SHA-256 leaves and nodes), E5b
   (SHAKE256) and E5c (SHA-256 in counter mode), built with `nvcc` 12.4 or later and `-dlto`,
   as NVIDIA's install guide asks. cuPQC is a device-side C++ library with no Python API, so
   the extension needs its own kernels and a ctypes or torch binding. None of it is written
   yet.

## Native batch

E5a_prime's `native_batch` calls a shared library built from two parts:

- `challenges_native.c`, in this directory. It makes the protocol's challenges (key `h`,
  label `0x10 ‖ m ‖ j`, the XOF bytes and their fp32 conversion) for a list of products in
  one call, on a pthread pool. ctypes releases the GIL for the whole call.
- BLAKE3's reference C implementation, release 1.8.7, from the `c/` directory of
  github.com/BLAKE3-team/BLAKE3. Its licence is CC0-1.0 or Apache-2.0, at the user's choice.

The repo doesn't hold BLAKE3's sources. To build the library, run:

```bash
.venv/bin/python -m benchmarks.hashing.native
```

The build runs in three steps:

1. It downloads the release archive of the pinned tag, `BLAKE3_TAG` in `native.py`, and
   checks its SHA-256 against `ARCHIVE_SHA256`. A mismatch stops the build before anything
   is extracted.
2. It extracts the C files it needs and the licences into
   `~/.cache/pots-benchmarks/blake3-1.8.7/`, and checks each file against its own pin in
   `SOURCE_SHA256`. Before a later build reuses that directory, it checks the pins again and
   downloads the archive again if any file is missing or changed.
3. It compiles them with `$CC` (default `cc`) into
   `~/.cache/pots-benchmarks/libpots_challenges-blake3-1.8.7-<platform>-<arch>-<hash>.so`. The
   hash covers the C source, the build command and the output of `$CC --version`, so editing
   either or upgrading the compiler builds a new library.

To put the cache elsewhere, set `$POTS_BENCH_CACHE`. Later runs reuse the library and need no
network.

BLAKE3 picks its SIMD code at run time. On arm64 the build includes NEON. On x86-64 it
includes BLAKE3's SSE2, SSE4.1, AVX2 and AVX-512 assembly, and the library uses the widest
one the CPU supports. On other machines it builds the portable C code. Each `native_batch`
row records `simd_degree`, the number of lanes of the path in use: 16 for AVX-512, 8 for
AVX2, 4 for SSE4.1, SSE2 or NEON. On every path but AVX-512, BLAKE3 computes each 64-byte output
block of an XOF one at a time, so challenge generation gains from cores, not from SIMD width.

The fast test suite never builds the library. Its native tests skip, with that reason, when
the library is missing.

## Report layout

The report is one JSON object:

```text
schema        "benchmarks.hashing/1"
created_utc   ISO time
argv          the command line
smoke         true for --smoke
args          every option after presets, plus threads_resolved and experiments
inputs.steps  per step: model, n_s, n, dtypes, leaves, total_bytes, bytes_by_section,
              distinct_sizes, largest_leaf
slots         E0, E1, E2, E3, E4, E5a, E5a_prime, E5b, E5c, E6, always all present
elapsed_s     total, and per experiment
complete      false until every selected experiment has run
```

A slot is `{title, status, note, results}`. A result is `{name, status, params, values,
units, reason, note}`, where `units` names the unit of each value.

A status is one of the following:

- `measured`: timed on this machine.
- `estimated`: derived from measurements, such as an extrapolation or a weighted sum. `note`
  says how.
- `skipped`: not run here; `reason` says why.
- `not_implemented`: not built; `reason` says what it needs.

A slot takes the first status that any of its results has, in that order.
