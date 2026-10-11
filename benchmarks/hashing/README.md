# Hashing benchmark

Standalone scripts for the H100 hashing benchmark of `docs/verification/HASHING_PLAN_DRAFT.md`
(Section 6). They measure what the full-scale run's hashing and challenge generation cost on
the CPU, and they record the machine. The GPU experiments are empty slots in the report.

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
- One SGD step per full-scale model on `cuda`.

The report goes to `$VERIF_OUTPUT_DIR/benchmarks/hashing_<UTC time>.json`, or to `--out`.

Before the session, prepare three things:

- E6 loads each model with `from_pretrained` at its pinned revision (`setup/model.py`). Run
  with network access, or fill the Hugging Face cache first.
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
every layer for the native batch, and E6 at 2 × 16 on the CPU. It took 70 s on an Apple M5
Pro. Any flag given explicitly overrides the preset; see `--help`.

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
  mode, sizes its pool by logical CPUs instead. `notes` says that `pcie_gen_current` and the
  clock are read at idle.

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

E2, E3 and E4: GPU hashing, overlap with a GEMM loop, and host-GPU copies
: Not implemented. They need CUDA code.

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
: Not implemented. They need CUDA code.

E6, step time
: One plain SGD step per model, eager and then sdpa attention, after warmup steps. It trains
  as FS3's step does: fp32 master weights, forward and loss under bf16 autocast, backward and
  the update outside it. Around the timing it sets `setup_determinism`'s two matmul precision
  settings, fp32 accumulation for cuBLAS's bf16 matmuls (F5a) and fp32 matmul precision
  "highest", and restores them after; each row records them and whether deterministic
  algorithms are on. The verified step isn't timed.

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
