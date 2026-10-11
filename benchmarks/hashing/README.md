# Hashing benchmark

Standalone scripts for the H100 hashing benchmark of `docs/verification/HASHING_PLAN_DRAFT.md`
(Section 6). They measure what the full-scale run's hashing and challenge generation cost on
the CPU, and they record the machine. The GPU experiments are empty slots in the report.

The package imports `setup`, PyTorch, transformers, NumPy, `hashlib` and `blake3`. From
`verification` it imports only the leaf-size list (`verification/computation/leaf_sizes.py`);
`tests/test_layering.py` enforces both rules.

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

Before the session, prepare two things:

- E6 loads each model with `from_pretrained` at its pinned revision (`setup/model.py`). Run
  with network access, or fill the Hugging Face cache first.
- `b3sum`, BLAKE3's command-line tool, gives BLAKE3's native ceiling. If it isn't on `PATH`,
  that result says skipped.

To try the runner on a laptop, use the smoke preset:

```bash
HF_HUB_OFFLINE=1 VERIF_OUTPUT_DIR=/tmp/out .venv/bin/python -m benchmarks.hashing.run --smoke
```

The smoke preset uses a 256 MiB buffer, one layer of the challenge list, and E6 at 2 × 16 on
the CPU. It took 67 s on an Apple M5 Pro. Any flag given explicitly overrides the preset; see
`--help`.

A step is named by its model (`llama-3.2-1b`), at `--n-s` × `--n` with bf16 products. To give
a step its own batch and product dtype, write `model:N_SxN:DTYPE`. For example,
`smollm2-135m:4x128:fp32` is the test-scale run's step. Weights are fp32 in both runs.

`--experiments E0,E1,E5,E6` picks experiments. The slots of the others say skipped. If an
experiment fails, its slots say skipped with the error, and the other experiments still run.

## Experiments

The report has one slot per experiment of Section 6.1:

E0, machine record
: CPU model, cores and core types, SHA and AVX-512 instruction flags, RAM, OS, Python and
  package versions, and the OpenSSL versions of `hashlib` and the CLI. It also holds GPU,
  driver, CUDA, PCIe link and NUMA fields. Fields that a machine lacks are `null` and are
  listed in `absent`.

E1, CPU hashing throughput
: SHA-256 (OpenSSL through `hashlib`) and BLAKE3 (`blake3`), in four measurements:
  - *Chunked*: the buffer cut into chunks hashed as `H(0x02 ‖ chunk)`, split over the threads.
  - *Whole objects*: one row per real leaf size of each step, hashed with the leaf encoding
    and `merkle.hash_leaves`'s threading policy. `step_from_sizes` weights each size by its
    count to estimate a step's time. `step_direct` hashes every leaf of a step and measures
    it.
  - *Native ceilings*: `openssl speed -multi` and `b3sum`.
  - *Under a copy*: the chunked measurement while a host-to-GPU copy runs. It's skipped
    without CUDA.

E2, E3 and E4: GPU hashing, overlap with a GEMM loop, and host-GPU copies
: Not implemented. They need CUDA code.

E5a, challenge generation on the current Python path
: The protocol's challenge functions, reproduced call for call, including the fp32
  conversion. A test checks them bit for bit against `challenge_matrix`. When `--e5-layers`
  times only some layers, the per-step time is extrapolated and says estimated.

E5a_prime, CPU-parallel baselines for challenge generation
: The same calls over a thread pool and over a process pool. `native_batch` is a proposal,
  not implemented: a C batch over BLAKE3's sources.

E5b and E5c: challenge generation on the GPU with cuPQC
: Not implemented. They need CUDA code.

E6, step time
: One plain bf16 SGD step per model, eager and then sdpa attention, after warmup steps.
  The verified step isn't timed.

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
