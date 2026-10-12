"""The hashing benchmark's runner: every experiment slot into one JSON file.

    python -m benchmarks.hashing.run                 # the H100 session (Section 6 sizes)
    python -m benchmarks.hashing.run --smoke         # small sizes, minutes on a laptop
    python -m benchmarks.hashing.native              # E5(a′)'s native library, built alone

The defaults are the H100 session's: a 4 GB buffer, 1/8/16/all-core threads, the five chunk
sizes, the four full-scale models at 128 × 128 (fp32 weights, bf16 products), k = 21, and
Llama-3.2-1B's whole product list for E5. ``--smoke`` shrinks the buffer, the samples and E5's
share, and times E6 at 2 × 16 on the CPU. Any option given explicitly overrides the preset.
E5(a′)'s native batch fetches and builds BLAKE3's C library on first use (``native.py``); pass
``--no-native-batch`` to run without it.

A step is named ``model``, at ``--n-s`` × ``--n`` with bf16 products, or ``model:N_SxN:DTYPE``
with its own batch and product dtype (``fp32`` or ``bf16``); weights are fp32 in both runs.
``smollm2-135m:4x128:fp32`` is the test-scale run's step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from benchmarks.hashing import challenges, cpu_hash, native, step_time
from benchmarks.hashing.machine import machine_record
from benchmarks.hashing.report import (SCHEMA, SLOTS, not_implemented_slot, result, slot,
                                       write_report)
from benchmarks.hashing.sizes import FULL_SCALE, MODELS, TEST_SCALE, step
from setup.config import load_config

MB = 1 << 20
TEST_STEP = f"{TEST_SCALE}:4x128:fp32"  # the test-scale run's step: 7,661 leaves, 2.62 GB
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

H100 = {
    "buffer_mb": 4096,
    "threads": "1,8,16,all",
    "chunk_kb": "8,16,64,256,1024",
    "reps": 3,
    "n_s": 128,
    "n": 128,
    "size_models": ",".join((*FULL_SCALE, TEST_STEP)),
    "direct_models": ",".join((*FULL_SCALE, TEST_STEP)),
    "direct_reps": 1,
    "sample_mb": 512,
    "object_cap_mb": 0,  # 0: hash every object at its real size
    "max_objects": cpu_hash.MAX_OBJECTS,
    "native": True,
    "native_threads": "1,8,16,all",
    "copy_mb": 1024,
    "e5_model": "llama-3.2-1b",
    "k": 21,
    "e5_layers": 0,  # 0: every layer
    "e5_processes": "all",
    "native_batch": True,
    "e5_native_layers": 0,  # 0: every layer; the native batch is fast enough to run them all
    "e6_models": ",".join(FULL_SCALE),
    "e6_n_s": 128,
    "e6_n": 128,
    "e6_device": "cuda",
    "e6_warmup": 2,
    "e6_reps": 5,
}

SMOKE = {
    **H100,
    "buffer_mb": 256,
    "reps": 2,
    "direct_models": TEST_STEP,
    "sample_mb": 32,
    "object_cap_mb": 256,
    "max_objects": 1 << 16,
    "native_threads": "all",
    "copy_mb": 256,
    "e5_layers": 1,
    "e6_models": f"qwen2.5-0.5b,{TEST_SCALE}",
    "e6_n_s": 2,
    "e6_n": 16,
    "e6_device": "cpu",
    "e6_warmup": 1,
    "e6_reps": 3,
}

EXPERIMENTS = ("E0", "E1", "E5", "E6")

# The slots this part of FS10 doesn't build: GPU code (cuPQC, a BLAKE3 kernel, CUDA streams).
GPU_ONLY = {
    "E2": "GPU hashing needs cuPQC and CUDA code; not written in FS10's CPU part",
    "E3": "GPU overlap needs CUDA streams and E2's kernels; not written in FS10's CPU part",
    "E4": "copy benchmarks run on the H100 with CUDA code; not written in FS10's CPU part",
    "E5b": "cuPQC SHAKE256 on the GPU; not written in FS10's CPU part",
    "E5c": "cuPQC SHA-256 counter mode on the GPU; not written in FS10's CPU part",
}


def _ints(text: str, cores: int) -> list[int]:
    return [cores if t.strip() == "all" else int(t) for t in text.split(",") if t.strip()]


def _models(text: str) -> list[str]:
    """The step names of a comma list (``none`` for none), each model checked."""
    keys = [m.strip() for m in text.split(",") if m.strip() and m.strip() != "none"]
    bad = [k for k in keys if k.split(":")[0] not in MODELS]
    if bad:
        raise SystemExit(f"unknown models {bad}; known: {sorted(MODELS)}")
    return keys


def _step_spec(name: str, n_s: int, n: int) -> tuple[str, int, int, torch.dtype]:
    """``(model, n_s, n, product dtype)`` of a step name (module docstring)."""
    parts = name.split(":")
    if len(parts) == 1:
        return parts[0], n_s, n, torch.bfloat16
    try:
        model, batch, dtype = parts
        s, t = (int(x) for x in batch.split("x"))
        return model, s, t, DTYPES[dtype]
    except (ValueError, KeyError):
        raise SystemExit(f"bad step {name!r}: expected model or model:N_SxN:fp32|bf16") from None


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m benchmarks.hashing.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true", help="small sizes for a laptop")
    p.add_argument("--out", type=Path, help="the JSON report (default: "
                   "$VERIF_OUTPUT_DIR/benchmarks/hashing_<UTC time>.json)")
    p.add_argument("--experiments", default=",".join(EXPERIMENTS),
                   help=f"which of {EXPERIMENTS} to run; the others' slots say skipped")
    a = p.add_argument
    a("--buffer-mb", type=int, help="E1's buffer, MiB (H100: 4096)")
    a("--threads", help="thread counts, capped at the cores (H100: 1,8,16,all)")
    a("--chunk-kb", help="E1's chunk sizes, KiB (H100: 8,16,64,256,1024)")
    a("--reps", type=int, help="repetitions per timed measurement (H100: 3)")
    a("--n-s", type=int, help="sequences per step for the leaf sizes (H100: 128)")
    a("--n", type=int, help="tokens per sequence for the leaf sizes (H100: 128)")
    a("--size-models", help="steps for E1's whole objects by size")
    a("--direct-models", help="steps that E1 hashes whole, leaf by leaf")
    a("--direct-reps", type=int, help="repetitions of a direct step (H100: 1)")
    a("--sample-mb", type=int, help="bytes hashed per leaf size, MiB (H100: 512)")
    a("--object-cap-mb", type=int, help="largest object hashed whole, MiB; 0 for none")
    a("--max-objects", type=int, help="most objects per whole-object measurement "
      f"(H100: {cpu_hash.MAX_OBJECTS})")
    a("--native", action=argparse.BooleanOptionalAction, default=None,
      help="run openssl speed and b3sum (H100: on)")
    a("--native-threads", help="thread counts for the native ceilings")
    a("--copy-mb", type=int, help="the host-to-GPU copy's buffer, MiB (CUDA only)")
    a("--scratch-dir", help="where b3sum's file goes (default /dev/shm, else the temp dir)")
    a("--e5-model", help="the step whose product list E5 uses (H100: llama-3.2-1b)")
    a("--k", type=int, help="challenge vectors per product (H100: 21)")
    a("--e5-layers", type=int, help="layers E5 times, the rest extrapolated; 0 for all")
    a("--e5-processes", help="E5(a′)'s process-pool size, or 'all'")
    a("--native-batch", action=argparse.BooleanOptionalAction, default=None,
      help="E5(a′)'s native BLAKE3 batch, fetched and built on first use (H100: on)")
    a("--e5-native-layers", type=int, help="layers the native batch times; 0 for all")
    a("--e6-models", help="models for E6 ('none' for no model)")
    a("--e6-n-s", type=int, help="E6's sequences per step (H100: 128)")
    a("--e6-n", type=int, help="E6's tokens per sequence (H100: 128)")
    a("--e6-device", help="E6's device (H100: cuda)")
    a("--e6-warmup", type=int, help="E6's untimed steps")
    a("--e6-reps", type=int, help="E6's timed steps")
    return p


def resolve(ns: argparse.Namespace) -> dict[str, Any]:
    """The preset with every explicitly given option on top."""
    preset = SMOKE if ns.smoke else H100
    return {key: getattr(ns, key) if getattr(ns, key, None) is not None else default
            for key, default in preset.items()}


def run(argv: list[str] | None = None, log=print) -> Path:
    ns = parser().parse_args(argv)
    args = resolve(ns)
    wanted = {e.strip() for e in ns.experiments.split(",") if e.strip()}
    unknown = wanted - set(EXPERIMENTS)
    if unknown:
        raise SystemExit(f"unknown experiments {sorted(unknown)}; known: {EXPERIMENTS}")
    started = time.perf_counter()
    created = dt.datetime.now(dt.UTC)
    out = ns.out or (load_config().output_dir / "benchmarks" /
                     f"hashing_{created.strftime('%Y%m%dT%H%M%SZ')}.json")

    def say(msg: str) -> None:
        log(f"[{time.perf_counter() - started:7.1f} s] {msg}")

    machine = machine_record()
    cores = machine["cores_for_threads"]
    threads = cpu_hash.thread_counts(cores, _ints(args["threads"], cores))
    steps: dict[str, Any] = {}  # the steps whose leaf sizes E1 and E5 use, built once

    def get_step(name: str):
        if name not in steps:
            model, n_s, n, p_dtype = _step_spec(name, args["n_s"], args["n"])
            steps[name] = step(model, n_s=n_s, n=n, compute_dtype=p_dtype)
        return steps[name]

    def e0() -> dict:
        return {"E0": slot("E0", [result("machine", "measured", values={"record": machine},
                                         units={"record": "fields as named; None when absent"})])}

    def e1() -> dict:
        return {"E1": slot("E1", cpu_hash.run_e1(
            buffer_bytes=args["buffer_mb"] * MB,
            chunks=[int(c) * 1024 for c in args["chunk_kb"].split(",")],
            threads=threads, reps=args["reps"],
            size_steps=[get_step(m) for m in _models(args["size_models"])],
            direct_steps=[get_step(m) for m in _models(args["direct_models"])],
            direct_reps=args["direct_reps"], sample_bytes=args["sample_mb"] * MB,
            object_cap=args["object_cap_mb"] * MB or None,
            native_threads=cpu_hash.thread_counts(cores, _ints(args["native_threads"], cores)),
            native=bool(args["native"]), copy_bytes=args["copy_mb"] * MB,
            scratch_dir=ns.scratch_dir, max_objects=args["max_objects"], log=say))}

    def e5() -> dict:
        (name,) = _models(args["e5_model"])
        s, k = get_step(name), args["k"]
        layers = args["e5_layers"] or None
        procs = cores if str(args["e5_processes"]) == "all" else int(args["e5_processes"])
        pool_threads = [t for t in threads if t > 1]
        say(f"E5(a): {name} k={k}, layers {layers or 'all'}")
        e5a = slot("E5a", challenges.run_e5a(s, k=k, layers=layers))
        say(f"E5(a′): thread pools {pool_threads}, {procs} processes")
        rows = challenges.run_e5a_prime(s, k=k, layers=layers, threads=pool_threads,
                                        processes=procs)
        if not args["native_batch"]:
            rows.append(result("native_batch", "skipped",
                               reason="turned off with --no-native-batch"))
        else:
            native_layers = args["e5_native_layers"] or None
            try:
                lib = native.load(native.build(log=say))
            except Exception as e:  # no network or no compiler: keep the other rows
                say(f"E5(a′) native batch unavailable:\n{traceback.format_exc()}")
                rows.append(result("native_batch", "skipped",
                                   reason=f"build failed: {type(e).__name__}: {e}"))
            else:
                say(f"E5(a′): native batch, threads {threads}, layers "
                    f"{native_layers or 'all'}, SIMD degree {lib.simd_degree}")
                try:
                    rows += native.run_native(s, lib, k=k, layers=native_layers,
                                              threads=threads)
                except Exception as e:  # keep the pool rows
                    say(f"E5(a′) native batch failed:\n{traceback.format_exc()}")
                    rows.append(result("native_batch", "skipped",
                                       reason=f"error: {type(e).__name__}: {e}"))
        return {"E5a": e5a, "E5a_prime": slot("E5a_prime", rows)}

    def e6() -> dict:
        device = args["e6_device"]
        if device == "cuda" and not torch.cuda.is_available():
            return {"E6": slot("E6", [result(
                "sgd_step", "skipped", reason="E6 asks for cuda and no CUDA device is visible; "
                                              "pass --e6-device cpu to time it on the CPU")])}
        return {"E6": slot("E6", step_time.run_e6(
            _models(args["e6_models"]), n_s=args["e6_n_s"], n=args["e6_n"], device=device,
            warmup=args["e6_warmup"], reps=args["e6_reps"], log=say))}

    experiments = {"E0": (("E0",), e0), "E1": (("E1",), e1), "E5": (("E5a", "E5a_prime"), e5),
                   "E6": (("E6",), e6)}
    slots: dict[str, dict] = {key: not_implemented_slot(key, reason)
                              for key, reason in GPU_ONLY.items()}
    for exp, (keys, _) in experiments.items():
        reason = ("pending: the run hasn't reached this experiment" if exp in wanted
                  else "not selected with --experiments")
        slots.update({key: slot(key, [result(key, "skipped", reason=reason)]) for key in keys})
    timing: dict[str, float] = {}

    def write(complete: bool) -> Path:
        """The report as it stands, rewritten after each experiment: a late crash keeps the
        finished rows."""
        return write_report({
            "schema": SCHEMA,
            "created_utc": created.isoformat(),
            "argv": sys.argv if argv is None else ["benchmarks.hashing.run", *argv],
            "smoke": ns.smoke,
            "complete": complete,
            "args": {**args, "threads_resolved": threads, "experiments": sorted(wanted)},
            "inputs": {"steps": {k: s.summary() for k, s in steps.items()}},
            "slots": {key: slots[key] for key in SLOTS},
            "elapsed_s": {"total": time.perf_counter() - started, **timing},
        }, out)

    for exp, (keys, fn) in experiments.items():
        if exp not in wanted:
            continue
        t0 = time.perf_counter()
        try:
            slots.update(fn())
        except Exception as e:  # one experiment's failure must not lose the others' results
            say(f"{exp} failed:\n{traceback.format_exc()}")
            slots.update({key: slot(key, [result(key, "skipped",
                                                  reason=f"error: {type(e).__name__}: {e}")])
                          for key in keys})
        timing[exp] = time.perf_counter() - t0
        write(complete=False)
        say(f"{exp} done; report so far in {out}")

    path = write(complete=True)
    say(f"wrote {path}")
    return path


if __name__ == "__main__":
    run()
