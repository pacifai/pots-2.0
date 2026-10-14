"""E6, step time per model: one plain SGD step at full scale's precision, eager then sdpa
attention.

The model is an unmodified ``from_pretrained`` load at its pinned revision (``setup.model``),
trained on random token ids with the model's own causal-LM loss, the way FS3's step trains
(F6b): fp32 master weights, forward and loss under ``torch.autocast`` to the compute dtype
(bf16), backward outside it, SGD in fp32 on the masters. The two matmul precision settings
``setup_determinism`` makes are set around the timing and restored after: fp32 accumulation
for cuBLAS's bf16 matmuls (F5a) and ``float32_matmul_precision("highest")`` (S4c). This is the
prover's lower bound; the gap between eager and sdpa is the cost F16 reports as context. The
verified step (capture, checks) is not timed here.

``setup_determinism`` also turns deterministic algorithms on, which this timing leaves as the
process has it; each row records the state.
"""

from __future__ import annotations

import contextlib
import gc
import statistics
import time
from collections.abc import Callable, Iterator, Sequence

import torch

from benchmarks.hashing.report import result
from benchmarks.hashing.sizes import MODELS
from setup.model import load_pretrained

ATTN_IMPLS = ("eager", "sdpa")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def matmul_precision() -> Iterator[dict]:
    """``setup_determinism``'s two matmul precision settings for the block, restored after;
    yields them for a row's params."""
    cuda_matmul = torch.backends.cuda.matmul
    saved = (cuda_matmul.allow_bf16_reduced_precision_reduction,
             torch.get_float32_matmul_precision())
    cuda_matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield {"allow_bf16_reduced_precision_reduction": False,
               "float32_matmul_precision": "highest"}
    finally:
        cuda_matmul.allow_bf16_reduced_precision_reduction = saved[0]
        torch.set_float32_matmul_precision(saved[1])


def _autocast(device: torch.device, compute_dtype: torch.dtype):
    """``DecoderComputation.autocast``'s rule: autocast when compute is narrower than the fp32
    masters, nothing otherwise."""
    if compute_dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=compute_dtype)


def time_steps(model: torch.nn.Module, *, n_s: int, n: int, warmup: int, reps: int,
               compute_dtype: torch.dtype = torch.bfloat16, lr: float = 1e-5,
               seed: int = 0) -> list[float]:
    """Seconds for each of ``reps`` SGD steps after ``warmup`` untimed ones; forward and loss
    run under autocast to ``compute_dtype``, backward and the update outside it."""
    device = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.config.vocab_size, (n_s, n), generator=g).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    times = []
    for i in range(warmup + reps):
        _sync(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with _autocast(device, compute_dtype):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return times


def _error(e: BaseException) -> str:
    """The exception's type and the first line of its message, for a skipped row's reason."""
    msg = (str(e).splitlines() or [""])[0][:200]
    return f"{type(e).__name__}: {msg}"


def run_e6(models: Sequence[str], *, n_s: int, n: int, device: str, warmup: int, reps: int,
           compute_dtype: torch.dtype = torch.bfloat16,
           log: Callable[[str], None] = print) -> list[dict]:
    if not models:
        return [result("sgd_step", "skipped", reason="no models selected")]
    dev = torch.device(device)
    out = []
    for key in models:
        pin = MODELS[key]
        for impl in ATTN_IMPLS:
            params = {"model": key, "repo": pin.repo, "revision": pin.revision, "attn": impl,
                      "master_dtype": "float32",
                      "compute_dtype": str(compute_dtype).removeprefix("torch."),
                      "autocast": compute_dtype != torch.float32, "n_s": n_s, "n": n,
                      "device": device, "warmup": warmup, "reps": reps, "optimizer": "SGD"}
            log(f"E6 {key} {impl} {n_s}x{n} on {device}")
            try:
                model = load_pretrained(pin.repo, pin.revision, attn_impl=impl,
                                        dtype=torch.float32)
            except Exception as e:  # e.g. offline with only the config cached
                out.append(result("sgd_step", "skipped", params=params,
                                  reason=f"load failed: {_error(e)}"))
                continue
            try:
                model.to(dev)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                with matmul_precision() as precision:
                    params.update(precision)
                    params["deterministic_algorithms"] = \
                        torch.are_deterministic_algorithms_enabled()
                    times = time_steps(model, n_s=n_s, n=n, warmup=warmup, reps=reps,
                                       compute_dtype=compute_dtype)
                values = {"seconds_median": statistics.median(times), "seconds_min": min(times),
                          "seconds": times,
                          "tokens_per_s": n_s * n / statistics.median(times)}
                units = {"seconds_median": "s per step", "seconds_min": "s per step",
                         "seconds": "s per step", "tokens_per_s": "tokens/s"}
                if dev.type == "cuda":
                    values["peak_memory_bytes"] = torch.cuda.max_memory_allocated(dev)
                    units["peak_memory_bytes"] = "bytes"
                out.append(result("sgd_step", "measured", params=params, values=values,
                                  units=units))
            except Exception as e:  # e.g. out of memory: keep the other models' rows
                out.append(result("sgd_step", "skipped", params=params,
                                  reason=f"error: {_error(e)}"))
            finally:
                del model
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    out.append(result("verified_step", "not_implemented",
                      reason="the verified step (capture, checks split by stage) needs the "
                             "verified run on CUDA (FS9's GPU path), and the benchmark "
                             "imports nothing from the protocol but the leaf-size list"))
    return out
