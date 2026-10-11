"""E6, step time per model: one plain SGD step in bf16, eager then sdpa attention.

The model is an unmodified ``from_pretrained`` load at its pinned revision (``setup.model``),
in bf16, trained on random token ids with the model's own causal-LM loss. This is the prover's
lower bound; the gap between eager and sdpa is the cost F16 reports as context. The verified
step (capture, checks) is not timed here.
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable, Sequence

import torch

from benchmarks.hashing.report import result
from benchmarks.hashing.sizes import MODELS
from setup.model import load_pretrained

ATTN_IMPLS = ("eager", "sdpa")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_steps(model: torch.nn.Module, *, n_s: int, n: int, warmup: int, reps: int,
               lr: float = 1e-5, seed: int = 0) -> list[float]:
    """Seconds for each of ``reps`` SGD steps after ``warmup`` untimed ones."""
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
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        _sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return times


def run_e6(models: Sequence[str], *, n_s: int, n: int, device: str, warmup: int, reps: int,
           log: Callable[[str], None] = print) -> list[dict]:
    if not models:
        return [result("sgd_step", "skipped", reason="no models selected")]
    dev = torch.device(device)
    out = []
    for key in models:
        pin = MODELS[key]
        for impl in ATTN_IMPLS:
            params = {"model": key, "repo": pin.repo, "revision": pin.revision, "attn": impl,
                      "dtype": "bfloat16", "n_s": n_s, "n": n, "device": device,
                      "warmup": warmup, "reps": reps, "optimizer": "SGD"}
            log(f"E6 {key} {impl} {n_s}x{n} on {device}")
            try:
                model = load_pretrained(pin.repo, pin.revision, attn_impl=impl,
                                        dtype=torch.bfloat16)
            except Exception as e:  # offline with only the config cached, transformers
                # 4.57 raises AttributeError rather than OSError; any load failure is a skip
                msg = (str(e).splitlines() or [""])[0][:200]
                out.append(result("sgd_step", "skipped", params=params,
                                  reason=f"weights not loadable (not in the local cache while "
                                         f"offline?): {type(e).__name__}: {msg}"))
                continue
            try:
                model.to(dev)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                times = time_steps(model, n_s=n_s, n=n, warmup=warmup, reps=reps)
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
            finally:
                del model
                gc.collect()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
    out.append(result("verified_step", "not_implemented",
                      reason="the verified step (capture, checks split by stage) is out of "
                             "scope for FS10's CPU part"))
    return out
