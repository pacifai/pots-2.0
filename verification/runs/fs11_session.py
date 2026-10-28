"""FS11's H100 sessions: the parts in order, each with its own log, under one deadline.

    .venv/bin/python -m verification.runs.fs11_session run --upload DIR --out DIR --deadline EPOCH
        [--session {fs11,recheck,profile-check,nvlink-probe,kernel-speed,fs32-pair,
                    device-leaves,four-h100,four-h100-profile}]
    .venv/bin/python -m verification.runs.fs11_session fetch-models --upload DIR --out DIR
    .venv/bin/python -m verification.runs.fs11_session env --out DIR

``fs11_job.sh`` runs ``run`` on the pod, then packs the output and ends the pod; see
``verification/README.md``, "Workflow: the first H100 session". ``fs11_prep`` builds the upload
on the Mac; both sessions run from the same upload. ``--session`` (the job's ``FS11_SESSION``)
picks the plan: ``fs11``, the first session (:func:`default_plan`, the default), or
``recheck``, the re-check (:func:`recheck_plan`, below), or ``profile-check``, the profile
check (:func:`profile_check_plan`, below), or ``nvlink-probe``, the four-card hardware probe
(:func:`nvlink_probe_plan`, below), ``kernel-speed``, the GPU BLAKE3 kernel's speed on one
H100 (:func:`kernel_speed_plan`, below: the upload holds the repo, the native sources and the
Mac's ``kernel_speed_sizes.json``, no models; ``fs11_prep --session kernel-speed`` packs it),
``fs32-pair``, the sharded run on two GPUs (:func:`fs32_pair_plan`, below), or
``device-leaves``, FS30's check of the device leaves (:func:`device_leaves_plan`, below), or
``four-h100``, FS35's sequential and pipelined steps on four H100s (:func:`four_h100_plan`,
below; it refuses a machine with fewer than four CUDA devices, and one without an H100 unless
``FS11_ALLOW_ANY_GPU=1``; ``summary.json["four_h100"]`` is :func:`four_h100`'s result and its
log ends with :func:`_four_lines`'s ``FS35 ok`` / ``FS35 FAIL`` lines), or
``four-h100-profile``, FS42 (d)'s profiling run on the same layout (:func:`four_h100_profile_plan`:
per model a plain run, the card and the two arms at four steps, the sequential arm tracing
step 4; ``summary.json["four_h100_profile"]`` is :func:`four_h100_profile`'s result and the log
ends with :func:`_profile_lines`'s ``FS42 ok`` / ``FS42 FAIL`` lines).
The first session's parts, in this
order (every plan the session runs, but the self-test's, starts with **native_build**:
``python -m verification.commitment.native``, which compiles the native BLAKE3 library from
the upload's ``native_cache/`` into it once, offline, so no later part's timing includes the
compile). A plan for two or more GPUs (``nvlink-probe``, ``fs32-pair``, ``four-h100``,
``four-h100-profile``) lists the
F22 peer-copy byte check, ``peer_check``, first among its own parts, ahead of the models part
(the native build, which every plan gets, runs ahead of it), and a pair that lands wrong bytes
stops the session (:func:`peer_check_part`, :func:`peer_check_stop`):

1. **models.** ``fetch-models``: check every packed file of the upload against the manifest's
   SHA-256, then fetch each public model's weights at its pinned commit with ``token=False``
   and check them against the SHA-256 that ``fs11_prep`` recorded from the Mac's cache. A file
   that doesn't match is deleted. A gated model (:data:`GATED`) is never fetched: its files
   come packed in the upload. Only this part may reach the network.
   After this part, on every run (a rerun that skips it included), :func:`check_upload`
   checks every file the later parts read against the manifest again. A packed file that is
   missing or differs keeps every later part from starting; a model whose files are missing
   or differ keeps its own bring-up and smooth parts from starting.
2. **benchmark.** ``benchmarks.hashing.run`` at its H100 preset (FS10), report in
   ``benchmark/hashing.json``. Its GPU-hashing slots need cuPQC (:func:`cupqc_status`). It
   exits 0 whatever its slots did, so the part counts as done only when the report is
   ``complete`` and no result ended in an error (:func:`benchmark_failure`).
3. **bring-up**, one per model: ``bring_up --no-tamper``, one verified step at the full-scale
   config (:data:`FULL_SCALE_ENV`) on the refusal task's ``D``, with ``metrics.py``'s peak
   memory. Its exit code also folds in the concentration guard, which may fail at bf16 (F20);
   the step's acceptance, peaks and times are in its log either way.
4. **smooth**, one per model and task: ``smooth_check``, F6c's 10 plain steps.

The re-check runs the models part, then the bring-up and smooth parts again with settings
added to the full-scale config (:data:`RECHECK_ENV`): the output layer's product checked in 8
row blocks (``VERIF_HEAD_BLOCKS``), and the allocator fix ("fix 2"), which is two settings: the
CUDA allocator's expandable segments (``PYTORCH_ALLOC_CONF``, which the allocator reads at
CUDA's first use, so it is set in the part's environment, with the deprecated
``PYTORCH_CUDA_ALLOC_CONF`` unset) and the cache release between prover and verifier
(``VERIF_RELEASE_CACHE=1``, the default, pinned). :data:`FIX_AB_MODEL`, the cheapest model,
is brought up first, so a failure of the 8 row blocks on CUDA shows within minutes. Right after
its bring-up comes the same bring-up with the whole fix off
(:data:`FIX_OFF_ENV`), its own part and run directory: ``summary.json``'s ``fix2_ab``
(:func:`fix2_ab`) sets the two side by side, so the fix's cost to a verified step reads off
directly. Last comes a timing A/B of the allocator setting alone: per model,
``plain_baseline``'s 10 plain steps twice, once with both allocator variables unset and
once with expandable segments, the first arm alternating from model to model
(:data:`ALLOC_ARMS`). It has no benchmark part.

The profile check runs the models part, the probe, then SmolLM2 at test scale's config on
CUDA (:data:`SMOLLM2_ENV`, ``k`` 15) in four runs of ``run_verified`` and ``plain_baseline``
(the trace-slice, trace-all and untraced runs, and the plain run), after a ``calibrate`` that
writes the band file they link even when the bf16 guard fails (``--write-bands-despite-guard``,
with a ``.guard_failed.json`` sidecar), and ``check5_profile`` on the untraced run's stored
step; then, per full-scale model, a ``plain_baseline`` run and
``bring_up --no-tamper --steps 3 --metrics --check5-repeats 3``, both at
:data:`FULL_SCALE_ENV` plus :data:`RECHECK_ENV`. The tracing variables (``VERIF_TRACE_STEP``,
``VERIF_TRACE_C5``) are in the part environments and read by the runs. It has no benchmark part,
and its parts keep the sampler on.

The hardware probe (F22's HW2 and the first half of HW3) runs the models part, then
``verification.runs.nvlink_probe`` (NVLink topology and peer-copy bandwidth, report
``nvlink/nvlink.json``), then, per model of :data:`RUN_PINS` and per card ``0 … n−1``
(``FS31_CARDS``, default 4), ``verification.runs.card_identity`` with ``CUDA_VISIBLE_DEVICES`` set
to the card, at :data:`FULL_SCALE_ENV` plus :data:`RECHECK_ENV`. ``summary.json``'s ``hw3_cards``
(:func:`hw3_cards`) compares the cards' roots per model. Unlike the other sessions, it refuses
unless torch sees at least ``FS31_CARDS`` CUDA devices (:func:`refusal`); the H100 check stays,
and ``FS11_ALLOW_ANY_GPU=1`` overrides that one.

The two-GPU session (FS32f, F22) runs the models part, the whole fast suite (``-rf``) and the
slow two-GPU tests, SmolLM2's calibration, then ``card_identity`` and a 3-step ``run_verified``
unsharded and again with ``VERIF_VERIFIER_DEVICES=cuda:0,cuda:1``, and the judged cheats with
that setting; then FS33d's pipelined arms (sharded, ``VERIF_PIPELINED=1``: ``run_pipelined``,
``run_pipelined_metrics`` and the cheats again), each compared with its sharded sequential
arm. ``summary.json``'s ``fs32_pair`` (:func:`fs32_pair`) compares the two arms' roots
(step 1's transcript root, the final weights' root, the per-step losses, the kept step 3's
root) and per-step verdicts, and holds each cheat's outcome (which must be ``exact``) and the
failing tests; the job log ends with one ``FS32`` line per item. It refuses unless torch sees
two CUDA devices, whatever their type. Each of its parts records every GPU's peak memory
(``parts/<name>/gpu_peak.json``, MiB by ``nvidia-smi`` index, the card's whole use).

The device-leaves session (FS30) runs the models part, the CUDA tests (the kernel's two test
files, FS30's own, and the GPU-memory test of the capture) with a junit report, then one
``bring_up --no-tamper --device-leaves`` per model at :data:`FULL_SCALE_ENV` plus
:data:`RECHECK_ENV`, the cheapest model first. Each bring-up prints one ``device leaves:`` line:
the prover's root (arrival digests and the device kernel), the root the CPU path gives over the
same leaves, how many arrival digests the commit could use, how many outputs the capture
hashed on arrival, how many leaves the commit actually took from the digests, the product
leaves ``K`` (O18) and the ``dK`` leaves ``D``.
``summary.json``'s ``device_leaves`` (:func:`device_leaves_gate`) fails closed. It refuses
without CUDA and without an H100 (``FS11_ALLOW_ANY_GPU=1`` overrides). Its upload is
``fs11_prep``'s default, with the four models.

Every part runs in its own process group, with ``HF_HUB_OFFLINE=1`` (all but the first), no
Hugging Face token and no Runpod key in its environment. A part's log is
``parts/<name>/log.txt``, and ``current_part`` holds the running part's process group id.
``parts/<name>/DONE`` is written when it succeeds, and a rerun into the same output directory
skips it. A part that fails doesn't stop the others. A part still running at ``--deadline``
(or past its own time limit) is killed, and the parts after it are recorded as not started.
A pytest part that wrote its junit report and has not exited :data:`EXIT_HANG_GRACE_S` (180 s)
later has hung at exit: its ``/proc`` snapshot (``hang.txt``) is written, SIGUSR1 makes it dump
every thread's stack into ``stacks.txt`` (``POTS_STACKS_FILE``), it is killed, and its status is
``exit_hang``, a failure like ``killed_timeout`` and ``killed_deadline``, which get the same
snapshot first. ``exit-probe`` (:func:`run_exit_probe`) is run by hand to find a module that
hangs at exit: each test module in its own pytest process.
``summary.json`` is rewritten after each part.

Before any part, ``run`` prints the environment (GPU, driver, CUDA, torch, free disk and RAM)
to ``environment.json`` and refuses to start unless CUDA is available and the GPU is an H100;
``FS11_ALLOW_ANY_GPU=1`` overrides. ``run --selftest`` (``FS11_SELFTEST``) skips that check
and runs :func:`selftest_plan`'s stand-in parts, which need only the standard library. Exits 0
when every part succeeded, 1 when any failed, and :data:`EXIT_REFUSED` when the environment
check refused.

``run`` also refuses when the running Python's version is not the one the upload's manifest
records (:func:`python_mismatch`; the self-test is exempt): 3.14.0 hangs at exit (gh-146313), and
``fs11_setup.sh`` builds the pod's venv at the Mac's version. ``FS11_ALLOW_PYTHON_MISMATCH=1``
overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import ParseError

__all__ = ["PROFILE_CHECK_TIMEOUT_S", "PROFILE_HELD", "PROFILE_PINS", "SMOLLM2_ENV", "RUN_PINS",
           "GATED", "TASKS", "FULL_SCALE_ENV", "RECHECK_ENV", "ALLOC_ARMS",
           "alloc_env", "FIX_AB_MODEL", "FIX_OFF_ENV", "SESSIONS", "EXIT_REFUSED", "Part",
           "default_plan", "recheck_plan", "profile_check_plan", "nvlink_probe_plan",
           "kernel_speed_plan", "device_leaves_plan", "device_leaves_gate",
           "kernel_speed_gate", "SIZES_NAME", "load_plan",
           "selftest_plan", "fs31_cards", "hw3_cards", "fs32_pair_plan", "fs32_pair",
           "four_h100_plan", "four_h100", "four_default_deadline_s", "FOUR_TEST_FILES",
           "four_h100_profile_plan", "four_h100_profile", "four_profile_default_deadline_s",
           "with_native_build",
           "choose_plan", "run_session", "fix2_ab",
           "fetch_models", "check_upload", "benchmark_failure", "environment", "cupqc_status",
           "data_key", "main"]

# The four full-scale models at their commits (tests/verification/full_scale_helpers.PINS, in
# the order the session brings them up).
RUN_PINS: tuple[tuple[str, str], ...] = (
    ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
    ("tiiuae/Falcon3-1B-Instruct", "28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
)
# Repos behind a licence click. Their files travel in the upload from the Mac's cache; the pod
# never fetches them and never holds a token.
GATED = frozenset({"meta-llama/Llama-3.2-1B-Instruct"})
# |D| per task at full scale (FS6): Alpaca's 369 records, AdvBench's 233.
TASKS = {"refusal": 369, "jailbreak": 233}
BRING_UP_TASK = "refusal"
# The full-scale run config (F6b's dtypes, F8a's k at T = 10, η = 10⁻³ fixed by F6c).
FULL_SCALE_ENV = {
    "VERIF_DEVICE": "cuda", "VERIF_MASTER_DTYPE": "float32", "VERIF_COMPUTE_DTYPE": "bfloat16",
    "VERIF_ATTN_IMPL": "eager", "VERIF_BATCH": "128", "VERIF_SEQ_LEN": "128", "VERIF_K": "21",
    "VERIF_ETA": "1e-3", "VERIF_STEPS": "10", "VERIF_SEED": "1", "VERIF_POISON_COUNT": "1",
}
# Each part's own time limit, seconds; the session deadline cuts any of them shorter.
TIMEOUT_S = {"models": 1800, "benchmark": 5400, "bring_up": 2400, "smooth": 1200}
# The re-check's additions to FULL_SCALE_ENV, for its bring-up and smooth parts: the output
# layer's product in 8 row blocks, and the allocator fix: expandable segments (four of the
# first session's seven out-of-memory errors left 6–12 GiB "reserved but unallocated") and
# the cache release between prover and verifier (the default, pinned against the pod's
# environment; only the bring-up reads it).
EXPANDABLE = "expandable_segments:True"


def alloc_env(conf: str | None) -> dict[str, str | None]:
    """The part environment's allocator setting: ``PYTORCH_ALLOC_CONF`` (torch 2.9's own name,
    which wins over the other) at ``conf``, and the deprecated ``PYTORCH_CUDA_ALLOC_CONF`` unset,
    so the pod's environment can't add to it. ``None`` unsets both."""
    return {"PYTORCH_ALLOC_CONF": conf, "PYTORCH_CUDA_ALLOC_CONF": None}


RECHECK_ENV: dict[str, str | None] = {"VERIF_HEAD_BLOCKS": "8", **alloc_env(EXPANDABLE),
                                      "VERIF_RELEASE_CACHE": "1"}
# The verified-step A/B of the whole allocator fix: this model's bring-up runs again right
# after its own, with RECHECK_ENV's two fix settings undone (None unsets). The fix is kept only
# if it costs no speed; Qwen2.5-0.5B fit in the first session without it, so this arm runs.
FIX_AB_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIX_OFF_ENV: dict[str, str | None] = {**alloc_env(None), "VERIF_RELEASE_CACHE": "0"}
# The re-check's allocator A/B: each arm's name and its PYTORCH_ALLOC_CONF (alloc_env; None
# unsets both variables, whatever the pod's environment holds). Even-numbered models in
# RUN_PINS run them in this order, odd-numbered ones in reverse, so an order effect (a warm
# file cache, the GPU's clocks) doesn't always favour the same arm.
ALLOC_ARMS: tuple[tuple[str, str | None], ...] = (("alloc_default", None),
                                                   ("alloc_expandable", EXPANDABLE))
# The re-check's time limits: about 3–5 times each part's expected time (the first session's
# smooth parts took 25–44 s, its one finished bring-up 191 s; a plain arm takes about a minute).
RECHECK_TIMEOUT_S = {"models": 1800, "bring_up": 1800, "smooth": 600, "plain": 600}
PLAIN_RUN = "plain_baseline"  # plain_baseline.RUN_NAME, its records' directory
# The profile check's time limits, seconds: SmolLM2's parts at test scale, and each full-scale
# model's plain run and bring-up. The full-scale plain run is the re-check's plain arm (a minute
# for its 10 timed steps, two memory-pass steps and one count-pass step), so it keeps that
# arm's 600 s. The bring-up's 3600 s: Llama-3.2-1B's two-step bring-up with repeats took 1152 s
# on an H100 and the third, untraced step adds 200-300 s, so about 1450 s, 2.5 times under.
PROFILE_CHECK_TIMEOUT_S = {"models": 1800, "probe": 600, "calibrate": 900, "run": 900,
                           "plain": 600, "check5_profile": 900, "bring_up": 3600}
# The full-scale models the profile check runs, in RUN_PINS order. Qwen2.5-1.5B is held out
# (user, 2026-10-19) until the architecture improves: its verifier ran out of GPU memory at
# check 5's first output-layer backward, short by 1.16 GiB, and the fix (the prover's fp32
# weights to host RAM during the verifier's turn) waits for that work.
PROFILE_HELD = frozenset({"Qwen/Qwen2.5-1.5B-Instruct"})
PROFILE_PINS: tuple[tuple[str, str], ...] = tuple(p for p in RUN_PINS if p[0] not in PROFILE_HELD)
# SmolLM2's run config in the profile check: test scale's batch, sequence length, steps and
# seed (the config's defaults, so set none), on CUDA at the full-scale dtypes. k is 15: a bf16
# calibration of SmolLM2 passed its guard but asked for k 15, and the band file's context holds
# k, so every part that reads the file runs at it.
SMOLLM2_ENV = {"VERIF_DEVICE": "cuda", "VERIF_MASTER_DTYPE": "float32",
               "VERIF_COMPUTE_DTYPE": "bfloat16", "VERIF_ATTN_IMPL": "eager", "VERIF_K": "15"}
SMOLLM2_TASK = "refusal"
SMOLLM2_SEED = 1  # the config's default seed, which names the honest run's directory
# run_verified.KEEP_STEP, the step its run stores at the band file's calibration seed (a test
# holds the two equal; this module imports no torch).
STORED_STEP = 4
WEIGHTS_FILE = "final_weights.json"  # plain_baseline.WEIGHTS_FILE: a verified run's final weights
# Environment variables a part never inherits.
SECRETS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH",
           "RUNPOD_API_KEY")
CUPQC_ENV = "CUPQC_SDK_DIR"
CUPQC_SLOTS = ("E2", "E3 (its hashing half)", "E5b", "E5c")
EXIT_REFUSED = 4
KILL_GRACE_S = 30  # SIGTERM, then SIGKILL this much later


def data_key(repo: str, task: str) -> str:
    """The name of one model and task's data and run directory: ``<repo name>_<task>``."""
    return f"{repo.split('/')[-1]}_{task}"


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _say(msg: str) -> None:
    try:  # UTC, as the job's; a full disk under the log mustn't stop the session
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)
    except OSError:
        pass


# ---- the environment ---------------------------------------------------------------------


def _meminfo() -> dict[str, int]:
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return {}
    out = {}
    for line in lines:
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[name] = int(parts[0]) * 1024
    return out


def _nvidia_smi() -> list[dict[str, str]] | str:
    try:
        text = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=60,
            check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable: {type(e).__name__}"
    keys = ("name", "driver", "memory")
    return [dict(zip(keys, (x.strip() for x in line.split(",")))) for line in text.splitlines()
            if line.strip()]


def environment(out: Path) -> dict[str, Any]:
    """The pod as the session sees it: GPU, driver, CUDA, torch, free disk and RAM. Names no
    secret: of the environment it records only the Runpod pod id and the cuPQC path. Without
    torch (the self-test on a bare pod) the torch fields are ``None`` and CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        torch = None
    cuda = torch is not None and torch.cuda.is_available()
    mem = _meminfo()
    phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") \
        if hasattr(os, "sysconf") else None
    out.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(out)
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "torch": torch and torch.__version__, "torch_cuda": torch and torch.version.cuda,
        "cuda_available": cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda else [],
        "nvidia_smi": _nvidia_smi(),
        "disk_free_bytes": disk.free, "disk_total_bytes": disk.total,
        "ram_total_bytes": mem.get("MemTotal", phys),
        "ram_available_bytes": mem.get("MemAvailable"),
        "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
        "cupqc": cupqc_status(),
    }


def refusal(env: Mapping[str, Any], allow_any_gpu: bool, min_cards: int | None = None,
            *, need: str | None = None) -> str | None:
    """Why the session won't start on this machine, or ``None``. ``min_cards`` (the hardware
    probe's ``FS31_CARDS``) is the CUDA devices torch must see; ``allow_any_gpu`` overrides the
    H100 check, not this."""
    seen = len(env["gpus"]) if env["cuda_available"] else 0
    if min_cards is not None and seen < min_cards:
        want = f"{CARDS_ENV} is {min_cards}" if need is None else f"{need} needs {min_cards}"
        return f"torch sees {seen} CUDA devices, {want}"
    if allow_any_gpu:
        return None
    if not env["cuda_available"]:
        return "torch.cuda.is_available() is false"
    if not any("H100" in g for g in env["gpus"]):
        return f"no H100 among the GPUs {env['gpus']}"
    return None


PYTHON_MISMATCH_ENV = "FS11_ALLOW_PYTHON_MISMATCH"


def python_mismatch(upload: Path, running: str) -> str | None:
    """Why the pod's Python is not the packing machine's, or ``None``. The upload's manifest
    records the Mac's version (``python_version``). CPython 3.14.0 can hang at exit (the
    resource tracker's deadlock, gh-146313), and the cure, 3.14.7, is only a venv built with
    ``fs11_setup.sh`` away. An older upload without the key is refused with its own message."""
    try:
        want = json.loads((upload / "manifest.json").read_text()).get("python_version")
    except (OSError, ValueError, AttributeError) as e:
        return (f"the upload's manifest.json is unreadable ({type(e).__name__}: {e}), so the "
                "Python version it records can't be checked")
    if not want:
        return ("the upload's manifest.json records no Python version (an older upload): "
                "repack it with fs11_prep, which records the Mac's")
    if want != running:
        return (f"the pod's Python {running} is not the Mac's {want}: another Python can behave "
                "differently at exit (3.14.0 hangs there, gh-146313); build the venv with "
                "fs11_setup.sh")
    return None


def _gib(x: int | None) -> str:
    return "?" if x is None else f"{x / 2**30:.1f} GiB"


def print_environment(env: Mapping[str, Any]) -> None:
    smi = env["nvidia_smi"]
    driver = smi if isinstance(smi, str) else ", ".join(sorted({g["driver"] for g in smi}))
    _say(f"GPU {env['gpus'] or 'none'}; driver {driver}; CUDA {env['torch_cuda']} "
         f"(available: {env['cuda_available']}); torch {env['torch']}; python {env['python']}")
    _say(f"disk free {_gib(env['disk_free_bytes'])} of {_gib(env['disk_total_bytes'])}; RAM "
         f"available {_gib(env['ram_available_bytes'])} of {_gib(env['ram_total_bytes'])}; "
         f"{env['cpu_count']} CPUs; pod {env['runpod_pod_id']}")


def cupqc_status() -> dict[str, Any]:
    """Whether cuPQC is installed, and what that means for the benchmark's GPU-hashing slots.

    cuPQC's licence is the user's call, deferred to just before the rental. ``$CUPQC_SDK_DIR``
    names an installed SDK. The benchmark has no cuPQC extension yet (``benchmarks/hashing/
    gpu.py``), so those slots are skipped either way; the session records why."""
    sdk = os.environ.get(CUPQC_ENV)
    found = bool(sdk) and Path(sdk).is_dir()
    if not sdk:
        why = (f"${CUPQC_ENV} is not set: cuPQC is not installed (its licence is deferred), so "
               "these slots are skipped")
    elif not found:
        why = f"${CUPQC_ENV}={sdk} is not a directory, so these slots are skipped"
    else:
        why = (f"cuPQC found at {sdk}, but the benchmark has no cuPQC extension to call it "
               "(benchmarks/hashing/gpu.py), so these slots are still skipped")
    return {"env": CUPQC_ENV, "path": sdk, "found": found, "slots": list(CUPQC_SLOTS),
            "reason": why}


# ---- the parts --------------------------------------------------------------------------


@dataclass
class Part:
    name: str
    argv: list[str]  # "{python}" stands for the session's interpreter
    env: dict[str, str | None] = field(default_factory=dict)  # None unsets
    timeout_s: float | None = None
    # models, peer_check, benchmark, bring_up, smooth, plain, probe, kernel_speed, other
    kind: str = "other"
    info: dict[str, Any] = field(default_factory=dict)  # model, task, run_dir, report


def _run_dir(out: Path, upload: Path, key: str, name: str | None = None) -> Path:
    """``runs/<name>/`` (default ``runs/<key>/``), a part's ``VERIF_OUTPUT_DIR``, its ``data``
    linked to the upload's ``data/<key>``."""
    d = out / "runs" / (name or key)
    d.mkdir(parents=True, exist_ok=True)
    link = d / "data"
    if not link.exists() and not link.is_symlink():
        link.symlink_to((upload / "data" / key).resolve(), target_is_directory=True)
    return d


def _models_part(upload: Path, out: Path, timeout_s: float) -> Part:
    return Part("models", ["{python}", "-m", "verification.runs.fs11_session", "fetch-models",
                           "--upload", str(upload), "--out", str(out / "models")],
                env={"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None, "VERIF_PIPELINED": None},
                timeout_s=timeout_s, kind="models",
                info={"report": str(out / "models" / "models.json")})


PEER_CHECK_NAME = "peer_check"
PEER_CHECK_TIMEOUT_S = 300


def peer_check_part(out: Path, timeout_s: float = PEER_CHECK_TIMEOUT_S) -> Part:
    """F22's peer-copy byte check (``verification/runs/peer_check.py``) as a part, its report
    ``<out>/peer_check/peer_check.json``. Every plan on two or more GPUs lists it first, before
    its models part (it needs no model): put ``peer_check_part(out)`` at the head of the plan.
    A pair that lands wrong bytes stops the session (:func:`peer_check_stop`)."""
    report = out / PEER_CHECK_NAME / "peer_check.json"
    return Part(PEER_CHECK_NAME,
                ["{python}", "-m", "verification.runs.peer_check", "--out", str(report)],
                env={**PCI_ORDER_ENV, "VERIF_PIPELINED": None}, timeout_s=timeout_s,
                kind="peer_check",
                info={"report": str(report)})


def _peer_check_extras(report: Path) -> dict[str, Any]:
    """The check's verdict from its report: ``all_ok``; for each wrong pair ``wrong`` (the cases
    whose bytes compared wrong) and ``raised`` (the cases that raised or gave a bad result, each
    to its ``"Type: message"``); ``card_suspect`` (the pairs whose host-staged control compared
    wrong); ``cannot`` (the pairs where the control raised or every direct case raised: the check
    could not copy there, so nothing is known of the peer path); the lowest direct ``gbps``; the
    ``reason`` when no pair could be checked; ``{"peer_check": None}`` when the part left no
    report."""
    if not report.exists():
        return {"peer_check": None}
    doc = json.loads(report.read_text())
    pairs = doc.get("pairs") or {}
    rates = [p["gbps"] for p in pairs.values() if p.get("gbps")]
    wrong: dict[str, list[str]] = {}
    raised: dict[str, dict[str, str]] = {}
    suspect: list[str] = []
    cannot: list[str] = []
    for k, p in pairs.items():
        if p.get("ok"):
            continue
        cases = p.get("cases") or {}
        errors = {c: e for c, e in (p.get("errors") or {}).items() if c in cases}
        direct = [c for c in cases if c != "host_staged"]
        wrong[k] = [c for c, ok in cases.items() if not ok and c not in errors]
        if errors:
            raised[k] = errors
        if "host_staged" in errors or (direct and all(c in errors for c in direct)):
            cannot.append(k)
        elif p.get("card_suspect"):
            suspect.append(k)
    return {"peer_check": {
        "all_ok": doc.get("all_ok") is True, "reason": doc.get("reason"),
        "wrong": wrong, "raised": raised, "card_suspect": suspect, "cannot": cannot,
        "min_gbps": min(rates) if rates else None, "seconds": doc.get("seconds"),
        "gbps": {k: p["gbps"] for k, p in pairs.items() if p.get("gbps")},
        "devices": doc.get("devices")}}


def _peer_error(case: str, err: str) -> str:
    from verification.runs.nvlink_probe import peer_error_text  # stdlib only, off module level

    return peer_error_text(case, err)


def peer_wrong_text(check: Mapping[str, Any] | None,
                    status: str | None = None) -> str:
    """The plain-words account of a failed check (from :func:`_peer_check_extras`), each pair as
    its own fault: wrong bytes (the cases), a case that raised (its error), the check unable to
    copy at all (the control or every case raised: the error, and no verdict on the peer path),
    or the control wrong (the card itself is suspect). A report that names no wrong pair says the
    check ended without a verdict (and, given the part's ``status``, how the part ended)."""
    if check is None:
        return "the peer-copy byte check left no report"
    if check.get("reason") and not check.get("wrong"):
        return f"the peer-copy byte check could not run ({check['reason']})"
    suspect = set(check.get("card_suspect") or ())
    cannot = set(check.get("cannot") or ())
    raised = check.get("raised") or {}
    bytes_bits: list[str] = []
    other: list[str] = []
    for k, c in (check.get("wrong") or {}).items():
        errs = [_peer_error(n, e) for n, e in (raised.get(k) or {}).items()]
        if k in cannot:
            other.append(f"pair {k}: the check could not copy, so the peer path is not judged "
                         f"({errs[0] if errs else 'nothing ran'})")
        elif k in suspect:
            other.append(f"pair {k}: the host-staged control was wrong: the card itself is "
                         f"suspect, not the peer path")
        elif c:
            bytes_bits.append(f"{k} ({'; '.join([', '.join(c), *errs])})")
        else:
            other.append(f"pair {k}: " + ("; ".join(errs) or "no case ran"))
    head = ["peer copies landed wrong bytes on pair " + "; ".join(bytes_bits)] if bytes_bits else []
    if head or other:
        return "; ".join(head + other)
    ended = f"the part ended {status}, and " if status else ""
    return (f"the peer-copy byte check ended without a verdict ({ended}its report names no "
            f"wrong pair)")


def peer_check_stop(rec: Mapping[str, Any]) -> str | None:
    """Why the session stops after a ``peer_check`` part's record ``rec``, or ``None`` when the
    check passed (``ok`` or done earlier, ``all_ok`` in its report)."""
    check = (rec.get("extras") or {}).get("peer_check")
    if rec["status"] in _PEER_DONE and check and check.get("all_ok"):
        return None
    why = peer_wrong_text(check, None if rec["status"] in _PEER_DONE else rec["status"])
    if rec["status"] not in _PEER_DONE and "ended without a verdict" not in why:
        why += f" (the part ended {rec['status']})"
    return f"{why}; no later part ran, a rate or result over wrong copies means nothing"


_PEER_DONE = ("ok", "skipped_done")


def _probe_part(out: Path) -> Part:
    """FS22's probe at :data:`FULL_SCALE_ENV`'s settings, writing ``<out>/probe/probe.json``."""
    report = out / "probe" / "probe.json"
    return Part("probe", ["{python}", "-m", "verification.runs.probe", "--out", str(report)],
                env=dict(FULL_SCALE_ENV), timeout_s=600, kind="probe",
                info={"report": str(report)})


def _probe_extras(report: Path) -> dict[str, Any]:
    if not report.exists():
        return {"probe": None}
    doc = json.loads(report.read_text())
    rates, hash_ = doc["rates"], doc["rates"].get("hash")

    def gb(key: str) -> str:
        return "n/a" if rates.get(key) is None else f"{rates[key] / 1e9:.1f}"

    def tflops(key: str) -> str:
        return "n/a" if rates.get(key) is None else f"{rates[key] / 1e12:.1f}"

    hash_gb = "n/a" if not hash_ else f"{1 / hash_['per_byte_s'] / 1e9:.1f}"
    line = (f"probe ({doc['device']}, {doc['threads']} threads): bf16 {tflops('gemm_bf16_flops')} "
            f"TFLOP/s, fp32 {tflops('gemm_fp32_flops')} TFLOP/s, HBM copy "
            f"{gb('hbm_copy_bytes')} GB/s, H2D pinned {gb('h2d_pinned_bytes')} GB/s, host memcpy "
            f"{gb('host_memcpy_bytes')} GB/s, hashing {hash_gb} GB/s; {len(doc['errors'])} "
            f"unmeasured, {doc['wall_s']:.0f} s")
    return {"record": str(report), "summary": line, "errors": doc["errors"]}


def _bring_up_parts(upload: Path, out: Path, extra: Mapping[str, str | None],
                    timeout_s: float, args: Sequence[str] = ()) -> list[Part]:
    """One ``bring_up --no-tamper`` (and ``args``) per model, on the refusal task, with
    ``extra`` added to the full-scale config."""
    parts = []
    for repo, rev in RUN_PINS:
        key = data_key(repo, BRING_UP_TASK)
        d = _run_dir(out, upload, key)
        parts.append(Part(f"bring_up_{repo.split('/')[-1]}",
                          ["{python}", "-m", "verification.runs.bring_up", "--no-tamper",
                           *args],
                          env={**_run_env(repo, rev, BRING_UP_TASK), **extra,
                               "VERIF_OUTPUT_DIR": str(d)},
                          timeout_s=timeout_s, kind="bring_up",
                          info={"model": repo, "task": BRING_UP_TASK, "run_dir": str(d)}))
    return parts


def _smooth_parts(upload: Path, out: Path, extra: Mapping[str, str | None],
                  timeout_s: float) -> list[Part]:
    """One ``smooth_check`` per model and task, with ``extra`` added to the full-scale config."""
    parts = []
    for repo, rev in RUN_PINS:
        for task in TASKS:
            key = data_key(repo, task)
            d = _run_dir(out, upload, key)
            parts.append(Part(f"smooth_{key}", ["{python}", "-m", "verification.runs.smooth_check"],
                              env={**_run_env(repo, rev, task), **extra,
                                   "VERIF_OUTPUT_DIR": str(d)},
                              timeout_s=timeout_s, kind="smooth",
                              info={"model": repo, "task": task, "run_dir": str(d),
                                    "report": str(d / "smooth_check" / f"{key}.json")}))
    return parts


def default_plan(upload: Path, out: Path) -> list[Part]:
    """The first H100 session's parts (module docstring), at :data:`FULL_SCALE_ENV`."""
    plan = [_models_part(upload, out, TIMEOUT_S["models"])]
    report = out / "benchmark" / "hashing.json"
    plan.append(Part("benchmark", ["{python}", "-m", "benchmarks.hashing.run", "--out",
                                   str(report)],
                     env={"VERIF_OUTPUT_DIR": str(out / "benchmark")},
                     timeout_s=TIMEOUT_S["benchmark"], kind="benchmark",
                     info={"report": str(report)}))
    plan += _bring_up_parts(upload, out, {}, TIMEOUT_S["bring_up"])
    plan += _smooth_parts(upload, out, {}, TIMEOUT_S["smooth"])
    return plan


def _plain_parts(upload: Path, out: Path) -> list[Part]:
    """The allocator A/B: two ``plain_baseline`` runs per model on the refusal task, one per
    :data:`ALLOC_ARMS` arm, each in its own output directory, with ``VERIF_HEAD_BLOCKS`` as in
    the re-check's other parts."""
    parts = []
    for i, (repo, rev) in enumerate(RUN_PINS):
        key = data_key(repo, BRING_UP_TASK)
        for arm, conf in ALLOC_ARMS if i % 2 == 0 else ALLOC_ARMS[::-1]:
            d = _run_dir(out, upload, key, f"{key}_plain_{arm}")
            parts.append(Part(f"plain_{repo.split('/')[-1]}_{arm}",
                              ["{python}", "-m", "verification.runs.plain_baseline"],
                              env={**_run_env(repo, rev, BRING_UP_TASK),
                                   "VERIF_HEAD_BLOCKS": RECHECK_ENV["VERIF_HEAD_BLOCKS"],
                                   **alloc_env(conf), "VERIF_OUTPUT_DIR": str(d)},
                              timeout_s=RECHECK_TIMEOUT_S["plain"], kind="plain",
                              info={"model": repo, "task": BRING_UP_TASK, "run_dir": str(d),
                                    "arm": arm, "alloc_conf": conf,
                                    "report": str(d / PLAIN_RUN / "records.jsonl")}))
    return parts


def _fix_off_part(on: Part, upload: Path, out: Path) -> Part:
    """``on``'s bring-up again with the whole allocator fix off (:data:`FIX_OFF_ENV`), in its
    own run directory."""
    key = data_key(on.info["model"], BRING_UP_TASK)
    d = _run_dir(out, upload, key, f"{key}_fix2off")
    return Part(f"{on.name}_fix2off", list(on.argv),
                env={**on.env, **FIX_OFF_ENV, "VERIF_OUTPUT_DIR": str(d)},
                timeout_s=on.timeout_s, kind="bring_up",
                info={**on.info, "run_dir": str(d), "arm": "fix2_off", "alloc_conf": None,
                      "release_cache": False})


def recheck_plan(upload: Path, out: Path) -> list[Part]:
    """The re-check's parts (module docstring): the models part; the bring-up and smooth parts
    at :data:`FULL_SCALE_ENV` plus :data:`RECHECK_ENV`, with :data:`FIX_AB_MODEL`'s bring-up
    first, followed by its fix-off arm, then the other models' bring-ups in
    :data:`RUN_PINS` order; then the allocator A/B. No benchmark.

    The cheapest model goes first so that a failure of the 8 head blocks on CUDA shows within
    minutes of the start."""
    plan = [_models_part(upload, out, RECHECK_TIMEOUT_S["models"])]
    ups = _bring_up_parts(upload, out, RECHECK_ENV, RECHECK_TIMEOUT_S["bring_up"])
    (on,) = [p for p in ups if p.info["model"] == FIX_AB_MODEL]
    on.info.update(arm="fix2_on", alloc_conf=EXPANDABLE, release_cache=True)
    plan += [on, _fix_off_part(on, upload, out)] + [p for p in ups if p is not on]
    plan += _smooth_parts(upload, out, RECHECK_ENV, RECHECK_TIMEOUT_S["smooth"])
    plan += _plain_parts(upload, out)
    return plan


def _smollm2_parts(upload: Path, out: Path) -> list[Part]:
    """The profile check's SmolLM2 parts (module docstring), in their order. ``runs/smollm2/``
    holds the data link and, once ``calibrate`` has run, ``bands.json``; each later run's
    directory links both (the config reads ``$VERIF_OUTPUT_DIR/data`` and ``bands.json``)."""
    from setup.data import TEST_SCALE_TOKENIZER as pin  # the pin's repo and revision
    key = data_key(pin.repo, SMOLLM2_TASK)
    base = {**SMOLLM2_ENV, "VERIF_MODEL": pin.repo, "VERIF_MODEL_REVISION": pin.revision}
    home = _run_dir(out, upload, key, "smollm2")

    def run_dir(name: str) -> Path:
        d = _run_dir(out, upload, key, name)
        bands = d / "bands.json"
        if not bands.is_symlink() and not bands.exists():
            bands.symlink_to(os.path.relpath(home / "bands.json", d))
        return d

    def part(name: str, module: str, d: Path, timeout_s: float, *args: str, kind: str = "other",
             trace: Mapping[str, str] | None = None, **info: Any) -> Part:
        return Part(name, ["{python}", "-m", f"verification.runs.{module}", *args],
                    env={**base, **(trace or {}), "VERIF_OUTPUT_DIR": str(d)},
                    timeout_s=timeout_s, kind=kind,
                    info={"model": pin.repo, "task": SMOLLM2_TASK, "run_dir": str(d), **info})

    t = PROFILE_CHECK_TIMEOUT_S
    steps = ("--steps", "6", "--metrics")
    untraced = run_dir("smollm2_untraced")
    honest = untraced / f"honest-{pin.repo.split('/')[-1]}-{SMOLLM2_TASK}-bpr0-s{SMOLLM2_SEED}"
    plain = run_dir("smollm2_plain")
    return [
        part("smollm2_calibrate", "calibrate", home, t["calibrate"],
             "--write-bands-despite-guard"),
        part("smollm2_trace_slice", "run_verified", run_dir("smollm2_trace_slice"), t["run"],
             *steps, trace={"VERIF_TRACE_STEP": "5"}),
        part("smollm2_trace_all", "run_verified", run_dir("smollm2_trace_all"), t["run"],
             *steps, trace={"VERIF_TRACE_STEP": "5", "VERIF_TRACE_C5": "all"}),
        part("smollm2_untraced", "run_verified", untraced, t["run"], *steps,
             trace={"VERIF_TRACE_STEP": "0"}),
        part("smollm2_plain", "plain_baseline", plain, t["plain"], kind="plain",
             report=str(plain / PLAIN_RUN / "records.jsonl")),
        part("smollm2_check5_profile", "check5_profile", untraced, t["check5_profile"],
             "--run-dir", str(honest), "--step", str(STORED_STEP), "--out",
             str(honest / "check5_profile.jsonl"), report=str(honest / "check5_profile.jsonl")),
    ]


def _full_scale_plain_part(repo: str, rev: str, upload: Path, out: Path,
                           timeout_s: float) -> Part:
    """One ``plain_baseline`` run of ``repo`` on the refusal task at :data:`FULL_SCALE_ENV` plus
    :data:`RECHECK_ENV`, in its own run directory."""
    key = data_key(repo, BRING_UP_TASK)
    d = _run_dir(out, upload, key, f"{key}_plain")
    return Part(f"plain_{repo.split('/')[-1]}",
                ["{python}", "-m", "verification.runs.plain_baseline"],
                env={**_run_env(repo, rev, BRING_UP_TASK), **RECHECK_ENV,
                     "VERIF_OUTPUT_DIR": str(d)},
                timeout_s=timeout_s, kind="plain",
                info={"model": repo, "task": BRING_UP_TASK, "run_dir": str(d),
                      "report": str(d / PLAIN_RUN / "records.jsonl")})


def profile_check_plan(upload: Path, out: Path) -> list[Part]:
    """The profile check's parts (module docstring): the models part and the probe; SmolLM2's
    calibration, three ``run_verified`` runs, the plain run and the stored step's profile; then,
    per model of :data:`PROFILE_PINS`, a full-scale ``plain_baseline`` run and
    ``bring_up --no-tamper --steps 3 --metrics --check5-repeats 3``, both at
    :data:`FULL_SCALE_ENV` plus :data:`RECHECK_ENV`; the bring-up traces step 2, a steady-state
    step, and runs its repeats on step 3. Every part runs with the sampler (``POTS_SAMPLER`` is
    never set)."""
    t = PROFILE_CHECK_TIMEOUT_S
    plan = [_models_part(upload, out, t["models"]), _probe_part(out)]
    plan += _smollm2_parts(upload, out)
    ups = _bring_up_parts(upload, out, {**RECHECK_ENV, "VERIF_TRACE_STEP": "2"}, t["bring_up"],
                          ("--steps", "3", "--metrics", "--check5-repeats", "3"))
    for (repo, rev), up in zip(RUN_PINS, ups, strict=True):
        if repo not in PROFILE_HELD:
            plan += [_full_scale_plain_part(repo, rev, upload, out, t["plain"]), up]
    return plan


# FS31's hardware probe: the number of cards each model runs on, set by ``FS31_CARDS`` (the plan
# is built before the GPUs are inspected), and the time limit of the probe and of each card run.
CARDS_ENV = "FS31_CARDS"
DEFAULT_CARDS = 4
NVLINK_TIMEOUT_S = 900
CARD_TIMEOUT_S = 900
GATE_CARDS = 4  # F22's four cards; ``FS31_CARDS`` other than this is a rehearsal, not the gate
# Card numbers follow ``nvidia-smi``'s (PCI bus order), so ``CUDA_VISIBLE_DEVICES=k`` and
# ``nvidia-smi -i k`` name the same card.
PCI_ORDER_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def fs31_cards(environ: Mapping[str, str] | None = None) -> int:
    """``FS31_CARDS``, the cards the hardware probe runs each model on (default 4)."""
    raw = (os.environ if environ is None else environ).get(CARDS_ENV, "").strip()
    if not raw:
        return DEFAULT_CARDS
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError(f"{CARDS_ENV} is {raw!r}; it must be a whole number of cards, at least 1")
    return n


def nvlink_probe_plan(upload: Path, out: Path) -> list[Part]:
    """The hardware probe's parts: the models part; the NVLink probe, which sees every card; then,
    per model of :data:`RUN_PINS` in order and per card ``0 … n−1`` (:func:`fs31_cards`), one
    ``card_identity`` run at :data:`FULL_SCALE_ENV` plus :data:`RECHECK_ENV` on the refusal task,
    the card chosen by ``CUDA_VISIBLE_DEVICES`` and the run in its own directory
    ``runs/<model>_<task>_card<k>/``. All four models run, Qwen2.5-1.5B included."""
    n = fs31_cards()
    report = out / "nvlink" / "nvlink.json"
    plan = [peer_check_part(out),
            _models_part(upload, out, PROFILE_CHECK_TIMEOUT_S["models"]),
            Part("nvlink_probe", ["{python}", "-m", "verification.runs.nvlink_probe", "--out",
                                  str(report)],
                 env=dict(PCI_ORDER_ENV), timeout_s=NVLINK_TIMEOUT_S, kind="nvlink",
                 info={"report": str(report)})]
    for repo, rev in RUN_PINS:
        key = data_key(repo, BRING_UP_TASK)
        for card in range(n):
            d = _run_dir(out, upload, key, f"{key}_card{card}")
            plan.append(Part(
                f"card_{repo.split('/')[-1]}_{card}",
                ["{python}", "-m", "verification.runs.card_identity", "--out",
                 str(d / "card.json")],
                env={**FULL_SCALE_ENV, **RECHECK_ENV, **_run_env(repo, rev, BRING_UP_TASK),
                     **PCI_ORDER_ENV, "CUDA_VISIBLE_DEVICES": str(card),
                     "VERIF_OUTPUT_DIR": str(d)},
                timeout_s=CARD_TIMEOUT_S, kind="card",
                info={"model": repo, "task": BRING_UP_TASK, "card": card, "run_dir": str(d),
                      "report": str(d / "card.json")}))
    return plan


# FS29's kernel-speed session: the sizes file ``kernel_speed sizes`` wrote on the Mac, at the
# upload's root (verification.runs.kernel_speed.SIZES_NAME), and the time limits of its parts.
# The kernel tests take a few minutes (the NVRTC compile and the native references); the speed
# run hashes four models' steps and their bands, each table once to warm up and 5 times timed,
# and checks each against the C library. Limits: native build 900 s (it sits beside this table,
# in with_native_build), tests 900 s, speed run 1,500 s: 3,300 s in all, inside the job's work
# window under its 5,400 s deadline (5,400 less 1,200 s of grace and 600 s of packing is 3,600 s).
# Expected: build about 1 min, tests about 5 min, speed run about 5 min; the limits only stop a
# hung part and need not all be spent, so they need not all fit twice over.
SIZES_NAME = "kernel_speed_sizes.json"
KERNEL_SPEED_TIMEOUT_S = {"tests": 900, "run": 1500}
KERNEL_TESTS = ("tests/verification/commitment/test_gpu_blake3.py",
                "tests/verification/commitment/test_gpu_blake3_edges.py")


def kernel_speed_plan(upload: Path, out: Path) -> list[Part]:
    """The kernel-speed session's parts: the kernel's tests on this card
    (``test_gpu_blake3.py`` and ``test_gpu_blake3_edges.py``), then ``kernel_speed run`` on the
    upload's sizes file, report ``kernel_speed/kernel_speed.json``. No models part: the run
    hashes random contents of the real leaf sizes. The native build comes first
    (:func:`with_native_build`)."""
    t = KERNEL_SPEED_TIMEOUT_S
    report = out / "kernel_speed" / "kernel_speed.json"
    return [Part("kernel_tests", ["{python}", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                  *KERNEL_TESTS], timeout_s=t["tests"]),
            Part("kernel_speed", ["{python}", "-m", "verification.runs.kernel_speed", "run",
                                  "--sizes", str(upload / SIZES_NAME), "--out", str(report)],
                 env=dict(PCI_ORDER_ENV), timeout_s=t["run"], kind="kernel_speed",
                 info={"report": str(report)})]


# FS30's device-leaves session: arrival hashing in capture and the device kernel at the commit,
# checked on the H100 against the CPU path's roots. Time limits, seconds: the CUDA tests (the
# kernel's two files, FS30's own tests and the capture's memory test) 1,200; each bring-up
# 1,800 (the re-check's limit, :data:`RECHECK_TIMEOUT_S`); the models part 1,800; the native
# build 900 (:func:`with_native_build`). Estimates, minutes, scaled by hand from the parts' own
# earlier figures (nothing was measured): models 5, native build 2, CUDA tests 8. A bring-up
# here is one step with no repeats, but it hashes the step's leaves a second time on the CPU
# path. The figures: the first session's one finished bring-up took 191 s (Qwen2.5-0.5B's size,
# about 3 min), and Llama-3.2-1B's two-step bring-up with repeats took 1,152 s (about 19 min;
# a step is 200-300 s of it, so one step without repeats is taken as about 10 min), Falcon3-1B
# about the same as Llama, Qwen2.5-1.5B about 14. Four bring-ups about 37 (10 + 10 + 3 + 14).
# Parts about 52 min, packing about 1, the grace window 20: about 73 min. The job's deadline is
# twice that, 146 min, rounded up to the next 100 s: 8,800 s; the parts stop by
# 8,800 - 1,200 - 600 = 7,000 s. (The first version of this plan took the four bring-ups as
# about 30 min, which its own Llama figure, 19 min alone, did not support; its deadline was
# 8,000 s.)
DEVICE_LEAVES_TIMEOUT_S = {"models": 1800, "tests": 1200, "bring_up": 1800}
DEVICE_LEAVES_DEADLINE_S = 8800
DEVICE_TESTS_FILE = "tests/verification/runs/test_device_leaves_cuda.py"
DEVICE_TESTS_MEMORY = ("tests/verification/prover/test_capture_cuda.py"
                       "::test_gpu_memory_does_not_hold_the_transcript")
DEVICE_LEAVES_TESTS = (*KERNEL_TESTS, DEVICE_TESTS_FILE, DEVICE_TESTS_MEMORY)


def device_leaves_plan(upload: Path, out: Path) -> list[Part]:
    """FS30's parts (module docstring): the models part, the CUDA tests with a junit report,
    then one ``bring_up --no-tamper --device-leaves`` per model of :data:`RUN_PINS` at
    :data:`FULL_SCALE_ENV` plus :data:`RECHECK_ENV`, :data:`FIX_AB_MODEL` (the cheapest) first.
    The native build comes first (:func:`with_native_build`)."""
    t = DEVICE_LEAVES_TIMEOUT_S
    junit = out / "parts" / "device_tests" / "junit.xml"
    tests = Part("device_tests", ["{python}", "-m", "pytest", "-rf", "-q", f"--junitxml={junit}",
                                  "-p", "no:cacheprovider", *DEVICE_LEAVES_TESTS],
                 timeout_s=t["tests"], kind="device_tests", info={"junit": str(junit)})
    ups = _bring_up_parts(upload, out, RECHECK_ENV, t["bring_up"], args=("--device-leaves",))
    (first,) = [p for p in ups if p.info["model"] == FIX_AB_MODEL]
    return [_models_part(upload, out, t["models"]), tests, first,
            *[p for p in ups if p is not first]]


_DEVICE_LEAVES_LINE = re.compile(
    r"^device leaves: step (\d+) accepted (True|False), prover root ([0-9a-f]{64}|none), "
    r"cpu root ([0-9a-f]{64}), arrival digests (\d+), hashed (\d+), commit known (\d+), "
    r"product leaves (\d+), dK leaves (\d+)$")


def _device_leaves_logged(lines: Sequence[str]) -> dict[str, Any] | None:
    """The last ``device leaves:`` line of a bring-up's log, as numbers, or ``None``."""
    got = [m for m in map(_DEVICE_LEAVES_LINE.match, lines) if m]
    if not got:
        return None
    m = got[-1]
    return {"step": int(m[1]), "accepted": m[2] == "True",
            "prover_root": None if m[3] == "none" else m[3], "cpu_root": m[4],
            "arrival": int(m[5]), "hashed": int(m[6]), "known": int(m[7]), "K": int(m[8]),
            "dK": int(m[9])}


def _device_tests_extras(junit: Path) -> dict[str, Any]:
    """:func:`_pytest_extras` and every case's outcome (``passed``, ``failed``, ``error``,
    ``skipped``) by ``classname::name``: the session's tests are few, and the gate needs to know
    that each FS30 test ran."""
    import xml.etree.ElementTree as ET

    out = _pytest_extras(junit)
    if out.get("junit", 0) is None:
        return out
    cases = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        tags = {c.tag for c in case}
        cases[f"{case.get('classname')}::{case.get('name')}"] = (
            "error" if "error" in tags else "failed" if "failure" in tags
            else "skipped" if "skipped" in tags else "passed")
    return {**out, "cases": cases}


_REPO_ROOT = Path(__file__).resolve().parents[2]
_collected: dict[str, tuple[list[str], str]] = {}


def _junit_name(mod: str, node: str) -> str:
    """``Class::test[id]`` (a collected node below its file) as junit names it: ``mod.Class::
    test[id]``, class segments dotted on to the module (the ``[id]`` may hold ``::``)."""
    head, bracket, ident = node.partition("[")
    *classes, test = head.split("::")
    return ".".join([mod, *classes]) + "::" + test + bracket + ident


def device_tests_expected(file: str = DEVICE_TESTS_FILE) -> tuple[list[str], str]:
    """``(names, error)``: every test case ``file`` defines, parametrized cases each, as the
    junit report names it (``dotted.module::name[id]``, ``dotted.module.Class::name`` in a
    class), from ``pytest --collect-only`` run at the repository root; ``error`` is empty unless
    the collection failed, timed out, could not start or found none. Read at gate time, so a
    renamed, deleted or added test changes what the gate asks for. Cached, a failure too. The
    subprocess gets the session's environment without :data:`SECRETS` (the gate has no upload
    directory for :func:`base_env`), offline, and no stdin."""
    if file not in _collected:
        env = {k: v for k, v in os.environ.items() if k not in SECRETS}
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
               file]
        mod = file.removesuffix(".py").replace("/", ".")
        try:
            r = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                               capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            _collected[file] = ([], f"pytest --collect-only {file} timed out after 300 s")
        except OSError as e:
            _collected[file] = ([], f"pytest --collect-only {file} could not start: {e}")
        else:
            names = [_junit_name(mod, ln[len(file) + 2:]) for ln in r.stdout.splitlines()
                     if ln.startswith(file + "::")]
            err = "" if r.returncode == 0 and names else (
                f"pytest --collect-only {file} exited {r.returncode} and found "
                f"{len(names)} tests")
            _collected[file] = (names, err)
    return _collected[file]


def device_leaves_gate(parts: Sequence[Mapping[str, Any]],
                       expected: Sequence[str] | None = None) -> dict[str, Any] | None:
    """FS30's gate from the parts' records, or ``None`` when the session has no tests part.

    It is ``pass`` only if the CUDA tests part finished ``ok`` with no failure or error, and
    FS30's own CUDA tests all ran and passed (none skipped). Each case the file defines, found
    by :func:`device_tests_expected` or given as ``expected``, is asked for by name, and a
    missing one is a reason of its own; an extra case in the file's module that passed is let
    through. Every model of :data:`RUN_PINS` must also have a bring-up that logged its line and
    shows the step accepted, the two roots equal, and the arrival digests in use and the leaves
    the commit took from them (``known``) each equal to ``K`` less the ``dK`` leaves ``D``, and the
    outputs hashed on arrival equal to ``K`` (the ``δK̃ᵀ`` outputs are hashed too, their digests
    unused). Anything missing fails.
    """
    tests = next((p for p in parts if p.get("kind") == "device_tests"), None)
    if tests is None:
        return None
    why: list[str] = []
    ex = tests.get("extras") or {}
    if tests.get("status") not in ("ok", "skipped_done"):
        why.append(f"the CUDA tests part is {tests.get('status')}")
    if ex.get("junit", 0) is None or "cases" not in ex:
        why.append("the CUDA tests left no junit report")
    else:
        if ex.get("failures") or ex.get("errors"):
            why.append(f"the CUDA tests have {ex.get('failures')} failures and "
                       f"{ex.get('errors')} errors")
        mine = {n: o for n, o in ex["cases"].items()
                if "test_device_leaves_cuda" in n
                or n.endswith("::test_gpu_memory_does_not_hold_the_transcript")}
        if not any("test_device_leaves_cuda" in n for n in mine):
            why.append("none of FS30's CUDA tests ran")
        if not any(n.endswith("test_gpu_memory_does_not_hold_the_transcript") for n in mine):
            why.append("the capture's GPU-memory test did not run")
        why += [f"{n}: {o}" for n, o in sorted(mine.items()) if o != "passed"]
        if expected is None:
            expected, err = device_tests_expected()
            if err:
                why.append(f"FS30's expected tests could not be collected: {err}")
        why += [f"{n}: expected but not in the junit report"
                for n in expected if n not in ex["cases"]]
    by_model = {(p.get("info") or {}).get("model"): p for p in parts if p.get("kind") == "bring_up"}
    models: dict[str, Any] = {}
    for repo, _rev in RUN_PINS:
        p = by_model.get(repo)
        got = None if p is None else (p.get("extras") or {}).get("device_leaves")
        if p is None:
            row = {"ok": False, "why": "no bring-up part"}
        elif got is None:
            row = {"ok": False, "why": f"no device leaves line (the part is {p.get('status')})"}
        else:
            bad = []
            if not got["accepted"]:
                bad.append("the step was not accepted")
            if got["prover_root"] is None or got["prover_root"] != got["cpu_root"]:
                bad.append("the roots differ")
            if got["arrival"] != got["K"] - got["dK"]:
                bad.append(f"{got['arrival']} arrival digests, expected "
                           f"{got['K'] - got['dK']} (K {got['K']} less {got['dK']} dK leaves)")
            if got["known"] != got["K"] - got["dK"]:
                bad.append(f"the commit took {got['known']} leaves from arrival digests, "
                           f"expected {got['K'] - got['dK']} (K {got['K']} less {got['dK']} dK "
                           f"leaves)")
            if got["hashed"] != got["K"]:
                bad.append(f"{got['hashed']} outputs hashed on arrival, expected {got['K']}")
            row = {**got, "ok": not bad, "why": "; ".join(bad)}
        models[repo] = row
        if not row["ok"]:
            why.append(f"{repo}: {row['why']}")
    ok = not why
    line = ("FS30 device leaves pass: every FS30 CUDA test passed, and for each of "
            f"{len(models)} models the roots agree with the arrival digests in use" if ok else
            "FS30 device leaves FAIL: " + "; ".join(why))
    return {"pass": ok, "reasons": why, "models": models, "line": line}


# FS32f's two-GPU session (``fs32-pair``): the sharded run on two cards, F22's two-A40 check.
# Time limits, seconds, of its parts; the fast suite's limit allows twice its estimated 25 min.
PAIR_TIMEOUT_S = {"models": 1800, "fast_suite": 3600, "two_gpu_slow": 1800, "calibrate": 900,
                  "card": 900, "run": 900, "run_metrics": 1800, "cheat": 900,
                  "fs34_tests": 900}
# Estimates, minutes, scaled by hand (none was measured on two cards): the peer-copy byte check
# 1 (two ordered pairs; its limit is 5), the models part 2-5, the
# native build 1-2, the fast suite 13-25 on 10 vCPUs (the CUDA tests run on it for the first
# time), the slow two-GPU tests about 10, SmolLM2's calibration about 5, each of two
# card_identity runs about 3, each of two 3-step runs about 6, each of three cheat runs about 4.
# FS33d's pipelined arms are estimated at their sequential counterparts' time (the pipelined loop
# overlaps prover and verifier, so it is no slower; an estimate, none was measured): the plain
# pipelined run about 6, the one with metrics on about 12 (its memory pass and counting pass add
# two more passes), the three pipelined cheats about 12 together. FS34's two-GPU join tests
# (``fs34_tests``: a tiny sharded step, a run_loop and the kernel's first build on each card)
# about 5, an estimate.
PAIR_ESTIMATE_MIN = {"peer_check": 1, "models": 5, "native_build": 2, "fast_suite": 25,
                     "two_gpu_slow": 10, "fs34_tests": 5, "calibrate": 5, "card": 6, "run": 12,
                     "cheat": 12, "run_pipelined": 6, "run_pipelined_metrics": 12,
                     "cheat_pipelined": 12}
PAIR_PACK_MIN, PAIR_GRACE_MIN = 1, 20  # the job's packing and grace window, minutes
PAIR_STEPS = 3
PAIR_CARDS = 2  # the session refuses a machine with fewer CUDA devices
PAIR_DEVICES = "cuda:0,cuda:1"  # VERIF_VERIFIER_DEVICES of the sharded arms
PAIR_CHEATS = ("A1", "A2", "A3", "hidden", "flip")  # the oracle table's names for the cheats
# The pipelined arms (FS33d): the same runs and cheats with ``VERIF_PIPELINED=1``, sharded.
PAIR_PIPELINED = {"VERIF_VERIFIER_DEVICES": PAIR_DEVICES, "VERIF_PIPELINED": "1"}
PAIR_CHEAT_PARTS = ("poisoned_step", "hidden_steps", "flipped_matmul")
PAIR_PIPELINED_CHEAT_PARTS = tuple(f"{n}_pipelined" for n in PAIR_CHEAT_PARTS)
PAIR_MODE_LINE = "loop mode: "  # the entry points print ``loop mode: pipelined|sequential``


def pair_default_deadline_s() -> int:
    """The session's default deadline: twice the estimated runtime (the parts' estimates, the
    job's packing and its grace window), in seconds, rounded up to 100 s. ``fs11_job.sh`` holds
    the same figure as ``DEFAULT_DEADLINE_S`` for ``fs32-pair`` and a test holds them equal."""
    minutes = sum(PAIR_ESTIMATE_MIN.values()) + PAIR_PACK_MIN + PAIR_GRACE_MIN
    return -(-2 * minutes * 60 // 100) * 100
# scenarios.OUTCOMES (this module imports no torch; a test holds the two equal).
ORACLE_OUTCOMES = ("exact", "accepted", "rejected", "early", "overrun", "wrong-check",
                   "wrong-kind", "wrong-product")
# the slow two-GPU tests: the FS32f ones (two_gpus) and FS32c's SmolLM2 replay (two_cuda_devices)
PAIR_SLOW_SELECT = "two_gpus or two_cuda_devices"


FS34_TESTS_FILE = "tests/verification/runs/test_fs34_join_cuda.py"


def _fs34_tests_part(out: Path, timeout_s: float, output_dir: Path) -> Part:
    """FS34's two-GPU join tests (:data:`FS34_TESTS_FILE`) with a junit report, by name: the
    comparison asks for each case the file defines and that none was skipped."""
    junit = out / "parts" / "fs34_tests" / "junit.xml"
    return Part("fs34_tests", ["{python}", "-m", "pytest", "-rf", "-q", f"--junitxml={junit}",
                               "-p", "no:cacheprovider", FS34_TESTS_FILE],
                env={"VERIF_OUTPUT_DIR": str(output_dir), "VERIF_VERIFIER_DEVICES": None,
                     "VERIF_PIPELINED": None},
                timeout_s=timeout_s, kind="fs34_tests",
                info={"junit": str(junit), "gpu_peak": True})


def fs34_gate(parts: Sequence[Mapping[str, Any]],
              expected: Sequence[str] | None = None) -> list[str]:
    """The reasons FS34's CUDA tests do not count, empty when they do: the part ran ``ok``, its
    junit report names every case :data:`FS34_TESTS_FILE` defines (found by
    :func:`device_tests_expected`, or given as ``expected``) and each one passed, none skipped,
    failed or errored. Fails closed on anything missing."""
    part = next((p for p in parts if p.get("name") == "fs34_tests"), {})
    ex = part.get("extras") or {}
    why: list[str] = []
    if part.get("status") not in _PAIR_OK:
        why.append(f"the FS34 tests part is {part.get('status', 'not_started')}")
    if ex.get("junit", 0) is None or "cases" not in ex:
        why.append("the FS34 tests left no junit report")
        return why
    if ex.get("failures") or ex.get("errors"):
        why.append(f"the FS34 tests have {ex.get('failures')} failures and "
                   f"{ex.get('errors')} errors")
    if expected is None:
        expected, err = device_tests_expected(FS34_TESTS_FILE)
        if err:
            why.append(f"FS34's expected tests could not be collected: {err}")
    why += [f"{n}: " + ("expected but not in the junit report" if n not in ex["cases"]
                         else ex["cases"][n]) for n in expected
            if ex["cases"].get(n) != "passed"]
    why += [f"{n}: {o}" for n, o in sorted(ex["cases"].items())
            if o != "passed" and n not in expected]
    return why


def fs32_pair_plan(upload: Path, out: Path) -> list[Part]:
    """The two-GPU session's parts: the F22 peer-copy byte check first (:func:`peer_check_part`;
    wrong bytes stop the session); the models part; the whole fast suite (``-rf``, junit XML);
    the slow two-GPU tests; SmolLM2's calibration (unsharded, writing the band file, even when
    the bf16 guard fails); ``card_identity`` (step 1's transcript root) and a 3-step
    ``run_verified`` unsharded and again with ``VERIF_VERIFIER_DEVICES``; and the judged cheats
    (``poisoned_step`` A1-A3, ``hidden_steps`` P11, ``flipped_matmul_sweep --no-sweep``) with it
    set. FS33d's pipelined arms follow (sharded, ``VERIF_PIPELINED=1``, each in its own run
    directory, ``peer_check`` still first): ``run_pipelined`` (``run_verified``, the same steps and
    kept step), ``run_pipelined_metrics`` (the same with metrics on, so the merged rows and the
    memory pass's per-device peaks are exercised), and the three cheats again as
    ``<cheat>_pipelined``. Only those set ``VERIF_PIPELINED``. Every part records its GPUs' peak
    memory. SmolLM2 runs at :data:`SMOLLM2_ENV`."""
    from setup.data import TEST_SCALE_TOKENIZER as pin  # the pin's repo and revision
    key = data_key(pin.repo, SMOLLM2_TASK)
    base = {**SMOLLM2_ENV, "VERIF_MODEL": pin.repo, "VERIF_MODEL_REVISION": pin.revision}
    sharded = {"VERIF_VERIFIER_DEVICES": PAIR_DEVICES}
    home = _run_dir(out, upload, key, "smollm2")
    t = PAIR_TIMEOUT_S

    def run_dir(name: str) -> Path:
        d = _run_dir(out, upload, key, name)
        bands = d / "bands.json"
        if not bands.is_symlink() and not bands.exists():
            bands.symlink_to(os.path.relpath(home / "bands.json", d))
        return d

    def part(name: str, module: str, d: Path, timeout_s: float, *args: str, kind: str,
             env: Mapping[str, str] | None = None, **info: Any) -> Part:
        return Part(name, ["{python}", "-m", f"verification.runs.{module}", *args],
                    env={**base, **(env or {}), "VERIF_OUTPUT_DIR": str(d),
                         "VERIF_VERIFIER_DEVICES": (env or {}).get("VERIF_VERIFIER_DEVICES"),
                         "VERIF_PIPELINED": (env or {}).get("VERIF_PIPELINED")},
                    timeout_s=timeout_s, kind=kind,
                    info={"model": pin.repo, "task": SMOLLM2_TASK, "run_dir": str(d),
                          "gpu_peak": True, **info})

    def pytest_part(name: str, timeout_s: float, *args: str) -> Part:
        junit = out / "parts" / name / "junit.xml"
        return Part(name, ["{python}", "-m", "pytest", "-rf", "-q", f"--junitxml={junit}", *args],
                    env={"VERIF_OUTPUT_DIR": str(_run_dir(out, upload, key, "pytest")),
                         "VERIF_VERIFIER_DEVICES": None, "VERIF_PIPELINED": None},
                    timeout_s=timeout_s, kind="pytest",
                    info={"junit": str(junit), "gpu_peak": True})

    honest = f"honest-{pin.repo.split('/')[-1]}-{SMOLLM2_TASK}-bpr0-s{SMOLLM2_SEED}"
    steps = ("--steps", str(PAIR_STEPS), "--keep-step", str(PAIR_STEPS), "--no-metrics")
    plan = [peer_check_part(out), _models_part(upload, out, t["models"]),
            pytest_part("fast_suite", t["fast_suite"]),
            pytest_part("two_gpu_slow", t["two_gpu_slow"], "-m", "slow", "-k", PAIR_SLOW_SELECT),
            _fs34_tests_part(out, t["fs34_tests"], _run_dir(out, upload, key, "pytest")),
            part("smollm2_calibrate", "calibrate", home, t["calibrate"],
                 "--write-bands-despite-guard", kind="other")]
    for arm, env in (("unsharded", {}), ("sharded", sharded)):
        d = run_dir(f"pair_card_{arm}")
        plan.append(part(f"card_{arm}", "card_identity", d, t["card"], "--out",
                         str(d / "card.json"), kind="pair_card", env=env, arm=arm,
                         report=str(d / "card.json")))
    for arm, env in (("unsharded", {}), ("sharded", sharded)):
        d = run_dir(f"pair_run_{arm}")
        plan.append(part(f"run_{arm}", "run_verified", d, t["run"], *steps, kind="pair_run",
                         env=env, arm=arm, weights=str(d / honest / WEIGHTS_FILE)))
    for name, metrics in (("run_pipelined", "--no-metrics"),
                          ("run_pipelined_metrics", "--metrics")):
        d = run_dir(f"pair_{name}")
        plan.append(part(name, "run_verified", d, t["run" if metrics == "--no-metrics"
                                                   else "run_metrics"],
                         *steps[:-1], metrics, kind="pair_run", env=PAIR_PIPELINED,
                         arm=name.removeprefix("run_"), weights=str(d / honest / WEIGHTS_FILE)))
    cheats = (("poisoned_step", "poisoned_step", ()), ("hidden_steps", "hidden_steps", ()),
              ("flipped_matmul", "flipped_matmul_sweep", ("--no-sweep",)))
    for name, module, args in cheats:
        d = run_dir(f"pair_{name}")
        plan.append(part(name, module, d, t["cheat"], *args, "--no-metrics", kind="cheat",
                         env=sharded))
    for name, module, args in cheats:
        d = run_dir(f"pair_{name}_pipelined")
        plan.append(part(f"{name}_pipelined", module, d, t["cheat"], *args, "--no-metrics",
                         kind="cheat", env=PAIR_PIPELINED, arm="pipelined"))
    for p in plan:
        p.info["gpu_peak"] = True
    return plan


# FS35's four-H100 session (``four-h100``): F22's machine, GPU 0 trains and GPUs 1-3 hold, hash
# and verify the transcript, sequential and pipelined, one verified step per model.
FOUR_PROVER = "cuda:0"
FOUR_VERIFIERS = "cuda:1,cuda:2,cuda:3"  # VERIF_VERIFIER_DEVICES of the sharded arms
FOUR_CARDS = 4  # the session refuses a machine with fewer CUDA devices
FOUR_STEPS = 3  # step 1 pays one-off costs; steady: steps 2-3 (sequential), step 2 (pipelined)
# Cheapest first (the order the models' parts run in), by the short name the parts carry.
FOUR_ORDER = ("Qwen2.5-0.5B", "Llama-3.2-1B", "Falcon3-1B", "Qwen2.5-1.5B")
# Test files the session runs beside FS34's: the per-device tools' CUDA- and NVML-only tests
# (the check-5 profile's CUDA file is left out until it exists). A missing file is named in the
# summary.
FOUR_TEST_FILES = ("tests/verification/runs/test_probe_cuda.py",
                   "tests/setup/test_machine_cuda.py", "tests/setup/test_sampler_gpus.py")
FOUR_RUN_NAME = "llama_step"  # the metrics directory of ``llama_step`` under VERIF_OUTPUT_DIR
# Estimates, minutes, none measured on four GPUs: the native build 2, the peer check 1, the
# models part 6, the tests 8, the probe 8 (60 s per device and the 12 pairs), each card part 3,
# each sequential and each pipelined run 8 (three steps, no counting pass, one memory pass).
FOUR_ESTIMATE_MIN = {"native_build": 2, "peer_check": 1, "models": 6, "four_tests": 8,
                     "probe": 8, "card": 3, "seq": 8, "pipe": 8}
# Time limits, seconds, about twice the estimates (the native build's is the build part's own).
FOUR_TIMEOUT_S = {"models": 720, "four_tests": 960, "probe": 960, "card": 360, "seq": 960,
                  "pipe": 960}
FOUR_PACK_MIN, FOUR_GRACE_MIN = 1, 20  # the job's packing and grace window, minutes


def four_model_key(repo: str) -> str:
    """The short name a model's parts carry: ``Qwen/Qwen2.5-0.5B-Instruct`` is ``Qwen2.5-0.5B``."""
    return repo.split("/")[-1].removesuffix("-Instruct")


def four_pins(pins: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """``pins`` with :data:`FOUR_ORDER`'s models first, in its order, and any other model after
    them in the order given (an unlisted model is not an error: it must not break every
    session at import)."""
    def rank(p: tuple[str, str]) -> int:
        k = four_model_key(p[0])
        return FOUR_ORDER.index(k) if k in FOUR_ORDER else len(FOUR_ORDER)

    return tuple(sorted(pins, key=rank))


FOUR_PINS: tuple[tuple[str, str], ...] = four_pins(RUN_PINS)


def four_estimated_min() -> int:
    """The plan's estimated runtime, minutes: every part's estimate, the models' parts once per
    model."""
    e = FOUR_ESTIMATE_MIN
    return (e["native_build"] + e["peer_check"] + e["models"] + e["four_tests"] + e["probe"]
            + len(FOUR_PINS) * (e["card"] + e["seq"] + e["pipe"]))


def four_default_deadline_s() -> int:
    """The session's default deadline: twice the estimated runtime (the parts' estimates, the
    job's packing and its grace window), in seconds, rounded up to 100 s. ``fs11_job.sh`` holds
    the same figure as ``DEFAULT_DEADLINE_S`` for ``four-h100`` and a test holds them equal."""
    minutes = four_estimated_min() + FOUR_PACK_MIN + FOUR_GRACE_MIN
    return -(-2 * minutes * 60 // 100) * 100


# FS42 (d)'s profiling run (``four-h100-profile``): the same machine and head as ``four-h100``,
# three models (F24, 2026-10-25, dropped Qwen2.5-1.5B from the full-scale runs: it runs out of
# memory pipelined) and, per model, a plain training run (three steps), the card and the two
# arms at four steps, the sequential arm tracing its step 4. Step 1 is the warm-up, steps 2-3 are
# the steady steps (untraced in both arms; the pipelined arm's step 4 is its drain), the traced
# step is the last.
FOUR_PROFILE_ORDER = ("Qwen2.5-0.5B", "Llama-3.2-1B", "Falcon3-1B")
FOUR_PROFILE_STEPS = 4
FOUR_PROFILE_TRACE_STEP = 4  # VERIF_TRACE_STEP of the sequential arms
FOUR_PROFILE_PLAIN_STEPS = 3  # the plain run: step 1 the warm-up, steps 2-3 the steady steps
FOUR_PROFILE_PINS: tuple[tuple[str, str], ...] = tuple(sorted(
    (p for p in RUN_PINS if four_model_key(p[0]) in FOUR_PROFILE_ORDER),
    key=lambda p: FOUR_PROFILE_ORDER.index(four_model_key(p[0]))))
# Estimates, minutes, from the parts FS35's four-H100 session measured on 2026-10-25 (four H100
# SXM): the native build 3 s, the peer check 12 s, the models 39 s, the four-GPU tests 246 s, the
# probe 334 s, per model the card 37-62 s and the sequential arm 91-156 s, the pipelined arm
# 109-178 s (three steps each, the memory pass included). Here the pipelined arm runs four steps
# (a step is about a quarter of those times, plus the memory pass and the start-up), so 4
# minutes; the models part is FS35's estimate, 6, and limit, 720 s (it fetches the whole
# manifest, and a slow download fails the part); the sequential arm adds the trace of one
# step (a full-scale step's trace is a JSON of about 1-1.5 GB, gzipped at level 1: an estimate,
# adding 2-3 minutes to a step of about 195 s), so 6 minutes and 720 s; the plain run (three steps
# and a memory pass on one GPU) is estimated, not measured, at 3.
FOUR_PROFILE_ESTIMATE_MIN = {"native_build": 1, "peer_check": 1, "models": 6, "four_tests": 6,
                             "probe": 7, "plain": 3, "card": 2, "seq": 6, "pipe": 4}
# Time limits, seconds, about twice the estimates (the native build's is the build part's own).
FOUR_PROFILE_TIMEOUT_S = {"models": 720, "four_tests": 720, "probe": 840, "plain": 360,
                          "card": 240, "seq": 720, "pipe": 480}


def four_profile_estimated_min() -> int:
    """``four-h100-profile``'s estimated runtime, minutes: every part's estimate, the models'
    parts once per model."""
    e = FOUR_PROFILE_ESTIMATE_MIN
    return (e["native_build"] + e["peer_check"] + e["models"] + e["four_tests"] + e["probe"]
            + len(FOUR_PROFILE_PINS) * (e["plain"] + e["card"] + e["seq"] + e["pipe"]))


def four_profile_default_deadline_s() -> int:
    """``four-h100-profile``'s default deadline: twice the estimated runtime (the parts'
    estimates, the job's packing and its grace window), in seconds, rounded up to 100 s.
    ``fs11_job.sh`` holds the same figure as ``DEFAULT_DEADLINE_S`` for the session and a test
    holds them equal."""
    minutes = four_profile_estimated_min() + FOUR_PACK_MIN + FOUR_GRACE_MIN
    return -(-2 * minutes * 60 // 100) * 100


def _four_plan(upload: Path, out: Path, *, pins: Sequence[tuple[str, str]],
               t: Mapping[str, int], steps: int, profile: bool) -> list[Part]:
    """The parts of the four-H100 sessions: ``pins``' models, ``steps`` verified steps per arm,
    time limits ``t``; with ``profile`` (``four-h100-profile``, :func:`four_h100_profile_plan`)
    a ``plain_<key>`` run (``plain_baseline`` on GPU 0 alone, :data:`FOUR_PROFILE_PLAIN_STEPS`
    steps, metrics) before each model's card, the sequential arm traced at
    :data:`FOUR_PROFILE_TRACE_STEP` and a ``samples.jsonl`` check on every part. Otherwise
    :func:`four_h100_plan`'s.
    The F22 peer-copy byte check first (the native build runs ahead of it); the models part;
    ``four_tests``, one pytest part with a junit report running
    every two-GPU test, slow or not (:data:`PAIR_SLOW_SELECT`: the non-slow ones put the
    verifiers on GPUs 1-3 here), FS34's file and :data:`FOUR_TEST_FILES`,
    on SmolLM2's test-scale data (its run
    directory's ``data/`` link, as the pair plan's pytest parts have it);
    the probe on all four GPUs (``VERIF_DEVICE`` GPU 0, :data:`FOUR_VERIFIERS` the rest, 60 s per
    device); then, per model cheapest first, on the refusal task at :data:`FULL_SCALE_ENV` plus
    :data:`RECHECK_ENV`: ``card_<key>`` (``card_identity`` on GPU 0 alone, one card's proved
    step-1 root), ``seq_<key>`` (``llama_step``, ``steps`` steps, metrics, no counting pass, sharded
    over GPUs 1-3) and ``pipe_<key>`` (the same with ``VERIF_PIPELINED=1``), each in its own run
    directory. Only the ``pipe_*`` parts set ``VERIF_PIPELINED``. Every part records its GPUs'
    peak memory."""
    from setup.data import TEST_SCALE_TOKENIZER as pin  # the tests' model: SmolLM2
    small = data_key(pin.repo, SMOLLM2_TASK)
    sharded = {"VERIF_DEVICE": FOUR_PROVER, "VERIF_VERIFIER_DEVICES": FOUR_VERIFIERS}
    junit = out / "parts" / "four_tests" / "junit.xml"
    expr = " or ".join([f"({PAIR_SLOW_SELECT})",
                        *(Path(f).stem for f in (FS34_TESTS_FILE, *FOUR_TEST_FILES))])
    tests = Part("four_tests", ["{python}", "-m", "pytest", "-rf", "-q", f"--junitxml={junit}",
                                "-p", "no:cacheprovider", "-m", "slow or not slow", "-k", expr],
                 env={**PCI_ORDER_ENV,
                      "VERIF_OUTPUT_DIR": str(_run_dir(out, upload, small, "four_pytest")),
                      "VERIF_VERIFIER_DEVICES": None, "VERIF_PIPELINED": None},
                 timeout_s=t["four_tests"], kind="four_tests", info={"junit": str(junit)})
    report = out / "probe" / "probe.json"
    probe = Part("probe", ["{python}", "-m", "verification.runs.probe", "--out", str(report),
                           "--seconds", "60"],
                 env={**FULL_SCALE_ENV, **PCI_ORDER_ENV, **sharded, "VERIF_PIPELINED": None},
                 timeout_s=t["probe"], kind="four_probe", info={"report": str(report)})
    plan = [peer_check_part(out), _models_part(upload, out, t["models"]), tests, probe]
    for repo, rev in pins:
        k, key = four_model_key(repo), data_key(repo, BRING_UP_TASK)
        base = {**FULL_SCALE_ENV, **RECHECK_ENV, **_run_env(repo, rev, BRING_UP_TASK),
                **PCI_ORDER_ENV}
        info = {"model": repo, "task": BRING_UP_TASK}
        if profile:  # the plain training the verifier's cost is compared with, on GPU 0 alone
            d = _run_dir(out, upload, key, f"four_{key}_plain")
            plan.append(Part(
                f"plain_{k}", ["{python}", "-m", "verification.runs.plain_baseline", "--steps",
                               str(FOUR_PROFILE_PLAIN_STEPS), "--metrics"],
                env={**base, "VERIF_DEVICE": FOUR_PROVER, "VERIF_OUTPUT_DIR": str(d),
                     "VERIF_VERIFIER_DEVICES": None, "VERIF_PIPELINED": None},
                timeout_s=t["plain"], kind="profile_plain",
                info={**info, "run_dir": str(d), "records": str(d / PLAIN_RUN / "records.jsonl")}))
        d = _run_dir(out, upload, key, f"four_{key}_card")
        plan.append(Part(f"card_{k}", ["{python}", "-m", "verification.runs.card_identity",
                                       "--out", str(d / "card.json")],
                         env={**base, "CUDA_VISIBLE_DEVICES": "0", "VERIF_OUTPUT_DIR": str(d),
                              "VERIF_VERIFIER_DEVICES": None, "VERIF_PIPELINED": None},
                         timeout_s=t["card"], kind="four_card",
                         info={**info, "run_dir": str(d), "report": str(d / "card.json")}))
        for arm, piped in (("seq", None), ("pipe", "1")):  # only the pipelined arm sets the mode
            d = _run_dir(out, upload, key, f"four_{key}_{arm}")
            env = {**base, **sharded, "VERIF_OUTPUT_DIR": str(d), "VERIF_PIPELINED": piped}
            if profile:  # only the sequential arm is traced, at its last step, check 5 whole
                env["VERIF_TRACE_STEP"] = str(FOUR_PROFILE_TRACE_STEP) if arm == "seq" else None
                env["VERIF_TRACE_C5"] = "all" if arm == "seq" else None
            plan.append(Part(
                f"{arm}_{k}", ["{python}", "-m", "verification.runs.llama_step", "--steps",
                               str(steps), "--metrics", "--no-count"],
                env=env, timeout_s=t[arm], kind="profile_run" if profile else "four_run",
                info={**info, "arm": arm, "model_key": k, "run_dir": str(d),
                      "records": str(d / FOUR_RUN_NAME / "records.jsonl")}))
    for p in plan:
        p.info["gpu_peak"] = True
        if profile:
            p.info["samples_check"] = True
    return plan


def four_h100_plan(upload: Path, out: Path) -> list[Part]:
    """The four-H100 session's parts (:func:`_four_plan`): three verified steps per arm,
    :data:`FOUR_ORDER`'s models."""
    return _four_plan(upload, out, pins=FOUR_PINS, t=FOUR_TIMEOUT_S, steps=FOUR_STEPS,
                      profile=False)


def four_h100_profile_plan(upload: Path, out: Path) -> list[Part]:
    """The ``four-h100-profile`` session's parts (:func:`_four_plan`): the same head, then per
    model of :data:`FOUR_PROFILE_ORDER` a plain run, the card, and the two arms at
    :data:`FOUR_PROFILE_STEPS` steps, the sequential one tracing step
    :data:`FOUR_PROFILE_TRACE_STEP`."""
    return _four_plan(upload, out, pins=FOUR_PROFILE_PINS, t=FOUR_PROFILE_TIMEOUT_S,
                      steps=FOUR_PROFILE_STEPS, profile=True)


# Each session: its name in summary.json, and its plan.
SESSIONS: dict[str, tuple[str, Callable[[Path, Path], list[Part]]]] = {
    "fs11": ("FS11", default_plan),
    "recheck": ("recheck", recheck_plan),
    "profile-check": ("profile-check", profile_check_plan),
    "nvlink-probe": ("nvlink-probe", nvlink_probe_plan),
    "kernel-speed": ("kernel-speed", kernel_speed_plan),
    "fs32-pair": ("fs32-pair", fs32_pair_plan),
    "device-leaves": ("device-leaves", device_leaves_plan),
    "four-h100": ("four-h100", four_h100_plan),
    "four-h100-profile": ("four-h100-profile", four_h100_profile_plan),
}
# The sessions that need two or more GPUs: each plan starts (after the native build) with
# ``peer_check_part``. A new multi-GPU session is listed here, and a test holds it to that.
MULTI_GPU_SESSIONS = ("nvlink-probe", "fs32-pair", "four-h100", "four-h100-profile")


def _run_env(repo: str, rev: str, task: str) -> dict[str, str | None]:
    return {**FULL_SCALE_ENV, "VERIF_MODEL": repo, "VERIF_MODEL_REVISION": rev,
            "VERIF_ATTACK": task, "VERIF_N_RECORDS": str(TASKS[task])}


def load_plan(path: Path) -> list[Part]:
    """A plan from JSON, a list of :class:`Part` fields: for a dry run with stand-in parts."""
    return [Part(**p) for p in json.loads(path.read_text())]


def selftest_plan() -> list[Part]:
    """``FS11_SELFTEST``'s stand-in parts: standard library only, a few seconds in all, so a
    cheap pod without the venv runs the job's whole path up to the real terminate."""
    return [Part("selftest_ok", ["{python}", "-c", "print('self-test part ran')"], timeout_s=60),
            Part("selftest_sleep", ["{python}", "-c", "import time; time.sleep(5)"],
                 timeout_s=60)]


NATIVE_BUILD_NAME = "native_build"


def _native_build_part() -> Part:
    """A fresh build part, so no two plans share its env and info dicts."""
    return Part(NATIVE_BUILD_NAME, ["{python}", "-m", "verification.commitment.native"],
                env={"VERIF_PIPELINED": None}, timeout_s=900)


def with_native_build(plan: Sequence[Part]) -> list[Part]:
    """``plan`` with the native library's build first, so no part's timing includes the compile
    (the sources are in the upload). A plan that already starts with it is unchanged."""
    plan = list(plan)
    return plan if plan and plan[0].name == NATIVE_BUILD_NAME else [_native_build_part(), *plan]


def choose_plan(build: Callable[[Path, Path], list[Part]], upload: Path, out: Path, *,
                plan_file: Path | None, selftest: bool) -> list[Part]:
    """The session's parts: a ``--plan`` file's, else the stand-ins (``selftest``) or
    ``build``'s. All but the stand-ins (a bare pod, no venv) start with the native build."""
    if selftest and not plan_file:
        return selftest_plan()
    plan = load_plan(plan_file) if plan_file else build(upload, out)
    return plan if selftest else with_native_build(plan)


# ---- CPU placement (O13) -------------------------------------------------------------------
# The session pins itself before any part starts, and every part (a child process) inherits
# the pin. A part's threads then run on the CPUs of the GPU's NUMA node, so the pinned host
# buffers, the hashing threads and the GPU's DMA share one node, and the thread count stays
# inside the container's CPU quota (a pod shows 160 CPUs and may be allowed 17). Everything
# here is stdlib-only and does nothing, with one logged line, where the files or the calls
# are missing (macOS, a pod without sysfs).

SYSFS_PCI = "/sys/bus/pci/devices"
CGROUP_CPU_MAX = "/sys/fs/cgroup/cpu.max"


def parse_cpulist(text: str) -> set[int]:
    """The CPU numbers of a kernel cpulist such as ``"40-79,120-159"`` or ``"3"``. An empty
    list gives the empty set; a malformed one raises ``ValueError``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        lo, dash, hi = part.partition("-")
        a, b = int(lo), int(hi) if dash else int(lo)
        if a < 0 or b < a:
            raise ValueError(f"bad cpulist range {part!r}")
        cpus.update(range(a, b + 1))
    return cpus


def format_cpulist(cpus: Sequence[int] | set[int]) -> str:
    """The inverse of :func:`parse_cpulist`: ``{1, 2, 3, 7}`` as ``"1-3,7"``."""
    out: list[str] = []
    xs = sorted(cpus)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def normalize_pci_bus_id(bus_id: str) -> str:
    """``nvidia-smi``'s ``"00000000:3B:00.0"`` (eight-digit domain, capitals) as sysfs names the
    device: ``"0000:3b:00.0"``. A bus id without a domain gets domain ``0000``."""
    parts = bus_id.strip().lower().split(":")
    if len(parts) == 2:
        parts.insert(0, "0")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"not a PCI bus id: {bus_id!r}")
    return f"{parts[0][-4:].zfill(4)}:{parts[1]}:{parts[2]}"


def parse_cgroup_cpu_max(text: str) -> float | None:
    """CPUs a cgroup v2 ``cpu.max`` allows: ``"1700000 100000"`` is 17.0, ``"max 100000"`` is
    ``None`` (no limit). A malformed file raises ``ValueError``."""
    fields = text.split()
    if len(fields) != 2:
        raise ValueError(f"cpu.max has {len(fields)} fields, expected 2")
    if fields[0] == "max":
        return None
    quota, period = int(fields[0]), int(fields[1])
    if quota <= 0 or period <= 0:
        raise ValueError(f"cpu.max {text.strip()!r}: quota and period must be positive")
    return quota / period


def cgroup_cpu_quota(path: str | os.PathLike[str] = CGROUP_CPU_MAX) -> float | None:
    """The CPUs the container's cgroup allows, or ``None`` (no limit, or no readable file)."""
    try:
        return parse_cgroup_cpu_max(Path(path).read_text())
    except (OSError, ValueError):
        return None


def session_threads(requested: int, quota: float | None, usable: int | None) -> int:
    """``requested`` threads, at most the whole CPUs of the cgroup quota and the CPUs the
    process may run on, and at least 1."""
    n = requested
    if quota is not None:
        n = min(n, int(quota))
    if usable:
        n = min(n, usable)
    return max(1, n)


def _usable_cpus() -> int | None:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


def gpu_pci_bus_id(index: int = 0, run: Callable[..., Any] = subprocess.run) -> str:
    """GPU ``index``'s PCI bus id in sysfs's spelling, from ``nvidia-smi``. Raises ``OSError``,
    ``subprocess.SubprocessError`` or ``ValueError`` when it can't be had."""
    text = run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
               capture_output=True, text=True, timeout=60, check=True).stdout
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if index >= len(lines):
        raise ValueError(f"nvidia-smi lists {len(lines)} GPUs, none at index {index}")
    return normalize_pci_bus_id(lines[index])


def pin_to_gpu_numa(*, sysfs: str = SYSFS_PCI, environ: Mapping[str, str] = os.environ,
                    run: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """Pin this process to the CPUs of the first visible GPU's NUMA node (the sysfs
    ``local_cpulist`` of its PCI device, intersected with the CPUs the process may use).

    Returns ``{"pinned", "cpus", "bus_id", "message"}``; ``cpus`` is the cpulist now in force
    when pinned, else ``None``, and ``message`` is the one line to log. Never raises: where it
    can't pin (no ``sched_setaffinity``, no ``nvidia-smi``, no sysfs file, an empty
    intersection) the affinity is left as it was and ``message`` says why."""
    def fail(why: str, bus_id: str | None = None) -> dict[str, Any]:
        return {"pinned": False, "cpus": None, "bus_id": bus_id,
                "message": f"CPU pin: none ({why})"}

    if not hasattr(os, "sched_setaffinity"):
        return fail("no sched_setaffinity on this platform")
    first = environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    index = int(first) if first.isdigit() else 0
    try:
        bus_id = gpu_pci_bus_id(index, run)
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return fail(f"no GPU bus id: {type(e).__name__}")
    try:
        local = parse_cpulist((Path(sysfs) / bus_id / "local_cpulist").read_text())
    except (OSError, ValueError) as e:
        return fail(f"no local_cpulist for {bus_id}: {type(e).__name__}", bus_id)
    allowed = set(os.sched_getaffinity(0))
    chosen = local & allowed
    if not chosen:
        return fail(f"{bus_id}'s CPUs {format_cpulist(local)} are outside this process's "
                    f"{format_cpulist(allowed)}", bus_id)
    try:
        os.sched_setaffinity(0, chosen)
    except OSError as e:
        return fail(f"sched_setaffinity failed: {e}", bus_id)
    cpus = format_cpulist(chosen)
    return {"pinned": True, "cpus": cpus, "bus_id": bus_id,
            "message": f"CPU pin: GPU {bus_id} node, CPUs {cpus} ({len(chosen)} of "
                       f"{len(allowed)} allowed)"}


def _threads(env: Mapping[str, str]) -> str:
    """``FS11_THREADS`` (default 8) capped by :func:`session_threads`; a non-number is kept."""
    threads = env.get("FS11_THREADS", "8")
    if threads.isdigit():
        threads = str(session_threads(int(threads), cgroup_cpu_quota(), _usable_cpus()))
    return threads


def base_env(upload: Path) -> dict[str, str]:
    """Every part's environment: the session's own, without :data:`SECRETS`, offline, with the
    upload's Hugging Face cache and BLAKE3 sources (the benchmark's and the native library's).
    The thread count (``FS11_THREADS``, default 8) is capped at the cgroup's CPU quota and the
    CPUs the session may run on (:func:`session_threads`)."""
    env = {k: v for k, v in os.environ.items() if k not in SECRETS}
    threads = _threads(env)
    env.update({
        "HF_HOME": str(upload / "hf"), "HF_HUB_CACHE": str(upload / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "POTS_BENCH_CACHE": str(upload / "bench_cache"),
        "POTS_NATIVE_CACHE": str(upload / "native_cache"),
        "PYTHONUNBUFFERED": "1", "VERIF_THREADS": threads, "OMP_NUM_THREADS": threads,
    })
    return env


# ---- what each part reports ---------------------------------------------------------------


_BRING_UP_LINE = re.compile(r"^(step \d+: (prover|verifier) .*|process lifetime peak RSS .*|"
                            r"(verdict|honest|guard|exit).*)$", re.IGNORECASE)
# llama_step.report_costs' lines: each side's peak (on CUDA with the reserved peak), and each
# step's wall clock, "step 1: prover 51.26 s (prove_step 50.00 s, commit 1.26 s); verifier
# 100.39 s (0 0.50, 1 1.20, …); hand-off 0.40 s" (a log from before the hand-off has none).
_PEAK = re.compile(r"^step (\d+): (prover|verifier) peak ([\d.]+) GB(?:, reserved ([\d.]+) GB)?")
_STEP_TIME = re.compile(r"^step (\d+): prover ([\d.]+) s \(prove_step ([\d.]+) s, commit "
                        r"([\d.]+) s\); verifier ([\d.]+) s \((.*)\)(?:; hand-off ([\d.]+) s)?$")
# The memory pass's line per top-level section as it closes (metrics.MemoryRecorder's echo):
# "   step 1, check 5: peak 71.56 GB, reserved 80.00 GB (start 1.90, end 1.95 GB; nested:
# 5.glue 30.12 (reserved 33.00) GB); raised OutOfMemoryError", or "   run, P4.tree: …" for a
# run-level section. The reserved figures are CUDA's only.
_SECTION = re.compile(r"^   (?:step (\d+)|run), (.+?): peak ([\d.]+|\?) GB(?:, reserved "
                      r"([\d.]+|\?) GB)? \(start ([\d.]+|\?), end ([\d.]+|\?) GB"
                      r"(?:; nested: (.*) GB)?\)(?:; raised (\w+))?$")
_NESTED = re.compile(r"^(.+) ([\d.]+|\?)(?: \(reserved ([\d.]+|\?)\))?$")


def _gb_or_none(x: str) -> float | None:
    return None if x == "?" else float(x)


def _section_peaks(lines: Sequence[str]) -> list[dict[str, Any]]:
    """The echo lines as records; ``reserved_gb`` and ``nested_reserved_gb`` are ``None`` and
    empty off CUDA."""
    out = []
    for m in map(_SECTION.match, lines):
        if not m:
            continue
        nested, nested_reserved = {}, {}
        for item in filter(None, (m[7] or "").split(", ")):
            n = _NESTED.match(item)
            if n:
                nested[n[1]] = _gb_or_none(n[2])
                if n[3] is not None:
                    nested_reserved[n[1]] = _gb_or_none(n[3])
        out.append({"step": None if m[1] is None else int(m[1]), "section": m[2],
                    "peak_gb": _gb_or_none(m[3]),
                    "reserved_gb": None if m[4] is None else _gb_or_none(m[4]),
                    "start_gb": _gb_or_none(m[5]), "end_gb": _gb_or_none(m[6]),
                    "nested_gb": nested, "nested_reserved_gb": nested_reserved,
                    "raised": m[8]})
    return out


def _step_times(lines: Sequence[str]) -> list[dict[str, Any]]:
    """Each step's wall clock from ``report_costs``: the prover (``prove_step`` and the
    commit), the verifier with each check's share, the hand-off between them (on CUDA it holds
    the cache release), and ``total_s``, the three together."""
    out = []
    for m in map(_STEP_TIME.match, lines):
        if not m:
            continue
        checks = {}
        for item in filter(None, m[6].split(", ")):
            name, _, s = item.rpartition(" ")
            checks[name] = float(s)
        prover, verifier = float(m[2]), float(m[5])
        handoff = None if m[7] is None else float(m[7])
        out.append({"step": int(m[1]), "prover_s": prover, "prove_step_s": float(m[3]),
                    "commit_s": float(m[4]), "verifier_s": verifier, "checks_s": checks,
                    "handoff_s": handoff,
                    "total_s": round(prover + verifier + (handoff or 0.0), 2)})
    return out


# Which allocator settings a run had: the environment record's fields (metrics.ALLOC_VARS and
# release_cache), or bring_up's "allocator: {...}" line, which prints the same as JSON.
_ALLOC_KEYS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "release_cache")
_ALLOC_LINE = re.compile(r"^allocator: (\{.*\})\s*$")


def _allocator(doc: Any) -> dict[str, Any] | None:
    """The allocator settings :data:`_ALLOC_KEYS` from an environment record, or ``None``
    when it isn't one (a malformed or older record)."""
    if not isinstance(doc, dict):
        return None
    return {k: doc.get(k) for k in _ALLOC_KEYS}


def _logged_allocator(lines: Sequence[str]) -> dict[str, Any] | None:
    for m in map(_ALLOC_LINE.match, lines):
        if m:
            try:
                return _allocator(json.loads(m[1]))
            except ValueError:
                return None
    return None


def _bring_up_extras(log: Path) -> dict[str, Any]:
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    peaks = [{"step": int(m[1]), "side": m[2], "peak_gb": float(m[3]),
              "reserved_gb": None if m[4] is None else float(m[4])}
             for m in map(_PEAK.match, lines) if m]
    return {"allocator": _logged_allocator(lines), "peaks": peaks,
            "step_times": _step_times(lines),
            "section_peaks": _section_peaks(lines),
            "device_leaves": _device_leaves_logged(lines),
            "lines": [l for l in lines if _BRING_UP_LINE.match(l)][-40:]}


def _number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _plain_row_ok(row: Mapping[str, Any]) -> bool:
    """Whether a ``records.jsonl`` row has the figures :func:`_plain_extras` reads as numbers:
    a time row's ``step`` and ``time_s``, a memory row's peaks (each a number or null)."""
    if row.get("record") == "time":
        return _number(row.get("step")) and _number(row.get("time_s"))
    if row.get("record") == "memory":
        return all(row.get(k) is None or _number(row.get(k))
                   for k in ("peak_bytes", "peak_reserved_bytes"))
    return True


def _plain_extras(report: Path) -> dict[str, Any]:
    """A plain arm's ``records.jsonl``: the timed run's per-step times (the ``step`` total of
    each step's ``P0`` phases, which on CUDA synchronize at their bounds), their mean and
    median, the memory pass's peak (on CUDA with its reserved peak, else ``None``), and the
    allocator settings its environment record holds. A line that isn't a JSON object (cut
    short when the run was killed mid-write), or a time or memory row whose figures aren't
    numbers, is skipped and counted in ``bad_lines``, so one bad line doesn't cost the arm."""
    if not report.exists():
        return {"record": None}
    rows, bad = [], 0
    for line in report.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if isinstance(row, dict) and _plain_row_ok(row):
            rows.append(row)
        else:
            bad += 1
    steps = sorted((r["step"], r["time_s"]) for r in rows
                   if r.get("record") == "time" and r.get("level") == "total"
                   and r.get("component") == "step" and r["step"] > 0)
    times = [s for _, s in steps]
    mem = [r for r in rows if r.get("record") == "memory" and r.get("peak_bytes") is not None]
    held = [r["peak_reserved_bytes"] for r in mem if r.get("peak_reserved_bytes") is not None]
    end = next((r for r in rows if r.get("record") == "run_end"), None)
    env = next((r for r in rows if r.get("record") == "environment"), None)
    return {"record": str(report), "bad_lines": bad, "allocator": _allocator(env),
            "completed": end is not None, "step_s": times,
            "mean_step_s": statistics.fmean(times) if times else None,
            "median_step_s": statistics.median(times) if times else None,
            "peak_bytes": max((r["peak_bytes"] for r in mem), default=None),
            "peak_reserved_bytes": max(held, default=None),
            "mem_source": mem[0].get("mem_source") if mem else None,
            "lifetime_maxrss_bytes": None if end is None else end.get("lifetime_maxrss_bytes")}


def _smooth_extras(report: Path) -> dict[str, Any]:
    if not report.exists():
        return {"record": None}
    doc = json.loads(report.read_text())
    return {"record": str(report), "allocator": _allocator(doc.get("environment")),
            "smooth": doc.get("smooth"), "reasons": doc.get("reasons"),
            "whole_list_losses": doc.get("whole_list_losses"),
            "batch_losses": doc.get("batch_losses"), "grad_norms": doc.get("grad_norms"),
            "peak_bytes": (doc.get("memory") or {}).get("peak_bytes"),
            "peak_reserved_bytes": (doc.get("memory") or {}).get("peak_reserved_bytes"),
            "total_s": (doc.get("timings_s") or {}).get("total")}


def _benchmark_extras(report: Path) -> dict[str, Any]:
    doc: dict[str, Any] = {"cupqc": cupqc_status()}
    if not report.exists():
        return {**doc, "report": None}
    rep = json.loads(report.read_text())
    skipped = []
    for key, slot in (rep.get("slots") or {}).items():
        for r in slot.get("results") or []:
            if r.get("status") in ("skipped", "not_implemented"):
                skipped.append({"slot": key, "result": r.get("name"),
                                "status": r.get("status"), "reason": r.get("reason")})
    return {**doc, "report": str(report), "complete": rep.get("complete"),
            "elapsed_s": rep.get("elapsed_s"), "skipped": skipped}


def benchmark_failure(report: Path) -> str | None:
    """Why the benchmark part failed though it exited 0, or ``None``: the report is missing or
    unreadable, isn't ``complete``, or holds a result whose reason starts ``error:``."""
    try:
        rep = json.loads(report.read_text())
    except (OSError, ValueError) as e:
        return f"report unreadable: {type(e).__name__}: {e}"
    if rep.get("complete") is not True:
        return "the report isn't complete"
    errors = [f"{key}/{r.get('name')}" for key, slot in (rep.get("slots") or {}).items()
              for r in slot.get("results") or []
              if str(r.get("reason") or "").startswith("error:")]
    if errors:
        return f"{len(errors)} results ended in an error ({', '.join(errors[:5])})"
    return None


def part_extras(part: Part, log: Path) -> dict[str, Any]:
    try:
        if part.kind == "bring_up":
            return _bring_up_extras(log)
        if part.kind == "smooth":
            return _smooth_extras(Path(part.info["report"]))
        if part.kind == "plain":
            return _plain_extras(Path(part.info["report"]))
        if part.kind == "benchmark":
            return _benchmark_extras(Path(part.info["report"]))
        if part.kind == "probe":
            return _probe_extras(Path(part.info["report"]))
        if part.kind == "nvlink":
            return _nvlink_extras(Path(part.info["report"]))
        if part.kind == "kernel_speed":
            return _kernel_speed_extras(Path(part.info["report"]))
        if part.kind in ("card", "pair_card", "four_card"):
            return _card_extras(Path(part.info["report"]))
        if part.kind == "four_tests":
            return _four_tests_extras(Path(part.info["junit"]))
        if part.kind == "four_probe":
            return _four_probe_extras(Path(part.info["report"]))
        if part.kind == "four_run":
            return _four_run_extras(log, Path(part.info["records"]))
        if part.kind == "profile_run":
            return _profile_run_extras(log, Path(part.info["records"]))
        if part.kind == "profile_plain":
            return _profile_plain_extras(Path(part.info["records"]))
        if part.kind == "pytest":
            return _pytest_extras(Path(part.info["junit"]))
        if part.kind in ("device_tests", "fs34_tests"):
            return _device_tests_extras(Path(part.info["junit"]))
        if part.kind == "pair_run":
            return _pair_run_extras(log, Path(part.info["weights"]))
        if part.kind == "cheat":
            return _cheat_extras(log)
        if part.kind == "peer_check":
            return _peer_check_extras(Path(part.info["report"]))
        if part.kind == "models":
            report = Path(part.info["report"])
            return json.loads(report.read_text()) if report.exists() else {"report": None}
    except (OSError, ValueError, KeyError, TypeError, ParseError) as e:
        return {"extras_error": f"{type(e).__name__}: {e}"}
    return {}


def _nvlink_extras(report: Path) -> dict[str, Any]:
    """HW2's verdict, its reasons and the slowest peer copy, from the probe's report."""
    if not report.exists():
        return {"hw2": None}
    hw2 = json.loads(report.read_text())["hw2"]
    return {"hw2": {k: hw2[k] for k in ("verdict", "pass", "reasons", "nv_links",
                                         "nv18_everywhere", "p2p_read_ok", "peer_copy_min_gbps",
                                         "slowest_pair", "pass_gbps", "fail_gbps", "n_cards",
                                         "expected_cards", "gate", "peer_bytes_ok")}}


def _kernel_speed_extras(report: Path) -> dict[str, Any]:
    """FS29's gate line, its verdict and the slowest model's rates, from the run's report; a
    missing or unreadable report gives ``{"gate": None}`` (and ``report_error`` when unreadable)."""
    if not report.exists():
        return {"gate": None}
    try:
        doc = json.loads(report.read_text())
        return {"gate": {k: doc["gate"].get(k) for k in (
                    "verdict", "slowest", "call_gbps", "call_best_gbps", "kernel_gbps",
                    "kernel_best_gbps", "reasons")},
                "gate_line": doc.get("gate_line")}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        return {"gate": None, "report_error": f"{type(e).__name__}: {e}"}


def kernel_speed_gate(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """FS29's gate from the parts' records, or ``None`` when the session has no speed part. The
    speed run's own verdict stands only when the native build and the kernel tests both finished
    ``ok`` (or were skipped as done); otherwise it is ``fail`` with that reason, the measured rates
    still on the line. With no readable report the verdict is ``no result``, never ``fail``."""
    by = {p.get("name"): p for p in parts}
    speed = next((p for p in parts if p.get("kind") == "kernel_speed"), None)
    if speed is None:
        return None
    status = {n: (by[n]["status"] if n in by else "absent") for n in ("native_build",
                                                                       "kernel_tests")}
    extras = speed.get("extras") or {}
    gate = extras.get("gate")
    line = extras.get("gate_line")
    out: dict[str, Any] = {"native_build": status["native_build"],
                           "kernel_tests": status["kernel_tests"], "speed_run": speed["status"]}
    if not gate or not line:
        why = extras.get("report_error") or f"the speed run's status is {speed['status']}"
        out.update(verdict="no result", reasons=[f"no readable report ({why})"],
                   line=f"FS29 gate: no result (the speed run left no readable report: {why}).")
        return out
    bad = [f"{n} {st}" for n, st in status.items() if st not in ("ok", "skipped_done")]
    if bad:
        out.update(verdict="fail", reasons=bad, line=(
            f"FS29 gate: fail. Because {' and '.join(bad)}, so the speed run's own verdict "
            f"({gate.get('verdict')}) does not count. " + line.split("FS29 gate: ", 1)[-1]))
    else:
        out.update(verdict=gate.get("verdict"), reasons=list(gate.get("reasons") or []), line=line)
    return out


def _card_extras(report: Path) -> dict[str, Any]:
    """One card's roots and losses, from its ``card_identity`` report."""
    if not report.exists():
        return {"card": None}
    doc = json.loads(report.read_text())
    return {"card": doc["card"].get("name"), "uuid": doc["card"].get("uuid"),
            "pci_bus_id": doc["card"].get("pci_bus_id"), "root": doc["proved"]["root"],
            "w_next_root": doc["proved"]["w_next_root"],
            "plain_w_next_root": doc["plain"]["w_next_root"],
            "loss": doc["proved"]["loss"], "plain_loss": doc["plain"]["loss"],
            "n_leaves": doc["n_leaves"], "wall_s": doc["wall_s"]}


def _pytest_extras(junit: Path) -> dict[str, Any]:
    """A pytest part's counts and failures from its junit XML: how many tests ran, failed,
    errored and skipped, each failure's name and seconds, and the slowest failing one."""
    import xml.etree.ElementTree as ET

    if not junit.exists():
        return {"junit": None}
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    failed = []
    for case in ET.parse(junit).getroot().iter("testcase"):
        counts["tests"] += 1
        bad = [c for c in case if c.tag in ("failure", "error")]
        counts["skipped"] += any(c.tag == "skipped" for c in case)
        counts["failures"] += any(c.tag == "failure" for c in case)
        counts["errors"] += any(c.tag == "error" for c in case)
        if bad:
            failed.append({"name": f"{case.get('classname')}::{case.get('name')}",
                           "time_s": float(case.get("time") or 0.0)})
    slowest = max(failed, key=lambda r: r["time_s"], default=None)
    return {**counts, "failed": failed, "slowest_failing": slowest}


_PAIR_STEP = re.compile(r"^step\s+(\d+) \([^)]*\): ((accepted|REJECTED);.*)$")
_PAIR_ROOT = re.compile(r"^step\s+(\d+) root ([0-9a-f]{64})$")
_PAIR_KEPT = re.compile(r"^stored step (\d+): .*root ([0-9a-f]{16})")


def _loop_mode(lines: Sequence[str]) -> str | None:
    """The mode the entry point printed (``loop mode: pipelined`` or ``sequential``), ``None``
    when there is no such line, and ``"multiple"`` when there is more than one (never a mode)."""
    modes = [x[len(PAIR_MODE_LINE):].strip() for x in lines if x.startswith(PAIR_MODE_LINE)]
    return "multiple" if len(modes) > 1 else (modes[0] if modes else None)


def _pair_run_extras(log: Path, weights: Path) -> dict[str, Any]:
    """A pair run's per-step lines (``step_lines``: everything after ``step t (...): ``),
    verdicts, transcript roots (``step_roots``, the ``step t root`` lines) and kept step's root
    (16 hex) from its log, each step's wall time (``step_times``: prover, verifier, hand-off,
    total, from ``report_costs``' lines), the printed ``loop_mode``, and the final weights' root
    and per-step losses from ``final_weights.json``."""
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    steps = {int(m[1]): m[2] for x in lines if (m := _PAIR_STEP.match(x))}
    roots = {int(m[1]): m[2] for x in lines if (m := _PAIR_ROOT.match(x))}
    kept = next((m[2] for x in lines if (m := _PAIR_KEPT.match(x))), None)
    doc = json.loads(weights.read_text()) if weights.exists() else {}
    return {"step_lines": [steps[t] for t in sorted(steps)],
            "verdicts": [steps[t].split(";")[0] for t in sorted(steps)],
            "step_roots": [roots[t] for t in sorted(roots)], "kept_root": kept,
            "final_root": doc.get("root"), "losses": doc.get("losses"),
            "step_times": _step_times(lines), "loop_mode": _loop_mode(lines)}


def _cheat_extras(log: Path) -> dict[str, Any]:
    """The oracle table's outcome per cheat: its rows are ``name declared actual outcome ρ_5 ρ_κ
    ρ_6`` (``cheats.report_oracle``), so the outcome is the fourth word from the end;
    ``oracle_lines`` keeps each whole row (words joined by one space) and ``loop_mode`` is the
    printed mode."""
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    out, rows = {}, {}
    for x in lines:
        w = x.split()
        if len(w) >= 7 and w[0] in PAIR_CHEATS and w[-4] in ORACLE_OUTCOMES:
            out[w[0]] = w[-4]
            rows[w[0]] = " ".join(w)
    return {"outcomes": out, "oracle_lines": rows, "loop_mode": _loop_mode(lines)}


def _read_peaks(path: Path) -> dict[str, int] | None:
    try:
        return json.loads(path.read_text())["peak_mib"]
    except (OSError, ValueError, KeyError):
        return None


def _gpu_used_mib() -> dict[str, int]:
    """Memory in use per GPU (MiB, by ``nvidia-smi`` index); empty without ``nvidia-smi``."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=15)
        return {a.strip(): int(b) for a, b in
                (row.split(",") for row in r.stdout.splitlines() if row.strip())}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


class _GpuPeaks:
    """A thread that samples every GPU's memory in use while a part runs and writes the
    peaks to ``path`` at :meth:`stop`: ``{"peak_mib": {index: MiB}, "samples": n}``. It is the
    card's whole use, not the part's process alone, and a sample sees what ``nvidia-smi``
    does, so a peak shorter than ``every`` seconds can be missed."""

    def __init__(self, path: Path, every: float = 1.0,
                 sample: Callable[[], dict[str, int]] = _gpu_used_mib) -> None:
        import threading

        self.path, self.every, self.sample = path, every, sample
        self.peaks: dict[str, int] = {}
        self.n = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            for k, v in self.sample().items():
                self.peaks[k] = max(self.peaks.get(k, 0), v)
            self.n += 1
            if self._stop.wait(self.every):
                return

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
        _write_json(self.path, {"peak_mib": self.peaks, "samples": self.n})


def _pair_part(parts: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    return next((p for p in parts if p["name"] == name), {})


PAIR_PARTS = (PEER_CHECK_NAME, "models", NATIVE_BUILD_NAME, "fast_suite", "two_gpu_slow",
              "fs34_tests",
              "smollm2_calibrate", "card_unsharded", "card_sharded", "run_unsharded",
              "run_sharded", "run_pipelined", "run_pipelined_metrics", *PAIR_CHEAT_PARTS,
              *PAIR_PIPELINED_CHEAT_PARTS)
# The mode each part's log must print: the pipelined arms' parts, and every other loop part's.
PAIR_PIPELINED_ARMS = ("pipelined", "pipelined_metrics")
PAIR_PIPELINED_PARTS = ("run_pipelined", "run_pipelined_metrics", *PAIR_PIPELINED_CHEAT_PARTS)
PAIR_SEQUENTIAL_PARTS = ("run_unsharded", "run_sharded", *PAIR_CHEAT_PARTS)
_PAIR_OK = ("ok", "skipped_done")


def _suite(parts: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any] | None:
    """A suite's counts, or ``None`` when its junit XML was never written."""
    ex = _pair_part(parts, name).get("extras") or {}
    if not ex.get("tests"):
        return None
    return {k: ex.get(k) for k in ("tests", "failures", "errors", "skipped")}


def fs32_pair(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The two-GPU session's comparison, or ``None`` without its parts. ``pass`` fails closed:
    every part of :data:`PAIR_PARTS` ran and succeeded; both suites left a report with tests in
    it, none failed or errored, and the two-GPU one skipped none; each arm's verdicts are
    ``accepted`` at every one of :data:`PAIR_STEPS` steps; the arms' roots and step lines are
    equal; every cheat is ``exact``. FS33d's pipelined arms (``run_pipelined``,
    ``run_pipelined_metrics``, the three ``*_pipelined`` cheats) are compared with the sharded
    sequential arm: the same verdicts, roots (final, kept, losses, every step's), whole step lines,
    and for each cheat ``exact`` with the same whole oracle row; each pipelined part's log holds
    ``loop mode: pipelined`` and each sequential loop part's ``loop mode: sequential``
    (``mode_problems``). ``step_times`` sets each arm's per-step wall time (prover, verifier,
    hand-off, total) side by side and ``gpu_peak`` each part's per-GPU peak, both outside
    ``pass``. The roots are step 1's transcript root, ``W_{t+1}``'s root
    and loss (``card_identity``), the final weights' root, per-step losses, kept step's root and
    every step's transcript root. A value is equal only when both arms have it. The plain and
    proved ``W_{t+1}`` of one arm (``plain_vs_proved``) are recorded, outside ``pass``."""
    if not any(p["kind"] == "pair_run" for p in parts):
        return None

    def arms(kind: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        return tuple((next((p for p in parts if p["kind"] == kind
                            and p["info"].get("arm") == a), {}).get("extras") or {})
                     for a in ("unsharded", "sharded"))  # type: ignore[return-value]

    def same(a: Mapping[str, Any], b: Mapping[str, Any], keys: Sequence[str]) -> dict[str, bool]:
        return {k: a.get(k) is not None and a.get(k) == b.get(k) for k in keys}

    def arm_of(arm: str) -> Mapping[str, Any]:
        return next((p for p in parts if p["kind"] == "pair_run"
                     and p["info"].get("arm") == arm), {}).get("extras") or {}

    card, run = arms("pair_card"), arms("pair_run")
    run_keys = ("final_root", "kept_root", "losses", "step_roots")
    roots = {**same(*card, ("root", "w_next_root", "loss")), **same(*run, run_keys)}
    roots["step_roots"] &= len(run[0].get("step_roots") or ()) == PAIR_STEPS
    pipe = {a: arm_of(a) for a in PAIR_PIPELINED_ARMS}
    for a, ex in pipe.items():  # each pipelined arm against the sharded sequential one
        roots.update({f"{a} {k}": ok for k, ok in same(run[1], ex, run_keys).items()})
        roots[f"{a} step_roots"] &= len(ex.get("step_roots") or ()) == PAIR_STEPS
    verdicts = tuple(r.get("verdicts") or [] for r in run)
    lines_equal = bool(run[0].get("step_lines")) and run[0].get("step_lines") == run[1].get(
        "step_lines")
    pipe_equal = {a: bool(ex.get("step_lines")) and ex.get("step_lines") == run[1].get("step_lines")
                  and (ex.get("verdicts") or []) == verdicts[1] for a, ex in pipe.items()}
    honest = all(v == ["accepted"] * PAIR_STEPS
                 for v in (*verdicts, *((ex.get("verdicts") or []) for ex in pipe.values())))
    outcomes: dict[str, str | None] = {}
    oracle: dict[str, str] = {}
    piped: dict[str, str | None] = {}
    piped_oracle: dict[str, str] = {}
    for p in parts:
        if p["kind"] == "cheat":
            ex = p.get("extras") or {}
            if p["info"].get("arm") == "pipelined":
                piped.update(ex.get("outcomes", {}))
                piped_oracle.update(ex.get("oracle_lines", {}))
            else:
                outcomes.update(ex.get("outcomes", {}))
                oracle.update(ex.get("oracle_lines", {}))
    oracle_equal = {n: n in oracle and oracle.get(n) == piped_oracle.get(n) for n in PAIR_CHEATS}
    modes = {n: ("pipelined" if n in PAIR_PIPELINED_PARTS else "sequential")
             for n in (*PAIR_SEQUENTIAL_PARTS, *PAIR_PIPELINED_PARTS)}
    mode_problems = {}
    for n, want in modes.items():
        part = _pair_part(parts, n)
        seen = (part.get("extras") or {}).get("loop_mode")
        if part.get("status") in _PAIR_OK and seen != want:
            mode_problems[n] = seen
    tests = {n: _suite(parts, n) for n in ("fast_suite", "two_gpu_slow")}
    failing = [dict(ex["slowest_failing"], part=n) for n in tests
               if (ex := _pair_part(parts, n).get("extras") or {}).get("slowest_failing")]
    names = {p["name"] for p in parts}
    problems = {n: _pair_part(parts, n).get("status", "not_started") for n in PAIR_PARTS
                if n not in names or _pair_part(parts, n).get("status") not in _PAIR_OK}
    suites_ok = all(t and not t["failures"] and not t["errors"] for t in tests.values())
    peer = (_pair_part(parts, PEER_CHECK_NAME).get("extras") or {}).get("peer_check")
    peer_ok = bool(peer) and peer.get("all_ok") is True
    slow_ran = bool(tests["two_gpu_slow"]) and tests["two_gpu_slow"]["skipped"] == 0
    fs34 = fs34_gate(parts)
    verdicts_equal = verdicts[0] == verdicts[1] and lines_equal and all(pipe_equal.values())
    step_times = {a: (e.get("step_times") or []) for a, e in
                  (("unsharded", run[0]), ("sharded", run[1]), *pipe.items())}
    gpu_peak = {n: (_pair_part(parts, n).get("extras") or {}).get("gpu_peak_mib")
                for n in PAIR_PARTS}
    return {"roots_equal": all(roots.values()), "roots": roots,
            "verdicts_equal": verdicts_equal, "honest": honest,
            "verdicts": {"unsharded": verdicts[0], "sharded": verdicts[1]},
            "plain_vs_proved": {a: None if c.get("plain_w_next_root") is None
                                else c.get("plain_w_next_root") == c.get("w_next_root")
                                for a, c in zip(("unsharded", "sharded"), card)},
            "cheats": {n: outcomes.get(n) for n in PAIR_CHEATS},
            "pipelined_cheats": {n: piped.get(n) for n in PAIR_CHEATS},
            "oracle_equal": oracle_equal, "mode_problems": mode_problems,
            "step_times": step_times, "gpu_peak": gpu_peak,
            "tests": tests, "fs34_problems": fs34, "part_problems": problems,
            "peer_bytes_ok": peer_ok,
            "peer_copies": None if peer_ok else peer_wrong_text(peer),
            "slowest_failing": max(failing, key=lambda r: r["time_s"], default=None),
            "pass": (all(roots.values()) and verdicts_equal and honest and not problems
                     and peer_ok and suites_ok and slow_ran and not fs34
                     and all(outcomes.get(n) == "exact" for n in PAIR_CHEATS)
                     and all(piped.get(n) == "exact" for n in PAIR_CHEATS)
                     and all(oracle_equal.values()) and not mode_problems)}


def _pair_lines(pair: Mapping[str, Any] | None) -> list[str]:
    """The two-GPU session's last job-log lines: one per acceptance item, then one
    ``FS32 FAIL`` line for each part that didn't run or left no test report."""
    if pair is None:
        return []
    r, v = pair["roots"], pair["verdicts"]
    lines = []
    if not pair["peer_bytes_ok"]:  # first: nothing below counts over wrong peer copies
        lines.append(f"FS32 FAIL: {pair['peer_copies']}")
    lines += ["FS32 roots equal: " + ("yes" if pair["roots_equal"] else "NO") + " ("
             + ", ".join(f"{k} {'=' if ok else '!='}" for k, ok in r.items()) + ")",
             "FS32 verdicts equal: " + ("yes" if pair["verdicts_equal"] else "NO")
             + f" (unsharded {v['unsharded']}, sharded {v['sharded']})"]
    lines += [f"FS32 {a} plain and proved W_{{t+1}}: "
              + {True: "equal", False: "differ", None: "unknown"}[eq]
              for a, eq in pair["plain_vs_proved"].items()]
    lines += [f"FS32 cheat {n}: {o or 'not run'}" + ("" if o == "exact" else " (NOT exact)")
              for n, o in pair["cheats"].items()]
    lines += [f"FS32 pipelined cheat {n}: {o or 'not run'}"
              + ("" if o == "exact" else " (NOT exact)")
              + ("" if pair["oracle_equal"][n] else "; oracle line differs from sequential")
              for n, o in pair["pipelined_cheats"].items()]
    for a, times in pair["step_times"].items():
        lines.append(f"FS32 step wall time {a} (prover/verifier/hand-off/total s): "
                     + (", ".join("/".join("?" if r.get(k) is None else f"{r[k]:.1f}"
                                           for k in _FIX_TIMES) for r in times) or "none"))
    for n, t in pair["tests"].items():
        if t is not None:
            lines.append(f"FS32 {n}: {t['tests']} tests, {t['failures']} failed, "
                         f"{t['errors']} errors, {t['skipped']} skipped")
    for n, seen in pair["mode_problems"].items():
        lines.append(f"FS32 FAIL: {n} " + {"sequential": "ran sequentially",
                                           "pipelined": "ran pipelined",
                                           "multiple": "printed more than one loop mode line",
                                           None: "printed no loop mode line"}.get(
                                               seen, f"printed loop mode {seen!r}"))
    for n, status in pair["part_problems"].items():
        lines.append(f"FS32 FAIL: {n} " + ("did not run" if status == "not_started"
                                           else f"ended {status}"))
    for n, t in pair["tests"].items():
        if t is None and n not in pair["part_problems"]:
            lines.append(f"FS32 FAIL: {n} left no test report")
    if not pair["honest"]:
        lines.append(f"FS32 FAIL: the honest run was not accepted at every one of "
                     f"{PAIR_STEPS} steps in both arms")
    lines += [f"FS32 FAIL: FS34 test {w}" for w in pair["fs34_problems"]]
    slow = pair["tests"]["two_gpu_slow"]
    if slow is not None and slow["skipped"]:
        lines.append(f"FS32 FAIL: two_gpu_slow skipped {slow['skipped']} of {slow['tests']} tests")
    s = pair["slowest_failing"]
    lines.append("FS32 slowest failing test: " + ("none" if s is None else
                 f"{s['name']} ({s['time_s']:.1f} s, {s['part']})"))
    return lines


# ---- FS35's four-H100 session: the parts' extras and the result ------------------------------


def _four_records(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of a ``records.jsonl`` (a line cut short by a kill is skipped)."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _role_gpus(roles: Sequence[Mapping[str, Any]] | None, side: str) -> int:
    """The GPUs of the environment record's ``gpu_roles`` that hold ``side`` (``prover`` or
    ``verifier``); 0 without the record."""
    if not isinstance(roles, list):
        return 0
    return sum(side in (g.get("roles") or ()) for g in roles if isinstance(g, dict))


def _accumulated_gpu_s(side_s: Mapping[str, float],
                       roles: Sequence[Mapping[str, Any]] | None) -> float | None:
    """One step's accumulated GPU time, as the cost decision defines it (each GPU's held time
    summed over the GPUs a part holds: three verifier GPUs for 1 s are 3 s): the prover side's
    time times the number of prover GPUs plus the verifier side's time times the number of
    verifier GPUs, both counts from the environment record's ``gpu_roles``. ``side_s`` holds the
    step's ``total`` rows' ``time_s`` by side, each the sum of that side's top-level sections
    (the ``run`` side and the harness's own sections are in neither). ``None`` when either side's
    row or a role count is missing."""
    if not isinstance(roles, list) or not all(s in side_s for s in ("prover", "verifier")):
        return None
    n = {s: _role_gpus(roles, s) for s in ("prover", "verifier")}
    if not all(n.values()):
        return None
    return side_s["prover"] * n["prover"] + side_s["verifier"] * n["verifier"]


def _four_run_extras(log: Path, records: Path) -> dict[str, Any]:
    """A ``llama_step`` arm's result: from its log the verdict (``accepted``: the ``actual:
    accept`` and ``PASS`` lines), each step's transcript root (``step_roots``), each step's
    prover, verifier, hand-off and total time (``step_times``, ``report_costs``' lines) and the
    printed ``loop_mode``; from its ``records.jsonl`` the environment record's GPUs
    (``machine.gpus``: index and UUID; ``gpu_roles``), per step the ``step`` row's
    ``gpu_held_upper_s`` (``metrics.py``'s ``gpu_held_s``: each section's seconds times every
    device its process lists, an upper bound, not the cost decision's quantity), the
    ``accumulated_gpu_s`` (:func:`_accumulated_gpu_s`) and the merged step's wall extent
    (``step_wall`` rows' ``time_s``, the latency) and every step's interval to the step before's
    close (``step_interval`` rows' ``time_s``, FS42), the verifier side's ``total`` row per
    step (``verifier_s``) and the memory pass's peak per device
    (``device_peak_bytes`` and ``device_peak_reserved_bytes`` of the prover's and the verifier's
    ``total`` rows, the larger over the steps)."""
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    roots = {int(m[1]): m[2] for x in lines if (m := _PAIR_ROOT.match(x))}
    stripped = [x.strip() for x in lines]
    actual = next((x[len("actual:"):].strip() for x in stripped if x.startswith("actual:")), None)
    rows = _four_records(records)
    env = next((r for r in rows if r.get("record") == "environment"), None) or {}
    gpus = (env.get("machine") or {}).get("gpus")
    held: dict[int, float] = {}
    wall: dict[int, float] = {}
    interval: dict[int, float] = {}
    side_s: dict[int, dict[str, float]] = {}
    peaks: dict[str, dict[str, int | None]] = {}
    for r in rows:
        step = r.get("step")
        if not isinstance(step, int) or step < 1:
            continue
        if r.get("record") == "time" and r.get("level") == "total":
            if r.get("component") == "step" and _number(r.get("gpu_held_s")):
                held[step] = r["gpu_held_s"]
            if r.get("component") == "step_wall" and _number(r.get("time_s")):
                wall[step] = r["time_s"]
            if r.get("component") == "step_interval" and _number(r.get("time_s")):
                interval[step] = r["time_s"]
            if r.get("side") in ("prover", "verifier") and _number(r.get("time_s")):
                side_s.setdefault(step, {})[r["side"]] = r["time_s"]
        if (r.get("record") == "memory" and r.get("level") == "total"
                and r.get("component") in ("prover", "verifier")
                and isinstance(r.get("devices"), list)):
            for field, key in (("device_peak_bytes", "peak_bytes"),
                               ("device_peak_reserved_bytes", "peak_reserved_bytes")):
                vals = r.get(field)
                if not isinstance(vals, list) or len(vals) != len(r["devices"]):
                    continue
                for dev, v in zip(r["devices"], vals, strict=True):
                    row = peaks.setdefault(dev, {"peak_bytes": None, "peak_reserved_bytes": None})
                    if _number(v) and (row[key] is None or v > row[key]):
                        row[key] = v
    return {"accepted": actual == "accept" and "PASS" in stripped, "actual": actual,
            "step_roots": [roots[t] for t in sorted(roots)], "loop_mode": _loop_mode(lines),
            "step_times": _step_times(lines), "record": str(records) if rows else None,
            "gpus": None if not isinstance(gpus, list) else [
                {"index": g.get("index"), "uuid": g.get("uuid")} for g in gpus
                if isinstance(g, dict)],
            "gpu_roles": env.get("gpu_roles"),
            "gpu_held_upper_s": [held[t] for t in sorted(held)],
            "accumulated_gpu_s": [a for t in sorted(side_s)
                                  if (a := _accumulated_gpu_s(side_s[t], env.get("gpu_roles")))
                                  is not None],
            "verifier_s": [side_s[t]["verifier"] for t in sorted(side_s)
                           if "verifier" in side_s[t]],
            "step_wall_s": [wall[t] for t in sorted(wall)],
            "step_interval_s": [interval[t] for t in sorted(interval)], "device_peaks": peaks}


# FS42 (``four-h100-profile``): the profile's inputs, as the metrics module names them in a
# run directory (``<part run dir>/llama_step/`` for an arm, ``.../plain_baseline/`` for the
# plain run).
PROFILE_MARKERS = "markers.jsonl"  # metrics.MetricsWriter.write_markers
PROFILE_LAYOUT = "c5_layout.json"  # metrics.MetricsWriter.write_c5_layout
PROFILE_TRACE_DIR = "trace"  # metrics.TRACE_DIR; the traced step's step_NNNNN.json.gz in it
# The work rows' sections: the ``count`` rows the analysis' floors read (shape-derived under
# ``--no-count``, FS42 (b)); a ``storage`` row beside them holds the transcript bytes.
PROFILE_WORK_SECTIONS = ("train.forward", "train.backward", "P3.commit", "2")


def _samples_header(path: Path) -> dict[str, Any] | None:
    """A part's ``samples.jsonl`` header: its ``version`` and how many GPUs it names (``gpus``,
    ``None`` when the header has none); ``None`` when there is no file or its first line is not
    a header."""
    try:
        with open(path) as f:
            head = json.loads(f.readline())
    except (OSError, ValueError):
        return None
    if not isinstance(head, dict) or head.get("kind") != "header":
        return None
    gpus = head.get("gpus")
    return {"version": head.get("version"), "gpus": len(gpus) if isinstance(gpus, list) else None}


def _profile_markers(path: Path) -> dict[str, Any]:
    """``markers.jsonl``'s ``step`` lines: ``markers_steps`` (the steps with a line) and
    ``traced_steps`` (those whose line says ``traced``); a line cut short is skipped."""
    steps: dict[int, bool] = {}
    for line in path.read_text(errors="replace").splitlines() if path.is_file() else []:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("kind") == "step" and isinstance(row.get("step"), int):
            steps[row["step"]] = bool(row.get("traced"))
    return {"markers_file": path.is_file(), "markers_steps": sorted(steps),
            "traced_steps": sorted(t for t, v in steps.items() if v)}


def _profile_run_extras(log: Path, records: Path) -> dict[str, Any]:
    """:func:`_four_run_extras` and, under ``profile``, what the profile reads from the arm's
    run directory (``records``' directory): the markers' steps and the traced ones, the steps
    with a ``step_interval`` time row, the work sections with a ``count`` row and the number of
    ``storage`` rows, whether the layout file exists and is not empty, and the non-empty trace
    files (``trace_files``) with the trace's own ``error`` from ``trace/meta.json``."""
    out = _four_run_extras(log, records)
    d = records.parent
    rows = _four_records(records)
    trace = d / PROFILE_TRACE_DIR
    meta: Any = None
    try:
        meta = json.loads((trace / "meta.json").read_text())
    except (OSError, ValueError):
        pass
    layout = d / PROFILE_LAYOUT
    out["profile"] = {
        **_profile_markers(d / PROFILE_MARKERS),
        "interval_steps": sorted({r["step"] for r in rows if r.get("record") == "time"
                                  and r.get("level") == "total"
                                  and r.get("component") == "step_interval"
                                  and isinstance(r.get("step"), int) and _number(r.get("time_s"))}),
        "count_sections": sorted({r["component"] for r in rows if r.get("record") == "count"
                                  and r.get("component") in PROFILE_WORK_SECTIONS}),
        "storage_rows": sum(r.get("record") == "storage" for r in rows),
        "layout": layout.is_file() and layout.stat().st_size > 0,
        "trace_files": sorted(p.name for p in trace.glob("step_*.json.gz")
                              if p.stat().st_size > 0) if trace.is_dir() else [],
        "trace_error": meta.get("error") if isinstance(meta, dict) else None}
    return out


def _profile_plain_extras(report: Path) -> dict[str, Any]:
    """:func:`_plain_extras` and what the profile reads of the plain run: the steps with a
    ``P0.*`` time row (``p0_steps``), the ``step`` total row per step (``step_totals_s``, the
    row ``time`` / ``total`` / ``step``), their ``steady_s`` (mean of steps 2 and 3, ``None``
    without both) and the size of ``final_weights.json`` beside the records."""
    out = _plain_extras(report)
    rows = _four_records(report)
    time_rows = [r for r in rows if r.get("record") == "time" and isinstance(r.get("step"), int)
                 and r["step"] >= 1 and _number(r.get("time_s"))]
    totals = {r["step"]: r["time_s"] for r in time_rows
              if r.get("level") == "total" and r.get("component") == "step"}
    weights = report.parent / WEIGHTS_FILE
    out.update({"p0_steps": sorted({r["step"] for r in time_rows
                                    if str(r.get("component", "")).startswith("P0.")}),
                "step_totals_s": [totals[t] for t in sorted(totals)],
                "steady_s": statistics.fmean([totals[2], totals[3]])
                if 2 in totals and 3 in totals else None,
                "weights_bytes": weights.stat().st_size if weights.is_file() else 0})
    return out


def _four_tests_extras(junit: Path) -> dict[str, Any]:
    """The tests part's cases (:func:`_device_tests_extras`) and, for FS34's file and each of
    :data:`FOUR_TEST_FILES`, how many cases the junit report holds and how many were skipped
    (``files``); ``missing_files`` names a file that does not exist in the checkout (the run
    selects them by name, so a missing one collects nothing, and says nothing)."""
    out = _device_tests_extras(junit)
    wanted = (FS34_TESTS_FILE, *FOUR_TEST_FILES)
    out["missing_files"] = [f for f in wanted if not (_REPO_ROOT / f).exists()]
    cases = out.get("cases")
    if cases is not None:
        files = {}
        for f in wanted:
            mod = f.removesuffix(".py").replace("/", ".") + "::"
            mine = [o for n, o in cases.items() if n.startswith(mod)]
            files[f] = {"cases": len(mine), "skipped": sum(o == "skipped" for o in mine)}
        out["files"] = files
    return out


def _four_probe_extras(report: Path) -> dict[str, Any]:
    """:func:`_probe_extras` and, from the probe's version 2 report (I3), each measured device's
    headline rates (``per_device``: bf16 GEMM, HBM copy, the hash kernel's seconds per byte) and
    the ordered pairs' peer rates in bytes/s (``peer``, ``peer_errors``)."""
    out = _probe_extras(report)
    if not report.exists():
        return out
    doc = json.loads(report.read_text())
    per: dict[str, Any] = {}
    for name in doc.get("devices") or []:
        rates = (doc.get("per_device", {}).get(name) or {}).get("rates") or {}
        per[name] = {"gemm_bf16_flops": rates.get("gemm_bf16_flops"),
                     "hbm_copy_bytes": rates.get("hbm_copy_bytes"),
                     "hash_kernel_per_byte_s": (rates.get("hash_kernel") or {}).get("per_byte_s")}
    return {**out, "version": doc.get("version"), "devices": doc.get("devices"),
            "per_device": per, "peer": doc.get("peer") or {},
            "peer_errors": doc.get("peer_errors") or {}}


FOUR_ARMS = ("seq", "pipe")


def _four_gpu_problems(ex: Mapping[str, Any]) -> list[str]:
    """Why an arm's environment record does not name the four GPUs as the session lays them out
    (empty when it does): ``machine.gpus`` four entries with distinct UUIDs, ``gpu_roles`` with
    ``cuda:0`` the prover and ``cuda:1`` to ``cuda:3`` verifiers, each role entry's UUID the
    machine record's for its index."""
    why: list[str] = []
    gpus = ex.get("gpus")
    uuids = {g.get("index"): g.get("uuid") for g in gpus or []}
    if gpus is None:
        return ["the record has no machine.gpus"]
    if len(gpus) != FOUR_CARDS or len(uuids) != FOUR_CARDS or not all(uuids.values()):
        why.append(f"machine.gpus has {len(gpus)} entries ({len(uuids)} indices)")
    if len(set(uuids.values())) != len(uuids):
        why.append("machine.gpus has a UUID twice")
    roles = ex.get("gpu_roles")
    if not isinstance(roles, list):
        return [*why, "the record has no gpu_roles"]
    want = {"cuda:0": ["prover"], **{f"cuda:{i}": ["verifier"] for i in (1, 2, 3)}}
    got = {r.get("device"): r for r in roles if isinstance(r, dict)}
    if len(roles) != FOUR_CARDS or set(got) != set(want):
        why.append(f"gpu_roles lists {sorted(map(str, got))}")
    for dev, role in want.items():
        r = got.get(dev)
        if r is None:
            continue
        if r.get("roles") != role:
            why.append(f"{dev} has roles {r.get('roles')}, not {role}")
        if uuids.get(r.get("index")) != r.get("uuid") or not r.get("uuid"):
            why.append(f"{dev}'s UUID is not the machine record's")
    if len({r.get("uuid") for r in got.values()}) != len(got):
        why.append("gpu_roles has a UUID twice")
    return why


def _four_tests_problems(part: Mapping[str, Any], expected: Sequence[str] | None) -> list[str]:
    """Why the tests part does not count, empty when it does: it ran ``ok``, its junit report
    holds tests and no failure or error, no FS34 case (found by :func:`device_tests_expected`, or
    given as ``expected``) is missing, failed, errored or skipped, every file of
    :data:`FOUR_TEST_FILES` exists and left a case in the report, and no selected case at all was
    skipped (on four GPUs every one runs; the first few skipped cases are named, then a count)
    and no such file has every case skipped. Fails closed."""
    ex = part.get("extras") or {}
    why = [f"{f} is missing from the checkout" for f in ex.get("missing_files") or ()]
    if part.get("status") not in _PAIR_OK:
        why.append(f"the tests part is {part.get('status', 'not_started')}")
    if ex.get("junit", 0) is None or "cases" not in ex:
        return [*why, "the tests left no junit report"]
    if not ex.get("tests"):
        why.append("the junit report has no tests")
    if ex.get("failures") or ex.get("errors"):
        why.append(f"{ex.get('failures')} failures and {ex.get('errors')} errors")
    if expected is None:
        expected, err = device_tests_expected(FS34_TESTS_FILE)
        if err:
            why.append(f"FS34's expected tests could not be collected: {err}")
    why += [f"FS34 case {n}: " + ("not in the junit report" if n not in ex["cases"]
                                  else ex["cases"][n]) for n in expected
            if ex["cases"].get(n) != "passed"]
    why += [f"{f} left no case in the junit report" for f in FOUR_TEST_FILES
            if not (ex.get("files") or {}).get(f, {}).get("cases")
            and f"{f} is missing from the checkout" not in why]
    skipped = sorted(n for n, o in ex["cases"].items() if o == "skipped")
    count = max(len(skipped), ex.get("skipped") or 0)
    if count:
        shown = ", ".join(skipped[:3]) + (f" and {count - 3} more" if count > 3 else "")
        why.append(f"{count} case(s) skipped, none may be on four GPUs"
                   + (f": {shown}" if shown else ""))
    why += [f"every case of {f} was skipped" for f in FOUR_TEST_FILES
            if (m := (ex.get("files") or {}).get(f, {})).get("cases")
            and m.get("skipped") == m["cases"]]
    return why


def _steady_slice(steps: int, pipelined: bool = False, traced_last: bool = False) -> slice:
    """The steady steps of an arm of ``steps`` steps, as a slice of its per-step list: step 1 is
    the warm-up and the last step is left out where it is not steady, steps 2..``steps`` for a
    sequential arm, steps 2..``steps`` - 1 for a pipelined one (its last step is the drain, verified
    with the prover idle, about 11% short on the rehearsal) and for an arm that traces its last
    step. Three steps pipelined: step 2 alone; four steps: steps 2-3."""
    return slice(1, steps - 1 if pipelined or traced_last else steps)


def _steady_steps_text(steps: int, pipelined: bool = False, traced_last: bool = False) -> str:
    """The steady steps of :func:`_steady_slice` as the logs name them: ``"steps 2-3"`` or
    ``"step 2"``, or ``"no steady step"`` where the set is empty (two pipelined steps)."""
    n = len(range(steps)[_steady_slice(steps, pipelined, traced_last)])
    if n == 0:
        return "no steady step"
    return "step 2" if n == 1 else f"steps 2-{n + 1}"


def _steady_s(ex: Mapping[str, Any], steps: int = FOUR_STEPS, pipelined: bool = False,
              traced_last: bool = False) -> tuple[float | None, str | None]:
    """An arm's steady step time, the mean of its steady steps (:func:`_steady_slice`) of its
    ``steps`` steps, and what it is: the interval between consecutive verified steps
    (``step_interval``, EQ14's wall time per step, FS42) where the records hold it, else the
    merged step's wall extent (``step_wall``, records written before FS42), else the times'
    total."""
    sl = _steady_slice(steps, pipelined, traced_last)
    if len(range(steps)[sl]) == 0:
        return None, None
    ivs = ex.get("step_interval_s") or []
    if len(ivs) >= steps:
        return statistics.fmean(ivs[sl]), "step_interval"
    wall = ex.get("step_wall_s") or []
    if len(wall) >= steps:
        return statistics.fmean(wall[sl]), "step_wall"
    times = ex.get("step_times") or []
    if len(times) >= steps and all(r.get("total_s") is not None for r in times):
        return statistics.fmean(r["total_s"] for r in times[sl]), "total"
    return None, None


def _pos(x: Any) -> bool:
    """A finite positive number."""
    return _number(x) and math.isfinite(x) and x > 0


def _steady_mean(per_step: Sequence[float] | None, steps: int = FOUR_STEPS,
                 pipelined: bool = False, traced_last: bool = False) -> float | None:
    """The mean of the steady steps (:func:`_steady_slice`) of a per-step list of ``steps``
    values, else ``None``."""
    vals = list(per_step or ())[_steady_slice(steps, pipelined, traced_last)]
    return statistics.fmean(vals) if vals and len(per_step or ()) >= steps else None


def _four_missing_numbers(models: Mapping[str, Mapping[str, Any]],
                          probe: Mapping[str, Any], check_gbps: Mapping[str, Any]) -> list[str]:
    """What the plan's acceptance ("each model's step time, per-GPU peaks and peer-copy rates
    reported") lacks, empty when it has all: per model and arm a steady step time and a peak for
    each of the four GPUs (``gpu_peak_mib``, the whole card's, or else ``device_peaks``, this
    process's), and once a finite positive peer rate for all 12 ordered pairs (the probe's
    ``peer``, or else ``peer_check``'s per-pair ``gbps``)."""
    why: list[str] = []
    for k, row in models.items():
        for a, arm in row["arms"].items():
            if arm["steady_s"] is None:
                why.append(f"{k} {a}: no steady step time")
            card = arm.get("gpu_peak_mib") or {}
            procs = arm.get("device_peaks") or {}
            lost = [i for i in range(FOUR_CARDS)
                    if not _pos(card.get(str(i)))
                    and not _pos((procs.get(f"cuda:{i}") or {}).get("peak_bytes"))]
            if lost:
                why.append(f"{k} {a}: no peak for GPU " + ", ".join(map(str, lost)))
    pairs = [(x, y) for x in range(FOUR_CARDS) for y in range(FOUR_CARDS) if x != y]
    got_probe = sum(_pos((probe.get("peer") or {}).get(f"cuda:{x}->cuda:{y}")) for x, y in pairs)
    got_check = sum(_pos(check_gbps.get(f"{x}->{y}")) for x, y in pairs)
    if max(got_probe, got_check) < len(pairs):
        why.append(f"peer rates: the probe has {got_probe} of {len(pairs)} pairs, the peer "
                   f"check {got_check}")
    return why


def _four_model_items(parts: Sequence[Mapping[str, Any]], k: str, row: dict[str, Any],
                      steps: int, traced_last: bool = False) -> list[tuple[bool, str]]:
    """One model's acceptance items of the four-H100 sessions (``steps`` verified steps per
    arm; ``traced_last``: the sequential arm traces its last step, :func:`_steady_slice` leaves
    it out of the steady steps): both arms accepted at all ``steps`` steps in their loop mode, the
    arms' step roots equal step for step, step 1's root the card's proved root, and the four
    GPUs named in each arm's environment record. Fills ``row`` (the model's entry in the result)
    and returns the items. The arms' parts are ``seq_<k>`` and ``pipe_<k>``, the card's
    ``card_<k>``."""
    items: list[tuple[bool, str]] = []
    card = (_pair_part(parts, f"card_{k}").get("extras") or {})
    arm_ex = {a: (_pair_part(parts, f"{a}_{k}").get("extras") or {}) for a in FOUR_ARMS}
    mode = {"seq": "sequential", "pipe": "pipelined"}
    accepted = {a: bool(e.get("accepted")) and len(e.get("step_roots") or ()) == steps
                and e.get("loop_mode") == mode[a] for a, e in arm_ex.items()}
    seq_roots, pipe_roots = (arm_ex[a].get("step_roots") or [] for a in FOUR_ARMS)
    equal = len(seq_roots) == steps and seq_roots == pipe_roots
    card_ok = bool(card.get("root")) and bool(seq_roots) and seq_roots[0] == card["root"] \
        and bool(pipe_roots) and pipe_roots[0] == card["root"]
    gpu_why = {a: _four_gpu_problems(e) for a, e in arm_ex.items()}
    pipe = {a: a == "pipe" for a in arm_ex}
    last = {a: traced_last and a == "seq" for a in arm_ex}
    steady = {a: _steady_s(e, steps, pipe[a], last[a]) for a, e in arm_ex.items()}
    row.update({
        "accepted": accepted, "roots_equal": equal, "step1_equals_card": card_ok,
        "card_root": card.get("root"), "gpu_problems": gpu_why,
        "arms": {a: {"step_roots": e.get("step_roots"), "step_times": e.get("step_times"),
                     "step_wall_s": e.get("step_wall_s"),
                     "step_interval_s": e.get("step_interval_s"),
                     "latency_steady_s": _steady_mean(e.get("step_wall_s"), steps, True)
                     if a == "pipe" else None,
                     "steady_steps": _steady_steps_text(steps, pipe[a], last[a]),
                     "gpu_held_upper_s": e.get("gpu_held_upper_s"),
                     "accumulated_gpu_s": e.get("accumulated_gpu_s"),
                     "verifier_steady_s": _steady_mean(e.get("verifier_s"), steps, pipe[a],
                                                       last[a]),
                     "accumulated_steady_s": _steady_mean(e.get("accumulated_gpu_s"), steps,
                                                          pipe[a], last[a]),
                     "steady_s": steady[a][0], "steady_source": steady[a][1],
                     "device_peaks": e.get("device_peaks"), "loop_mode": e.get("loop_mode"),
                     "gpu_peak_mib": (_pair_part(parts, f"{a}_{k}").get("extras")
                                      or {}).get("gpu_peak_mib")}
                 for a, e in arm_ex.items()}})
    items += [
        (all(accepted.values()),
         f"{k}: both arms accepted at all {steps} steps" if all(accepted.values())
         else f"{k}: " + ", ".join(f"the {a} arm was not accepted at all {steps} steps "
                                   "in its loop mode" for a, ok in accepted.items()
                                   if not ok)),
        (equal, f"{k}: the arms' step roots are equal" if equal
         else f"{k}: the sequential and pipelined arms' step roots differ"),
        (card_ok, f"{k}: step 1's root is the card's proved root" if card_ok
         else f"{k}: step 1's root differs from card_{k}'s proved root"),
        (not any(gpu_why.values()), f"{k}: the environment records name the four GPUs"
         if not any(gpu_why.values()) else f"{k}: " + "; ".join(
             f"{a} arm: " + ", ".join(w) for a, w in gpu_why.items() if w))]
    return items


def four_h100(parts: Sequence[Mapping[str, Any]],
              expected: Sequence[str] | None = None) -> dict[str, Any] | None:
    """The four-H100 session's result, or ``None`` without its parts. ``pass`` fails closed:
    every part of :func:`four_h100_parts` ran ``ok`` (or was done earlier); the peer copies
    landed right bytes; the tests part counts (:func:`_four_tests_problems`); and per model both
    arms are accepted at all :data:`FOUR_STEPS` steps and ran in their mode (``loop mode:``
    line), the two arms' step roots are equal step for step, step 1's root is the card part's
    proved root, and each arm's environment record names the four GPUs
    (:func:`_four_gpu_problems`); and the reported numbers are all there
    (:func:`_four_missing_numbers`: per model and arm a steady step time and a peak per GPU,
    once 12 peer rates). ``items`` lists each acceptance item as ``(ok, text)``, in the order
    the log prints them. Outside ``pass``: per model and arm each step's prover, verifier,
    hand-off and total time (``step_times``), ``step_wall_s``, ``step_interval_s`` and
    ``gpu_held_upper_s`` (an upper bound) per step, ``accumulated_gpu_s`` per step and its steady
    mean
    (``accumulated_steady_s``, the arm's steady steps, ``steady_steps``: steps 2..n for a
    sequential arm, 2..n-1 for a pipelined one, whose last step is the drain;
    :func:`_steady_slice`, :func:`_accumulated_gpu_s`), ``steady_s`` (the same steps,
    :func:`_steady_s`: the step interval; ``latency_steady_s``, the pipelined arm's
    ``step_wall`` mean), the memory pass's peak per device (``device_peaks``) and the whole
    card's peaks (``gpu_peak_mib``, per GPU); and once the probe's per-device headline rates, the
    12 peer rates and ``peer_check``'s per-pair rates."""
    if not any(p["kind"] == "four_run" for p in parts):
        return None
    names = {p["name"] for p in parts}
    models = {four_model_key(r): {} for r, _ in FOUR_PINS}
    want = four_h100_parts()
    problems = {n: _pair_part(parts, n).get("status", "not_started") for n in want
                if n not in names or _pair_part(parts, n).get("status") not in _PAIR_OK}
    peer = (_pair_part(parts, PEER_CHECK_NAME).get("extras") or {}).get("peer_check")
    peer_ok = bool(peer) and peer.get("all_ok") is True
    tests_problems = _four_tests_problems(_pair_part(parts, "four_tests"), expected)
    tests_ex = _pair_part(parts, "four_tests").get("extras") or {}
    items: list[tuple[bool, str]] = [
        (not problems, "every part ran and succeeded" if not problems else "; ".join(
            f"{n} " + ("did not run" if s == "not_started" else f"ended {s}")
            for n, s in problems.items())),
        (peer_ok, "the peer copies landed the right bytes" if peer_ok else
         f"peer copies: {peer_wrong_text(peer)}"),
        (not tests_problems,
         f"the tests: {tests_ex.get('tests')} tests, {tests_ex.get('failures')} failed, "
         f"{tests_ex.get('errors')} errors, {tests_ex.get('skipped')} skipped; FS34's cases "
         "all passed" if not tests_problems else "the tests: " + "; ".join(tests_problems))]
    for k, row in models.items():
        items += _four_model_items(parts, k, row, FOUR_STEPS)
    probe = _pair_part(parts, "probe").get("extras") or {}
    check_rates = (peer or {}).get("gbps") or {}
    missing = _four_missing_numbers(models, probe, check_rates)
    items.append((not missing, "the step times, per-GPU peaks and peer rates are all reported"
                  if not missing else "numbers missing: " + "; ".join(missing)))
    return {"pass": all(ok for ok, _ in items), "items": [[ok, text] for ok, text in items],
            "missing_numbers": missing,
            "models": models, "part_problems": problems, "peer_bytes_ok": peer_ok,
            "tests": {k: tests_ex.get(k) for k in ("tests", "failures", "errors", "skipped",
                                                   "files", "missing_files")},
            "tests_problems": tests_problems,
            "probe": {k: probe.get(k) for k in ("version", "devices", "per_device", "peer",
                                                "peer_errors")},
            "peer_check_gbps": check_rates,
            "gpu_peak": {n: (_pair_part(parts, n).get("extras") or {}).get("gpu_peak_mib")
                         for n in want}}


def four_h100_parts() -> tuple[str, ...]:
    """The names of the session's parts, the native build and the peer check first."""
    return (NATIVE_BUILD_NAME, PEER_CHECK_NAME, "models", "four_tests", "probe",
            *(f"{a}_{four_model_key(r)}" for r, _ in FOUR_PINS for a in ("card", "seq", "pipe")))


def _four_peer_range(res: Mapping[str, Any]) -> str:
    """The slowest and fastest ordered pair, from the probe's 12 rates (else from the peer
    check's per-pair rates), in GB/s."""
    rates = {k: v / 1e9 for k, v in ((res.get("probe") or {}).get("peer") or {}).items()
             if _number(v)}
    src = "probe"
    if not rates:
        rates = {k: float(v) for k, v in (res.get("peer_check_gbps") or {}).items()
                 if _number(v)}
        src = "peer check"
    if not rates:
        return "FS35 peer rates: none measured"
    lo, hi = min(rates, key=lambda k: rates[k]), max(rates, key=lambda k: rates[k])
    return (f"FS35 peer rates ({src}, {len(rates)} pairs): slowest {lo} {rates[lo]:.1f} GB/s, "
            f"fastest {hi} {rates[hi]:.1f} GB/s")


def _four_lines(res: Mapping[str, Any] | None) -> list[str]:
    """The session's last job-log lines: one ``FS35 ok`` or ``FS35 FAIL`` per acceptance item,
    one per model and arm with the wall time (the steady step) beside the accumulated GPU time
    (steady: the mean of steps 2-3 for a sequential arm, step 2 alone for a pipelined one) and
    the four GPUs' whole-card peaks, then the peer-rate range."""
    if res is None:
        return []
    lines = [f"FS35 {'ok' if ok else 'FAIL'}: {text}" for ok, text in res["items"]]
    for k, row in res["models"].items():
        for a, arm in row["arms"].items():
            s, src, acc = arm["steady_s"], arm["steady_source"], arm.get("accumulated_steady_s")
            lat = arm.get("latency_steady_s")
            st = arm.get("steady_steps") or "steps 2-3"
            peaks = arm.get("gpu_peak_mib") or {}
            lines.append(
                f"FS35 {k} {a}: steady step "
                + ("unknown" if s is None else
                   f"{s:.1f} s (interval between verified steps, mean of {st})"
                   if src == "step_interval" else f"{s:.1f} s ({src}, mean of {st})")
                + ("" if a != "pipe" else "; latency " + (
                    "unknown" if lat is None else f"{lat:.1f} s (step_wall)"))
                + "; accumulated GPU time "
                + ("unknown" if acc is None else f"{acc:.1f} s (mean of {st})")
                + "; GPU peaks "
                + (", ".join(f"{i}: {peaks[i] / 1024:.1f} GiB" if i in peaks else f"{i}: ?"
                             for i in map(str, range(FOUR_CARDS))) if peaks else "unknown"))
    lines.append(_four_peer_range(res))
    return lines


def four_profile_parts() -> tuple[str, ...]:
    """The names of ``four-h100-profile``'s parts, the native build and the peer check first;
    per model the plain run, the card, then the two arms."""
    return (NATIVE_BUILD_NAME, PEER_CHECK_NAME, "models", "four_tests", "probe",
            *(f"{a}_{four_model_key(r)}" for r, _ in FOUR_PROFILE_PINS
              for a in ("plain", "card", "seq", "pipe")))


def _missing_steps(have: Sequence[int], steps: int = FOUR_PROFILE_STEPS) -> str:
    """``"steps 2, 4"``: the steps of 1..``steps`` not in ``have``."""
    gone = [t for t in range(1, steps + 1) if t not in set(have)]
    return "step" + ("s " if len(gone) > 1 else " ") + ", ".join(map(str, gone))


def _profile_arm_problems(ex: Mapping[str, Any], arm: str) -> list[str]:
    """What an arm's output lacks of the profile's inputs (empty when none): a markers line, a
    ``step_interval`` time row for every step; a ``count`` row for each of
    :data:`PROFILE_WORK_SECTIONS` and a ``storage`` row; the layout file; the sequential arm's
    trace of step :data:`FOUR_PROFILE_TRACE_STEP` (its markers line ``traced`` and no other
    step's, the trace file there and not empty, the trace's own error empty); and for the
    pipelined arm no traced step."""
    pr = ex.get("profile")
    if not isinstance(pr, dict):
        return ["no profile inputs read (the arm left no output)"]
    why: list[str] = []
    steps = FOUR_PROFILE_STEPS
    if not pr.get("markers_file"):
        why.append(f"no {PROFILE_MARKERS}")
    elif len(set(pr.get("markers_steps") or ()) & set(range(1, steps + 1))) < steps:
        why.append(f"{PROFILE_MARKERS} has no line for "
                   + _missing_steps(pr.get("markers_steps") or ()))
    if len(set(pr.get("interval_steps") or ()) & set(range(1, steps + 1))) < steps:
        why.append("no step_interval time row for "
                   + _missing_steps(pr.get("interval_steps") or ()))
    lost = [s for s in PROFILE_WORK_SECTIONS if s not in (pr.get("count_sections") or ())]
    if lost:
        why.append("no count row for " + ", ".join(lost))
    if not pr.get("storage_rows"):
        why.append("no storage row")
    if not pr.get("layout"):
        why.append(f"no {PROFILE_LAYOUT} (or it is empty)")
    traced = list(pr.get("traced_steps") or ())
    name = f"step_{FOUR_PROFILE_TRACE_STEP:05d}.json.gz"
    if arm == "seq":
        if traced != [FOUR_PROFILE_TRACE_STEP]:
            why.append(f"the markers trace {traced or 'no step'}, not step "
                       f"{FOUR_PROFILE_TRACE_STEP} alone")
        if name not in (pr.get("trace_files") or ()):
            why.append(f"no trace file {PROFILE_TRACE_DIR}/{name} (or it is empty)")
        if pr.get("trace_error"):
            why.append(f"the trace reports an error: {pr['trace_error']}")
    elif traced:
        why.append(f"the markers show a traced step ({', '.join(map(str, traced))}), "
                   "a pipelined run is never traced")
    return why


def _profile_plain_problems(ex: Mapping[str, Any]) -> list[str]:
    """What the plain run lacks (empty when nothing): its records, a completed run, ``P0.*``
    time rows and a ``step`` total for every step, ``final_weights.json``."""
    if not ex.get("record"):
        return ["no plain run records"]
    why: list[str] = []
    if not ex.get("completed"):
        why.append("the plain run did not complete")
    steps = FOUR_PROFILE_PLAIN_STEPS
    if len(set(ex.get("p0_steps") or ()) & set(range(1, steps + 1))) < steps:
        why.append("no P0.* time row for " + _missing_steps(ex.get("p0_steps") or ()))
    if len(ex.get("step_totals_s") or ()) < steps or ex.get("steady_s") is None:
        why.append(f"fewer than {steps} step total rows (or none for steps 2 and 3)")
    if not ex.get("weights_bytes"):
        why.append(f"no {WEIGHTS_FILE}")
    return why


def _samples_problem(ex: Mapping[str, Any]) -> str | None:
    """Why a part's ``samples.jsonl`` does not count (version 2, four GPUs in its header)."""
    s = ex.get("samples")
    if not isinstance(s, dict):
        return "no samples.jsonl"
    if s.get("version") != 2 or s.get("gpus") != FOUR_CARDS:
        return (f"samples.jsonl is version {s.get('version')} with {s.get('gpus')} GPUs in its "
                f"header, not version 2 with {FOUR_CARDS}")
    return None


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b is not None and _pos(a) and _pos(b) else None


def four_h100_profile(parts: Sequence[Mapping[str, Any]],
                      expected: Sequence[str] | None = None) -> dict[str, Any] | None:
    """The ``four-h100-profile`` session's result, or ``None`` without its parts. ``pass``
    fails closed: every part of :func:`four_profile_parts` ran ``ok`` (or was done earlier); the
    peer copies landed right bytes; the tests part counts (:func:`_four_tests_problems`); per
    model FS35's items at :data:`FOUR_PROFILE_STEPS` steps (:func:`_four_model_items`); the
    plain run's records and weights (:func:`_profile_plain_problems`) and each arm's profile
    inputs (:func:`_profile_arm_problems`); every part's ``samples.jsonl``
    (:func:`_samples_problem`; one line per part that fails); the probe's version 2 report with
    four ``per_device`` entries; and the reported numbers (:func:`_four_missing_numbers`).
    ``items`` lists each acceptance item as ``(ok, text)``, in the order the log prints them.
    Outside ``pass``, per model: each arm's numbers as the FS35 result has them (the steady step
    from the step interval, the pipelined latency, the accumulated GPU time, the four GPUs'
    peaks), ``verifier_steady_s`` (the verifier side's ``total`` row, mean of the arm's steady
    steps; the sequential arm's traced step 4 and the pipelined arm's drain step 4 stay out,
    so steps 2-3), the
    plain run's ``steady_s`` (mean of the ``step`` total over steps 2-3) and per arm
    ``verify_over_train`` (verifier over the plain step, in wall time: the verifier side's step
    time over the plain step's, whatever the number of verifier GPUs),
    ``verify_over_train_accumulated`` (the verifier side's step time times its GPUs, from the
    arm's ``gpu_roles``, over the plain step on its one GPU: the accumulated GPU time ratio
    that EQ14 puts beside PoTS's verify over train) and ``wall_over_train`` (the arm's step
    interval over the plain step)."""
    if not any(p.get("kind") == "profile_run" for p in parts):
        return None
    names = {p["name"] for p in parts}
    models = {four_model_key(r): {} for r, _ in FOUR_PROFILE_PINS}
    want = four_profile_parts()
    problems = {n: _pair_part(parts, n).get("status", "not_started") for n in want
                if n not in names or _pair_part(parts, n).get("status") not in _PAIR_OK}
    peer = (_pair_part(parts, PEER_CHECK_NAME).get("extras") or {}).get("peer_check")
    peer_ok = bool(peer) and peer.get("all_ok") is True
    tests_problems = _four_tests_problems(_pair_part(parts, "four_tests"), expected)
    tests_ex = _pair_part(parts, "four_tests").get("extras") or {}
    items: list[tuple[bool, str]] = [
        (not problems, "every part ran and succeeded" if not problems else "; ".join(
            f"{n} " + ("did not run" if s == "not_started" else f"ended {s}")
            for n, s in problems.items())),
        (peer_ok, "the peer copies landed the right bytes" if peer_ok else
         f"peer copies: {peer_wrong_text(peer)}"),
        (not tests_problems,
         f"the tests: {tests_ex.get('tests')} tests, {tests_ex.get('failures')} failed, "
         f"{tests_ex.get('errors')} errors, {tests_ex.get('skipped')} skipped; FS34's cases "
         "all passed" if not tests_problems else "the tests: " + "; ".join(tests_problems))]
    for k, row in models.items():
        items += _four_model_items(parts, k, row, FOUR_PROFILE_STEPS, traced_last=True)
        plain = _pair_part(parts, f"plain_{k}").get("extras") or {}
        why_plain = _profile_plain_problems(plain)
        steady_plain = plain.get("steady_s")
        row["plain"] = {"steady_s": steady_plain, "steady_row": "time / total / step, mean of "
                        "steps 2-3", "step_totals_s": plain.get("step_totals_s"),
                        "problems": why_plain}
        items.append((not why_plain, f"{k} plain: the records and final weights are there"
                      if not why_plain else f"{k} plain: " + "; ".join(why_plain)))
        row["inputs"] = {}
        for a, arm in row["arms"].items():
            ex = _pair_part(parts, f"{a}_{k}").get("extras") or {}
            why = _profile_arm_problems(ex, a)
            row["inputs"][a] = why
            arm["verify_over_train"] = _ratio(arm["verifier_steady_s"], steady_plain)
            n_ver = _role_gpus(ex.get("gpu_roles"), "verifier")
            arm["verify_over_train_accumulated"] = _ratio(
                arm["verifier_steady_s"] * n_ver if arm["verifier_steady_s"] is not None
                and n_ver else None, steady_plain)
            arm["wall_over_train"] = (_ratio(arm["steady_s"], steady_plain)
                                      if arm["steady_source"] == "step_interval" else None)
            items.append((not why, f"{k} {a}: the profile's inputs are there" if not why
                          else f"{k} {a}: " + "; ".join(why)))
    sample_why = {n: w for n in want if n != NATIVE_BUILD_NAME
                  and (w := _samples_problem(_pair_part(parts, n).get("extras") or {}))}
    items += [(False, f"{n}: {w}") for n, w in sample_why.items()]
    if not sample_why:
        items.append((True, "every part left a version 2 samples.jsonl with four GPUs"))
    probe = _pair_part(parts, "probe").get("extras") or {}
    probe_ok = probe.get("version") == 2 and len(probe.get("per_device") or {}) == FOUR_CARDS
    items.append((probe_ok, "the probe report is version 2 with four per_device entries"
                  if probe_ok else f"probe.json is version {probe.get('version')} with "
                  f"{len(probe.get('per_device') or {})} per_device entries, not version 2 "
                  f"with {FOUR_CARDS}"))
    check_rates = (peer or {}).get("gbps") or {}
    missing = _four_missing_numbers(models, probe, check_rates)
    items.append((not missing, "the step times, per-GPU peaks and peer rates are all reported"
                  if not missing else "numbers missing: " + "; ".join(missing)))
    return {"pass": all(ok for ok, _ in items), "items": [[ok, text] for ok, text in items],
            "missing_numbers": missing, "models": models, "part_problems": problems,
            "peer_bytes_ok": peer_ok,
            "tests": {k: tests_ex.get(k) for k in ("tests", "failures", "errors", "skipped",
                                                   "files", "missing_files")},
            "tests_problems": tests_problems, "samples_problems": sample_why,
            "probe": {k: probe.get(k) for k in ("version", "devices", "per_device", "peer",
                                                "peer_errors")},
            "peer_check_gbps": check_rates,
            "gpu_peak": {n: (_pair_part(parts, n).get("extras") or {}).get("gpu_peak_mib")
                         for n in want}}


def _profile_lines(res: Mapping[str, Any] | None) -> list[str]:
    """The job log's last lines of ``four-h100-profile``: one ``FS42 ok`` or ``FS42 FAIL`` per
    acceptance item, then per model and arm the step (the interval between verified steps), the
    verifier side's time per step, the plain run's step and the two ratios."""
    if res is None:
        return []

    def sec(x: float | None) -> str:
        return "unknown" if x is None else f"{x:.1f} s"

    def rat(x: float | None) -> str:
        return "unknown" if x is None else f"{x:.2g}"

    lines = [f"FS42 {'ok' if ok else 'FAIL'}: {text}" for ok, text in res["items"]]
    for k, row in res["models"].items():
        plain = (row.get("plain") or {}).get("steady_s")
        for a, arm in row["arms"].items():
            src = arm.get("steady_source")
            what = ("interval between verified steps" if src == "step_interval"
                    else f"{src}, not the interval" if src else "no step time")
            lines.append(
                f"FS42 {k} {a}: step {sec(arm.get('steady_s'))} ({what}), verifier "
                f"{sec(arm.get('verifier_steady_s'))} per step, plain training {sec(plain)} per "
                f"step; verifier ÷ training {rat(arm.get('verify_over_train'))} (wall time), "
                f"{rat(arm.get('verify_over_train_accumulated'))} (accumulated GPU time); step "
                f"÷ training {rat(arm.get('wall_over_train'))}")
    return lines


_FIX_TIMES = ("prover_s", "verifier_s", "handoff_s", "total_s")
_FIX_PEAKS = ("peak_gb", "reserved_gb")


def _fix_cost(on: Mapping[str, Any] | None, off: Mapping[str, Any] | None,
              keys: Sequence[str]) -> dict[str, float | None]:
    return {k: None if on is None or off is None or on.get(k) is None or off.get(k) is None
            else round(on[k] - off[k], 2) for k in keys}


def fix2_ab(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The verified-step A/B of the allocator fix, from the summary's part records, or ``None``
    when the plan has no fix-off part. The bring-up with the fix on (``info["arm"]``
    ``fix2_on``) beside the one with it off (``fix2_off``), each with the allocator settings
    it was planned with and those its log says it ran with (``ran_with``): per step, the
    prover, verifier,
    hand-off and total wall clock (``total_s``, the three together, from the memory pass as the
    bring-up reports it); per step and side, the allocated and reserved peaks; and
    ``fix_cost``, on minus off (positive: the fix costs that much time or memory)."""
    arms = {(p.get("info") or {}).get("arm"): p for p in parts if p.get("kind") == "bring_up"}
    off, on = arms.get("fix2_off"), arms.get("fix2_on")
    if off is None:
        return None
    extras = {"on": (on or {}).get("extras") or {}, "off": off.get("extras") or {}}

    def head(p: Mapping[str, Any] | None) -> dict[str, Any] | None:
        return None if p is None else {
            "part": p["name"], "status": p["status"],
            **{k: p["info"].get(k) for k in ("alloc_conf", "release_cache")},
            "ran_with": (p.get("extras") or {}).get("allocator")}

    times = {a: {r["step"]: {k: r.get(k) for k in _FIX_TIMES} for r in x.get("step_times", [])}
             for a, x in extras.items()}
    peaks = {a: {(r["step"], r["side"]): {k: r.get(k) for k in _FIX_PEAKS}
                 for r in x.get("peaks", [])} for a, x in extras.items()}
    return {
        "model": off["info"].get("model"), "fix_on": head(on), "fix_off": head(off),
        "steps": [{"step": t, "fix_on": times["on"].get(t), "fix_off": times["off"].get(t),
                   "fix_cost": _fix_cost(times["on"].get(t), times["off"].get(t), _FIX_TIMES)}
                  for t in sorted(times["on"].keys() | times["off"].keys())],
        "peaks": [{"step": t, "side": side, "fix_on": peaks["on"].get((t, side)),
                   "fix_off": peaks["off"].get((t, side)),
                   "fix_cost": _fix_cost(peaks["on"].get((t, side)), peaks["off"].get((t, side)),
                                         _FIX_PEAKS)}
                  for t, side in sorted(peaks["on"].keys() | peaks["off"].keys())],
    }


def _leaf_label(layout: Mapping[str, Any], index: int) -> tuple[str, str]:
    """The section and name of leaf ``index`` under a card report's ``layout``: product leaves
    by ``K`` and ``product_leaf_names`` (O18); an old report's ``M`` and ``product_names`` still
    read."""
    n_s, n_w = layout["n_s"], layout["n_w"]
    k = layout["K"] if "K" in layout else layout["M"]
    names = layout["product_leaf_names"] if "K" in layout else layout["product_names"]
    if index < n_s:
        return "record", f"record {index}"
    if index < n_s + n_w:
        return "w_t", layout["weight_names"][index - n_s]
    if index < n_s + n_w + k:
        return "product", names[index - n_s - n_w]
    return "w_next", layout["weight_names"][index - n_s - n_w - k]


def _first_difference(ref: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """Where two cards' leaf hashes first differ, from their ``.digests`` files (32 bytes a
    leaf, in leaf order), with that leaf's section and name and the differing leaves per section."""
    a = Path(ref["info"]["report"]).with_suffix(".digests").read_bytes()
    b = Path(other["info"]["report"]).with_suffix(".digests").read_bytes()
    out: dict[str, Any] = {"card": other["info"]["card"], "reference_card": ref["info"]["card"]}
    if len(a) != len(b):
        return {**out, "leaf": None, "note": f"digest files of {len(a)} and {len(b)} bytes"}
    layout = json.loads(Path(other["info"]["report"]).read_text())["layout"]
    bad = [i for i in range(len(a) // 32) if a[32 * i:32 * i + 32] != b[32 * i:32 * i + 32]]
    if not bad:
        return {**out, "leaf": None, "note": "every leaf hash is equal"}
    by_section: dict[str, int] = {}
    for i in bad:
        sec = _leaf_label(layout, i)[0]
        by_section[sec] = by_section.get(sec, 0) + 1
    section, name = _leaf_label(layout, bad[0])
    return {**out, "leaf": bad[0], "section": section, "name": name,
            "n_differing_leaves": len(bad), "by_section": by_section}


def hw3_cards(parts: Sequence[Mapping[str, Any]], expected: int | None = None
              ) -> dict[str, Any] | None:
    """F22's HW3, first half, from the summary's part records: ``None`` when the plan has no card
    part. Per model: ``roots`` (the committed step's root per card, ``None`` for a card with no
    report), ``n_cards`` (the cards compared), ``uuids``, ``distinct_cards`` (every UUID present
    and none repeated: the runs landed on different physical cards), ``ran`` (every card's part
    succeeded), ``same_bits`` (``expected`` cards, at least 2, distinct, all ran, the roots equal)
    and, where roots differ, ``first_difference``: the first leaf at which a card's hashes differ
    from the first card's, its section and name. top-level ``same_bits`` is true when every model
    has it; ``pass`` also needs ``gate``.
    ``expected`` (default ``FS31_CARDS``) is recorded with ``n_cards`` (the fewest over the
    models); ``gate`` is false for anything but :data:`GATE_CARDS` (a rehearsal, never a pass)."""
    cards = [p for p in parts if p.get("kind") == "card"]
    if not cards:
        return None
    n_expected = fs31_cards() if expected is None else expected
    models: dict[str, dict[str, Any]] = {}
    for model in dict.fromkeys(p["info"]["model"] for p in cards):
        mine = sorted((p for p in cards if p["info"]["model"] == model),
                      key=lambda p: p["info"]["card"])
        roots = {str(p["info"]["card"]): (p.get("extras") or {}).get("root") for p in mine}
        uuids = {str(p["info"]["card"]): (p.get("extras") or {}).get("uuid") for p in mine}
        ran = all(p["status"] in ("ok", "skipped_done") for p in mine)
        distinct = {r for r in roots.values() if r is not None}
        problems: list[str] = []
        if len(mine) < 2:
            problems.append(f"only {len(mine)} card compared, at least 2 are needed")
        if len(mine) != n_expected:
            problems.append(f"{len(mine)} cards compared, {CARDS_ENV} is {n_expected}")
        absent = [k for k, u in uuids.items() if not u]
        if absent:
            problems.append("no UUID for card " + ", ".join(absent))
        seen = [u for u in uuids.values() if u]
        again = sorted({u for u in seen if seen.count(u) > 1})
        if again:
            problems.append("the same UUID on more than one card: " + ", ".join(again))
        same = (ran and None not in roots.values() and len(distinct) == 1 and not problems)
        rec: dict[str, Any] = {"roots": roots, "n_cards": len(mine), "uuids": uuids,
                               "distinct_cards": not absent and not again, "ran": ran,
                               "same_bits": same, "problems": problems}
        if len(distinct) > 1:
            held = [p for p in mine if roots[str(p["info"]["card"])] is not None]
            ref = held[0]
            rec["first_difference"] = next(
                (_first_difference(ref, p) for p in held[1:]
                 if roots[str(p["info"]["card"])] != roots[str(ref["info"]["card"])]), None)
        models[model] = rec
    same_all = all(m["same_bits"] for m in models.values())
    return {"models": models, "same_bits": same_all, "pass": same_all and n_expected == GATE_CARDS,
            "n_cards": min(m["n_cards"] for m in models.values()), "expected_cards": n_expected,
            "gate": n_expected == GATE_CARDS}


def _hw_lines(parts: Sequence[Mapping[str, Any]], hw3: Mapping[str, Any] | None) -> list[str]:
    """The job log's HW2 and HW3 lines (and, for the kernel-speed session, its gate line): pass or
    FAIL, with the reasons; a review for HW2's middle speed band; and, when ``FS31_CARDS`` is not
    the four cards, a rehearsal in place of a pass."""
    from verification.runs.nvlink_probe import hw2_line  # stdlib only, kept off module level

    lines = []
    speed = kernel_speed_gate(parts)
    if speed is not None:  # FS29's session has no HW2 or HW3; its one line is the gate
        lines.append(speed["line"])
    leaves = device_leaves_gate(parts)
    if leaves is not None:  # FS30's session: its one line is the gate
        lines.append(leaves["line"])
    probe = next((p for p in parts if p.get("kind") == "nvlink"), None)
    if probe is not None:
        hw2 = (probe.get("extras") or {}).get("hw2")
        peer = next((p for p in parts if p.get("kind") == "peer_check"), None)
        if peer is not None and peer_check_stop(peer):
            # the session stopped on it: the probe never ran, so say why, plainly
            why = peer_wrong_text((peer.get("extras") or {}).get("peer_check"))
            lines.append("HW2 FAIL: " + why
                         + "; the NVLink probe did not run")
        elif hw2 is None:
            lines.append(f"HW2 FAIL: the NVLink probe left no verdict ({probe['status']})")
        else:
            lines.append(hw2_line(hw2))
    if hw3 is not None:
        if "hw3_cards_error" in hw3:
            lines.append(f"HW3 part 1 FAIL: the comparison raised ({hw3['hw3_cards_error']})")
        elif hw3.get("same_bits", hw3["pass"]):
            n = hw3.get("expected_cards", GATE_CARDS)
            head = ("HW3 part 1 pass" if hw3.get("gate", True)
                    else f"HW3 part 1 rehearsal ({n} of {GATE_CARDS} cards), not the gate")
            lines.append(f"{head}: the same bits on every card for all "
                         f"{len(hw3['models'])} models")
        else:
            why = []
            for model, m in hw3["models"].items():
                if m["same_bits"]:
                    continue
                d = m.get("first_difference")
                found = list(m.get("problems") or [])
                if d is not None:
                    found.append(f"card {d['card']} differs from card {d['reference_card']} "
                                 f"first at leaf {d.get('leaf')} ({d.get('section')} "
                                 f"{d.get('name')})")
                if not m["ran"]:
                    found.append("a card did not run or has no report")
                elif not found:
                    found.append("no common root")
                detail = "; ".join(found)
                why.append(f"{model.split('/')[-1]}: {detail}")
            lines.append("HW3 part 1 FAIL: " + "; ".join(why))
    return lines


# ---- the session ------------------------------------------------------------------------


class _Stop(Exception):
    pass


def _kill(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the part's process group, then SIGKILL it after :data:`KILL_GRACE_S`."""
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_S), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _start_sampler(proc: subprocess.Popen[bytes], out: Path, cwd: Path, python: str) -> Any:
    """The part's sampler (``setup.sampler``, writing ``out``), or ``None`` when ``POTS_SAMPLER=0``
    or it can't start: a sampler failure never fails the part."""
    if os.environ.get("POTS_SAMPLER") == "0":
        return None
    try:
        from setup.sampler import start_sampler
        return start_sampler(proc.pid, out, cwd=cwd, python=python)
    except Exception:  # noqa: BLE001
        return None


def _stop_sampler(handle: Any) -> None:
    if handle is not None:
        try:
            handle.stop()
        except Exception:  # noqa: BLE001
            pass


EXIT_HANG_GRACE_S = 180  # a pytest part's process may outlive its junit report this long
PROBE_HANG_GRACE_S = 60  # the same, for one run of the exit probe (:func:`run_exit_probe`)
STACKS_WAIT_S = 5  # after SIGUSR1, the process gets this long to write its stacks
STACKS_ENV = "POTS_STACKS_FILE"  # ``tests/exit_guard.py``: where a pytest process dumps stacks
GRACE_ENV = "POTS_EXIT_GRACE_S"  # ``tests/exit_guard.py``: its watchdog's seconds after the run
# The watchdog's grace under the session: the watch's own grace plus this, so the watch takes the
# /proc snapshot (every native thread, the kernel's wait) while the process still hangs; the
# watchdog's dump holds Python threads only and ends the process before the watch could look.
GUARD_LAG_S = 120
PROC = "/proc"  # where :func:`proc_snapshot` reads (a test points it at a fake tree)


def _proc_text(path: str) -> str:
    """A ``/proc`` file as one line of text, or ``<ErrorName: reason>`` (a container hides
    some of them, ``stack`` and ``syscall`` among them)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        return f"<{type(e).__name__}: {e.strerror or e}>"
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def _stat_fields(path: str) -> tuple[str, str, str] | None:
    """``(state, ppid, pgrp)`` from a ``/proc/.../stat`` file, or ``None``."""
    text = _proc_text(path)
    try:
        rest = text[text.rindex(")") + 2:].split()
        return rest[0], rest[1], rest[2]
    except (ValueError, IndexError):
        return None


def proc_snapshot(pgid: int) -> str:
    """Every process of process group ``pgid`` from ``/proc``: pid, parent, state and command
    line; per thread its id, ``comm``, state, ``wchan``, and ``syscall`` and ``stack`` where
    readable. Standard library only; every read is guarded, an unreadable one is recorded."""
    lines = [f"snapshot {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} of process "
             f"group {pgid}"]
    try:
        pids = sorted(int(n) for n in os.listdir(PROC) if n.isdigit())
    except OSError as e:
        return lines[0] + f"\nno {PROC}: {type(e).__name__}: {e}\n"
    for pid in pids:
        fields = _stat_fields(f"{PROC}/{pid}/stat")
        if fields is None or fields[2] != str(pgid):
            continue
        lines.append(f"process {pid} ppid {fields[1]} state {fields[0]} cmdline "
                     f"{_proc_text(f'{PROC}/{pid}/cmdline')[:400]}")
        try:
            tids = sorted(int(n) for n in os.listdir(f"{PROC}/{pid}/task") if n.isdigit())
        except OSError as e:
            lines.append(f"  threads unreadable: {type(e).__name__}: {e}")
            continue
        for tid in tids:
            base = f"{PROC}/{pid}/task/{tid}"
            tf = _stat_fields(f"{base}/stat")
            lines.append(f"  thread {tid} comm {_proc_text(f'{base}/comm')} state "
                         f"{tf[0] if tf else '?'} wchan {_proc_text(f'{base}/wchan')} "
                         f"syscall {_proc_text(f'{base}/syscall')}")
            stack = _proc_text(f"{base}/stack")
            if stack:
                lines.append(f"    stack {stack}")
    return "\n".join(lines) + "\n"


@dataclass
class Watched:
    """How :func:`watch_process` ended: ``status`` is ``ok`` or ``failed`` (the process
    exited by itself, by its exit code), ``exit_hang`` (it outlived its junit report by the
    grace and was killed) or ``limit`` (the time limit came, and the process was killed);
    ``junit_to_exit_s`` is the time from the report to the end of the process, when there was
    a report."""
    status: str
    rc: int | None
    junit_to_exit_s: float | None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def watch_process(proc: subprocess.Popen[bytes], *, t0: float, limit: float,
                  junit: Path | None = None, grace_s: float = EXIT_HANG_GRACE_S,
                  hang_file: Path | None = None, stacks: bool = False) -> Watched:
    """Wait for ``proc`` (the leader of its own process group) until it exits, until ``limit``
    (Unix time) or, with ``junit``, until ``grace_s`` after that file appears with an mtime
    after ``t0``: a pytest process that wrote its report and does not exit is hung in its
    teardown. A process that is still alive then is ended: its group's :func:`proc_snapshot`
    goes to ``hang_file``, ``SIGUSR1`` asks the leader alone (a spawned child would die of it)
    for every thread's stack (``stacks``: the process is a pytest run, whose
    ``tests/exit_guard.py`` writes them to ``POTS_STACKS_FILE``), and after
    :data:`STACKS_WAIT_S` the group is killed (:func:`_kill`). Both paths, the hang and the
    limit, leave the same evidence. The one place the session and the exit probe watch a
    process in; a caller kills the group itself when this raises."""
    seen: float | None = None  # the report's mtime

    def ended(status: str, rc: int | None) -> Watched:
        return Watched(status, rc, None if seen is None else max(0.0, time.time() - seen))

    def evidence() -> None:
        if hang_file is not None:
            try:
                with open(hang_file, "a") as h:
                    h.write(proc_snapshot(proc.pid))
            except OSError:
                pass
        if stacks and hasattr(signal, "SIGUSR1") and proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGUSR1)
                proc.wait(timeout=STACKS_WAIT_S)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

    while True:
        try:
            rc = proc.wait(timeout=max(0.0, min(5.0, limit - time.time())))
            return ended("ok" if rc == 0 else "failed", rc)
        except subprocess.TimeoutExpired:
            pass
        if junit is not None and seen is None:
            m = _mtime(junit)
            if m is not None and m >= t0:
                seen = m
        hung = seen is not None and time.time() - seen >= grace_s
        if hung or time.time() >= limit:
            evidence()
            _kill(proc)
            return ended("exit_hang" if hung else "limit", proc.returncode)


def _run_part(part: Part, env: dict[str, str], cwd: Path, log: Path, *, deadline: float,
              python: str, current: Path | None = None) -> tuple[str, int | None, float]:
    """Run one part to its end, its time limit or the deadline: (status, exit code, wall s).

    ``current``, while the part runs, holds its process group id, so ``fs11_job.sh`` can kill
    the part too when the session itself hangs. A sampler (:func:`_start_sampler`) writes the
    part's ``samples.jsonl`` beside its log. A pytest part (``info["junit"]``) gets
    ``POTS_STACKS_FILE=parts/<name>/stacks.txt``, and :func:`watch_process` ends it as
    ``exit_hang`` when it outlives its junit report by :data:`EXIT_HANG_GRACE_S`; at that
    kill and at a time-limit or deadline kill ``parts/<name>/hang.txt`` holds the process
    snapshot, and, for a pytest part, ``stacks.txt`` every thread's stack."""
    argv = [python if a == "{python}" else a for a in part.argv]
    penv = dict(env)
    for k, v in part.env.items():
        if v is None:
            penv.pop(k, None)
        else:
            penv[k] = v
    junit = Path(part.info["junit"]) if part.info.get("junit") else None
    if junit is not None:
        penv[STACKS_ENV] = str(log.parent / "stacks.txt")
        penv[GRACE_ENV] = str(EXIT_HANG_GRACE_S + GUARD_LAG_S)
    t0 = time.time()
    limit = deadline if part.timeout_s is None else min(deadline, t0 + part.timeout_s)
    with open(log, "ab") as f:
        f.write(f"$ {' '.join(argv)}\n".encode())
        f.flush()
        proc = subprocess.Popen(argv, cwd=cwd, env=penv, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        if current is not None:  # before the sampler, whose start can take a while
            try:
                current.write_text(f"{proc.pid}\n")
            except OSError:
                pass
        sampler = _start_sampler(proc, log.parent / "samples.jsonl", cwd, python)
        peaks = _GpuPeaks(log.parent / "gpu_peak.json") if part.info.get("gpu_peak") else None
        try:
            w = watch_process(proc, t0=t0, limit=limit, junit=junit, grace_s=EXIT_HANG_GRACE_S,
                              hang_file=log.parent / "hang.txt", stacks=junit is not None)
            status = ("killed_deadline" if limit == deadline else "killed_timeout"
                      ) if w.status == "limit" else w.status
            if w.status in ("limit", "exit_hang"):
                what = ("hung at exit, " + f"{w.junit_to_exit_s:.0f} s after its junit report"
                        if w.status == "exit_hang" else f"after {time.time() - t0:.0f} s")
                f.write(f"\n[fs11] {status} {what}\n".encode())
            return status, w.rc, time.time() - t0
        except BaseException:  # SIGTERM to the session (_Stop) or Ctrl-C: take the part down too
            _kill(proc)
            f.write(f"\n[fs11] killed: the session was stopped after {time.time() - t0:.0f} s\n"
                    .encode())
            raise
        finally:
            _stop_sampler(sampler)
            if peaks is not None:
                peaks.stop()
            if current is not None:
                current.unlink(missing_ok=True)


# ---- the exit probe ---------------------------------------------------------------------

EXIT_PROBE_REPORT = "exit_probe.json"
EXIT_PROBE_RUN_S = 900  # the limit of one run alone; the whole-list run gets what is left
EXIT_PROBE_MIN_S = 120  # no run starts with less of the part's budget left than this
# The modules the exit probe runs, each alone and then all together (30). Chosen from evidence:
# the fast suite's process hung after its summary on the 2026-10-23 two-A100 session; on one A40
# the measuring-kernel session ran the last nine of these (and others) through the session's
# pytest parts and each exited 2-3 s after its summary, as did the two-A100 ``fs34_tests`` and
# ``two_gpu_slow`` parts. So the suspects come first: the two-device modules that run beside
# those parts but not in them (``test_shared_cuda``, ``test_sharded_run_cuda``), then the modules
# that start child processes (the pipelined loop, the shared transcript, FS33d's entry points,
# the metrics and FS34 modules), then the other modules with CUDA tests or tests that skip on a
# machine with fewer than two cards. The A40 modules close the list: they are the ones known to
# exit on one card, so a hang there points at the second card.
EXIT_PROBE_MODULES = (
    "tests/verification/transcript/test_shared_cuda.py",
    "tests/verification/runs/test_sharded_run_cuda.py",
    "tests/verification/runs/test_pipelined_loop.py",
    "tests/verification/runs/test_pipelined_loop_tester.py",
    "tests/verification/transcript/test_shared.py",
    "tests/verification/transcript/test_shared_tester.py",
    "tests/verification/runs/test_fs34_join_cuda.py",
    "tests/verification/runs/test_fs33d_entry_points.py",
    "tests/verification/runs/test_fs33d_entry_points_tester.py",
    "tests/verification/runs/test_metrics_devices.py",
    "tests/verification/runs/test_metrics_devices_tester.py",
    "tests/verification/commitment/test_fs34_routing.py",
    "tests/verification/commitment/test_fs34_tester.py",
    "tests/verification/runs/test_fs33_fix1.py",
    "tests/verification/runs/test_fs33_fix2.py",
    "tests/verification/runs/test_peer_check.py",
    "tests/verification/commitment/test_hostcopy.py",
    "tests/verification/commitment/test_leaves_native.py",
    "tests/verification/computation/test_decoder_split_replay.py",
    "tests/verification/prover/test_capture_routing.py",
    "tests/verification/runs/test_device_leaves_session_tester.py",
    "tests/verification/verifier/matmul_check/test_measure_kernel_cuda.py",
    "tests/verification/verifier/test_kernel_step.py",
    "tests/verification/prover/test_capture_cuda.py",
    "tests/verification/verifier/test_check_devices.py",
    "tests/verification/runs/test_loop.py",
    "tests/verification/runs/test_pipelined_loop_cuda.py",
    "tests/verification/runs/test_metrics_devices_cuda.py",
    "tests/verification/runs/test_device_leaves_cuda.py",
    "tests/verification/verifier/matmul_check/test_freivalds.py",
)


def run_exit_probe(out: Path, modules: Sequence[str], *, cwd: Path, python: str,
                   budget_s: float, grace_s: float = PROBE_HANG_GRACE_S,
                   run_limit_s: float = EXIT_PROBE_RUN_S,
                   pytest_args: Sequence[str] = ("-rf", "-q"),
                   say: Callable[[str], None] = _say) -> dict[str, Any]:
    """Find what keeps a pytest process from exiting (the pair session's fast suite hung for
    the rest of its time limit after its summary). Each of ``modules`` (test files, relative
    to ``cwd``) runs alone in its own pytest process, one after another, each with its own
    junit report, ``POTS_STACKS_FILE`` and :func:`watch_process` at ``grace_s``; if no module
    hung alone, all of them run in one process last, to see whether the hang needs two of them
    together. ``out/<run>/`` holds a run's ``junit.xml``, ``log.txt``, ``stacks.txt`` and, for a
    hang, ``hang.txt``. ``out/exit_probe.json`` (written after every run) has, per run, the
    modules, the tests, failures, errors and skips, the status (``ok``, ``failed`` when a test
    or the process failed, ``exit_hang``, ``killed_timeout``), the seconds from the junit
    report to the exit and the snapshot and stacks files. A diagnostic: a failed test in it is
    reported, not judged. Returns the report. A run starts only while ``budget_s`` has at least
    :data:`EXIT_PROBE_MIN_S` left; the rest are recorded as ``not_started``."""
    t_start = time.time()
    runs: list[dict[str, Any]] = []
    report: dict[str, Any] = {"modules": list(modules), "runs": runs, "hung": [],
                              "whole_list": False}

    def one(name: str, mods: Sequence[str], cap_s: float | None) -> dict[str, Any]:
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        junit = d / "junit.xml"
        junit.unlink(missing_ok=True)
        left = budget_s - (time.time() - t_start)
        rec: dict[str, Any] = {"name": name, "modules": list(mods)}
        if left < EXIT_PROBE_MIN_S:
            return {**rec, "status": "not_started", "reason": "the part's budget was spent"}
        t0 = time.time()
        limit = t0 + (left if cap_s is None else min(left, cap_s)) - 5
        env = {**os.environ, STACKS_ENV: str(d / "stacks.txt"),
               GRACE_ENV: str(grace_s + GUARD_LAG_S)}
        with open(d / "log.txt", "ab") as f:
            proc = subprocess.Popen([python, "-m", "pytest", *pytest_args,
                                     f"--junitxml={junit}", *mods], cwd=cwd, env=env,
                                    stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                w = watch_process(proc, t0=t0, limit=limit, junit=junit, grace_s=grace_s,
                                  hang_file=d / "hang.txt", stacks=True)
            except BaseException:
                _kill(proc)
                raise
        counts = _pytest_extras(junit)
        return {**rec, "status": "killed_timeout" if w.status == "limit" else w.status,
                "exit_code": w.rc, "wall_s": round(time.time() - t0, 1),
                "tests": counts.get("tests"), "failures": counts.get("failures"),
                "errors": counts.get("errors"), "skipped": counts.get("skipped"),
                "junit_to_exit_s": None if w.junit_to_exit_s is None
                else round(w.junit_to_exit_s, 1),
                "snapshot": str(d / "hang.txt") if (d / "hang.txt").exists() else None,
                "stacks": str(d / "stacks.txt") if (d / "stacks.txt").exists() else None,
                "log": str(d / "log.txt")}

    def record(rec: dict[str, Any]) -> None:
        runs.append(rec)
        report["hung"] = [r["name"] for r in runs if r["status"] == "exit_hang"]
        _write_json(out / EXIT_PROBE_REPORT, report)
        say(f"exit probe {rec['name']}: {rec['status']}"
            + ("" if rec.get("tests") is None else f", {rec['tests']} tests"))

    for mod in modules:
        record(one(Path(mod).stem, [mod], run_limit_s))
    if not report["hung"] and modules:
        report["whole_list"] = True
        record(one("whole_list", modules, None))
    _write_json(out / EXIT_PROBE_REPORT, report)
    return report


def exit_probe_lines(report: Mapping[str, Any] | None) -> list[str]:
    """The job log's lines for the probe's report: one per run that hung at exit, then the
    summary line (``none hung`` when none did). A run that was not started is not counted."""
    if not report:
        return ["FS32 exit probe: left no report"]
    ran = [r for r in report.get("runs", []) if r.get("status") != "not_started"]
    hung = [r for r in ran if r.get("status") == "exit_hang"]
    lines = [f"FS32 exit probe: {r['name']} ({', '.join(r['modules'])}) hung at exit, "
             f"{r.get('junit_to_exit_s')} s after its junit report; snapshot {r.get('snapshot')}, "
             f"stacks {r.get('stacks')}" for r in hung]
    skipped = len(report.get("runs", [])) - len(ran)
    tail = f" ({skipped} not started)" if skipped else ""
    lines.append(f"FS32 exit probe: {len(hung)} of {len(ran)} runs hung at exit: "
                 + (", ".join(r["name"] for r in hung) if hung else "none hung") + tail)
    return lines


def run_session(plan: Sequence[Part], *, upload: Path, out: Path, deadline: float, cwd: Path,
                python: str = sys.executable, env_doc: Mapping[str, Any] | None = None,
                say: Callable[[str], None] = _say, session: str = "FS11") -> dict[str, Any]:
    """Run ``plan`` in order (module docstring) and return the summary, also written to
    ``out/summary.json`` after each part. ``session`` is the summary's name for the session
    (:data:`SESSIONS`). A plan with a fix-off bring-up also gets ``fix2_ab`` (:func:`fix2_ab`),
    and the job log one line per step with the two arms' totals; a plan with card parts gets
    ``hw3_cards`` (:func:`hw3_cards`), and the hardware probe's job log ends with a line each for
    HW2 and HW3."""
    env = base_env(upload)
    summary: dict[str, Any] = {
        "session": session, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)),
        "upload": str(upload), "out": str(out), "environment": env_doc,
        "cupqc": cupqc_status(), "parts": [], "finished": False,
    }
    say(f"cuPQC: {summary['cupqc']['reason']} ({', '.join(CUPQC_SLOTS)})")
    path = out / "summary.json"
    check: dict[str, Any] | None = None  # check_upload's result, once the models part is past
    for part in plan:
        d = out / "parts" / part.name
        d.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = {"name": part.name, "kind": part.kind, "info": part.info,
                               "log": str(d / "log.txt")}
        why = blocked(part, check)
        if summary.get("stopped"):
            rec.update(status="not_started", exit_code=None, reason=summary["stopped"])
        elif (d / "DONE").exists():
            # Done in an earlier run, whose own check passed: its result stands.
            rec.update(status="skipped_done", exit_code=0)
            say(f"{part.name}: done in an earlier run, skipped")
        elif why:
            rec.update(status="not_started", exit_code=None, reason=why)
            say(f"{part.name}: not started, {why}")
        elif time.time() >= deadline:
            rec.update(status="not_started", exit_code=None,
                       reason="the session deadline passed before it started")
            say(f"{part.name}: not started, the deadline passed")
        else:
            say(f"{part.name}: start (log {d / 'log.txt'})")
            t0 = time.time()
            try:
                status, rc, wall = _run_part(part, env, cwd, d / "log.txt", deadline=deadline,
                                             python=python, current=out / "current_part")
            except (_Stop, KeyboardInterrupt) as e:
                # Record the cut; the parts after it then say why they didn't start.
                summary["stopped"] = f"the session was stopped ({e or 'SIGINT'})"
                status, rc, wall = "killed_signal", None, time.time() - t0
            rec.update(status=status, exit_code=rc, wall_s=round(wall, 1))
            if status == "ok" and part.kind == "benchmark":
                # benchmarks.hashing.run exits 0 whatever its slots did; its report decides.
                bad = benchmark_failure(Path(part.info["report"]))
                if bad:
                    status = "failed"
                    rec.update(status=status, reason=bad)
            if status == "ok":
                (d / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
            say(f"{part.name}: {status}, exit {rc}, {wall:.0f} s"
                + (f" ({rec['reason']})" if rec.get("reason") else ""))
        if part.kind == "benchmark" and rec["status"] != "skipped_done":
            with open(d / "log.txt", "a") as f:
                f.write(f"[fs11] cuPQC slots {', '.join(CUPQC_SLOTS)}: "
                        f"{summary['cupqc']['reason']}\n")
        rec["extras"] = part_extras(part, d / "log.txt")
        if part.info.get("gpu_peak"):
            rec["extras"]["gpu_peak_mib"] = _read_peaks(d / "gpu_peak.json")
        if part.info.get("samples_check"):
            rec["extras"]["samples"] = _samples_header(d / "samples.jsonl")
        summary["parts"].append(rec)
        try:
            ab = fix2_ab(summary["parts"])
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            # as part_extras: a malformed entry costs the A/B, not the session
            summary["fix2_ab"] = {"fix2_ab_error": f"{type(e).__name__}: {e}"}
            ab = None
        if ab is not None:
            summary["fix2_ab"] = ab
            for r in ab["steps"] if part.info.get("arm") == "fix2_off" else ():
                if r["fix_cost"]["total_s"] is not None:
                    say(f"allocator fix A/B, step {r['step']}: {r['fix_on']['total_s']:.2f} s "
                        f"with the fix, {r['fix_off']['total_s']:.2f} s without "
                        f"({r['fix_cost']['total_s']:+.2f} s)")
        try:
            hw3 = hw3_cards(summary["parts"])
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            hw3 = {"hw3_cards_error": f"{type(e).__name__}: {e}"}
        if hw3 is not None:
            summary["hw3_cards"] = hw3
        try:
            pair = fs32_pair(summary["parts"])
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            pair = {"fs32_pair_error": f"{type(e).__name__}: {e}"}
        if pair is not None:
            summary["fs32_pair"] = pair
        try:
            four = four_h100(summary["parts"])
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError,
                statistics.StatisticsError) as e:
            four = {"four_h100_error": f"{type(e).__name__}: {e}"}
        if four is not None:
            summary["four_h100"] = four
        try:
            profile = four_h100_profile(summary["parts"])
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError,
                statistics.StatisticsError) as e:
            profile = {"four_h100_profile_error": f"{type(e).__name__}: {e}"}
        if profile is not None:
            summary["four_h100_profile"] = profile
        if part.kind == "peer_check" and rec["status"] != "not_started":
            # Wrong bytes between the cards: nothing after this part counts. A rerun that
            # skipped the part (DONE) reads its report again.
            why_stop = peer_check_stop(rec)
            if why_stop and not summary.get("stopped"):
                summary["stopped"] = why_stop
                say(f"STOP: {why_stop}")
        if part.kind == "models" and rec["status"] != "not_started":
            # On every run, a rerun that skipped the part included: the later parts read
            # these files, and a DONE marker says nothing about them now.
            check = check_upload(upload)
            summary["upload_check"] = check
            say(f"upload check ({check['s']:.0f} s): {len(check['packed'])} packed files "
                f"missing or differing; models with problems: "
                f"{', '.join(check['models']) or 'none'}")
        _write_json(path, summary)
    speed_gate = kernel_speed_gate(summary["parts"])
    if speed_gate is not None:
        summary["kernel_speed_gate"] = speed_gate
    leaves_gate = device_leaves_gate(summary["parts"])
    if leaves_gate is not None:
        summary["device_leaves"] = leaves_gate
    for line in _hw_lines(summary["parts"], summary.get("hw3_cards")):
        say(line)
    pair = summary.get("fs32_pair")
    if pair is not None and "fs32_pair_error" in pair:
        say(f"FS32 FAIL: the comparison raised ({pair['fs32_pair_error']})")
    else:
        for line in _pair_lines(pair):
            say(line)
    four = summary.get("four_h100")
    if four is not None and "four_h100_error" in four:
        say(f"FS35 FAIL: the comparison raised ({four['four_h100_error']})")
    else:
        for line in _four_lines(four):
            say(line)
    prof = summary.get("four_h100_profile")
    if prof is not None and "four_h100_profile_error" in prof:
        say(f"FS42 FAIL: the acceptance raised ({prof['four_h100_profile_error']})")
    else:
        for line in _profile_lines(prof):
            say(line)
    summary["finished"] = True
    summary["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["failed"] = [p["name"] for p in summary["parts"]
                         if p["status"] not in ("ok", "skipped_done")]
    _write_json(path, summary)
    return summary


# ---- fetch-models -----------------------------------------------------------------------


def _hf_download(repo: str, filename: str, revision: str, hub: Path) -> Path:
    from huggingface_hub import hf_hub_download

    # token=False: no token is sent, whatever the environment holds.
    return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=hub, token=False))


def _drop(path: Path) -> None:
    """Delete a snapshot file and the blob it links to."""
    blob = Path(os.path.realpath(path))
    for p in (path, blob):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def snapshot_path(hub: Path, repo: str, revision: str, rel: str) -> Path:
    return hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / rel


def packed_problems(upload: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Each packed file of ``upload`` that is missing or differs from the manifest's SHA-256."""
    bad = []
    for arc, digest in manifest["files"].items():
        p = upload / arc
        if not p.is_file():
            bad.append(f"{arc}: missing")
        elif sha256_file(p) != digest:
            bad.append(f"{arc}: SHA-256 differs from the manifest")
    return bad


def check_upload(upload: Path) -> dict[str, Any]:
    """Every file the later parts read, against the manifest: the packed files, and each
    model's fetched weights in the upload's Hugging Face cache. The session runs it after the
    models part on every run, a rerun that skips that part included.

    ``packed`` lists the packed files that are missing or differ (or an unreadable manifest):
    the upload can't be trusted, and no later part starts. ``models`` maps a repo to its
    problems (missing on the Mac, a weight not fetched or not matching): that model's parts
    don't start. ``native`` says whether the native library's pinned sources are present
    and intact (``bad`` lists the files that aren't); it blocks nothing, the build part
    fails with its own message. The benchmark still runs; a model whose weights are missing
    fails to load there and says so in its row."""
    t0 = time.time()
    models: dict[str, list[str]] = {}
    try:
        manifest = json.loads((upload / "manifest.json").read_text())
        packed = packed_problems(upload, manifest)
        hub = upload / "hf" / "hub"
        for m in manifest["models"]:
            # A benchmark-only pin may lack weights on purpose (the unsloth mirror's configs
            # size E6's leaves; its row times Meta's weights), so only a run model's count.
            probs = [f"missing on the Mac: {x}" for x in m.get("missing") or []
                     if "run" in m.get("roles", ["run"])]
            for f in m["files"]:
                if f["packed"]:
                    continue  # in manifest["files"], checked above
                p = snapshot_path(hub, m["repo"], m["revision"], f["path"])
                if not p.is_file():
                    probs.append(f"{f['path']}: not in the cache")
                elif sha256_file(p) != f["sha256"]:
                    probs.append(f"{f['path']}: SHA-256 differs from the Mac's")
            if probs:
                models[m["repo"]] = probs
    except (OSError, ValueError, KeyError, TypeError) as e:
        packed = [f"manifest.json unreadable: {type(e).__name__}: {e}"]
    from verification.commitment import native  # imports numpy and torch: not at module level

    src = native.source_dir(upload / "native_cache")
    bad = native.bad_sources(src) if src.is_dir() else ["the directory"]
    return {"packed": packed, "models": models,
            "native": {"dir": f"native_cache/{src.name}", "present": src.is_dir(), "bad": bad},
            "s": round(time.time() - t0, 1)}


def blocked(part: Part, check: Mapping[str, Any] | None) -> str | None:
    """Why :func:`check_upload`'s result keeps ``part`` from starting, or ``None``."""
    if check is None:
        return None
    if check["packed"]:
        return (f"the upload doesn't match its manifest ({len(check['packed'])} files, e.g. "
                f"{check['packed'][0]})")
    model = part.info.get("model")
    if model in check["models"]:
        probs = check["models"][model]
        return f"{model}'s files are missing or differ ({len(probs)}, e.g. {probs[0]})"
    return None


def fetch_models(upload: Path, out: Path, *,
                 download: Callable[[str, str, str, Path], Path] = _hf_download,
                 say: Callable[[str], None] = _say) -> int:
    """Check the upload and fetch the public weights (module docstring, part 1). Writes
    ``out/models.json``; returns 0 when every file is present and matches."""
    manifest = json.loads((upload / "manifest.json").read_text())
    hub = upload / "hf" / "hub"
    problems: list[str] = []
    t0 = time.time()
    bad = packed_problems(upload, manifest)
    problems += bad
    say(f"upload: {len(manifest['files']) - len(bad)} of {len(manifest['files'])} packed files "
        f"match the manifest ({time.time() - t0:.0f} s)")
    fetched = []
    for m in manifest["models"]:
        repo, rev = m["repo"], m["revision"]
        for f in m["files"]:
            if f["packed"]:
                continue
            p = snapshot_path(hub, repo, rev, f["path"])
            if repo in GATED or m.get("gated"):
                problems.append(f"{repo}/{f['path']}: gated, never fetched on the pod")
                continue
            if p.is_file() and sha256_file(p) == f["sha256"]:
                fetched.append({"repo": repo, "path": f["path"], "status": "present"})
                continue
            t1 = time.time()
            try:
                p = download(repo, f["path"], rev, hub)
            except Exception as e:  # recorded; the parts that need the file fail on their own
                problems.append(f"{repo}/{f['path']}: fetch failed: {type(e).__name__}: {e}")
                continue
            got = sha256_file(p)
            ok = got == f["sha256"]
            if not ok:
                _drop(p)
                problems.append(f"{repo}/{f['path']}: fetched SHA-256 {got}, the Mac's "
                                f"{f['sha256']}; deleted")
            fetched.append({"repo": repo, "path": f["path"], "status": "fetched" if ok else
                            "mismatch", "bytes": f["bytes"], "s": round(time.time() - t1, 1)})
            say(f"{repo}/{f['path']}: {'fetched and matches' if ok else 'MISMATCH, deleted'} "
                f"({time.time() - t1:.0f} s)")
    doc = {"files_checked": len(manifest["files"]), "fetched": fetched, "problems": problems,
           "missing_on_mac": manifest.get("missing", [])}
    _write_json(out / "models.json", doc)
    for p in problems:
        say(f"problem: {p}")
    return 0 if not problems else 1


# ---- the command line -------------------------------------------------------------------


def _on_term(signum: int, frame: Any) -> None:
    raise _Stop(signal.Signals(signum).name)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the session's parts in order")
    r.add_argument("--upload", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--deadline", type=float, required=True,
                   help="Unix time at which a running part is killed and no other starts")
    r.add_argument("--plan", type=Path, default=None,
                   help="the parts as JSON instead of the session's (for a dry run)")
    r.add_argument("--selftest", action="store_true",
                   help="FS11_SELFTEST: stand-in parts (unless --plan), no GPU check")
    r.add_argument("--session", choices=sorted(SESSIONS), default="fs11",
                   help="FS11_SESSION: the first session's plan (fs11, the default), the "
                        "re-check's (recheck), the profile check's (profile-check), the "
                        "four-card hardware probe's (nvlink-probe), the BLAKE3 kernel's speed "
                        "(kernel-speed), the two-GPU sharded run's (fs32-pair), FS30's "
                        "device-leaves check (device-leaves), the four-H100 session's "
                        "(four-h100) or its profiling run's (four-h100-profile); --plan and "
                        "--selftest replace its parts")
    x = sub.add_parser("exit-probe", help="run each test module in its own pytest process "
                                          "and report which one cannot exit")
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("--budget-s", type=float, required=True,
                   help="seconds in all; a run starts only with some of them left")
    x.add_argument("--modules", nargs="+", default=list(EXIT_PROBE_MODULES),
                   help="test files, relative to the working directory")
    f = sub.add_parser("fetch-models", help="check the upload, fetch the public weights")
    f.add_argument("--upload", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("env", help="print the environment and the H100 check")
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "fetch-models":
        return fetch_models(args.upload.resolve(), args.out)
    if args.cmd == "exit-probe":
        signal.signal(signal.SIGTERM, _on_term)
        try:
            run_exit_probe(args.out.resolve(), args.modules, cwd=Path.cwd(),
                           python=sys.executable, budget_s=args.budget_s)
        except _Stop:
            return 1
        return 0
    out = args.out.resolve()
    selftest = args.cmd == "run" and args.selftest
    pairing = False
    label, build = SESSIONS[args.session] if args.cmd == "run" else SESSIONS["fs11"]
    allow = os.environ.get("FS11_ALLOW_ANY_GPU") == "1" or selftest
    pin = pin_to_gpu_numa()  # before anything starts threads; the parts inherit it
    env_doc = environment(out)
    quota, usable = cgroup_cpu_quota(), _usable_cpus()
    env_doc["cpu_pin"] = {**{k: v for k, v in pin.items() if k != "message"},
                          "cgroup_cpu_quota": quota, "usable_cpus": usable,
                          "threads": _threads(os.environ)}
    _write_json(out / "environment.json", env_doc)
    print_environment(env_doc)
    _say(f"{pin['message']}; threads {env_doc['cpu_pin']['threads']}"
         f" (cgroup CPU quota {'none' if quota is None else f'{quota:g}'})")
    try:
        probing = args.cmd == "run" and args.session == "nvlink-probe" and not selftest
        pairing = args.cmd == "run" and args.session == "fs32-pair" and not selftest
        four = (args.cmd == "run" and args.session in ("four-h100", "four-h100-profile")
                and not selftest)
        if pairing:  # any GPU type, but two of them
            allow = True
        # the four-H100 session needs four CUDA devices (FS11_ALLOW_ANY_GPU does not lift that)
        # and H100s unless FS11_ALLOW_ANY_GPU=1
        why = refusal(env_doc, allow, fs31_cards() if probing else PAIR_CARDS if pairing
                      else FOUR_CARDS if four else None,
                      need="the fs32-pair session" if pairing
                      else f"the {args.session} session" if four else None)
    except ValueError as e:
        why = str(e)
    py_refused = False
    if args.cmd == "run" and not selftest and not why:
        why = python_mismatch(args.upload.resolve(), platform.python_version())
        py_refused = bool(why)
        if why and os.environ.get(PYTHON_MISMATCH_ENV) == "1":
            _say(f"{PYTHON_MISMATCH_ENV}=1: {why}")
            why, py_refused = None, False
    if selftest:
        _say(f"self-test: stand-in parts in place of the {label} session's, the H100 check is "
             "skipped")
    elif pairing:
        _say("fs32-pair: any GPU type, at least 2 CUDA devices")
    elif four and allow:
        _say(f"FS11_ALLOW_ANY_GPU=1: the H100 check is overridden, {FOUR_CARDS} CUDA devices "
             "are still needed")
    elif allow:
        _say("FS11_ALLOW_ANY_GPU=1: the H100 check is overridden")
    if args.cmd == "env":
        _say("refused: " + why if why else "the H100 check passes")
        return EXIT_REFUSED if why else 0
    if why:
        override = PYTHON_MISMATCH_ENV if py_refused else "FS11_ALLOW_ANY_GPU"
        _say(f"refused to start: {why} ({override}=1 overrides)")
        _write_json(out / "summary.json", {"session": label, "refused": why,
                                           "environment": env_doc, "parts": [],
                                           "finished": True})
        return EXIT_REFUSED
    upload = args.upload.resolve()
    plan = choose_plan(build, upload, out, plan_file=args.plan, selftest=selftest)
    _say(f"session {label}: {len(plan)} part{'' if len(plan) == 1 else 's'}")
    signal.signal(signal.SIGTERM, _on_term)
    summary = run_session(plan, upload=upload, out=out, deadline=args.deadline,
                          cwd=Path.cwd(), env_doc=env_doc, session=label)
    failed = summary["failed"]
    _say("all parts succeeded" if not failed else f"failed or cut short: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
