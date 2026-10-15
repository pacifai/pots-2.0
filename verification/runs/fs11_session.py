"""FS11's H100 sessions: the parts in order, each with its own log, under one deadline.

    .venv/bin/python -m verification.runs.fs11_session run --upload DIR --out DIR --deadline EPOCH
                                                           [--session {fs11,recheck}]
    .venv/bin/python -m verification.runs.fs11_session fetch-models --upload DIR --out DIR
    .venv/bin/python -m verification.runs.fs11_session env --out DIR

``fs11_job.sh`` runs ``run`` on the pod, then packs the output and ends the pod; see
``verification/README.md``, "Workflow: the first H100 session". ``fs11_prep`` builds the upload
on the Mac; both sessions run from the same upload. ``--session`` (the job's ``FS11_SESSION``)
picks the plan: ``fs11``, the first session (:func:`default_plan`, the default), or
``recheck``, the re-check (:func:`recheck_plan`, below). The first session's parts, in this
order:

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
CUDA allocator's expandable segments (``PYTORCH_CUDA_ALLOC_CONF``, which the allocator reads at
CUDA's first use, so it is set in the part's environment) and the cache release between prover
and verifier (``VERIF_RELEASE_CACHE=1``, the default, pinned). Right after
:data:`FIX_AB_MODEL`'s bring-up comes the same bring-up with the whole fix off
(:data:`FIX_OFF_ENV`), its own part and run directory: ``summary.json``'s ``fix2_ab``
(:func:`fix2_ab`) sets the two side by side, so the fix's cost to a verified step reads off
directly. Last comes a timing A/B of the allocator setting alone: per model,
``plain_baseline``'s 10 plain steps twice, once with ``PYTORCH_CUDA_ALLOC_CONF`` unset and
once with expandable segments, the first arm alternating from model to model
(:data:`ALLOC_ARMS`). It has no benchmark part.

Every part runs in its own process group, with ``HF_HUB_OFFLINE=1`` (all but the first), no
Hugging Face token and no Runpod key in its environment. A part's log is
``parts/<name>/log.txt``, and ``current_part`` holds the running part's process group id.
``parts/<name>/DONE`` is written when it succeeds, and a rerun into the same output directory
skips it. A part that fails doesn't stop the others. A part still running at ``--deadline``
(or past its own time limit) is killed, and the parts after it are recorded as not started.
``summary.json`` is rewritten after each part.

Before any part, ``run`` prints the environment (GPU, driver, CUDA, torch, free disk and RAM)
to ``environment.json`` and refuses to start unless CUDA is available and the GPU is an H100;
``FS11_ALLOW_ANY_GPU=1`` overrides. ``run --selftest`` (``FS11_SELFTEST``) skips that check
and runs :func:`selftest_plan`'s stand-in parts, which need only the standard library. Exits 0
when every part succeeded, 1 when any failed, and :data:`EXIT_REFUSED` when the environment
check refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

__all__ = ["RUN_PINS", "GATED", "TASKS", "FULL_SCALE_ENV", "RECHECK_ENV", "ALLOC_ARMS",
           "FIX_AB_MODEL", "FIX_OFF_ENV", "SESSIONS", "EXIT_REFUSED", "Part", "default_plan",
           "recheck_plan", "load_plan", "selftest_plan", "run_session", "fix2_ab",
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
RECHECK_ENV = {"VERIF_HEAD_BLOCKS": "8", "PYTORCH_CUDA_ALLOC_CONF": EXPANDABLE,
               "VERIF_RELEASE_CACHE": "1"}
# The verified-step A/B of the whole allocator fix: this model's bring-up runs again right
# after its own, with RECHECK_ENV's two fix settings undone (None unsets). The fix is kept only
# if it costs no speed; Qwen2.5-0.5B fit in the first session without it, so this arm runs.
FIX_AB_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FIX_OFF_ENV: dict[str, str | None] = {"PYTORCH_CUDA_ALLOC_CONF": None, "VERIF_RELEASE_CACHE": "0"}
# The re-check's allocator A/B: each arm's name and its PYTORCH_CUDA_ALLOC_CONF (None unsets it,
# whatever the pod's environment holds). Even-numbered models in RUN_PINS run them in this
# order, odd-numbered ones in reverse, so an order effect (a warm file cache, the GPU's clocks)
# doesn't always favour the same arm.
ALLOC_ARMS: tuple[tuple[str, str | None], ...] = (("alloc_default", None),
                                                   ("alloc_expandable", EXPANDABLE))
# The re-check's time limits: about 3–5 times each part's expected time (the first session's
# smooth parts took 25–44 s, its one finished bring-up 191 s; a plain arm takes about a minute).
RECHECK_TIMEOUT_S = {"models": 1800, "bring_up": 1800, "smooth": 600, "plain": 600}
PLAIN_RUN = "plain_baseline"  # plain_baseline.RUN_NAME, its records' directory
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


def refusal(env: Mapping[str, Any], allow_any_gpu: bool) -> str | None:
    """Why the session won't start on this machine, or ``None``."""
    if allow_any_gpu:
        return None
    if not env["cuda_available"]:
        return "torch.cuda.is_available() is false"
    if not any("H100" in g for g in env["gpus"]):
        return f"no H100 among the GPUs {env['gpus']}"
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
    kind: str = "other"  # models, benchmark, bring_up, smooth, plain, other
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
                env={"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None},
                timeout_s=timeout_s, kind="models",
                info={"report": str(out / "models" / "models.json")})


def _bring_up_parts(upload: Path, out: Path, extra: Mapping[str, str | None],
                    timeout_s: float) -> list[Part]:
    """One ``bring_up --no-tamper`` per model, on the refusal task, with ``extra`` added to the
    full-scale config."""
    parts = []
    for repo, rev in RUN_PINS:
        key = data_key(repo, BRING_UP_TASK)
        d = _run_dir(out, upload, key)
        parts.append(Part(f"bring_up_{repo.split('/')[-1]}",
                          ["{python}", "-m", "verification.runs.bring_up", "--no-tamper"],
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
                                   "PYTORCH_CUDA_ALLOC_CONF": conf, "VERIF_OUTPUT_DIR": str(d)},
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
    followed by its fix-off arm; then the allocator A/B. No benchmark."""
    plan = [_models_part(upload, out, RECHECK_TIMEOUT_S["models"])]
    for on in _bring_up_parts(upload, out, RECHECK_ENV, RECHECK_TIMEOUT_S["bring_up"]):
        plan.append(on)
        if on.info["model"] == FIX_AB_MODEL:
            on.info.update(arm="fix2_on", alloc_conf=EXPANDABLE, release_cache=True)
            plan.append(_fix_off_part(on, upload, out))
    plan += _smooth_parts(upload, out, RECHECK_ENV, RECHECK_TIMEOUT_S["smooth"])
    plan += _plain_parts(upload, out)
    return plan


# Each session: its name in summary.json, and its plan.
SESSIONS: dict[str, tuple[str, Callable[[Path, Path], list[Part]]]] = {
    "fs11": ("FS11", default_plan),
    "recheck": ("recheck", recheck_plan),
}


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


def base_env(upload: Path) -> dict[str, str]:
    """Every part's environment: the session's own, without :data:`SECRETS`, offline, with the
    upload's Hugging Face cache and BLAKE3 sources."""
    env = {k: v for k, v in os.environ.items() if k not in SECRETS}
    threads = env.get("FS11_THREADS", "8")
    env.update({
        "HF_HOME": str(upload / "hf"), "HF_HUB_CACHE": str(upload / "hf" / "hub"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "POTS_BENCH_CACHE": str(upload / "bench_cache"),
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


def _bring_up_extras(log: Path) -> dict[str, Any]:
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    peaks = [{"step": int(m[1]), "side": m[2], "peak_gb": float(m[3]),
              "reserved_gb": None if m[4] is None else float(m[4])}
             for m in map(_PEAK.match, lines) if m]
    return {"peaks": peaks, "step_times": _step_times(lines),
            "section_peaks": _section_peaks(lines),
            "lines": [l for l in lines if _BRING_UP_LINE.match(l)][-40:]}


def _plain_extras(report: Path) -> dict[str, Any]:
    """A plain arm's ``records.jsonl``: the timed run's per-step times (the ``step`` total of
    each step's ``P0`` phases, which on CUDA synchronize at their bounds), their mean and
    median, and the memory pass's peak (on CUDA with its reserved peak, else ``None``)."""
    if not report.exists():
        return {"record": None}
    rows = [json.loads(line) for line in report.read_text().splitlines() if line.strip()]
    steps = sorted((r["step"], r["time_s"]) for r in rows
                   if r.get("record") == "time" and r.get("level") == "total"
                   and r.get("component") == "step" and r.get("step", 0) > 0)
    times = [s for _, s in steps]
    mem = [r for r in rows if r.get("record") == "memory" and r.get("peak_bytes") is not None]
    held = [r["peak_reserved_bytes"] for r in mem if r.get("peak_reserved_bytes") is not None]
    end = next((r for r in rows if r.get("record") == "run_end"), None)
    return {"record": str(report), "completed": end is not None, "step_s": times,
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
    return {"record": str(report), "smooth": doc.get("smooth"), "reasons": doc.get("reasons"),
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
        if part.kind == "models":
            report = Path(part.info["report"])
            return json.loads(report.read_text()) if report.exists() else {"report": None}
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"extras_error": f"{type(e).__name__}: {e}"}
    return {}


_FIX_TIMES = ("prover_s", "verifier_s", "handoff_s", "total_s")
_FIX_PEAKS = ("peak_gb", "reserved_gb")


def _fix_cost(on: Mapping[str, Any] | None, off: Mapping[str, Any] | None,
              keys: Sequence[str]) -> dict[str, float | None]:
    return {k: None if on is None or off is None or on.get(k) is None or off.get(k) is None
            else round(on[k] - off[k], 2) for k in keys}


def fix2_ab(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The verified-step A/B of the allocator fix, from the summary's part records, or ``None``
    when the plan has no fix-off part. The bring-up with the fix on (``info["arm"]``
    ``fix2_on``) beside the one with it off (``fix2_off``): per step, the prover, verifier,
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
            **{k: p["info"].get(k) for k in ("alloc_conf", "release_cache")}}

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


def _run_part(part: Part, env: dict[str, str], cwd: Path, log: Path, *, deadline: float,
              python: str, current: Path | None = None) -> tuple[str, int | None, float]:
    """Run one part to its end, its time limit or the deadline: (status, exit code, wall s).

    ``current``, while the part runs, holds its process group id, so ``fs11_job.sh`` can kill
    the part too when the session itself hangs."""
    argv = [python if a == "{python}" else a for a in part.argv]
    penv = dict(env)
    for k, v in part.env.items():
        if v is None:
            penv.pop(k, None)
        else:
            penv[k] = v
    t0 = time.time()
    limit = deadline if part.timeout_s is None else min(deadline, t0 + part.timeout_s)
    with open(log, "ab") as f:
        f.write(f"$ {' '.join(argv)}\n".encode())
        f.flush()
        proc = subprocess.Popen(argv, cwd=cwd, env=penv, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        if current is not None:
            try:
                current.write_text(f"{proc.pid}\n")
            except OSError:
                pass
        try:
            while True:
                try:
                    rc = proc.wait(timeout=max(0.0, min(5.0, limit - time.time())))
                    return ("ok" if rc == 0 else "failed"), rc, time.time() - t0
                except subprocess.TimeoutExpired:
                    if time.time() >= limit:
                        _kill(proc)
                        why = "killed_deadline" if limit == deadline else "killed_timeout"
                        f.write(f"\n[fs11] {why} after {time.time() - t0:.0f} s\n".encode())
                        return why, proc.returncode, time.time() - t0
        except BaseException:  # SIGTERM to the session (_Stop) or Ctrl-C: take the part down too
            _kill(proc)
            f.write(f"\n[fs11] killed: the session was stopped after {time.time() - t0:.0f} s\n"
                    .encode())
            raise
        finally:
            if current is not None:
                current.unlink(missing_ok=True)


def run_session(plan: Sequence[Part], *, upload: Path, out: Path, deadline: float, cwd: Path,
                python: str = sys.executable, env_doc: Mapping[str, Any] | None = None,
                say: Callable[[str], None] = _say, session: str = "FS11") -> dict[str, Any]:
    """Run ``plan`` in order (module docstring) and return the summary, also written to
    ``out/summary.json`` after each part. ``session`` is the summary's name for the session
    (:data:`SESSIONS`). A plan with a fix-off bring-up also gets ``fix2_ab`` (:func:`fix2_ab`),
    and the job log one line per step with the two arms' totals."""
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
        summary["parts"].append(rec)
        ab = fix2_ab(summary["parts"])
        if ab is not None:
            summary["fix2_ab"] = ab
            for r in ab["steps"] if part.info.get("arm") == "fix2_off" else ():
                if r["fix_cost"]["total_s"] is not None:
                    say(f"allocator fix A/B, step {r['step']}: {r['fix_on']['total_s']:.2f} s "
                        f"with the fix, {r['fix_off']['total_s']:.2f} s without "
                        f"({r['fix_cost']['total_s']:+.2f} s)")
        if part.kind == "models" and rec["status"] != "not_started":
            # On every run, a rerun that skipped the part included: the later parts read
            # these files, and a DONE marker says nothing about them now.
            check = check_upload(upload)
            summary["upload_check"] = check
            say(f"upload check ({check['s']:.0f} s): {len(check['packed'])} packed files "
                f"missing or differing; models with problems: "
                f"{', '.join(check['models']) or 'none'}")
        _write_json(path, summary)
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
    don't start. The benchmark still runs; a model whose weights are missing fails to load
    there and says so in its row."""
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
    return {"packed": packed, "models": models, "s": round(time.time() - t0, 1)}


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
                   help="FS11_SESSION: the first session's plan (fs11, the default) or the "
                        "re-check's (recheck); --plan and --selftest replace its parts")
    f = sub.add_parser("fetch-models", help="check the upload, fetch the public weights")
    f.add_argument("--upload", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("env", help="print the environment and the H100 check")
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "fetch-models":
        return fetch_models(args.upload.resolve(), args.out)
    out = args.out.resolve()
    selftest = args.cmd == "run" and args.selftest
    label, build = SESSIONS[args.session] if args.cmd == "run" else SESSIONS["fs11"]
    allow = os.environ.get("FS11_ALLOW_ANY_GPU") == "1" or selftest
    env_doc = environment(out)
    _write_json(out / "environment.json", env_doc)
    print_environment(env_doc)
    why = refusal(env_doc, allow)
    if selftest:
        _say(f"self-test: stand-in parts in place of the {label} session's, the H100 check is "
             "skipped")
    elif allow:
        _say("FS11_ALLOW_ANY_GPU=1: the H100 check is overridden")
    if args.cmd == "env":
        _say("refused: " + why if why else "the H100 check passes")
        return EXIT_REFUSED if why else 0
    if why:
        _say(f"refused to start: {why} (FS11_ALLOW_ANY_GPU=1 overrides)")
        _write_json(out / "summary.json", {"session": label, "refused": why,
                                           "environment": env_doc, "parts": [],
                                           "finished": True})
        return EXIT_REFUSED
    upload = args.upload.resolve()
    if args.plan:
        plan = load_plan(args.plan)
    else:
        plan = selftest_plan() if selftest else build(upload, out)
    _say(f"session {label}: {len(plan)} part{'' if len(plan) == 1 else 's'}")
    signal.signal(signal.SIGTERM, _on_term)
    summary = run_session(plan, upload=upload, out=out, deadline=args.deadline,
                          cwd=Path.cwd(), env_doc=env_doc, session=label)
    failed = summary["failed"]
    _say("all parts succeeded" if not failed else f"failed or cut short: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
