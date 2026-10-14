"""FS11's H100 session: the parts in order, each with its own log, under one deadline.

    .venv/bin/python -m verification.runs.fs11_session run --upload DIR --out DIR --deadline EPOCH
    .venv/bin/python -m verification.runs.fs11_session fetch-models --upload DIR --out DIR
    .venv/bin/python -m verification.runs.fs11_session env --out DIR

``fs11_job.sh`` runs ``run`` on the pod, then packs the output and ends the pod; see
``verification/README.md``, "Workflow: the first H100 session". ``fs11_prep`` builds the upload
on the Mac. The parts, in this order:

1. **models.** ``fetch-models``: check every packed file of the upload against the manifest's
   SHA-256, then fetch each public model's weights at its pinned commit with ``token=False``
   and check them against the SHA-256 that ``fs11_prep`` recorded from the Mac's cache. A file
   that doesn't match is deleted. A gated model (:data:`GATED`) is never fetched: its files
   come packed in the upload. Only this part may reach the network.
2. **benchmark.** ``benchmarks.hashing.run`` at its H100 preset (FS10), report in
   ``benchmark/hashing.json``. Its GPU-hashing slots need cuPQC (:func:`cupqc_status`).
3. **bring-up**, one per model: ``bring_up --no-tamper``, one verified step at the full-scale
   config (:data:`FULL_SCALE_ENV`) on the refusal task's ``D``, with ``metrics.py``'s peak
   memory. Its exit code also folds in the concentration guard, which may fail at bf16 (F20);
   the step's acceptance, peaks and times are in its log either way.
4. **smooth**, one per model and task: ``smooth_check``, F6c's 10 plain steps.

Every part runs in its own process group, with ``HF_HUB_OFFLINE=1`` (all but the first), no
Hugging Face token and no Runpod key in its environment. A part's log is
``parts/<name>/log.txt``. ``parts/<name>/DONE`` is written when it exits 0, and a rerun into
the same output directory skips it. A part that fails doesn't stop the others. A part still
running at ``--deadline`` (or past its own time limit) is killed, and the parts after it are
recorded as not started. ``summary.json`` is rewritten after each part.

Before any part, ``run`` prints the environment (GPU, driver, CUDA, torch, free disk and RAM)
to ``environment.json`` and refuses to start unless CUDA is available and the GPU is an H100;
``FS11_ALLOW_ANY_GPU=1`` overrides. Exits 0 when every part succeeded, 1 when any failed, and
:data:`EXIT_REFUSED` when the environment check refused.
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
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["RUN_PINS", "GATED", "TASKS", "FULL_SCALE_ENV", "EXIT_REFUSED", "Part",
           "default_plan", "load_plan", "run_session", "fetch_models", "environment",
           "cupqc_status", "data_key", "main"]

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
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)  # UTC, as the job's


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
    secret: of the environment it records only the Runpod pod id and the cuPQC path."""
    import torch

    cuda = torch.cuda.is_available()
    mem = _meminfo()
    phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") \
        if hasattr(os, "sysconf") else None
    out.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(out)
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(), "platform": platform.platform(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
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
    kind: str = "other"  # models, benchmark, bring_up, smooth, other
    info: dict[str, Any] = field(default_factory=dict)  # model, task, run_dir, report


def _run_dir(out: Path, upload: Path, key: str) -> Path:
    """``runs/<key>/``, a part's ``VERIF_OUTPUT_DIR``, its ``data`` linked to the upload's."""
    d = out / "runs" / key
    d.mkdir(parents=True, exist_ok=True)
    link = d / "data"
    if not link.exists() and not link.is_symlink():
        link.symlink_to((upload / "data" / key).resolve(), target_is_directory=True)
    return d


def default_plan(upload: Path, out: Path) -> list[Part]:
    py = "{python}"
    plan = [Part("models", [py, "-m", "verification.runs.fs11_session", "fetch-models",
                            "--upload", str(upload), "--out", str(out / "models")],
                 env={"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None},
                 timeout_s=TIMEOUT_S["models"], kind="models",
                 info={"report": str(out / "models" / "models.json")})]
    report = out / "benchmark" / "hashing.json"
    plan.append(Part("benchmark", [py, "-m", "benchmarks.hashing.run", "--out", str(report)],
                     env={"VERIF_OUTPUT_DIR": str(out / "benchmark")},
                     timeout_s=TIMEOUT_S["benchmark"], kind="benchmark",
                     info={"report": str(report)}))
    for repo, rev in RUN_PINS:
        key = data_key(repo, BRING_UP_TASK)
        d = _run_dir(out, upload, key)
        plan.append(Part(f"bring_up_{repo.split('/')[-1]}",
                         [py, "-m", "verification.runs.bring_up", "--no-tamper"],
                         env={**_run_env(repo, rev, BRING_UP_TASK), "VERIF_OUTPUT_DIR": str(d)},
                         timeout_s=TIMEOUT_S["bring_up"], kind="bring_up",
                         info={"model": repo, "task": BRING_UP_TASK, "run_dir": str(d)}))
    for repo, rev in RUN_PINS:
        for task in TASKS:
            key = data_key(repo, task)
            d = _run_dir(out, upload, key)
            plan.append(Part(f"smooth_{key}", [py, "-m", "verification.runs.smooth_check"],
                             env={**_run_env(repo, rev, task), "VERIF_OUTPUT_DIR": str(d)},
                             timeout_s=TIMEOUT_S["smooth"], kind="smooth",
                             info={"model": repo, "task": task, "run_dir": str(d),
                                   "report": str(d / "smooth_check" / f"{key}.json")}))
    return plan


def _run_env(repo: str, rev: str, task: str) -> dict[str, str | None]:
    return {**FULL_SCALE_ENV, "VERIF_MODEL": repo, "VERIF_MODEL_REVISION": rev,
            "VERIF_ATTACK": task, "VERIF_N_RECORDS": str(TASKS[task])}


def load_plan(path: Path) -> list[Part]:
    """A plan from JSON, a list of :class:`Part` fields: for a dry run with stand-in parts."""
    return [Part(**p) for p in json.loads(path.read_text())]


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
_PEAK = re.compile(r"^step (\d+): (prover|verifier) peak ([\d.]+) GB")


def _bring_up_extras(log: Path) -> dict[str, Any]:
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    peaks = [{"step": int(m[1]), "side": m[2], "peak_gb": float(m[3])}
             for m in map(_PEAK.match, lines) if m]
    return {"peaks": peaks, "lines": [l for l in lines if _BRING_UP_LINE.match(l)][-40:]}


def _smooth_extras(report: Path) -> dict[str, Any]:
    if not report.exists():
        return {"record": None}
    doc = json.loads(report.read_text())
    return {"record": str(report), "smooth": doc.get("smooth"), "reasons": doc.get("reasons"),
            "whole_list_losses": doc.get("whole_list_losses"),
            "batch_losses": doc.get("batch_losses"),
            "peak_bytes": (doc.get("memory") or {}).get("peak_bytes"),
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


def part_extras(part: Part, log: Path) -> dict[str, Any]:
    try:
        if part.kind == "bring_up":
            return _bring_up_extras(log)
        if part.kind == "smooth":
            return _smooth_extras(Path(part.info["report"]))
        if part.kind == "benchmark":
            return _benchmark_extras(Path(part.info["report"]))
        if part.kind == "models":
            report = Path(part.info["report"])
            return json.loads(report.read_text()) if report.exists() else {"report": None}
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"extras_error": f"{type(e).__name__}: {e}"}
    return {}


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
              python: str) -> tuple[str, int | None, float]:
    """Run one part to its end, its time limit or the deadline: (status, exit code, wall s)."""
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


def run_session(plan: Sequence[Part], *, upload: Path, out: Path, deadline: float, cwd: Path,
                python: str = sys.executable, env_doc: Mapping[str, Any] | None = None,
                say: Callable[[str], None] = _say) -> dict[str, Any]:
    """Run ``plan`` in order (module docstring) and return the summary, also written to
    ``out/summary.json`` after each part."""
    env = base_env(upload)
    summary: dict[str, Any] = {
        "session": "FS11", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)),
        "upload": str(upload), "out": str(out), "environment": env_doc,
        "cupqc": cupqc_status(), "parts": [], "finished": False,
    }
    say(f"cuPQC: {summary['cupqc']['reason']} ({', '.join(CUPQC_SLOTS)})")
    path = out / "summary.json"
    for part in plan:
        d = out / "parts" / part.name
        d.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = {"name": part.name, "kind": part.kind, "info": part.info,
                               "log": str(d / "log.txt")}
        if summary.get("stopped"):
            rec.update(status="not_started", exit_code=None, reason=summary["stopped"])
        elif (d / "DONE").exists():
            rec.update(status="skipped_done", exit_code=0)
            say(f"{part.name}: done in an earlier run, skipped")
        elif time.time() >= deadline:
            rec.update(status="not_started", exit_code=None,
                       reason="the session deadline passed before it started")
            say(f"{part.name}: not started, the deadline passed")
        else:
            say(f"{part.name}: start (log {d / 'log.txt'})")
            t0 = time.time()
            try:
                status, rc, wall = _run_part(part, env, cwd, d / "log.txt", deadline=deadline,
                                             python=python)
            except (_Stop, KeyboardInterrupt) as e:
                # Record the cut; the parts after it then say why they didn't start.
                summary["stopped"] = f"the session was stopped ({e or 'SIGINT'})"
                status, rc, wall = "killed_signal", None, time.time() - t0
            rec.update(status=status, exit_code=rc, wall_s=round(wall, 1))
            if status == "ok":
                (d / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))
            say(f"{part.name}: {status}, exit {rc}, {wall:.0f} s")
        if part.kind == "benchmark" and rec["status"] != "skipped_done":
            with open(d / "log.txt", "a") as f:
                f.write(f"[fs11] cuPQC slots {', '.join(CUPQC_SLOTS)}: "
                        f"{summary['cupqc']['reason']}\n")
        rec["extras"] = part_extras(part, d / "log.txt")
        summary["parts"].append(rec)
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


def fetch_models(upload: Path, out: Path, *,
                 download: Callable[[str, str, str, Path], Path] = _hf_download,
                 say: Callable[[str], None] = _say) -> int:
    """Check the upload and fetch the public weights (module docstring, part 1). Writes
    ``out/models.json``; returns 0 when every file is present and matches."""
    manifest = json.loads((upload / "manifest.json").read_text())
    hub = upload / "hf" / "hub"
    problems: list[str] = []
    t0 = time.time()
    bad = []
    for arc, digest in manifest["files"].items():
        p = upload / arc
        if not p.is_file():
            bad.append(f"{arc}: missing")
        elif sha256_file(p) != digest:
            bad.append(f"{arc}: SHA-256 differs from the manifest")
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
    f = sub.add_parser("fetch-models", help="check the upload, fetch the public weights")
    f.add_argument("--upload", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("env", help="print the environment and the H100 check")
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "fetch-models":
        return fetch_models(args.upload.resolve(), args.out)
    out = args.out.resolve()
    allow = os.environ.get("FS11_ALLOW_ANY_GPU") == "1"
    env_doc = environment(out)
    _write_json(out / "environment.json", env_doc)
    print_environment(env_doc)
    why = refusal(env_doc, allow)
    if allow:
        _say("FS11_ALLOW_ANY_GPU=1: the H100 check is overridden")
    if args.cmd == "env":
        _say("refused: " + why if why else "the H100 check passes")
        return EXIT_REFUSED if why else 0
    if why:
        _say(f"refused to start: {why} (FS11_ALLOW_ANY_GPU=1 overrides)")
        _write_json(out / "summary.json", {"session": "FS11", "refused": why,
                                           "environment": env_doc, "parts": [],
                                           "finished": True})
        return EXIT_REFUSED
    upload = args.upload.resolve()
    plan = load_plan(args.plan) if args.plan else default_plan(upload, out)
    signal.signal(signal.SIGTERM, _on_term)
    summary = run_session(plan, upload=upload, out=out, deadline=args.deadline,
                          cwd=Path.cwd(), env_doc=env_doc)
    failed = summary["failed"]
    _say("all parts succeeded" if not failed else f"failed or cut short: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
