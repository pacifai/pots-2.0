"""E5(a′)'s native batch: challenge generation in C over BLAKE3's reference library.

    python -m benchmarks.hashing.native      # fetch, check and build; prints the library

BLAKE3's C sources are fetched at build time and never stored in the repo. The release
archive of :data:`BLAKE3_TAG` is checked against :data:`ARCHIVE_SHA256` before anything is
extracted. The system C compiler (``$CC``, else ``cc``) builds them with this package's
``challenges_native.c`` into one shared library, loaded through ctypes, which releases the GIL
for the whole call. BLAKE3's own dispatch picks the SIMD path at run time: NEON on arm64, and
SSE2, SSE4.1, AVX2 or AVX-512 on x86-64 through BLAKE3's assembly files.

The cache is ``$POTS_BENCH_CACHE``, else ``~/.cache/pots-benchmarks``. The library's file name
carries the tag, the machine and a hash of the C source and the build command, so changing
either builds a new library.
"""

from __future__ import annotations

import ctypes
import hashlib
import io
import os
import platform
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from benchmarks.hashing.challenges import Call, _extrapolate, _timed, _validate, plan, products, \
    select
from benchmarks.hashing.sizes import Step

BLAKE3_TAG = "1.8.7"
BLAKE3_COMMIT = "f3149ec5bb5449af877ba20377a11008ff499fa2"
ARCHIVE_URL = f"https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/{BLAKE3_TAG}.tar.gz"
ARCHIVE_SHA256 = "c6782a28842b1c0478524ac06a4f2ede784038ee298d6e2162c0b089c4306a3c"
SOURCE = Path(__file__).with_name("challenges_native.c")

_COMMON = ["blake3.c", "blake3_dispatch.c", "blake3_portable.c"]
_X86_ASM = ["blake3_sse2_x86-64_unix.S", "blake3_sse41_x86-64_unix.S",
            "blake3_avx2_x86-64_unix.S", "blake3_avx512_x86-64_unix.S"]
_NO_SIMD = ["-DBLAKE3_USE_NEON=0", "-DBLAKE3_NO_SSE2", "-DBLAKE3_NO_SSE41", "-DBLAKE3_NO_AVX2",
            "-DBLAKE3_NO_AVX512"]
_KEEP = ("blake3.h", "blake3_impl.h", *_COMMON, "blake3_neon.c", *_X86_ASM)
_LICENSES = ("LICENSE_A2", "LICENSE_A2LLVM", "LICENSE_CC0")


def cache_dir() -> Path:
    return Path(os.environ.get("POTS_BENCH_CACHE") or Path.home() / ".cache" / "pots-benchmarks")


def target(machine: str | None = None) -> tuple[str, list[str], list[str]]:
    """``(arch, BLAKE3 files, flags)`` for a machine name as ``platform.machine()`` gives it."""
    m = (machine or platform.machine()).lower()
    if m in ("arm64", "aarch64"):
        return "arm64", [*_COMMON, "blake3_neon.c"], ["-DBLAKE3_USE_NEON=1"]
    if m in ("x86_64", "amd64") and sys.platform != "win32":
        return "x86_64", [*_COMMON, *_X86_ASM], []
    return m or "unknown", list(_COMMON), list(_NO_SIMD)


def _command(cc: str, out: Path, src: Path, files: Sequence[str], flags: Sequence[str]) -> list[str]:
    return [cc, "-O3", "-shared", "-fPIC", *flags, f"-I{src}", "-o", str(out),
            str(SOURCE), *(str(src / f) for f in files), "-lpthread"]


def library_path(cache: Path | None = None, cc: str | None = None) -> Path:
    cache = cache or cache_dir()
    cc = cc or os.environ.get("CC", "cc")
    arch, files, flags = target()
    key = hashlib.sha256(SOURCE.read_bytes() + repr(
        (BLAKE3_TAG, ARCHIVE_SHA256, cc, files, flags)).encode()).hexdigest()[:12]
    return cache / f"libpots_challenges-blake3-{BLAKE3_TAG}-{sys.platform}-{arch}-{key}.so"


def find_library() -> Path | None:
    """The built library for this machine, or None."""
    path = library_path()
    return path if path.exists() else None


def source_dir(cache: Path | None = None) -> Path:
    return (cache or cache_dir()) / f"blake3-{BLAKE3_TAG}"


def check_archive(data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(f"BLAKE3 {BLAKE3_TAG} archive has SHA-256 {digest}, expected "
                           f"{ARCHIVE_SHA256}; not extracted")


def _tls() -> ssl.SSLContext:
    """certifi's CA bundle when it's installed (python.org's macOS build ships none), else the
    system's. The archive's integrity rests on :data:`ARCHIVE_SHA256`, not on TLS."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch(dest: Path, *, log: Callable[[str], None] = print) -> Path:
    """Download the pinned archive, check its hash, and extract ``c/``'s needed files and the
    licences into ``dest``."""
    log(f"fetching {ARCHIVE_URL}")
    with urllib.request.urlopen(ARCHIVE_URL, timeout=120, context=_tls()) as r:
        data = r.read()
    check_archive(data)
    tmp = Path(tempfile.mkdtemp(prefix="blake3-", dir=dest.parent))
    try:
        root = f"BLAKE3-{BLAKE3_TAG}/"
        wanted = {root + "c/" + f for f in _KEEP} | {root + f for f in _LICENSES}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name in wanted]
            missing = wanted - {m.name for m in members}
            if missing:
                raise RuntimeError(f"archive lacks {sorted(missing)}")
            for m in members:
                m.name = Path(m.name).name
            tar.extractall(tmp, members=members, filter="data")
        (tmp / "ARCHIVE_SHA256").write_text(ARCHIVE_SHA256 + "\n")
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def build(*, log: Callable[[str], None] = print) -> Path:
    """The library for this machine, fetched and built into the cache when it's missing."""
    out = library_path()
    if out.exists():
        return out
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    src = source_dir(cache)
    marker = src / "ARCHIVE_SHA256"
    if not marker.exists() or marker.read_text().strip() != ARCHIVE_SHA256:
        fetch(src, log=log)
    cc = os.environ.get("CC", "cc")
    _, files, flags = target()
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    cmd = _command(cc, tmp, src, files, flags)
    log("building " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"build failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}")
    os.replace(tmp, out)
    return out


class Native:
    """The loaded library."""

    def __init__(self, path: Path) -> None:
        self.path = path
        lib = ctypes.CDLL(str(path))
        u32 = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
        lib.pots_challenges.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_size_t, u32, u32,
            np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"), ctypes.c_uint32]
        lib.pots_challenges.restype = ctypes.c_int
        lib.pots_simd_degree.restype = ctypes.c_size_t
        lib.pots_blake3_version.restype = ctypes.c_char_p
        self._lib = lib
        self.simd_degree = int(lib.pots_simd_degree())
        self.blake3_version = lib.pots_blake3_version().decode()

    def challenge_matrices(self, h: bytes, ms: Sequence[int], k: int, width: int, *,
                           threads: int = 1) -> torch.Tensor:
        """The protocol's ``challenge_matrices``, bit for bit: ``[len(ms), width, k]``."""
        if not 1 <= k <= 255:
            raise ValueError(f"k must be in [1, 255], got {k}")
        for m in ms:
            _validate(h, m, width)
        n = len(ms)
        out = np.empty(n * width * k, dtype=np.float32)
        rc = self._lib.pots_challenges(
            bytes(h), k, n, np.asarray(ms, dtype=np.uint32), np.full(n, width, dtype=np.uint32),
            np.arange(n, dtype=np.uint64) * np.uint64(width * k), out, max(1, int(threads)))
        if rc != 0:
            raise RuntimeError(f"pots_challenges returned {rc}")
        return torch.from_numpy(out).view(n, width, k)

    def challenge_matrix(self, h: bytes, m: int, k: int, width: int, *,
                         threads: int = 1) -> torch.Tensor:
        """The protocol's ``challenge_matrix``, bit for bit: ``[width, k]``."""
        _validate(h, m, width)
        return self.challenge_matrices(h, [m], k, width, threads=threads)[0]


def load(path: Path | None = None) -> Native:
    return Native(path or build())


def run_native(step: Step, lib: Native, *, k: int, layers: int | None, threads: Sequence[int],
               seed: int = 0) -> list[dict]:
    """E5(a′)'s native rows: check 5's calls through the library, one call at a time as check 5
    makes them, at each thread count."""
    h = np.random.default_rng(seed).bytes(32)
    calls = plan(products(step))
    chosen, n_sel, n_total = select(calls, layers)
    calls_step = sum(c.xof_calls for c in calls) * k

    def run_call(h: bytes, k: int, c: Call, t: int) -> None:
        if c.batched:
            lib.challenge_matrices(h, c.ms, k, c.width, threads=t)
        else:
            lib.challenge_matrix(h, c.ms[0], k, c.width, threads=t)

    base = {**step.params(), "k": k, "blake3": lib.blake3_version, "tag": BLAKE3_TAG,
            "simd_degree": lib.simd_degree, "machine": platform.machine()}
    out = []
    for t in threads:
        run_call(h, k, chosen[0], t)  # warm: page in the code, start the allocator
        head, lay, n_head, n_lay = _timed(chosen, h, k,
                                          run_call=lambda h, k, c, t=t: run_call(h, k, c, t))
        out += _extrapolate("native_batch", step, {**base, "threads": t}, head, lay, n_sel,
                            n_total, n_head + n_lay, calls_step,
                            note="BLAKE3's C library through ctypes, one native call per check-5 "
                                 "call, its vectors spread over the threads; float32 out")
    return out


def main(argv: list[str] | None = None) -> int:
    t0 = time.perf_counter()
    path = build()
    lib = load(path)
    print(f"{path}\nBLAKE3 {lib.blake3_version}, SIMD degree {lib.simd_degree}, "
          f"{time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
