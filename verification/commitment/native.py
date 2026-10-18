"""The protocol's native BLAKE3 batch (decision F15b): challenges, leaf hashing with the
finiteness test, and the Merkle tree, in C over BLAKE3's reference library.

    python -m verification.commitment.native      # fetch, check and build; prints the library

BLAKE3's C sources are fetched at build time and never stored in the repo. The release
archive of :data:`BLAKE3_TAG` is checked against :data:`ARCHIVE_SHA256` before anything is
extracted. The system C compiler (``$CC``, else ``cc``) builds them with this package's
``native.c`` into one shared library, loaded through ctypes, which releases the GIL for the
whole call. BLAKE3's own dispatch picks the SIMD path at run time: NEON on arm64, and SSE2,
SSE4.1, AVX2 or AVX-512 on x86-64 through BLAKE3's assembly files.

Nothing is built, fetched or loaded on import: :func:`get` does that on first call, and a
failed build raises with no fallback. The cache is ``$POTS_NATIVE_CACHE``, else
``~/.cache/pots-2.0/native``. Every extracted file is pinned by its SHA-256
(:data:`SOURCE_SHA256`), checked after extraction and again before a cached source directory
is reused; a mismatch fetches the archive again. The library's file name carries the tag, the
machine and a hash of the C source, the build command and the compiler's ``--version``
output, so changing any of them builds a new library.

This module imports nothing from ``verification`` or ``benchmarks``, and its
``challenge_matrices`` matches ``verifier.matmul_check.challenges.challenge_matrices`` bit
for bit.
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
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


BLAKE3_TAG = "1.8.7"
BLAKE3_COMMIT = "f3149ec5bb5449af877ba20377a11008ff499fa2"
ARCHIVE_URL = f"https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/{BLAKE3_TAG}.tar.gz"
ARCHIVE_SHA256 = "c6782a28842b1c0478524ac06a4f2ede784038ee298d6e2162c0b089c4306a3c"
SOURCE = Path(__file__).with_name("native.c")

_COMMON = ["blake3.c", "blake3_dispatch.c", "blake3_portable.c"]
_X86_ASM = ["blake3_sse2_x86-64_unix.S", "blake3_sse41_x86-64_unix.S",
            "blake3_avx2_x86-64_unix.S", "blake3_avx512_x86-64_unix.S"]
_NO_SIMD = ["-DBLAKE3_USE_NEON=0", "-DBLAKE3_NO_SSE2", "-DBLAKE3_NO_SSE41", "-DBLAKE3_NO_AVX2",
            "-DBLAKE3_NO_AVX512"]
_KEEP = ("blake3.h", "blake3_impl.h", *_COMMON, "blake3_neon.c", *_X86_ASM)
_LICENSES = ("LICENSE_A2", "LICENSE_A2LLVM", "LICENSE_CC0")
# The SHA-256 of every file kept from the archive at BLAKE3_TAG.
SOURCE_SHA256 = {
    "LICENSE_A2": "00fcc7a934ddbc9ece2a7cc063ac788e284b703b1d705ccbba72d462aa97921e",
    "LICENSE_A2LLVM": "a5695f57ea0c221e0e8b7d784ff774c35e88c3d3270353646a925880bb3492cc",
    "LICENSE_CC0": "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499",
    "blake3.c": "b118ddf7cf9e6e5ef3fded72dcb1acf9dfdc4ea923cbe4605900ad6ee9afe1af",
    "blake3.h": "81bbb11ad1909341565dce4914cc0e5c29718eca27e6950b28f9226a91aaf8c5",
    "blake3_avx2_x86-64_unix.S":
        "a2d95519fd9845cf80252110e21a93124e81c03da836eaac0bca0537be4f8324",
    "blake3_avx512_x86-64_unix.S":
        "c641987dd82fabed59d527f1a3413c4a2ff3df919af6d13ff2788c95e4035c0d",
    "blake3_dispatch.c": "134f21550138c0af6312925c988aeee35df287e4119e8ad1d206fccdb2238fe3",
    "blake3_impl.h": "d388dca3574602c8849805ea3c8c0a12d082ac0e428756f305e111942f099af4",
    "blake3_neon.c": "67e88d017a82df1b76d5f841e7b7215135a007486ac14c5be1407e1bca5201b0",
    "blake3_portable.c": "2bc25b0dad67b4329d0b49cfa075ab2b0d04e424addbddc4e9c389c52a192524",
    "blake3_sse2_x86-64_unix.S":
        "94baa998f4114bf57b99e1791a21e9f0bcdc4bd5803864eb30a2477440b7380c",
    "blake3_sse41_x86-64_unix.S":
        "135048e4a81a6f9e86960b115141874a55954dbfd2cd1e8e9818837b46b498ad",
}


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def cache_dir() -> Path:
    return Path(os.environ.get("POTS_NATIVE_CACHE")
                or Path.home() / ".cache" / "pots-2.0" / "native")


def target(machine: str | None = None) -> tuple[str, list[str], list[str]]:
    """``(arch, BLAKE3 files, flags)`` for a machine name as ``platform.machine()`` gives it."""
    m = (machine or platform.machine()).lower()
    if m in ("arm64", "aarch64"):
        return "arm64", [*_COMMON, "blake3_neon.c"], ["-DBLAKE3_USE_NEON=1"]
    if m in ("x86_64", "amd64") and sys.platform != "win32":
        return "x86_64", [*_COMMON, *_X86_ASM], []
    return m or "unknown", list(_COMMON), list(_NO_SIMD)


def _command(cc: str, out: Path, src: Path, files: Sequence[str],
             flags: Sequence[str]) -> list[str]:
    return [cc, "-O3", "-shared", "-fPIC", *flags, f"-I{src}", "-o", str(out),
            str(SOURCE), *(str(src / f) for f in files), "-lpthread"]


@lru_cache(maxsize=None)
def compiler_version(cc: str) -> str:
    """``cc --version``'s output, or "" when the compiler is missing or won't say."""
    if shutil.which(cc) is None:
        return ""
    try:
        proc = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def library_path(cache: Path | None = None, cc: str | None = None) -> Path:
    cache = cache or cache_dir()
    cc = cc or os.environ.get("CC", "cc")
    arch, files, flags = target()
    key = hashlib.sha256(SOURCE.read_bytes() + repr(
        (BLAKE3_TAG, ARCHIVE_SHA256, cc, compiler_version(cc), files, flags)).encode()
    ).hexdigest()[:12]
    return cache / f"libpots_native-blake3-{BLAKE3_TAG}-{sys.platform}-{arch}-{key}.so"


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


def bad_sources(src: Path) -> list[str]:
    """The pinned files in ``src`` that are missing or don't match :data:`SOURCE_SHA256`, and
    ``ARCHIVE_SHA256`` when the marker doesn't name the pinned archive."""
    bad = []
    marker = src / "ARCHIVE_SHA256"
    if not marker.is_file() or marker.read_text().strip() != ARCHIVE_SHA256:
        bad.append("ARCHIVE_SHA256")
    for name, digest in SOURCE_SHA256.items():
        path = src / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return bad


def _tls() -> ssl.SSLContext:
    """certifi's CA bundle when it's installed (python.org's macOS build ships none), else the
    system's. The archive's integrity rests on :data:`ARCHIVE_SHA256`, not on TLS."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch(dest: Path, *, log: Callable[[str], None] = _stderr) -> Path:
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
        bad = bad_sources(tmp)
        if bad:
            raise RuntimeError(f"extracted files don't match SOURCE_SHA256: {bad}")
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def build(*, log: Callable[[str], None] = _stderr) -> Path:
    """The library for this machine, fetched and built into the cache when it's missing."""
    out = library_path()
    if out.exists():
        return out
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    src = source_dir(cache)
    bad = bad_sources(src) if src.exists() else ["the directory"]
    if bad:
        log(f"BLAKE3 sources in {src} missing or changed ({', '.join(bad)}); fetching again")
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


KIND_NONE, KIND_F32, KIND_BF16, KIND_F16 = 0, 1, 2, 3
FINITE_KIND: dict[torch.dtype, int] = {
    torch.float32: KIND_F32, torch.bfloat16: KIND_BF16, torch.float16: KIND_F16,
    torch.int32: KIND_NONE,
}

_ptr = np.ctypeslib.ndpointer


class Native:
    """The loaded library."""

    def __init__(self, path: Path) -> None:
        self.path = path
        lib = ctypes.CDLL(str(path))
        u8, u32, u64 = (_ptr(t, flags="C_CONTIGUOUS") for t in (np.uint8, np.uint32, np.uint64))
        lib.pots_challenges.argtypes = [
            ctypes.c_char_p, ctypes.c_uint32, ctypes.c_size_t, u32, u32, u64,
            _ptr(np.float32, flags="C_CONTIGUOUS"), ctypes.c_uint32]
        lib.pots_challenges.restype = ctypes.c_int
        lib.pots_hash_leaves.argtypes = [
            ctypes.c_size_t, ctypes.c_char_p, u64, u32, u32, u64, u64, u8, u8, ctypes.c_uint32]
        lib.pots_hash_leaves.restype = ctypes.c_size_t
        lib.pots_all_finite.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint8, ctypes.c_uint32]
        lib.pots_all_finite.restype = ctypes.c_int
        lib.pots_merkle_tree.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p]
        lib.pots_merkle_tree.restype = None
        lib.pots_simd_degree.restype = ctypes.c_size_t
        lib.pots_blake3_version.restype = ctypes.c_char_p
        self._lib = lib
        self.simd_degree = int(lib.pots_simd_degree())
        self.blake3_version = lib.pots_blake3_version().decode()

    def challenge_matrices(self, h: bytes, ms: Sequence[int], k: int, width: int, *,
                           threads: int) -> torch.Tensor:
        """The protocol's ``challenge_matrices``, bit for bit: float32 ``[len(ms), width, k]``."""
        if len(h) != 32:
            raise ValueError("h must be 32 bytes")
        if not 1 <= k <= 255:
            raise ValueError(f"k must be in [1, 255], got {k}")
        n = len(ms)
        if n == 0:
            return torch.empty(0, width, k)
        out = np.empty(n * width * k, dtype=np.float32)
        rc = self._lib.pots_challenges(
            bytes(h), k, n, np.asarray(ms, dtype=np.uint32), np.full(n, width, dtype=np.uint32),
            np.arange(n, dtype=np.uint64) * np.uint64(width * k), out, max(1, int(threads)))
        if rc != 0:
            raise RuntimeError(f"pots_challenges returned {rc}")
        return torch.from_numpy(out).view(n, width, k)

    def hash_leaves(self, hdr_buf: bytes, hdr_off: np.ndarray, hdr_len: np.ndarray,
                    hdr_idx: np.ndarray, payload_ptr: np.ndarray, payload_len: np.ndarray,
                    kind: np.ndarray, *, threads: int) -> tuple[np.ndarray, int]:
        """``(digests, first_bad)``: leaf ``i``'s digest is ``BLAKE3(0x00 ‖ header ‖ payload)``
        as ``hash_leaf`` gives it, ``digests`` is uint8 ``[n, 32]``, and the header is
        ``hdr_buf[hdr_off[u]:][:hdr_len[u]]`` for ``u = hdr_idx[i]``. The payload is
        ``payload_len[i]`` bytes at address ``payload_ptr[i]``, tested as ``kind[i]`` says
        (``KIND_*``).

        ``first_bad`` is the first leaf whose payload holds a NaN or Inf, or ``n`` when none
        does; the digests before it are valid and the rest are unspecified. The caller keeps
        the payload objects alive (and unmodified) until the call returns.
        """
        n = len(payload_ptr)
        if not len(payload_len) == len(kind) == len(hdr_idx) == n:
            raise ValueError("payload_ptr, payload_len, kind and hdr_idx must have one length")
        if len(hdr_off) != len(hdr_len):
            raise ValueError("hdr_off and hdr_len must have one length")
        if n and (int(hdr_idx.max()) >= len(hdr_off)
                  or int((hdr_off.astype(np.uint64) + hdr_len).max()) > len(hdr_buf)):
            raise ValueError("a header index or range is out of bounds")
        out = np.empty((n, 32), dtype=np.uint8)
        bad = self._lib.pots_hash_leaves(
            n, bytes(hdr_buf), hdr_off, hdr_len, hdr_idx, payload_ptr, payload_len, kind, out,
            max(1, int(threads)))
        return out, int(bad)

    def all_finite(self, ptr: int, nbytes: int, kind: int, *, threads: int) -> bool:
        """Whether every value of the ``nbytes`` bytes at address ``ptr`` is finite."""
        return bool(self._lib.pots_all_finite(ptr, nbytes, kind, max(1, int(threads))))

    def merkle_tree(self, leaves: bytes, n: int) -> bytes:
        """The ``2n − 1`` subtree hashes of the RFC 6962 tree over ``n`` leaf digests, 32 bytes
        each, in pre-order: the root first, then for a range at ``p`` its left half at
        ``p + 1`` and its right half at ``p + 2k``, ``k`` the left half's leaf count."""
        if n < 1 or len(leaves) != 32 * n:
            raise ValueError(f"need n >= 1 and 32·n bytes of leaves, got n={n}, {len(leaves)}")
        out = ctypes.create_string_buffer(32 * (2 * n - 1))
        self._lib.pots_merkle_tree(n, bytes(leaves), out)
        return out.raw


_LIB: Native | None = None
_LOCK = threading.Lock()


def get() -> Native:
    """The library, built into the cache when missing and loaded once per process."""
    global _LIB
    with _LOCK:
        if _LIB is None:
            try:
                _LIB = Native(build())
            except Exception as e:
                raise RuntimeError(
                    "the native BLAKE3 library is required (decision F15b): it needs a C "
                    "compiler ($CC, else cc) and, for the first build only, network access to "
                    f"github.com. {type(e).__name__}: {e}") from e
        return _LIB


_disabled = 0
_disabled_lock = threading.Lock()


def enabled() -> bool:
    """False inside :func:`disabled`; the counting pass uses it to count in Python."""
    return _disabled == 0


@contextmanager
def disabled() -> Iterator[None]:
    """Make :func:`enabled` return False inside the block. Nestable."""
    global _disabled
    with _disabled_lock:
        _disabled += 1
    try:
        yield
    finally:
        with _disabled_lock:
            _disabled -= 1


def main() -> int:
    t0 = time.perf_counter()
    lib = get()
    print(f"{lib.path}\nBLAKE3 {lib.blake3_version}, SIMD degree {lib.simd_degree}, "
          f"{time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
