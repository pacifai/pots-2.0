"""Our own BLAKE3 CUDA kernel for transcript leaves (decision F23), with its ctypes loader.

    python -m verification.commitment.gpu_blake3 check            # sweep against the C library
    python -m verification.commitment.gpu_blake3 bench --out F    # GB/s by leaf size, as JSON

Leaf ``i``'s digest is ``BLAKE3(0x00 ‖ header ‖ payload)``, unkeyed, 32 bytes, bit for bit what
``native.Native.hash_leaves`` returns; the finiteness test of the float kinds rides in the
same pass over the payload. The kernels are in ``gpu_blake3.cu`` and ``gpu_blake3_core.h``,
compiled at the first call by NVRTC (``nvrtc*`` through ctypes) into a CUBIN for the device's
architecture, loaded and launched through the CUDA driver (``cu*``) on torch's primary
context and current stream. Nothing here needs a package beyond torch and numpy, and nothing
is compiled or loaded on import. The CUBIN is cached on disk (``native.cache_dir()``), keyed
by the sources, the architecture, NVRTC's version and every compile option.

Work decomposition (the core's header has the details): one thread hashes one 1 KiB chunk and
a block reduces 16 groups of ``G = 8`` chunks in shared memory (stage 0, which reads the
payload once, with 16 neighbouring threads on 64 contiguous bytes), then one 64-thread block
reduces ``R = 64`` nodes per tile in each later stage. A table takes ``1 + ceil(log_R(n0))``
launches, where ``n0`` is its largest leaf's group count: one for leaves up to 8 KiB, two up to
512 KiB, four for a 622 MiB leaf. Tiles of every leaf are numbered flat through prefix sums, so a launch covers
all leaves at once.

Temporary memory for a table of total payload ``P`` bytes: 32 bytes per node of a leaf that
needs more than one stage, which is ``P / 256`` for stage 0 (one node per 8 KiB group) and
1/64 of that again per later stage, so about 0.4 % of ``P``. Per leaf there are 32 bytes of
descriptor, 32 of digest, and 16 per stage of offsets and tile prefix.

This module imports nothing from ``verification`` except ``native``, for ``FINITE_KIND``, the
leaf kinds and the cache directory.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import glob
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verification.commitment import native
from verification.commitment.native import FINITE_KIND

HERE = Path(__file__).resolve().parent
CORE_NAME = "gpu_blake3_core.h"
CU_NAME = "gpu_blake3.cu"
G = 8  # chunks per group, POTS_G in the core
R = 64  # nodes per reduce tile, POTS_R in the core
CHUNK_THREADS = 128
CHUNK_LEN = 1024
MAX_CHUNKS = 1 << 32  # a chunk's counter is 32 bits in the kernel
NO_STREAM = 0
NVRTC_ERROR_INVALID_OPTION = 5
NVRTC_ERROR_BUILTIN_FAILURE = 7  # libnvrtc-builtins could not be loaded

KERNELS = ("pots_chunk_stage", "pots_reduce_stage")


class Params(ctypes.Structure):
    """``pots_params_t`` of the core: one stage's kernel argument, passed by value."""

    _fields_ = [
        ("n", ctypes.c_uint32), ("stage", ctypes.c_uint32), ("total_tiles", ctypes.c_uint64),
        ("leaf", ctypes.c_uint64), ("tile_prefix", ctypes.c_uint64), ("in_off", ctypes.c_uint64),
        ("out_off", ctypes.c_uint64), ("nodes", ctypes.c_uint64), ("digests", ctypes.c_uint64),
        ("first_bad", ctypes.c_uint64),
    ]


PARAM_FIELDS = tuple(name for name, _ in Params._fields_)
# ``pots_leaf_t`` of the core.
LEAF_DTYPE = np.dtype([("payload_ptr", "<u8"), ("payload_len", "<u8"), ("hdr_ptr", "<u8"),
                       ("hdr_len", "<u4"), ("kind", "<u4")])


def source_text() -> tuple[str, str]:
    """``(core header, kernel file)`` as strings."""
    return (HERE / CORE_NAME).read_text(), (HERE / CU_NAME).read_text()


def source_hash() -> str:
    core, cu = source_text()
    return hashlib.sha256(core.encode() + b"\0" + cu.encode()).hexdigest()


# ---- the table and its plan: numpy only, shared by the GPU path and the host test ----

def check_table(hdr_buf_len: int, hdr_off: np.ndarray, hdr_len: np.ndarray, hdr_idx: np.ndarray,
                payload_ptr: np.ndarray, payload_len: np.ndarray, kind: np.ndarray) -> None:
    """The validation of ``native.Native.hash_leaves``, plus the kernel's own limits."""
    n = len(payload_ptr)
    if not len(payload_len) == len(kind) == len(hdr_idx) == n:
        raise ValueError("payload_ptr, payload_len, kind and hdr_idx must have one length")
    if len(hdr_off) != len(hdr_len):
        raise ValueError("hdr_off and hdr_len must have one length")
    if n == 0:
        return
    if int(hdr_idx.max()) >= len(hdr_off):
        raise ValueError("a header index or range is out of bounds")
    if len(hdr_off):
        # ``hdr_off + hdr_len`` could wrap around uint64, so compare ``hdr_off`` with the room left.
        if int(hdr_len.max()) > hdr_buf_len or np.any(
                hdr_off.astype(np.uint64) > np.uint64(hdr_buf_len) - hdr_len.astype(np.uint64)):
            raise ValueError("a header index or range is out of bounds")
    if int(kind.max()) > native.KIND_F16:
        raise ValueError("kind must be one of native.KIND_*")
    hl = hdr_len[hdr_idx].astype(np.uint64)
    if int(hl.max()) > 255:
        raise ValueError("a header is longer than 255 bytes")
    if np.any((kind != native.KIND_NONE) & (((1 + hl) % 4) != 0)):
        raise ValueError("a float leaf's 1 + header length must be a multiple of 4")
    if np.any(payload_len > np.uint64(MAX_CHUNKS * CHUNK_LEN - 512)):
        raise ValueError("a leaf is longer than 4 TiB")


def chunk_counts(hdr_len_per_leaf: np.ndarray, payload_len: np.ndarray) -> np.ndarray:
    """Chunks ``C = ceil((1 + hdr_len + payload_len) / 1024)`` of each leaf."""
    length = hdr_len_per_leaf.astype(np.uint64) + payload_len + np.uint64(1)
    return (length + np.uint64(CHUNK_LEN - 1)) // np.uint64(CHUNK_LEN)


@dataclass
class Plan:
    """The stages of a table. ``tile_prefix[s]`` has ``n + 1`` entries and ``out_off[s]`` has
    ``n``, both uint64; stage ``s``'s ``in_off`` is stage ``s - 1``'s ``out_off``. ``nodes``
    is the node count of the temporary buffer."""

    tile_prefix: list[np.ndarray]
    out_off: list[np.ndarray]
    nodes: int

    @property
    def stages(self) -> int:
        return len(self.tile_prefix)

    def total_tiles(self, s: int) -> int:
        return int(self.tile_prefix[s][-1])


def make_plan(chunks: np.ndarray) -> Plan:
    """Level counts ``ceil(C / G)`` then ``ceil(n / R)`` per level, a stage per level. ``G`` and
    ``R`` are powers of two, and the shifts come from them, as the core's from its logs."""
    g, r = G, R
    gl, rl = np.uint64(g.bit_length() - 1), np.uint64(r.bit_length() - 1)
    levels = [(chunks + np.uint64(g - 1)) >> gl]
    while int(levels[-1].max(initial=0)) > 1:
        levels.append((levels[-1] + np.uint64(r - 1)) >> rl)
    prefix: list[np.ndarray] = []
    out_off: list[np.ndarray] = []
    base = 0
    for s, lv in enumerate(levels):
        tiles = lv if s == 0 else np.where(levels[s - 1] > 1, lv, np.uint64(0))
        p = np.zeros(len(lv) + 1, dtype=np.uint64)
        np.cumsum(tiles, dtype=np.uint64, out=p[1:])
        prefix.append(p)
        size = np.where(lv > 1, lv, np.uint64(0))
        o = np.zeros(len(lv), dtype=np.uint64)
        np.cumsum(size[:-1], dtype=np.uint64, out=o[1:])
        out_off.append(o + np.uint64(base))
        base += int(size.sum(dtype=np.uint64))
    return Plan(prefix, out_off, base)


def leaf_records(hdr_base: int, hdr_off: np.ndarray, hdr_len: np.ndarray, hdr_idx: np.ndarray,
                 payload_ptr: np.ndarray, payload_len: np.ndarray,
                 kind: np.ndarray) -> np.ndarray:
    """The ``pots_leaf_t`` array of a table whose header buffer is at address ``hdr_base``."""
    rec = np.empty(len(payload_ptr), dtype=LEAF_DTYPE)
    rec["payload_ptr"] = payload_ptr
    rec["payload_len"] = payload_len
    rec["hdr_ptr"] = np.uint64(hdr_base) + hdr_off.astype(np.uint64)[hdr_idx]
    rec["hdr_len"] = hdr_len[hdr_idx]
    rec["kind"] = kind
    return rec


def grid_blocks(stage: int, total_tiles: int) -> int:
    """Blocks of a stage's launch: 16 tiles to a block at stage 0, one tile after."""
    return -(-total_tiles // (CHUNK_THREADS // G)) if stage == 0 else total_tiles


def stage_params(plan: Plan, s: int, n: int, addr: dict[str, int]) -> Params:
    """Stage ``s``'s ``Params``; ``addr`` has the device addresses of ``leaf``,
    ``tile_prefix`` and ``out_off`` (lists by stage), ``nodes``, ``digests``, ``first_bad``."""
    tiles = plan.total_tiles(s)
    return Params(n=n, stage=s, total_tiles=tiles, leaf=addr["leaf"],
                  tile_prefix=addr["tile_prefix"][s], in_off=addr["out_off"][s - 1] if s else 0,
                  out_off=addr["out_off"][s], nodes=addr["nodes"], digests=addr["digests"],
                  first_bad=addr["first_bad"])


# ---- NVRTC and the driver, through ctypes ----

def _library_dirs(sub: str) -> list[Path]:
    """Directories where pip's ``nvidia-*`` wheels put their ``lib``: next to the ``nvidia``
    package, and in the site-packages that holds torch."""
    found: list[Path] = []
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        found += [Path(p) / sub / "lib" for p in spec.submodule_search_locations]
    spec = importlib.util.find_spec("torch")
    if spec is not None and spec.origin:
        found.append(Path(spec.origin).resolve().parent.parent / "nvidia" / sub / "lib")
    return found


_I, _V, _S, _Z, _U = (ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                      ctypes.c_uint)
_P = ctypes.POINTER
# ``name: (restype, argtypes)``. Handles are 64 bits, which ctypes' default int conversion would
# cut to 32. ``nvrtcResult`` and ``CUresult`` are ints. No driver call takes a ``CUdeviceptr``:
# the kernels' pointers travel inside ``Params``.
NVRTC_SIGNATURES: dict[str, tuple[Any, list[Any]]] = {
    "nvrtcVersion": (_I, [_P(_I), _P(_I)]),
    "nvrtcGetErrorString": (_S, [_I]),
    "nvrtcCreateProgram": (_I, [_P(_V), _S, _S, _I, _P(_S), _P(_S)]),
    "nvrtcCompileProgram": (_I, [_V, _I, _P(_S)]),
    "nvrtcGetProgramLogSize": (_I, [_V, _P(_Z)]),
    "nvrtcGetProgramLog": (_I, [_V, _V]),
    "nvrtcGetCUBINSize": (_I, [_V, _P(_Z)]),
    "nvrtcGetCUBIN": (_I, [_V, _V]),
    "nvrtcGetPTXSize": (_I, [_V, _P(_Z)]),
    "nvrtcGetPTX": (_I, [_V, _V]),
    "nvrtcDestroyProgram": (_I, [_P(_V)]),
}
DRIVER_SIGNATURES: dict[str, tuple[Any, list[Any]]] = {
    "cuInit": (_I, [_U]),
    "cuGetErrorName": (_I, [_I, _P(_S)]),
    "cuDeviceGet": (_I, [_P(_I), _I]),
    "cuDeviceGetAttribute": (_I, [_P(_I), _I, _I]),
    "cuDevicePrimaryCtxRetain": (_I, [_P(_V), _I]),
    "cuCtxPushCurrent_v2": (_I, [_V]),
    "cuCtxPopCurrent_v2": (_I, [_P(_V)]),
    "cuModuleLoadData": (_I, [_P(_V), _V]),
    "cuModuleGetFunction": (_I, [_P(_V), _V, _S]),
    "cuLaunchKernel": (_I, [_V, _U, _U, _U, _U, _U, _U, _U, _V, _P(_V), _P(_V)]),
}


def declare(lib: Any, table: dict[str, tuple[Any, list[Any]]]) -> Any:
    """Set ``restype`` and ``argtypes`` of every function of ``table`` on ``lib``."""
    for name, (restype, argtypes) in table.items():
        try:
            fn = getattr(lib, name)
        except AttributeError:
            raise RuntimeError(f"{lib} has no {name}; it is too old for gpu_blake3") from None
        fn.restype, fn.argtypes = restype, argtypes
    return lib


def nvrtc_dirs() -> list[Path]:
    """Where ``load_nvrtc`` looks for pip's NVRTC, and where NVRTC finds its builtins."""
    return _library_dirs("cuda_nvrtc")


def load_nvrtc() -> ctypes.CDLL:
    """``libnvrtc.so.12``: by name, else from pip's ``nvidia/cuda_nvrtc/lib``."""
    tried: list[str] = []
    try:
        return declare(ctypes.CDLL("libnvrtc.so.12"), NVRTC_SIGNATURES)
    except OSError as e:
        tried.append(str(e))
    for d in _library_dirs("cuda_nvrtc"):
        for path in sorted(glob.glob(str(d / "libnvrtc.so*"))):
            if "builtins" in os.path.basename(path):
                continue
            try:
                return declare(ctypes.CDLL(path), NVRTC_SIGNATURES)
            except OSError as e:
                tried.append(str(e))
    raise RuntimeError(
        "cannot load libnvrtc.so.12 (" + "; ".join(tried) + "). It ships in pip's "
        "nvidia-cuda-nvrtc-cu12 wheel, in nvidia/cuda_nvrtc/lib next to libnvrtc-builtins.so, "
        "which NVRTC loads itself and so needs in the same directory.")


def load_driver() -> ctypes.CDLL:
    try:
        return declare(ctypes.CDLL("libcuda.so.1"), DRIVER_SIGNATURES)
    except OSError as e:
        raise RuntimeError(f"cannot load the CUDA driver libcuda.so.1: {e}") from e


def builtins_hint(rc: int) -> str:
    """For ``NVRTC_ERROR_BUILTIN_OPERATION_FAILURE``: NVRTC could not load its builtins."""
    if rc != NVRTC_ERROR_BUILTIN_FAILURE:
        return ""
    where = ", ".join(str(d) for d in nvrtc_dirs()) or "no nvidia/cuda_nvrtc/lib directory found"
    return (" (libnvrtc-builtins could not be loaded; it must sit next to libnvrtc, in the "
            f"library path or in {where})")


class Nvrtc:
    """The few NVRTC calls this module makes. ``lib`` is the loaded library (or a fake)."""

    def __init__(self, lib: Any) -> None:
        self.lib = lib

    def _check(self, rc: int, what: str) -> None:
        if rc:
            text = ""
            try:
                text = (self.lib.nvrtcGetErrorString(rc) or b"").decode()
            except (AttributeError, TypeError):
                pass
            raise RuntimeError(f"{what} failed: NVRTC error {rc} {text}{builtins_hint(rc)}")

    def error_string(self, rc: int) -> str:
        try:
            return (self.lib.nvrtcGetErrorString(rc) or b"").decode()
        except (AttributeError, TypeError):
            return ""

    def version(self) -> tuple[int, int]:
        major, minor = ctypes.c_int(), ctypes.c_int()
        self._check(self.lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)),
                    "nvrtcVersion")
        return major.value, minor.value

    def compile(self, src: str, name: str, headers: dict[str, str],
                options: Sequence[str], want: str) -> tuple[int, bytes, str]:
        """``(rc, image, log)``: compile ``src`` with ``headers`` in memory. ``want`` is
        ``"cubin"`` or ``"ptx"``. A nonzero ``rc`` is returned for the caller to judge; every
        other failure raises."""
        lib = self.lib
        prog = ctypes.c_void_p()
        names = [h.encode() for h in headers]
        texts = [t.encode() for t in headers.values()]
        n = len(names)
        self._check(lib.nvrtcCreateProgram(
            ctypes.byref(prog), src.encode(), name.encode(), n,
            (ctypes.c_char_p * n)(*texts), (ctypes.c_char_p * n)(*names)), "nvrtcCreateProgram")
        try:
            opts = [o.encode() for o in options]
            rc = lib.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
            size = ctypes.c_size_t()
            self._check(lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size)), "log size")
            log = ctypes.create_string_buffer(max(size.value, 1))
            self._check(lib.nvrtcGetProgramLog(prog, log), "nvrtcGetProgramLog")
            text = log.value.decode(errors="replace")
            if rc:
                return rc, b"", text
            get_size, get = ((lib.nvrtcGetCUBINSize, lib.nvrtcGetCUBIN) if want == "cubin"
                             else (lib.nvrtcGetPTXSize, lib.nvrtcGetPTX))
            self._check(get_size(prog, ctypes.byref(size)), "image size")
            image = ctypes.create_string_buffer(size.value)
            self._check(get(prog, image), "nvrtcGetCUBIN" if want == "cubin" else "nvrtcGetPTX")
            return 0, image.raw[:size.value], text
        finally:
            lib.nvrtcDestroyProgram(ctypes.byref(prog))


class Driver:
    """The CUDA driver calls this module makes, on a device's primary context (torch's)."""

    def __init__(self, lib: Any) -> None:
        self.lib = lib
        self._ctx: dict[int, Any] = {}
        self.check(lib.cuInit(0), "cuInit")

    def check(self, rc: int, what: str) -> None:
        if rc:
            name = ctypes.c_char_p()
            try:
                self.lib.cuGetErrorName(rc, ctypes.byref(name))
            except (AttributeError, TypeError):
                pass
            raise RuntimeError(f"{what} failed: {(name.value or b'').decode() or rc} ({rc})")

    def device(self, ordinal: int) -> int:
        dev = ctypes.c_int()
        self.check(self.lib.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        return dev.value

    def compute_capability(self, dev: int) -> tuple[int, int]:
        vals = []
        for attr in (75, 76):  # COMPUTE_CAPABILITY_MAJOR, _MINOR
            v = ctypes.c_int()
            self.check(self.lib.cuDeviceGetAttribute(ctypes.byref(v), attr, dev),
                       "cuDeviceGetAttribute")
            vals.append(v.value)
        return vals[0], vals[1]

    def context(self, dev: int) -> Any:
        if dev not in self._ctx:
            ctx = ctypes.c_void_p()
            self.check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
                       "cuDevicePrimaryCtxRetain")
            self._ctx[dev] = ctx
        return self._ctx[dev]

    def push(self, dev: int) -> None:
        self.check(self.lib.cuCtxPushCurrent_v2(self.context(dev)), "cuCtxPushCurrent")

    def pop(self) -> None:
        ctx = ctypes.c_void_p()
        self.check(self.lib.cuCtxPopCurrent_v2(ctypes.byref(ctx)), "cuCtxPopCurrent")

    def load_module(self, image: bytes) -> Any:
        mod = ctypes.c_void_p()
        self.check(self.lib.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(image)),
                   "cuModuleLoadData")
        return mod

    def function(self, mod: Any, name: str) -> Any:
        fn = ctypes.c_void_p()
        self.check(self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
                   f"cuModuleGetFunction({name})")
        return fn

    def launch(self, fn: Any, grid: int, block: int, params: Params, stream: int) -> None:
        argv = (ctypes.c_void_p * 1)(ctypes.addressof(params))  # one pointer to the one argument
        self.check(self.lib.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0,
                                           ctypes.c_void_p(stream), argv, None), "cuLaunchKernel")


def arch_options(cc: tuple[int, int], arch_kind: str) -> list[str]:
    """NVRTC options for a device of compute capability ``cc``: ``sm_XY`` makes a CUBIN and
    ``compute_XY`` makes PTX that the driver compiles at load."""
    flag = f"{'sm' if arch_kind == 'cubin' else 'compute'}_{cc[0]}{cc[1]}"
    return [f"--gpu-architecture={flag}", "--std=c++14"]


@dataclass
class Built:
    """A compiled kernel image."""

    image: bytes
    kind: str  # "cubin" or "ptx"
    arch: str  # "sm_86"
    nvrtc: tuple[int, int]
    cached: bool = False
    stem: Path | None = None  # the cache entry's path without suffix

    def forget(self) -> None:
        """Delete this image's cache entry, for an image the driver would not load."""
        if self.stem is not None:
            for kind in ("cubin", "ptx"):
                self.stem.with_suffix("." + kind).unlink(missing_ok=True)


def cache_key(arch: str, nvrtc: tuple[int, int], options: Sequence[Sequence[str]] = ()) -> str:
    """Hash of the sources, the architecture, NVRTC's version and every compile option."""
    text = f"{source_hash()}|{arch}|{nvrtc[0]}.{nvrtc[1]}|{json.dumps([list(o) for o in options])}"
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def build_image(nvrtc: Nvrtc, cc: tuple[int, int], *, cache: Path | None = None,
                use_cache: bool = True) -> Built:
    """Compile the kernels for compute capability ``cc``: a CUBIN for ``sm_XY``, else PTX for
    ``compute_XY`` when NVRTC rejects that architecture. A compile error raises with NVRTC's
    error string and the log. ``use_cache=False`` skips the read and rewrites the entry."""
    version = nvrtc.version()
    arch = f"sm_{cc[0]}{cc[1]}"
    opts = {k: arch_options(cc, k) for k in ("cubin", "ptx")}
    key = cache_key(arch, version, list(opts.values()))
    stem = None if cache is None else cache / f"gpu_blake3-{key}"
    if stem is not None and use_cache:
        for kind in ("cubin", "ptx"):
            p = stem.with_suffix("." + kind)
            if p.is_file():
                return Built(p.read_bytes(), kind, arch, version, cached=True, stem=stem)
    core, cu = source_text()
    headers = {CORE_NAME: core}
    rc, image, log = nvrtc.compile(cu, CU_NAME, headers, opts["cubin"], "cubin")
    kind = "cubin"
    if rc == NVRTC_ERROR_INVALID_OPTION:
        rc, image, log = nvrtc.compile(cu, CU_NAME, headers, opts["ptx"], "ptx")
        kind = "ptx"
    if rc:
        raise RuntimeError(
            f"NVRTC {version[0]}.{version[1]} cannot compile {CU_NAME} for {arch}: "
            f"error {rc} {nvrtc.error_string(rc)}{builtins_hint(rc)}\n{log}")
    if stem is not None:
        try:
            stem.parent.mkdir(parents=True, exist_ok=True)
            tmp = stem.with_name(stem.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(image)
            os.replace(tmp, stem.with_suffix("." + kind))
        except OSError:
            pass
    return Built(image, kind, arch, version, stem=stem)


class Kernels:
    """A device's loaded module and its two kernel functions."""

    def __init__(self, driver: Driver, dev: int, built: Built) -> None:
        self.driver = driver
        self.dev = dev
        self.built = built
        driver.push(dev)
        try:
            self.module = driver.load_module(built.image)
            self.fn = {k: driver.function(self.module, k) for k in KERNELS}
        finally:
            driver.pop()

    def run(self, launches: Sequence[tuple[Params, int]], stream: int) -> None:
        """Launch each ``(params, blocks)`` in order on ``stream``. The ``Params`` objects
        must stay alive until the driver has copied them, which it does in the call."""
        self.driver.push(self.dev)
        try:
            for params, blocks in launches:
                chunk = params.stage == 0
                self.driver.launch(self.fn["pots_chunk_stage" if chunk else "pots_reduce_stage"],
                                   blocks, CHUNK_THREADS if chunk else R, params,
                                   stream)
        finally:
            self.driver.pop()


@functools.lru_cache(maxsize=None)
def _driver() -> Driver:
    return Driver(load_driver())


@functools.lru_cache(maxsize=None)
def _nvrtc() -> Nvrtc:
    return Nvrtc(load_nvrtc())


@functools.lru_cache(maxsize=None)
def _built(cc: tuple[int, int]) -> Built:
    return build_image(_nvrtc(), cc, cache=native.cache_dir())


def load_kernels(driver: Driver, dev: int, built: Built, rebuild: Callable[[], Built]) -> Kernels:
    """Load ``built``. A cached image the driver rejects is deleted and compiled afresh once."""
    try:
        return Kernels(driver, dev, built)
    except RuntimeError:
        if not built.cached:
            raise
        built.forget()
        return Kernels(driver, dev, rebuild())


@functools.lru_cache(maxsize=None)
def _kernels(ordinal: int) -> Kernels:
    driver = _driver()
    dev = driver.device(ordinal)
    cc = driver.compute_capability(dev)
    return load_kernels(driver, dev, _built(cc), lambda: build_image(
        _nvrtc(), cc, cache=native.cache_dir(), use_cache=False))


def describe(device: int | torch.device | None = None) -> dict[str, object]:
    """Source hash, NVRTC version and the architecture of ``device`` (the current one when
    ``None``), where they can be told. It loads NVRTC at most and never compiles."""
    info: dict[str, object] = {"source_sha256": source_hash(), "nvrtc": None, "arch": None}
    try:
        info["nvrtc"] = ".".join(map(str, _nvrtc().version()))
    except (RuntimeError, OSError) as e:
        info["nvrtc_error"] = str(e)
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(device)
        info["arch"] = f"sm_{cc[0]}{cc[1]}"
        info["device"] = torch.cuda.get_device_name(device)
    return info


# ---- the entry points ----

def _as_np(x: Any, dtype: Any) -> np.ndarray:
    """``x`` as ``dtype``; a value the type cannot hold raises instead of wrapping."""
    a = np.asarray(x)
    if a.dtype.kind in "iu" and a.size:
        info = np.iinfo(dtype)
        if int(a.min()) < 0:
            raise ValueError("lengths, offsets and addresses must not be negative")
        if int(a.max()) > int(info.max):
            raise ValueError(f"a value is above {int(info.max)}, the limit of {np.dtype(dtype)}")
    return a.astype(dtype, copy=False)


@dataclass
class Prepared:
    """A table on the device: ``launches`` is ready to run, ``keep`` holds its buffers."""

    n: int
    digests: torch.Tensor  # uint8 [max(n, 1), 32]
    first_bad: torch.Tensor  # int32 [1], the smallest bad leaf, else n
    launches: list[tuple[Params, int]]
    keep: list[Any]
    payload_bytes: int


class PendingHash:
    """A table enqueued by :meth:`GpuBlake3.enqueue` and not yet read. It holds every buffer the
    stream still touches (the payload tensors included, through ``keep``) until the result is
    read. ``done()`` asks the event and never waits; ``result()`` waits on the event alone."""

    def __init__(self, n: int, host: torch.Tensor, first_bad_at: int, digests_at: int,
                 event: Any, keep: list[Any]) -> None:
        self.n = n
        self._host = host
        self._first_bad_at = first_bad_at
        self._digests_at = digests_at
        self.event = event
        self.keep = keep
        self._result: tuple[list[bytes], int] | None = None

    def done(self) -> bool:
        return self.event is None or bool(self.event.query())

    def result(self) -> tuple[list[bytes], int]:
        """``(digests, first_bad)``: ``first_bad`` is the first leaf with a NaN or Inf, else
        ``n``, and ``digests`` are the valid ones, those below ``first_bad`` (the kernel leaves
        the rest unspecified). The same value on every call."""
        if self._result is None:
            if self.event is not None:
                self.event.synchronize()
            raw = self._host.numpy()
            bad = int(raw[self._first_bad_at:self._first_bad_at + 4].view(np.int32)[0])
            at = self._digests_at
            self._result = ([raw[at + 32 * i:at + 32 * i + 32].tobytes() for i in range(bad)],
                            bad)
            self.keep = []  # the stream is past its last use of the device buffers
        return self._result


def _align(x: int, a: int = 16) -> int:
    return -(-x // a) * a


class GpuBlake3:
    """Hashes tables of leaves on a CUDA device. ``GpuBlake3()`` loads nothing; the first
    ``hash_leaves`` compiles (or loads the cached CUBIN). ``kernels`` replaces the device's
    kernels with anything that has ``run(launches, stream)``: the host test passes the host
    build of the core and a CPU device."""

    def __init__(self, kernels: Any = None) -> None:
        self._kernels = kernels

    def kernels(self, device: torch.device) -> Any:
        if self._kernels is not None:
            return self._kernels
        index = torch.device(device).index
        return _kernels(torch.cuda.current_device() if index is None else index)

    def prepare(self, hdr_buf: Any, hdr_off: np.ndarray, hdr_len: np.ndarray,
                hdr_idx: np.ndarray, payload_ptr: np.ndarray, payload_len: np.ndarray,
                kind: np.ndarray, *, device: torch.device | str) -> Prepared:
        dev = torch.device(device)
        hdr_off, hdr_len = _as_np(hdr_off, np.uint64), _as_np(hdr_len, np.uint32)
        hdr_idx, kind = _as_np(hdr_idx, np.int64), _as_np(kind, np.uint8)
        payload_ptr, payload_len = _as_np(payload_ptr, np.uint64), _as_np(payload_len, np.uint64)
        if isinstance(hdr_buf, torch.Tensor):
            hdr_buf = hdr_buf.cpu().numpy().tobytes()
        hdr_bytes = bytes(hdr_buf)
        check_table(len(hdr_bytes), hdr_off, hdr_len, hdr_idx, payload_ptr, payload_len, kind)
        n = len(payload_ptr)
        hl = hdr_len[hdr_idx] if n else np.empty(0, np.uint32)
        plan = make_plan(chunk_counts(hl, payload_len))

        def up(a: np.ndarray) -> torch.Tensor:
            a = np.ascontiguousarray(a)
            return torch.from_numpy(a.view(np.int64) if a.dtype == np.uint64 else a).to(dev)

        hdr = torch.frombuffer(bytearray(hdr_bytes), dtype=torch.uint8).to(dev) \
            if hdr_bytes else torch.zeros(1, dtype=torch.uint8, device=dev)
        leaf = up(leaf_records(hdr.data_ptr(), hdr_off, hdr_len, hdr_idx, payload_ptr,
                               payload_len, kind).view(np.uint8))
        prefix = [up(p) for p in plan.tile_prefix]
        out_off = [up(o) for o in plan.out_off]
        nodes = torch.empty(max(plan.nodes, 1) * 32, dtype=torch.uint8, device=dev)
        digests = torch.empty((max(n, 1), 32), dtype=torch.uint8, device=dev)
        first_bad = torch.full((1,), n, dtype=torch.int32, device=dev)
        addr = {"leaf": leaf.data_ptr(), "tile_prefix": [p.data_ptr() for p in prefix],
                "out_off": [o.data_ptr() for o in out_off], "nodes": nodes.data_ptr(),
                "digests": digests.data_ptr(), "first_bad": first_bad.data_ptr()}
        launches = []
        for s in range(plan.stages):
            tiles = plan.total_tiles(s)
            if tiles:
                launches.append((stage_params(plan, s, n, addr), grid_blocks(s, tiles)))
        keep = [hdr, leaf, prefix, out_off, nodes]
        return Prepared(n, digests, first_bad, launches, keep, int(payload_len.sum()))

    def launch(self, prep: Prepared, device: torch.device | str) -> None:
        """Enqueue the stages on torch's current stream for ``device``."""
        if prep.launches:
            on_gpu = torch.device(device).type == "cuda"
            stream = torch.cuda.current_stream(device).cuda_stream if on_gpu else NO_STREAM
            self.kernels(torch.device(device)).run(prep.launches, stream)

    def finish(self, prep: Prepared, device: torch.device | str) -> tuple[torch.Tensor, int]:
        """Wait for the kernels, and read ``first_bad``."""
        bad = int(prep.first_bad.item())
        if prep.n == 0:
            return torch.empty((0, 32), dtype=torch.uint8, device=device), 0
        return prep.digests[: prep.n], bad

    def hash_leaves(self, hdr_buf: Any, hdr_off: np.ndarray, hdr_len: np.ndarray,
                    hdr_idx: np.ndarray, payload_ptr: np.ndarray, payload_len: np.ndarray,
                    kind: np.ndarray, *, device: torch.device | str
                    ) -> tuple[torch.Tensor, int]:
        """``(digests, first_bad)`` as ``native.Native.hash_leaves`` gives them, except that the
        payloads are at device addresses and ``digests`` is a uint8 ``[n, 32]`` tensor on
        ``device``. The kernels have finished when this returns; the caller keeps the payload
        tensors alive and unmodified until then. ``first_bad`` is the first leaf whose payload
        holds a NaN or Inf, or ``n``; the digests before it are valid, the rest unspecified."""
        prep = self.prepare(hdr_buf, hdr_off, hdr_len, hdr_idx, payload_ptr, payload_len, kind,
                            device=device)
        self.launch(prep, device)
        return self.finish(prep, device)

    def hash_tensor_leaves(self, headers: Sequence[bytes], tensors: Sequence[torch.Tensor],
                           kinds: Sequence[int] | None = None
                           ) -> tuple[torch.Tensor, int]:
        """``hash_leaves`` over contiguous CUDA tensors on one device: leaf ``i`` is
        ``headers[i]`` and ``tensors[i]``'s bytes, tested as ``kinds[i]`` says (by dtype through
        ``native.FINITE_KIND`` when ``kinds`` is None)."""
        n = len(tensors)
        if len(headers) != n or (kinds is not None and len(kinds) != n):
            raise ValueError("headers, tensors and kinds must have one length")
        if n == 0:
            return torch.empty((0, 32), dtype=torch.uint8), 0
        device = tensors[0].device
        if device.type != "cuda" and self._kernels is None:
            raise ValueError("tensors must be CUDA tensors")
        for t in tensors:
            if t.device != device:
                raise ValueError("tensors must share one device")
            if not t.is_contiguous():
                raise ValueError("tensors must be contiguous")
        if kinds is None:
            try:
                kinds = [FINITE_KIND[t.dtype] for t in tensors]
            except KeyError as e:
                raise ValueError(f"dtype {e.args[0]} has no leaf kind") from None
        hdr_len = np.fromiter((len(h) for h in headers), dtype=np.uint32, count=n)
        hdr_off = np.zeros(n, dtype=np.uint64)
        np.cumsum(hdr_len[:-1], out=hdr_off[1:])
        return self.hash_leaves(
            b"".join(headers), hdr_off, hdr_len, np.arange(n),
            np.fromiter((t.data_ptr() for t in tensors), dtype=np.uint64, count=n),
            np.fromiter((t.nbytes for t in tensors), dtype=np.uint64, count=n),
            np.asarray(kinds, dtype=np.uint8), device=device)


    def enqueue_table(self, hdr_buf: bytes, hdr_off: np.ndarray, hdr_len: np.ndarray,
                      hdr_idx: np.ndarray, payload_ptr: np.ndarray, payload_len: np.ndarray,
                      kind: np.ndarray, *, device: torch.device | str, stream: Any,
                      keep: Sequence[Any] = ()) -> PendingHash:
        """``hash_leaves`` that never waits for the host: the table is built in one pinned
        buffer and copied up with ``non_blocking``, the kernels run on ``stream`` (a
        ``torch.cuda.Stream``; None on the host test's CPU device), the digests and
        ``first_bad`` are copied back into pinned memory and an event is recorded after them.
        ``keep`` (the payload tensors) is held until the result is read."""
        dev = torch.device(device)
        on_gpu = dev.type == "cuda"
        hdr_off, hdr_len = _as_np(hdr_off, np.uint64), _as_np(hdr_len, np.uint32)
        hdr_idx, kind = _as_np(hdr_idx, np.int64), _as_np(kind, np.uint8)
        payload_ptr, payload_len = _as_np(payload_ptr, np.uint64), _as_np(payload_len, np.uint64)
        hdr_bytes = bytes(hdr_buf)
        check_table(len(hdr_bytes), hdr_off, hdr_len, hdr_idx, payload_ptr, payload_len, kind)
        n = len(payload_ptr)
        hl = hdr_len[hdr_idx] if n else np.empty(0, np.uint32)
        plan = make_plan(chunk_counts(hl, payload_len))
        # one layout for the host and the device buffers: headers, leaf records, tile prefixes,
        # offsets, then (copied back) first_bad and the digests; the nodes are device-only
        at_leaf = _align(len(hdr_bytes))
        at = at_leaf + 32 * n
        at_prefix = []
        for p in plan.tile_prefix:
            at_prefix.append(at)
            at += 8 * len(p)
        at_off = []
        for o in plan.out_off:
            at_off.append(at)
            at += 8 * len(o)
        at_bad = _align(at)
        at_dig = at_bad + 16
        total = at_dig + 32 * max(n, 1)
        with torch.cuda.stream(stream) if on_gpu else nullcontext():
            devbuf = torch.empty(total, dtype=torch.uint8, device=dev)
            nodes = torch.empty(max(plan.nodes, 1) * 32, dtype=torch.uint8, device=dev)
        host = torch.zeros(total, dtype=torch.uint8, pin_memory=on_gpu)
        h = host.numpy()
        base = devbuf.data_ptr()
        if hdr_bytes:
            h[:len(hdr_bytes)] = np.frombuffer(hdr_bytes, dtype=np.uint8)
        rec = leaf_records(base, hdr_off, hdr_len, hdr_idx, payload_ptr, payload_len, kind)
        h[at_leaf:at_leaf + 32 * n] = rec.view(np.uint8)
        for a, p in zip(at_prefix, plan.tile_prefix):
            h[a:a + 8 * len(p)] = np.ascontiguousarray(p).view(np.uint8)
        for a, o in zip(at_off, plan.out_off):
            h[a:a + 8 * len(o)] = np.ascontiguousarray(o).view(np.uint8)
        h[at_bad:at_bad + 4] = np.array([n], dtype=np.int32).view(np.uint8)
        addr = {"leaf": base + at_leaf, "tile_prefix": [base + a for a in at_prefix],
                "out_off": [base + a for a in at_off], "nodes": nodes.data_ptr(),
                "digests": base + at_dig, "first_bad": base + at_bad}
        launches = []
        for s in range(plan.stages):
            tiles = plan.total_tiles(s)
            if tiles:
                launches.append((stage_params(plan, s, n, addr), grid_blocks(s, tiles)))
        event = None
        with torch.cuda.stream(stream) if on_gpu else nullcontext():
            devbuf[:at_dig].copy_(host[:at_dig], non_blocking=on_gpu)
            if launches:
                raw = stream.cuda_stream if on_gpu else NO_STREAM
                self.kernels(dev).run(launches, raw)
            host[at_bad:].copy_(devbuf[at_bad:], non_blocking=on_gpu)
            if on_gpu:
                event = torch.cuda.Event()
                event.record(stream)
        return PendingHash(n, host, at_bad, at_dig, event, [devbuf, nodes, host, *keep])

    def enqueue(self, headers: Sequence[bytes], tensors: Sequence[torch.Tensor],
                kinds: Sequence[int] | None = None, *, stream: Any) -> PendingHash:
        """``hash_tensor_leaves`` without waiting (see :meth:`enqueue_table`): leaf ``i`` is
        ``headers[i]`` and the bytes of ``tensors[i]``, contiguous tensors on one device.
        Nothing here reads a device value or synchronizes. The caller keeps the tensors
        unmodified until the pending hash is done (it also holds them)."""
        n = len(tensors)
        if len(headers) != n or (kinds is not None and len(kinds) != n):
            raise ValueError("headers, tensors and kinds must have one length")
        if n == 0:
            raise ValueError("an empty table has nothing to enqueue")
        device = tensors[0].device
        if device.type != "cuda" and self._kernels is None:
            raise ValueError("tensors must be CUDA tensors")
        for t in tensors:
            if t.device != device:
                raise ValueError("tensors must share one device")
            if not t.is_contiguous():
                raise ValueError("tensors must be contiguous")
        if kinds is None:
            try:
                kinds = [FINITE_KIND[t.dtype] for t in tensors]
            except KeyError as e:
                raise ValueError(f"dtype {e.args[0]} has no leaf kind") from None
        hdr_len = np.fromiter((len(h) for h in headers), dtype=np.uint32, count=n)
        hdr_off = np.zeros(n, dtype=np.uint64)
        np.cumsum(hdr_len[:-1], out=hdr_off[1:])
        return self.enqueue_table(
            b"".join(headers), hdr_off, hdr_len, np.arange(n),
            np.fromiter((t.data_ptr() for t in tensors), dtype=np.uint64, count=n),
            np.fromiter((t.nbytes for t in tensors), dtype=np.uint64, count=n),
            np.asarray(kinds, dtype=np.uint8), device=device, stream=stream, keep=list(tensors))


def hash_leaves(*args: Any, **kw: Any) -> tuple[torch.Tensor, int]:
    return GpuBlake3().hash_leaves(*args, **kw)


def hash_tensor_leaves(headers: Sequence[bytes], tensors: Sequence[torch.Tensor],
                       kinds: Sequence[int] | None = None) -> tuple[torch.Tensor, int]:
    return GpuBlake3().hash_tensor_leaves(headers, tensors, kinds)


# ---- tables for the check and the benchmark ----

@dataclass
class Table:
    """A table whose payloads are slices of one byte arena (``arena``, a CPU uint8 tensor), so
    the same table can be pointed at a device copy of the arena."""

    arena: torch.Tensor
    hdr_buf: bytes
    hdr_off: np.ndarray
    hdr_len: np.ndarray
    hdr_idx: np.ndarray
    payload_off: np.ndarray  # offsets into the arena
    payload_len: np.ndarray
    kind: np.ndarray

    def ptrs(self, base: int) -> np.ndarray:
        return self.payload_off + np.uint64(base)

    def args(self, base: int) -> tuple[Any, ...]:
        return (self.hdr_buf, self.hdr_off, self.hdr_len, self.hdr_idx, self.ptrs(base),
                self.payload_len, self.kind)


def random_table(rng: np.random.Generator, n: int, max_len: int, *, bad: int = 0) -> Table:
    """``n`` leaves of random kind, header and payload length up to ``max_len`` at random
    byte offsets; payloads of the float kinds are finite, then ``bad`` random float leaves get
    a NaN or Inf at a random value."""
    kind = rng.integers(0, 4, n).astype(np.uint8)
    hl = np.where(kind == 0, rng.integers(0, 256, n), 3 + 4 * rng.integers(0, 6, n))
    hdr_len = hl.astype(np.uint32)
    hdr_off = np.zeros(n, dtype=np.uint64)
    if n:
        np.cumsum(hdr_len[:-1], out=hdr_off[1:])
    hdr_buf = rng.integers(0, 256, int(hdr_len.sum()), dtype=np.uint8).tobytes()
    size = np.where(rng.random(n) < 0.15, rng.integers(0, 4, n),
                    rng.integers(0, max_len + 1, n)).astype(np.uint64)
    pad = rng.integers(0, 8, n).astype(np.uint64)
    off = np.zeros(n, dtype=np.uint64)
    if n:
        np.cumsum((size + pad + 8)[:-1], out=off[1:])
        off += pad
    total = int((size + pad + 8).sum()) + 8
    arena = torch.from_numpy(rng.integers(0, 256, total, dtype=np.uint8))
    a = arena.numpy()
    for i in range(n):
        o, s = int(off[i]), int(size[i])
        if kind[i] == native.KIND_F32:
            a[o + 3:o + s:4] &= 0xBF
        elif kind[i]:
            a[o + 1:o + s:2] &= 0xBF
    floats = np.flatnonzero((kind != 0) & (size >= 4))
    for i in rng.permutation(floats)[:bad]:
        o, s = int(off[i]), int(size[i])
        w = int(rng.integers(0, s // 4)) * 4
        if kind[i] == native.KIND_F32:
            a[o + w + 3] |= 0x7F
            a[o + w + 2] |= 0x80
        elif kind[i] == native.KIND_BF16:
            a[o + w + 1] |= 0x7F
            a[o + w] |= 0x80
        else:
            a[o + w + 1] |= 0x7C
    return Table(arena, hdr_buf, hdr_off, hdr_len, np.arange(n, dtype=np.uint32), off, size, kind)


# ---- the command line ----

NEEDS_CUDA = "gpu_blake3: needs CUDA, and torch.cuda.is_available() is False here"


def mismatch(got: np.ndarray, got_bad: int, ref: np.ndarray, ref_bad: int) -> str | None:
    """``None`` when the kernel agrees with the C library, else a line naming the first
    differing leaf (digests are compared up to the first bad leaf) and both digests."""
    if got_bad != ref_bad:
        return f"first_bad {got_bad} vs {ref_bad}"
    end = min(ref_bad, len(ref))
    rows = np.flatnonzero((got[:end] != ref[:end]).any(axis=1)) if end else []
    if len(rows):
        i = int(rows[0])
        return f"leaf {i}: got {bytes(got[i]).hex()} vs {bytes(ref[i]).hex()}"
    return None


def check(seeds: int = 6) -> int:
    """Random tables through the kernel against the C library: digests, and ``first_bad``."""
    if not torch.cuda.is_available():
        print(NEEDS_CUDA, file=sys.stderr)
        return 2
    lib = native.get()
    g = GpuBlake3()
    dev = torch.device("cuda")
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for n, max_len, bad in ((1, 40 << 20, 0), (64, 200_000, 0), (200, 20_000, 5),
                                (3000, 3000, 20), (50, 3 << 20, 2), (0, 1, 0)):
            t = random_table(rng, n, max_len, bad=bad)
            ref, ref_bad = lib.hash_leaves(
                t.hdr_buf, t.hdr_off, t.hdr_len, t.hdr_idx, t.ptrs(t.arena.data_ptr()),
                t.payload_len, t.kind, threads=4)
            arena = t.arena.to(dev)
            got, got_bad = g.hash_leaves(*t.args(arena.data_ptr()), device=dev)
            got = got.cpu().numpy()
            why = mismatch(got, got_bad, ref, ref_bad)
            if why:
                print(f"MISMATCH seed={seed} n={n} max_len={max_len}: {why}", file=sys.stderr)
                return 1
        print(f"seed {seed}: ok")
    print(json.dumps(describe(dev)))
    return 0


BANDS = (  # name, leaves, bytes per leaf: about 1 GiB each, but the last
    ("32KiB x 32768", 32768, 32 << 10), ("1MiB x 1024", 1024, 1 << 20),
    ("16MiB x 64", 64, 16 << 20), ("256MiB x 4", 4, 256 << 20), ("622MiB x 1", 1, 622 << 20),
)


def bench(out: Path, repeats: int = 5) -> int:
    """GB/s of payload by leaf size, the kernels alone (CUDA events) and the whole call."""
    if not torch.cuda.is_available():
        print(NEEDS_CUDA, file=sys.stderr)
        return 2
    g = GpuBlake3()
    dev = torch.device("cuda")
    rows = []
    for name, n, size in BANDS:
        free = torch.cuda.mem_get_info()[0]
        if n * size * 1.01 + (64 << 20) > free:
            rows.append({"band": name, "skipped": "not enough free GPU memory"})
            continue
        x = torch.randn(n * size // 4, device=dev, dtype=torch.float32)
        hdr = bytes([0x40, 0x01, 0x01]) + (size // 4).to_bytes(4, "big")
        args = (hdr, np.zeros(1, np.uint64), np.array([len(hdr)], np.uint32), np.zeros(n, np.uint32),
                x.data_ptr() + np.arange(n, dtype=np.uint64) * size, np.full(n, size, np.uint64),
                np.full(n, native.KIND_F32, np.uint8))
        g.hash_leaves(*args, device=dev)  # compile and warm up
        prep = g.prepare(*args, device=dev)
        times = []
        for _ in range(repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.launch(prep, dev)
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / 1e3)
        t0 = time.perf_counter()
        g.hash_leaves(*args, device=dev)
        wall = time.perf_counter() - t0
        total = n * size
        best = min(times)
        rows.append({"band": name, "leaves": n, "leaf_bytes": size, "total_bytes": total,
                     "launches": len(prep.launches), "kernel_s_best": best,
                     "kernel_s_median": float(np.median(times)), "gbps_best": total / best / 1e9,
                     "gbps_median": total / float(np.median(times)) / 1e9,
                     "call_s": wall, "call_gbps": total / wall / 1e9})
        print(rows[-1], flush=True)
        del x, prep
        torch.cuda.empty_cache()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"describe": describe(dev), "bands": rows}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m verification.commitment.gpu_blake3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="random tables against the C library")
    c.add_argument("--seeds", type=int, default=6)
    b = sub.add_parser("bench", help="GB/s by leaf size")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args(argv)
    return check(a.seeds) if a.cmd == "check" else bench(a.out, a.repeats)


if __name__ == "__main__":
    sys.exit(main())
