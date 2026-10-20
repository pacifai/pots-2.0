"""The GPU BLAKE3 kernel (decision F23).

The kernels' work decomposition is the host build of ``gpu_blake3_core.h`` driven by
``gpu_blake3_host.c``, so every case below runs on any machine against the ``blake3`` package
and the C library, through the same ``GpuBlake3`` code the GPU path uses (``prepare``, the plan,
``finish``). The same cases run through the real kernel when a GPU is present. What only a GPU
can show (that NVRTC accepts ``gpu_blake3.cu``, and the speed) is not tested here."""

import ctypes
import subprocess
import sys
from pathlib import Path

import blake3
import numpy as np
import pytest
import torch

from verification.commitment import gpu_blake3 as gb
from verification.commitment import native
from verification.commitment.encoding import tensor_leaf_header
from verification.commitment.merkle import hash_tensor_leaf as _hash_tensor_leaf

ROOT = Path(__file__).resolve().parents[3]
HAVE_GPU = torch.cuda.is_available()
FLOATS = (torch.float32, torch.bfloat16, torch.float16)
DTYPES = (*FLOATS, torch.int32)


class HostKernels:
    """Runs a launch's stages through the host build of the core."""

    def __init__(self, lib):
        self.lib = lib

    def run(self, launches, stream):
        for params, _ in launches:
            self.lib.pots_host_stage(ctypes.byref(params))


@pytest.fixture(scope="module")
def hostlib(tmp_path_factory):
    out = tmp_path_factory.mktemp("host") / "gpu_blake3_host.so"
    src = ROOT / "tests/verification/commitment/gpu_blake3_host.c"
    cmd = ["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-shared", "-fPIC",
           f"-I{ROOT / 'verification/commitment'}", str(src), "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0 and not proc.stderr, proc.stderr
    lib = ctypes.CDLL(str(out))
    lib.pots_host_sizeof_params.restype = ctypes.c_size_t
    lib.pots_host_sizeof_leaf.restype = ctypes.c_size_t
    lib.pots_host_offsets.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
    lib.pots_host_offsets.restype = ctypes.c_size_t
    lib.host_g.restype = lib.host_r.restype = ctypes.c_uint
    lib.pots_host_stage.argtypes = [ctypes.c_void_p]
    return lib


@pytest.fixture(scope="module")
def ref():
    return native.get()


@pytest.fixture(params=["host", pytest.param("gpu", marks=pytest.mark.skipif(
    not HAVE_GPU, reason="needs CUDA"))])
def run(request, hostlib):
    """``run(table) -> (digests uint8 [n, 32] numpy, first_bad)`` on the host build or the GPU."""
    if request.param == "host":
        g = gb.GpuBlake3(HostKernels(hostlib))

        def host(t):
            d, bad = g.hash_leaves(*t.args(t.arena.data_ptr()), device="cpu")
            return d.numpy(), bad
        return host
    g = gb.GpuBlake3()

    def gpu(t):
        arena = t.arena.to("cuda")
        d, bad = g.hash_leaves(*t.args(arena.data_ptr()), device="cuda")
        return d.cpu().numpy(), bad
    return gpu


def make_table(leaves, *, pad=None, seed=0):
    """A ``Table`` of ``(header, payload, kind)`` leaves, each payload copied into the arena at
    a byte offset: ``pad`` bytes past a multiple of 8 (random in 0..7 when None)."""
    rng = np.random.default_rng(seed)
    n = len(leaves)
    hdr_len = np.array([len(h) for h, _, _ in leaves], dtype=np.uint32)
    hdr_off = np.zeros(n, dtype=np.uint64)
    if n:
        np.cumsum(hdr_len[:-1], out=hdr_off[1:])
    pads = [int(rng.integers(0, 8)) if pad is None else pad for _ in leaves]
    arena = bytearray(8)
    off = []
    for (_, p, _), q in zip(leaves, pads):
        arena += bytes(-len(arena) % 8 + q)
        off.append(len(arena))
        arena += bytes(p)
    arena += bytes(8)
    for (_, p, _), o in zip(leaves, off):
        arena[o:o + len(p)] = bytes(p)
    return gb.Table(
        torch.frombuffer(arena, dtype=torch.uint8).clone(), b"".join(h for h, _, _ in leaves),
        hdr_off, hdr_len, np.arange(n, dtype=np.uint32), np.array(off, dtype=np.uint64),
        np.array([len(p) for _, p, _ in leaves], dtype=np.uint64),
        np.array([k for _, _, k in leaves], dtype=np.uint8))


def digests_by_blake3(t):
    a = t.arena.numpy()
    out = []
    for i in range(len(t.payload_len)):
        h = t.hdr_buf[int(t.hdr_off[i]):][:int(t.hdr_len[i])]
        o, n = int(t.payload_off[i]), int(t.payload_len[i])
        out.append(blake3.blake3(b"\x00" + h + a[o:o + n].tobytes()).digest())
    return out


def by_native(ref, t):
    return ref.hash_leaves(t.hdr_buf, t.hdr_off, t.hdr_len, t.hdr_idx,
                           t.ptrs(t.arena.data_ptr()), t.payload_len, t.kind, threads=2)


def agree(run, ref, t):
    """The kernel's digests are the ``blake3`` package's, and its result is the C library's."""
    got, bad = run(t)
    want, want_bad = by_native(ref, t)
    n = len(t.payload_len)
    assert bad == want_bad
    end = min(want_bad, n)
    assert np.array_equal(got[:end], want[:end])
    for i, d in enumerate(digests_by_blake3(t)[:end]):
        assert got[i].tobytes() == d, f"leaf {i}"
    return got, bad


def real_tag(i):
    return 2 + i % 3  # one of the protocol's tensor tags


def hash_tensor_leaf(i, x):
    return _hash_tensor_leaf(real_tag(i), x)


def tensor_leaf(i, x):
    return (tensor_leaf_header(real_tag(i), x), x.reshape(-1).view(torch.uint8).numpy().tobytes(),
            native.FINITE_KIND[x.dtype])


def rand_bytes(n, seed):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


# ---- layout and plan ----

def test_params_layout_matches_the_core(hostlib):
    offsets = (ctypes.c_size_t * 10)()
    assert hostlib.pots_host_offsets(offsets) == 10
    assert list(offsets) == [getattr(gb.Params, f).offset for f in gb.PARAM_FIELDS]
    assert hostlib.pots_host_sizeof_params() == ctypes.sizeof(gb.Params)
    assert hostlib.pots_host_sizeof_leaf() == gb.LEAF_DTYPE.itemsize == 32
    assert (hostlib.host_g(), hostlib.host_r()) == (gb.G, gb.R)


@pytest.mark.parametrize("msg, stages", [
    (1, 1), (2, 1), (8 * 1024 - 1, 1), (8 * 1024, 1), (8 * 1024 + 1, 2), (32 * 1024, 2),
    (512 * 1024 - 1, 2), (512 * 1024, 2), (512 * 1024 + 1, 3), (622 << 20, 4)])
def test_launches_per_table(msg, stages):
    """Launches of a table whose one leaf has a message of ``msg`` bytes."""
    chunks = gb.chunk_counts(np.array([0], np.uint32), np.array([msg - 1], np.uint64))
    plan = gb.make_plan(chunks)
    assert plan.stages == stages
    assert plan.total_tiles(0) == -(-int(chunks[0]) // gb.G)


def test_plan_of_mixed_leaves():
    chunks = np.array([1, 9, 600, 1, 5000], dtype=np.uint64)
    plan = gb.make_plan(chunks)
    assert plan.stages == 3
    # groups: 1, 2, 75, 1, 625; stage 1 tiles: 0, 1, 2, 0, 10; stage 2: only the last leaf
    # (10 nodes of level 1) and the 600-chunk leaf (2 nodes) still have a tile to reduce
    assert [plan.total_tiles(s) for s in range(3)] == [704, 13, 2]
    assert plan.tile_prefix[1].tolist() == [0, 0, 1, 3, 3, 13]


def test_temporary_memory_is_a_small_fraction_of_the_payload():
    nbytes = 622 << 20
    plan = gb.make_plan(gb.chunk_counts(np.array([7], np.uint32), np.array([nbytes], np.uint64)))
    assert plan.nodes * 32 < nbytes * 0.005


# ---- digests against blake3 and the C library ----

def test_empty_table_and_zero_length_payloads(run, ref):
    d, bad = run(make_table([]))
    assert d.shape == (0, 32) and bad == 0
    t = make_table([(b"", b"", 0), (b"abc", b"", 0), (b"\x01\x02\x03", b"", 1),
                    (rand_bytes(255, 1), b"", 0), (b"", b"x", 0)])
    agree(run, ref, t)


CHUNK_COUNTS = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025)


def test_lengths_around_chunk_and_group_edges(run, ref):
    lens = sorted({1024 * c + d for c in CHUNK_COUNTS for d in (-1, 0, 1)})
    hls = (0, 3, 7, 255, 1, 64)
    leaves = []
    for i, m in enumerate(lens):
        hl = hls[i % len(hls)]
        if m - 1 - hl < 0:
            continue
        leaves.append((rand_bytes(hl, i), rand_bytes(m - 1 - hl, 1000 + i), 0))
    agree(run, ref, make_table(leaves, seed=1))


def test_lengths_around_block_edges(run, ref):
    leaves = []
    for b in list(range(1, 40)) + [63, 64, 65, 127, 128]:
        for d in (-1, 0, 1):
            m = 64 * b + d
            hl = (b + d) % 5 * 4 + 3 if (b + d) % 2 else 0
            if m - 1 - hl >= 0:
                leaves.append((rand_bytes(hl, m), rand_bytes(m - 1 - hl, m + 7), 0))
    agree(run, ref, make_table(leaves, seed=2))


def test_blake3_test_vector_inputs(run, ref):
    lens = (1, 1023, 1024, 1025, 2048, 2049, 3072, 3073, 4096, 4097, 5120, 5121, 6144, 6145,
            7168, 7169, 8192, 8193, 16384, 31744, 102400)
    msgs = [bytes(i % 251 for i in range(n)) for n in lens]
    t = make_table([(b"", m[1:], 0) for m in msgs], pad=0)
    got, _ = agree(run, ref, t)
    for g, m in zip(got, msgs):
        assert g.tobytes() == blake3.blake3(m).digest()


@pytest.mark.parametrize("pad", [0, 1, 2, 3, 4, 7])
def test_payload_addresses(run, ref, pad):
    leaves = [(h, rand_bytes(n, n + pad), k)
              for h, n, k in ((b"", 5000, 0), (b"\x01\x02\x03", 4093, 1), (b"abcdefg", 2049, 2),
                              (rand_bytes(11, 3), 8190, 3), (rand_bytes(4, 4), 9, 0))]
    agree(run, ref, make_table(leaves, pad=pad))


SHAPES = {1: (1000,), 2: (33, 65), 3: (5, 7, 9)}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_tensor_leaves_match_merkle(run, ref, dtype, ndim):
    g = torch.Generator().manual_seed(ndim)
    shapes = [SHAPES[ndim], tuple(3 for _ in range(ndim)), tuple(7 for _ in range(ndim)),
              tuple(40 if ndim < 3 else 17 for _ in range(ndim))]
    leaves, want = [], []
    for tag, shape in enumerate(shapes, start=1):
        x = (torch.randint(-(1 << 30), 1 << 30, shape, dtype=dtype, generator=g)
             if dtype == torch.int32 else torch.randn(shape, generator=g).to(dtype))
        leaves.append(tensor_leaf(tag, x))
        want.append(hash_tensor_leaf(tag, x))
    got, bad = run(make_table(leaves, seed=ndim))
    assert bad == len(shapes)
    assert [r.tobytes() for r in got] == want


BAD = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("value", BAD)
def test_non_finite_values_are_found_where_they_sit(run, ref, dtype, value):
    n = 5000  # 3 groups of 8 chunks
    first_group_edge = (8 * 1024 - 8) // dtype.itemsize  # first element of the second group
    spots = [0, n - 1, (1024 - 8) // dtype.itemsize, (1024 - 8) // dtype.itemsize - 1,
             first_group_edge, first_group_edge - 1]
    for spot in spots:
        x = torch.randn(n, generator=torch.Generator().manual_seed(spot)).to(dtype)
        bad = x.clone()
        bad[spot] = BAD[value]
        t = make_table([tensor_leaf(1, x), tensor_leaf(2, bad), tensor_leaf(3, x)], seed=spot)
        got, first = agree(run, ref, t)
        assert first == 1
        assert got[0].tobytes() == hash_tensor_leaf(1, x)


def test_the_earliest_bad_leaf_wins_and_earlier_digests_stand(run, ref):
    xs = [torch.randn(3000, generator=torch.Generator().manual_seed(i)) for i in range(7)]
    for i in (4, 2, 6):
        xs[i][2999 - i] = float("nan") if i != 2 else float("-inf")
    t = make_table([tensor_leaf(i + 1, x) for i, x in enumerate(xs)], seed=5)
    got, first = agree(run, ref, t)
    assert first == 2
    assert [g.tobytes() for g in got[:2]] == [hash_tensor_leaf(i + 1, xs[i]) for i in range(2)]


def test_a_trailing_half_value_is_not_tested(run, ref):
    # bf16 payload of 7 bytes holding 3 values and a lone byte of 0xFF: native.c ignores it
    t = make_table([(b"\x01\x02\x03", bytes([0, 0, 0, 0, 0, 0, 0xFF]), 2),
                    (b"\x01\x02\x03", bytes([0, 0, 0, 0, 0, 0, 0xFF, 0x7F]), 2)])
    _, bad = agree(run, ref, t)
    assert bad == 1


def test_negative_zero_subnormals_and_int_bit_patterns_pass(run, ref):
    f = torch.tensor([-0.0, 1e-45, -1e-40, 0.0, 1.17549435e-38], dtype=torch.float32)
    h = torch.tensor([-0.0, 6e-8, 5.96e-8, 6.1e-5], dtype=torch.float16)
    b = torch.tensor([-0.0, 9.2e-41, 1e-38], dtype=torch.bfloat16)
    ints = torch.tensor([0x7FC00000, 0x7F800000, -8388608, 0x7C00, 0x7F80, -1], dtype=torch.int32)
    t = make_table([tensor_leaf(k + 1, x) for k, x in enumerate((f, h, b, ints))])
    got, bad = agree(run, ref, t)
    assert bad == 4
    assert got[3].tobytes() == hash_tensor_leaf(4, ints)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_mixed_tables(run, ref, seed):
    rng = np.random.default_rng(seed)
    agree(run, ref, gb.random_table(rng, 60, 40_000, bad=0))
    agree(run, ref, gb.random_table(rng, 30, 20_000, bad=4))
    agree(run, ref, gb.random_table(rng, 400, 600, bad=30))


def test_a_multi_level_leaf_among_small_ones(run, ref):
    leaves = [(b"\x01\x02\x03", rand_bytes(n, n), 0) for n in (3, 1 << 20, 17, 600_000, 0)]
    agree(run, ref, make_table(leaves))


# ---- validation ----

def z(n, dtype=np.uint64):
    return np.zeros(n, dtype=dtype)


@pytest.mark.parametrize("args, message", [
    ((b"", z(0), z(0, np.uint32), z(2, np.uint32), z(2), z(3), z(2, np.uint8)), "one length"),
    ((b"", z(1), z(2, np.uint32), z(1, np.uint32), z(1), z(1), z(1, np.uint8)), "one length"),
    ((b"", z(1), z(1, np.uint32), np.ones(1, np.uint32), z(1), z(1), z(1, np.uint8)),
     "out of bounds"),
    ((b"abc", z(1), np.array([3], np.uint32), np.array([1], np.uint32), z(1), z(1),
      z(1, np.uint8)), "out of bounds"),
    ((b"abcd", np.array([2], np.uint64), np.array([3], np.uint32), z(1, np.uint32), z(1),
      z(1), z(1, np.uint8)), "out of bounds"),
    ((b"abc", z(1), np.array([2], np.uint32), z(1, np.uint32), z(1), z(1),
      np.array([1], np.uint8)), "multiple of 4"),
    ((b"abc", z(1), np.array([3], np.uint32), z(1, np.uint32), z(1), z(1),
      np.array([4], np.uint8)), "kind"),
    ((b"abc", z(1), np.array([3], np.uint32), z(1, np.uint32), z(1),
      np.array([1 << 43], np.uint64), z(1, np.uint8)), "4 TiB"),
])
def test_table_validation(args, message):
    with pytest.raises(ValueError, match=message):
        gb.GpuBlake3().hash_leaves(*args, device="cuda")


def test_a_header_of_3_bytes_with_a_float_kind_is_accepted(hostlib):
    g = gb.GpuBlake3(HostKernels(hostlib))
    d, bad = g.hash_leaves(b"abc", z(1), np.array([3], np.uint32), z(1, np.uint32), z(1), z(1),
                           np.array([1], np.uint8), device="cpu")
    assert bad == 1 and d.shape == (1, 32)


def test_hash_tensor_leaves_checks_its_tensors(hostlib):
    g = gb.GpuBlake3()
    with pytest.raises(ValueError, match="one length"):
        g.hash_tensor_leaves([b""], [])
    with pytest.raises(ValueError, match="CUDA"):
        g.hash_tensor_leaves([b"abc"], [torch.zeros(3)])
    d, bad = g.hash_tensor_leaves([], [])
    assert d.shape == (0, 32) and bad == 0
    h = gb.GpuBlake3(HostKernels(hostlib))
    x = torch.randn(10)
    for bad_input in (x[::2], ):
        with pytest.raises(ValueError, match="contiguous"):
            h.hash_tensor_leaves([tensor_leaf_header(2, x)], [bad_input])
    d, bad = h.hash_tensor_leaves([tensor_leaf_header(2, x)], [x])
    assert bad == 1 and d[0].numpy().tobytes() == hash_tensor_leaf(0, x)
    with pytest.raises(ValueError, match="dtype"):
        h.hash_tensor_leaves([b"abc"], [torch.zeros(3, dtype=torch.float64)])


# ---- the loader, against fake NVRTC and driver libraries ----

class FakeNvrtc:
    def __init__(self, *, reject_sm=False, compile_rc=0):
        self.reject_sm, self.compile_rc = reject_sm, compile_rc
        self.compiles: list[list[bytes]] = []
        self.created: list[tuple] = []
        self.destroyed = 0

    def nvrtcVersion(self, a, b):
        a._obj.value, b._obj.value = 12, 4
        return 0

    def nvrtcCreateProgram(self, prog, src, name, n, texts, names):
        self.created.append((src, name, [texts[i] for i in range(n)], [names[i] for i in range(n)]))
        prog._obj.value = 1
        return 0

    def nvrtcCompileProgram(self, prog, n, opts):
        o = [opts[i] for i in range(n)]
        self.compiles.append(o)
        if self.reject_sm and any(x.startswith(b"--gpu-architecture=sm_") for x in o):
            return 5
        return self.compile_rc

    def nvrtcGetProgramLogSize(self, prog, size):
        size._obj.value = len(self.log()) + 1
        return 0

    def log(self):
        return b"error: bad thing" if self.compile_rc else b""

    def nvrtcGetProgramLog(self, prog, buf):
        ctypes.memmove(buf, self.log() + b"\0", len(self.log()) + 1)
        return 0

    def image(self):
        return b"CUBIN" if b"sm_" in b"".join(self.compiles[-1]) else b"PTX"

    def nvrtcGetCUBINSize(self, prog, size):
        size._obj.value = len(self.image())
        return 0

    nvrtcGetPTXSize = nvrtcGetCUBINSize

    def nvrtcGetCUBIN(self, prog, buf):
        ctypes.memmove(buf, self.image(), len(self.image()))
        return 0

    nvrtcGetPTX = nvrtcGetCUBIN

    def nvrtcDestroyProgram(self, prog):
        self.destroyed += 1
        return 0

    def nvrtcGetErrorString(self, rc):
        return b"NVRTC_ERROR_FAKE"


def test_build_image_makes_a_cubin_for_the_device_arch(tmp_path):
    lib = FakeNvrtc()
    built = gb.build_image(gb.Nvrtc(lib), (8, 6), cache=tmp_path)
    assert (built.kind, built.arch, built.image, built.nvrtc) == ("cubin", "sm_86", b"CUBIN",
                                                                  (12, 4))
    assert lib.compiles == [[b"--gpu-architecture=sm_86", b"--std=c++14"]]
    src, name, texts, names = lib.created[0]
    assert names == [gb.CORE_NAME.encode()]
    assert texts == [(gb.HERE / gb.CORE_NAME).read_bytes()]
    assert src == (gb.HERE / gb.CU_NAME).read_bytes() and name == gb.CU_NAME.encode()
    assert lib.destroyed == 1


def test_build_image_falls_back_to_ptx_when_nvrtc_rejects_the_arch(tmp_path):
    lib = FakeNvrtc(reject_sm=True)
    built = gb.build_image(gb.Nvrtc(lib), (9, 0), cache=tmp_path)
    assert (built.kind, built.arch, built.image) == ("ptx", "sm_90", b"PTX")
    assert lib.compiles[1][0] == b"--gpu-architecture=compute_90"


def test_build_image_caches_on_disk_by_source_arch_and_version(tmp_path):
    first = FakeNvrtc()
    gb.build_image(gb.Nvrtc(first), (8, 0), cache=tmp_path)
    second = FakeNvrtc()
    built = gb.build_image(gb.Nvrtc(second), (8, 0), cache=tmp_path)
    assert built.cached and built.image == b"CUBIN" and not second.compiles
    gb.build_image(gb.Nvrtc(second), (8, 6), cache=tmp_path)
    assert len(second.compiles) == 1


def test_a_compile_error_raises_with_the_log(tmp_path):
    with pytest.raises(RuntimeError, match="error: bad thing"):
        gb.build_image(gb.Nvrtc(FakeNvrtc(compile_rc=6)), (8, 6), cache=tmp_path)
    assert list(tmp_path.iterdir()) == []


class FakeDriver:
    def __init__(self, fail=None):
        self.fail, self.calls, self.launched = fail, [], []

    def _rc(self, name):
        self.calls.append(name)
        return 7 if name == self.fail else 0

    def cuInit(self, flags):
        return self._rc("cuInit")

    def cuDeviceGet(self, dev, ordinal):
        dev._obj.value = ordinal + 10
        return self._rc("cuDeviceGet")

    def cuDeviceGetAttribute(self, v, attr, dev):
        v._obj.value = {75: 8, 76: 6}[attr]
        return self._rc("cuDeviceGetAttribute")

    def cuDevicePrimaryCtxRetain(self, ctx, dev):
        ctx._obj.value = 99
        return self._rc("cuDevicePrimaryCtxRetain")

    def cuCtxPushCurrent_v2(self, ctx):
        return self._rc("cuCtxPushCurrent_v2")

    def cuCtxPopCurrent_v2(self, ctx):
        return self._rc("cuCtxPopCurrent_v2")

    def cuModuleLoadData(self, mod, image):
        mod._obj.value = 5
        self.image = image.value
        return self._rc("cuModuleLoadData")

    def cuModuleGetFunction(self, fn, mod, name):
        fn._obj.value = 100 + len(name)
        return self._rc("cuModuleGetFunction:" + name.decode())

    def cuLaunchKernel(self, fn, gx, gy, gz, bx, by, bz, smem, stream, argv, extra):
        self.launched.append((fn.value, gx, gy, gz, bx, by, bz, smem, stream.value,
                              ctypes.string_at(argv[0], ctypes.sizeof(gb.Params))))
        return self._rc("cuLaunchKernel")

    def cuGetErrorName(self, rc, name):
        name._obj.value = b"CUDA_ERROR_FAKE"
        return 0


def test_driver_reads_the_compute_capability_and_loads_the_module():
    lib = FakeDriver()
    d = gb.Driver(lib)
    dev = d.device(1)
    assert dev == 11 and d.compute_capability(dev) == (8, 6)
    k = gb.Kernels(d, dev, gb.Built(b"IMAGE", "cubin", "sm_86", (12, 4)))
    assert lib.image == b"IMAGE"
    assert lib.calls[-6:] == ["cuDevicePrimaryCtxRetain", "cuCtxPushCurrent_v2", "cuModuleLoadData",
                              "cuModuleGetFunction:pots_chunk_stage",
                              "cuModuleGetFunction:pots_reduce_stage", "cuCtxPopCurrent_v2"]
    assert set(k.fn) == set(gb.KERNELS)


def test_kernel_arguments_are_packed_as_the_struct_and_launched_in_order():
    lib = FakeDriver()
    d = gb.Driver(lib)
    k = gb.Kernels(d, 0, gb.Built(b"I", "cubin", "sm_86", (12, 4)))
    p0 = gb.Params(n=3, stage=0, total_tiles=40, leaf=1, tile_prefix=2, in_off=0, out_off=3,
                   nodes=4, digests=5, first_bad=6)
    p1 = gb.Params(n=3, stage=1, total_tiles=2, leaf=1, tile_prefix=2, in_off=3, out_off=7,
                   nodes=4, digests=5, first_bad=6)
    lib.calls.clear()
    k.run([(p0, 3), (p1, 2)], stream=1234)
    assert lib.calls == ["cuCtxPushCurrent_v2", "cuLaunchKernel", "cuLaunchKernel",
                         "cuCtxPopCurrent_v2"]
    (f0, *a0, s0, b0), (f1, *a1, s1, b1) = lib.launched
    assert (a0, s0, b0) == ([3, 1, 1, 128, 1, 1, 0], 1234, bytes(p0))
    assert (a1, s1, b1) == ([2, 1, 1, 64, 1, 1, 0], 1234, bytes(p1))
    assert f0 != f1
    assert ctypes.sizeof(gb.Params) == 72


def test_a_driver_error_names_the_call_and_the_error():
    with pytest.raises(RuntimeError, match=r"cuInit failed: CUDA_ERROR_FAKE \(7\)"):
        gb.Driver(FakeDriver(fail="cuInit"))
    d = gb.Driver(FakeDriver(fail="cuModuleLoadData"))
    with pytest.raises(RuntimeError, match="cuModuleLoadData failed"):
        d.load_module(b"x")


def test_missing_nvrtc_gives_a_clear_error(monkeypatch):
    def refuse(name):
        raise OSError("not found")
    monkeypatch.setattr(gb.ctypes, "CDLL", refuse)
    monkeypatch.setattr(gb, "_library_dirs", lambda sub: [])
    with pytest.raises(RuntimeError, match="libnvrtc-builtins"):
        gb.load_nvrtc()


def test_describe_never_compiles(monkeypatch):
    lib = FakeNvrtc()
    monkeypatch.setattr(gb, "_nvrtc", lambda: gb.Nvrtc(lib))
    info = gb.describe()
    assert info["nvrtc"] == "12.4" and info["source_sha256"] == gb.source_hash()
    assert lib.compiles == [] and lib.created == []


def test_import_compiles_and_loads_nothing():
    code = ("import verification.commitment.gpu_blake3 as g;"
            "assert all(f.cache_info().currsize == 0 for f in (g._driver, g._nvrtc, g._built,"
            " g._kernels))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(HAVE_GPU, reason="needs a machine without CUDA")
@pytest.mark.parametrize("cmd", [["check"], ["bench", "--out", "unused.json"]])
def test_the_command_line_refuses_cleanly_without_cuda(cmd, capsys):
    assert gb.main(cmd) != 0
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "needs CUDA" in err


def test_the_cu_file_is_a_thin_layer_over_the_core_phases():
    cu = (gb.HERE / gb.CU_NAME).read_text()
    assert '#include "gpu_blake3_core.h"' in cu
    for name in gb.KERNELS:
        assert f"{name}(pots_params_t P)" in cu


STUB = """
struct pots_dim3 { unsigned x, y, z; };
static pots_dim3 blockIdx, threadIdx;
static void __syncthreads() {}
#define __global__
#define __shared__
#define __launch_bounds__(n)
"""


def test_the_cu_file_parses_as_c_plus_plus_against_a_stub_of_cuda(tmp_path):
    """NVRTC isn't here, so this catches what a compiler front end can: a misspelt phase or
    argument, a missing include. The device-only branches (``atomicMin``) are not seen."""
    stub = tmp_path / "stub.h"
    stub.write_text(STUB)
    cmd = ["c++", "-std=c++14", "-fsyntax-only", "-Wall", "-Wextra", "-x", "c++", "-include",
           str(stub), f"-I{gb.HERE}", str(gb.HERE / gb.CU_NAME)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("no C++ compiler")
    assert proc.returncode == 0, proc.stderr


# ---- on a GPU only ----

@pytest.mark.skipif(not HAVE_GPU, reason="needs CUDA")
def test_one_leaf_of_600_mb_matches_blake3():
    free = torch.cuda.mem_get_info()[0]
    if free < 8 << 30:
        pytest.skip("needs 8 GiB free")
    n = 622 << 20
    x = torch.randint(0, 256, (n,), dtype=torch.uint8, device="cuda")
    hdr = b"\x01\x02\x03"
    d, bad = gb.hash_leaves(hdr, z(1), np.array([3], np.uint32), z(1, np.uint32),
                            np.array([x.data_ptr()], np.uint64), np.array([n], np.uint64),
                            z(1, np.uint8), device="cuda")
    h = blake3.blake3(b"\x00" + hdr, max_threads=blake3.blake3.AUTO)
    h.update(x.cpu().numpy())
    assert bad == 0 and d[0].cpu().numpy().tobytes() == h.digest()


@pytest.mark.skipif(not HAVE_GPU, reason="needs CUDA")
def test_a_table_of_50000_leaves_of_32_kib(ref):
    n, size = 50_000, 32 << 10
    x = torch.randint(0, 256, (n * size,), dtype=torch.uint8, device="cuda")
    hdr = b"\x01\x02\x03"
    args = (hdr * n, np.arange(n, dtype=np.uint64) * 3, np.full(n, 3, np.uint32),
            np.arange(n, dtype=np.uint32), x.data_ptr() + np.arange(n, dtype=np.uint64) * size,
            np.full(n, size, np.uint64), np.zeros(n, np.uint8))
    d, bad = gb.hash_leaves(*args, device="cuda")
    host = x.cpu().numpy()
    want, _ = ref.hash_leaves(args[0], args[1], args[2], args[3],
                              host.ctypes.data + np.arange(n, dtype=np.uint64) * size,
                              args[5], args[6], threads=8)
    assert bad == 0 and np.array_equal(d.cpu().numpy(), want)
