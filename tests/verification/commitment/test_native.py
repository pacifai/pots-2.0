import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from verification.commitment import native
from verification.commitment.encoding import _all_finite, tensor_leaf_header
from verification.commitment.merkle import MerkleTree, hash_leaf, merkle_root
from verification.verifier.matmul_check.challenges import challenge_matrices

ROOT = Path(__file__).resolve().parents[3]
FLOATS = (torch.float32, torch.bfloat16, torch.float16)
DTYPES = (*FLOATS, torch.int32)
THREADS = (1, 2, 8)


@pytest.fixture(scope="module")
def lib():
    return native.get()


def rand(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.int32:
        return torch.randint(-(1 << 30), 1 << 30, shape, dtype=dtype, generator=g)
    return torch.randn(shape, generator=g).to(dtype)


def payload_bytes(t):
    return t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def hash_tensors(lib, tensors, *, threads, tag=0x03):
    """Run ``hash_leaves`` over ``tensors``, sharing one header entry per distinct header."""
    headers: dict[bytes, int] = {}
    idx = []
    for t in tensors:
        idx.append(headers.setdefault(tensor_leaf_header(tag, t), len(headers)))
    buf = b"".join(headers)
    lens = np.array([len(h) for h in headers], dtype=np.uint32)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.uint64)
    kinds = np.array([native.FINITE_KIND[t.dtype] for t in tensors], dtype=np.uint8)
    return lib.hash_leaves(
        buf, offs, lens, np.array(idx, dtype=np.uint32),
        np.array([t.data_ptr() for t in tensors], dtype=np.uint64),
        np.array([t.nbytes for t in tensors], dtype=np.uint64), kinds, threads=threads)


def expected(tensors, tag=0x03):
    return [hash_leaf(tensor_leaf_header(tag, t), payload_bytes(t)) for t in tensors]


SHAPES = [(0,), (1,), (3,), (7, 5), (16383,), (16384,), (16385,), (3, 5461, 3), (262143,),
          (262144,), (262145,), (5, 1 << 18)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("threads", THREADS)
def test_hash_leaves_matches_reference(lib, dtype, threads):
    ts = [rand(s, dtype, i) for i, s in enumerate(SHAPES)]
    ts += [rand((17,), dtype, 100 + i) for i in range(6)]  # shared header
    digests, bad = hash_tensors(lib, ts, threads=threads)
    assert bad == len(ts)
    assert [bytes(d) for d in digests] == expected(ts)


@pytest.mark.parametrize("dtype", DTYPES)
def test_hash_leaves_large(lib, dtype):
    t = rand(((20 << 20) // dtype.itemsize + 3,), dtype, 3)
    assert t.nbytes >= 20 << 20
    for threads in THREADS:
        digests, bad = hash_tensors(lib, [t, t[:5]], threads=threads)
        assert bad == 2
        assert [bytes(d) for d in digests] == expected([t, t[:5]])


def test_hash_leaves_views_of_one_storage(lib):
    big = rand((64, 33, 7), torch.float32, 1)
    ts = list(big.unbind(0))
    for threads in THREADS:
        digests, bad = hash_tensors(lib, ts, threads=threads)
        assert bad == len(ts)
        assert [bytes(d) for d in digests] == expected(ts)


def test_hash_leaves_many_small_leaves(lib):
    ts = [rand((13,), torch.float32, i) for i in range(5000)]
    digests, bad = hash_tensors(lib, ts, threads=8)
    assert bad == len(ts)
    assert [bytes(d) for d in digests] == expected(ts)


def test_hash_leaves_empty_and_empty_header(lib):
    digests, bad = lib.hash_leaves(
        b"", np.empty(0, np.uint64), np.empty(0, np.uint32), np.empty(0, np.uint32),
        np.empty(0, np.uint64), np.empty(0, np.uint64), np.empty(0, np.uint8), threads=4)
    assert digests.shape == (0, 32) and bad == 0
    t = rand((9,), torch.float32)
    digests, bad = lib.hash_leaves(
        b"", np.zeros(1, np.uint64), np.zeros(1, np.uint32), np.zeros(1, np.uint32),
        np.array([t.data_ptr()], np.uint64), np.array([t.nbytes], np.uint64),
        np.array([1], np.uint8), threads=1)
    assert bad == 1 and bytes(digests[0]) == hash_leaf(payload_bytes(t))


def test_hash_leaves_length_mismatch(lib):
    z = np.zeros(1, np.uint64)
    with pytest.raises(ValueError):
        lib.hash_leaves(b"", z, np.zeros(1, np.uint32), np.zeros(1, np.uint32), z,
                        np.zeros(2, np.uint64), np.zeros(1, np.uint8), threads=1)


def poison(t, pos, value):
    t = t.clone()
    t.reshape(-1)[pos] = value
    return t


@pytest.mark.parametrize("dtype", FLOATS)
@pytest.mark.parametrize("threads", THREADS)
def test_first_bad_is_the_first_leaf_in_order(lib, dtype, threads):
    rng = np.random.default_rng(5)
    for trial in range(6):
        ts = [rand((300,), dtype, i) for i in range(20)]
        for i in (9, 5):
            ts[i] = poison(ts[i], int(rng.integers(300)),
                           [float("nan"), float("inf"), -float("inf")][trial % 3])
        digests, bad = hash_tensors(lib, ts, threads=threads)
        assert bad == 5
        assert [bytes(d) for d in digests[:5]] == expected(ts[:5])


@pytest.mark.parametrize("dtype", FLOATS)
def test_nonfinite_as_last_element_of_a_large_leaf(lib, dtype):
    n = (1 << 16) * 3 // dtype.itemsize + 1
    for value in (float("nan"), float("inf"), -float("inf")):
        for threads in THREADS:
            t = poison(rand((n,), dtype), n - 1, value)
            assert t.nbytes > 1 << 16
            _, bad = hash_tensors(lib, [rand((8,), dtype), t], threads=threads)
            assert bad == 1
            assert not lib.all_finite(t.data_ptr(), t.nbytes, native.FINITE_KIND[dtype],
                                      threads=threads)


def test_int32_nan_patterns_are_not_flagged(lib):
    bits = torch.tensor([0x7fc00000, 0x7f800000, 0xff800000 - (1 << 32)], dtype=torch.int32)
    t = bits.repeat(1000)
    digests, bad = hash_tensors(lib, [t], threads=2)
    assert bad == 1 and bytes(digests[0]) == expected([t])[0]
    assert lib.all_finite(t.data_ptr(), t.nbytes, native.KIND_NONE, threads=2)


def test_negative_zero_and_subnormals_pass(lib):
    cases = {
        torch.float32: (-0.0, torch.tensor([1], dtype=torch.int32).view(torch.float32)),
        torch.bfloat16: (-0.0, torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)),
        torch.float16: (-0.0, torch.tensor([1], dtype=torch.int16).view(torch.float16)),
    }
    for dtype, (nz, sub) in cases.items():
        t = torch.cat([torch.tensor([nz], dtype=dtype), sub, rand((10,), dtype)])
        digests, bad = hash_tensors(lib, [t], threads=1)
        assert bad == 1 and bytes(digests[0]) == expected([t])[0]
        assert lib.all_finite(t.data_ptr(), t.nbytes, native.FINITE_KIND[dtype], threads=1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("threads", (1, 4, 16))
def test_all_finite_matches_encoding(lib, dtype, threads):
    cases = [rand(s, dtype, i) for i, s in enumerate(SHAPES)]
    cases.append(rand((3 << 20,), dtype, 9))  # several threads' worth
    if dtype in FLOATS:
        big = rand((3 << 20,), dtype, 9)
        for pos in (0, 1, (3 << 20) // 2, (3 << 20) - 1):
            for v in (float("nan"), float("inf"), -float("inf")):
                cases.append(poison(big, pos, v))
        cases.append(poison(rand((5,), dtype), 4, float("nan")))
    for t in cases:
        want = _all_finite(t) if dtype in FLOATS else True
        assert lib.all_finite(t.data_ptr(), t.nbytes, native.FINITE_KIND[dtype],
                              threads=threads) == want


def preorder(nodes, n):
    """``{(lo, hi): hash}`` read from the pre-order array."""
    out = {}

    def walk(lo, hi, p):
        out[(lo, hi)] = nodes[32 * p:32 * p + 32]
        if hi - lo > 1:
            k = 1 << ((hi - lo - 1).bit_length() - 1)
            walk(lo, lo + k, p + 1)
            walk(lo + k, hi, p + 2 * k)

    walk(0, n, 0)
    return out


@pytest.mark.parametrize("n", [*range(1, 71), 127, 128, 129, 1000, 4097])
def test_merkle_tree_matches_python(lib, n):
    hashes = [hash_leaf(i.to_bytes(4, "big")) for i in range(n)]
    nodes = lib.merkle_tree(b"".join(hashes), n)
    assert len(nodes) == 32 * (2 * n - 1)
    assert nodes[:32] == merkle_root(hashes)
    tree = MerkleTree(hashes)
    assert nodes == bytes(tree._buf)
    for (lo, hi), h in preorder(nodes, n).items():
        assert h == tree.node(lo, hi)


def test_merkle_tree_rejects_bad_input(lib):
    with pytest.raises(ValueError):
        lib.merkle_tree(b"", 0)
    with pytest.raises(ValueError):
        lib.merkle_tree(b"\x00" * 31, 1)


@pytest.mark.parametrize("threads", (1, 8))
@pytest.mark.parametrize("k", (1, 3, 21))
@pytest.mark.parametrize("width", (1, 5, 256, 257, 1000))
def test_challenge_matrices_bit_identical(lib, width, k, threads):
    h = bytes(range(32))
    for ms in ([1], [7, 2, 4_000_000_000], list(range(1, 40))):
        want = challenge_matrices(h, ms, k, width)
        got = lib.challenge_matrices(h, ms, k, width, threads=threads)
        assert got.shape == want.shape and got.dtype == torch.float32
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))


def test_challenge_matrices_guards(lib):
    for ms, width in (([1], 0), ([1], 1 << 32), ([0], 5), ([1 << 32], 5), ([3, -1], 5)):
        with pytest.raises(ValueError):
            lib.challenge_matrices(bytes(32), ms, 3, width, threads=1)


def test_all_finite_guards(lib):
    with pytest.raises(ValueError):
        lib.all_finite(1, -1, native.KIND_F32, threads=1)
    with pytest.raises(ValueError):
        lib.all_finite(0, 8, native.KIND_F32, threads=1)
    assert lib.all_finite(0, 0, native.KIND_F32, threads=1)


def test_challenge_matrices_empty(lib):
    out = lib.challenge_matrices(bytes(32), [], 3, 11, threads=2)
    assert out.shape == (0, 11, 3) and out.dtype == torch.float32


def test_import_builds_and_loads_nothing():
    code = ("import time, torch, numpy; t = time.perf_counter();"
            "import verification.commitment.native as n;"
            "assert n._LIB is None; assert time.perf_counter() - t < 0.5, 'slow import'")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_disabled_nests_and_restores():
    assert native.enabled()
    with native.disabled():
        assert not native.enabled()
        with native.disabled():
            assert not native.enabled()
        assert not native.enabled()
        with pytest.raises(RuntimeError):
            with native.disabled():
                raise RuntimeError
        assert not native.enabled()
    assert native.enabled()
    with pytest.raises(RuntimeError):
        with native.disabled():
            raise RuntimeError
    assert native.enabled()


# ---- describe ------------------------------------------------------------------------------


def test_describe_never_builds(tmp_path, monkeypatch):
    monkeypatch.setenv("POTS_NATIVE_CACHE", str(tmp_path))
    monkeypatch.setattr(native, "_LIB", None)
    d = native.describe()
    assert d == {"enabled": True, "library": None, "blake3": None, "simd_degree": None,
                 "error": None}
    assert list(tmp_path.iterdir()) == []


def test_describe_names_the_built_library(lib, monkeypatch):
    monkeypatch.setattr(native, "_LIB", None)  # found in the cache, not the loaded one
    for _ in range(2):
        d = native.describe()
        assert d["library"].endswith(".so") and d["library"] == lib.path.name
        assert d["blake3"] == "1.8.7" and d["simd_degree"] >= 1 and d["error"] is None
        monkeypatch.setattr(native, "_LIB", lib)
    assert native._LIB is lib


def test_describe_reports_a_load_error(tmp_path, monkeypatch):
    bad = tmp_path / "libbad.so"
    bad.write_bytes(b"not a library")
    monkeypatch.setattr(native, "_LIB", None)
    monkeypatch.setattr(native, "find_library", lambda: bad)
    d = native.describe()
    assert d["library"] is None and d["blake3"] is None and d["simd_degree"] is None
    assert d["error"]
    assert native._LIB is None


def test_describe_under_disabled():
    with native.disabled():
        assert native.describe()["enabled"] is False
    assert native.describe()["enabled"] is True
