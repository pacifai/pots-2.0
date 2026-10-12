"""E5(a′)'s native batch: the build steps without the network, and the library against the
protocol's challenges bit for bit when it's built."""

import hashlib
import io
import tarfile

import numpy as np
import pytest
import torch

from benchmarks.hashing import native, report
from benchmarks.hashing.sizes import Step
from tests.verification.decoder_helpers import tiny_config
from verification.computation.leaf_sizes import leaf_sizes
from verification.verifier.matmul_check import challenges as proto

H = bytes(range(32))
needs_library = pytest.mark.skipif(
    native.find_library() is None,
    reason="native BLAKE3 batch not built; run python -m benchmarks.hashing.native (one "
           "download from github.com)")


def _bits(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().numpy().view(np.int32)


# ---- build steps, no network ----------------------------------------------------------------


def test_target_picks_blake3s_simd_files():
    for m in ("arm64", "aarch64"):
        arch, files, flags = native.target(m)
        assert (arch, flags) == ("arm64", ["-DBLAKE3_USE_NEON=1"])
        assert "blake3_neon.c" in files and not any(f.endswith(".S") for f in files)
    arch, files, flags = native.target("x86_64")
    assert (arch, flags) == ("x86_64", [])
    assert {f"blake3_{s}_x86-64_unix.S" for s in ("sse2", "sse41", "avx2", "avx512")} <= set(files)
    arch, files, flags = native.target("riscv64")
    assert files == native._COMMON and "-DBLAKE3_NO_AVX512" in flags


def test_cache_and_library_names(tmp_path, monkeypatch):
    monkeypatch.setenv("POTS_BENCH_CACHE", str(tmp_path))
    assert native.cache_dir() == tmp_path
    path = native.library_path()
    assert path.parent == tmp_path and native.BLAKE3_TAG in path.name
    assert native.compiler_version("clang-99") == ""  # missing: no crash
    assert native.library_path(cc="clang-99") != path  # another compiler, another library
    assert native.find_library() is None


def test_a_compiler_upgrade_builds_a_new_library(monkeypatch):
    monkeypatch.setattr(native, "compiler_version", lambda cc: "clang version 21.0.0")
    before = native.library_path(cc="cc")
    monkeypatch.setattr(native, "compiler_version", lambda cc: "clang version 22.0.0")
    assert native.library_path(cc="cc") != before


def test_the_pins_cover_every_kept_file():
    assert set(native.SOURCE_SHA256) == {*native._KEEP, *native._LICENSES}


def _pins(monkeypatch) -> None:
    """Pin the fake files below, whose contents are their own names."""
    monkeypatch.setattr(native, "SOURCE_SHA256", {
        f: hashlib.sha256(f.encode()).hexdigest() for f in (*native._KEEP, *native._LICENSES)})


def _sources(src) -> None:
    """A cached source directory of fake files, as :func:`_pins` pins them."""
    src.mkdir(parents=True)
    for f in (*native._KEEP, *native._LICENSES):
        (src / f).write_bytes(f.encode())
    (src / "ARCHIVE_SHA256").write_text(native.ARCHIVE_SHA256)


def _archive(extra: dict[str, bytes]) -> bytes:
    root = f"BLAKE3-{native.BLAKE3_TAG}/"
    files = {root + "c/" + f: f.encode() for f in native._KEEP}
    files |= {root + f: f.encode() for f in native._LICENSES}
    files |= {root + name: data for name, data in extra.items()}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, data: bytes) -> None:
    monkeypatch.setattr(native.urllib.request, "urlopen", lambda *a, **kw: _Response(data))


def test_a_wrong_archive_is_refused_before_extraction(tmp_path, monkeypatch):
    _serve(monkeypatch, _archive({}))  # well formed, but not the pinned archive
    dest = tmp_path / "src"
    with pytest.raises(RuntimeError, match="not extracted"):
        native.fetch(dest, log=lambda _: None)
    assert list(tmp_path.iterdir()) == []


def test_fetch_extracts_only_the_needed_files(tmp_path, monkeypatch):
    data = _archive({"c/blake3_avx2_x86-64_windows_gnu.S": b"x", "README.md": b"x",
                     "c/blake3_c_rust_bindings/src/lib.rs": b"x"})
    monkeypatch.setattr(native, "ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    _pins(monkeypatch)
    _serve(monkeypatch, data)
    dest = native.fetch(tmp_path / "src", log=lambda _: None)
    assert {p.name for p in dest.iterdir()} == \
        {*native._KEEP, *native._LICENSES, "ARCHIVE_SHA256"}
    assert (dest / "blake3.c").read_bytes() == b"blake3.c"
    assert (dest / "ARCHIVE_SHA256").read_text().strip() == native.ARCHIVE_SHA256
    assert [p.name for p in tmp_path.iterdir()] == ["src"]  # no temp directory left


def test_a_file_off_its_pin_is_refused(tmp_path, monkeypatch):
    """The archive hash matches but one extracted file doesn't match its own pin."""
    data = _archive({})
    monkeypatch.setattr(native, "ARCHIVE_SHA256", hashlib.sha256(data).hexdigest())
    _pins(monkeypatch)
    monkeypatch.setitem(native.SOURCE_SHA256, "blake3.c", "0" * 64)
    _serve(monkeypatch, data)
    with pytest.raises(RuntimeError, match=r"\['blake3.c'\]"):
        native.fetch(tmp_path / "src", log=lambda _: None)
    assert list(tmp_path.iterdir()) == []


def test_changed_sources_are_fetched_again(tmp_path, monkeypatch):
    monkeypatch.setenv("POTS_BENCH_CACHE", str(tmp_path))
    _pins(monkeypatch)
    src = native.source_dir()
    _sources(src)
    assert native.bad_sources(src) == []
    (src / "blake3_dispatch.c").write_bytes(b"tampered")
    (src / "blake3.h").unlink()
    assert native.bad_sources(src) == ["blake3.h", "blake3_dispatch.c"]

    def refetch(dest, **kw):
        raise RuntimeError("fetched")

    monkeypatch.setattr(native, "fetch", refetch)
    with pytest.raises(RuntimeError, match="fetched"):
        native.build(log=lambda _: None)


def test_a_failed_compile_raises_and_leaves_no_library(tmp_path, monkeypatch):
    monkeypatch.setenv("POTS_BENCH_CACHE", str(tmp_path))
    monkeypatch.setenv("CC", "false")  # a "compiler" that always fails
    _pins(monkeypatch)
    src = native.source_dir()
    _sources(src)  # sources present and on their pins: no fetch
    monkeypatch.setattr(native, "fetch", lambda *a, **kw: pytest.fail("fetched"))
    with pytest.raises(RuntimeError, match="build failed"):
        native.build(log=lambda _: None)
    assert [p.name for p in tmp_path.iterdir()] == [src.name]


def test_a_built_library_is_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("POTS_BENCH_CACHE", str(tmp_path))
    path = native.library_path()
    path.write_bytes(b"")
    monkeypatch.setattr(native, "fetch", lambda *a, **kw: pytest.fail("fetched again"))
    assert native.build(log=lambda _: None) == path


# ---- the library, bit for bit against the protocol ------------------------------------------


@pytest.fixture(scope="module")
def lib():
    path = native.find_library()
    if path is None:  # never build here: building may download
        pytest.skip("native BLAKE3 batch not built")
    return native.Native(path)


@needs_library
@pytest.mark.parametrize("k", [1, 3, 21])
@pytest.mark.parametrize("width", [1, 3, 17, 64, 255, 256, 257, 1000, 5000])
def test_challenge_matrices_match_bit_for_bit(lib, width, k):
    ms = [1, 9, 393555, 2**32 - 1]
    theirs = proto.challenge_matrices(H, ms, k, width)
    for threads in (1, 4):
        ours = lib.challenge_matrices(H, ms, k, width, threads=threads)
        assert ours.dtype == theirs.dtype == torch.float32
        assert ours.shape == theirs.shape == (len(ms), width, k)
        assert np.array_equal(_bits(ours), _bits(theirs))


@needs_library
@pytest.mark.parametrize("m,k,width", [(1, 1, 1), (7, 21, 64), (2**32 - 1, 3, 17),
                                       (393555, 21, 2048), (5, 21, 128256)])
def test_challenge_matrix_matches_bit_for_bit(lib, m, k, width):
    theirs = proto.challenge_matrix(H, m, k, width)
    ours = lib.challenge_matrix(H, m, k, width, threads=8)
    assert ours.shape == theirs.shape == (width, k)
    assert np.array_equal(_bits(ours), _bits(theirs))


@needs_library
def test_many_items_on_many_threads(lib):
    ms = list(range(1, 301))  # one layer's worth of attention members, split over 16 threads
    ours = lib.challenge_matrices(H, ms, 21, 64, threads=16)
    assert np.array_equal(_bits(ours), _bits(proto.challenge_matrices(H, ms, 21, 64)))
    assert lib.challenge_matrices(H, [], 21, 64).shape == (0, 64, 21)


@needs_library
@pytest.mark.parametrize("call", [
    lambda f: f.challenge_matrix(H, 0, 21, 4),
    lambda f: f.challenge_matrix(H, 1, 0, 4),
    lambda f: f.challenge_matrix(H, 1, 256, 4),
    lambda f: f.challenge_matrix(H[:31], 1, 21, 4),
    lambda f: f.challenge_matrix(H, 1, 21, 0),
    lambda f: f.challenge_matrices(H, [1, 2**32], 21, 4),
    lambda f: f.challenge_matrices(H, [0], 0, 4),
])
def test_validation_matches(lib, call):
    with pytest.raises(ValueError) as theirs:
        call(proto)
    with pytest.raises(ValueError) as ours:
        call(lib)
    assert str(ours.value) == str(theirs.value)


@needs_library
def test_the_library_reports_blake3s_dispatch(lib):
    assert lib.blake3_version == native.BLAKE3_TAG
    assert lib.simd_degree in (1, 4, 8, 16)


@needs_library
def test_run_native_rows(lib):
    config = tiny_config()
    step = Step("tiny", 2, 12, torch.float32, torch.bfloat16,
                tuple(leaf_sizes(config, n_s=2, n=12)))
    rows = native.run_native(step, lib, k=3, layers=1, threads=[1, 2])
    assert [(r["name"], r["params"]["threads"]) for r in rows] == [
        ("native_batch", 1), ("native_batch_per_step", 1),
        ("native_batch", 2), ("native_batch_per_step", 2)]
    for r in rows:
        assert r["status"] in report.STATUSES and set(r["values"]) == set(r["units"])
        assert r["params"]["blake3"] == native.BLAKE3_TAG
