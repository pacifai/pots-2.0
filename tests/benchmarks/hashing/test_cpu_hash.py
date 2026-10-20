"""E1's hashing replicas against the protocol's, and its helpers."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.hashing import cpu_hash
from benchmarks.hashing.sizes import Step
from tests.verification.decoder_helpers import tiny_config
from verification.commitment import merkle
from verification.commitment.encoding import _all_finite, tensor_leaf_header
from verification.computation.leaf_sizes import leaf_sizes

MB = 1 << 20
ROOT = Path(__file__).resolve().parents[3]


def tiny_step() -> Step:
    config = tiny_config()
    return Step("tiny", 2, 12, torch.float32, torch.float32,
                tuple(leaf_sizes(config, n_s=2, n=12)))


@pytest.mark.parametrize("nbytes", [0, 1, 1000, 16 * MB - 1, 16 * MB, 17 * MB])
def test_hash_leaf_is_the_protocols(nbytes):
    """Below and above the 16 MiB multithreading threshold."""
    buf = cpu_hash.make_buffer(max(nbytes, 1))[:nbytes]
    parts = [b"\x02\x01\x01", memoryview(buf)]
    assert cpu_hash.hash_leaf("blake3", parts) == merkle.hash_leaf(*parts)


@pytest.mark.parametrize("variant", ["as_built", "capped"])
def test_hash_leaves_keep_order_and_values(variant):
    """Both BLAKE3 variants give the protocol's digests, in order, big leaves mixed in."""
    buf = cpu_hash.make_buffer(64 * MB)
    views = cpu_hash.Views(buf)
    leaves = [views.leaf(n, 11) for n in (100, 20 * MB, 5, 9 * MB, 9 * MB, 1, 17 * MB, 3)]
    expected = merkle.hash_leaves(leaves)
    pools = cpu_hash.Pools()
    try:
        for t in (1, 4):
            assert cpu_hash.hash_leaves("blake3", leaves, t, pools, variant) == expected
    finally:
        pools.close()


def test_unknown_variants_raise():
    with pytest.raises(ValueError):
        cpu_hash.hash_leaves("sha256", [], 2, cpu_hash.Pools(), "capped")
    with pytest.raises(ValueError):
        cpu_hash.variant_params("blake3", "uncapped", 2)
    as_built = cpu_hash.variant_params("blake3", "as_built", 1)
    assert "all logical CPUs" in as_built["large_leaf_threads"]
    assert cpu_hash.variant_params("blake3", "capped", 8)["large_leaf_threads"] == 8


CPU_PER_WALL = """
import time
from benchmarks.hashing.cpu_hash import Pools, Views, hash_leaves, make_buffer
views, pools = Views(make_buffer(64 << 20)), Pools()
for t in (1, 2):
    hash_leaves("blake3", [views.leaf(32 << 20, 7)], t, pools, "capped")  # warm-up
    leaves = [views.leaf(32 << 20, 7) for _ in range(6)]
    c0, w0 = time.process_time(), time.perf_counter()
    hash_leaves("blake3", leaves, t, pools, "capped")
    print(t, (time.process_time() - c0) / (time.perf_counter() - w0))
pools.close()
"""


def test_capped_blake3_uses_at_most_its_threads():
    """CPU time over wall time stays near ``threads`` for the capped variant, on leaves of
    32 MiB that the as-built variant hashes on every logical CPU. A fresh process, so no
    spinning OpenMP thread of an earlier test adds CPU time; the bound is loose."""
    proc = subprocess.run([sys.executable, "-c", CPU_PER_WALL], cwd=ROOT, capture_output=True,
                          text=True, check=True, timeout=120)
    ratios = {int(t): float(r) for t, r in (ln.split() for ln in proc.stdout.splitlines())}
    assert set(ratios) == {1, 2}
    for t, r in ratios.items():
        assert r <= 1.3 * t + 0.2, ratios


def test_pools_start_every_thread():
    pools = cpu_hash.Pools()
    try:
        assert len(pools.get(5)._threads) == 5
    finally:
        pools.close()


def test_sha256_leaf():
    import hashlib
    assert cpu_hash.hash_leaf("sha256", [b"ab", b"c"]) == hashlib.sha256(b"\x00abc").digest()


def test_views_stream_a_payload_larger_than_the_buffer():
    buf = cpu_hash.make_buffer(3 * MB)
    views = cpu_hash.Views(buf)
    parts = views.payload(7 * MB + 5)
    assert sum(p.nbytes for p in parts) == 7 * MB + 5
    leaf = views.leaf(2 * MB, 7)
    assert sum(memoryview(p).nbytes for p in leaf) == 2 * MB and len(leaf[0]) == 7


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_buffer_views_are_finite(dtype):
    buf = cpu_hash.make_buffer(3 * MB)
    views = cpu_hash.Views(buf)
    for numel in (1, 1000, MB // 2, 1000):
        t = views.tensor(dtype, numel)
        assert t.dtype == dtype and t.numel() == numel and bool(torch.isfinite(t).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_all_finite_is_the_protocols(dtype):
    cases = [torch.zeros(0), torch.randn(50), torch.tensor([1.0, float("nan")]),
             torch.tensor([float("inf"), 2.0]), torch.tensor([-float("inf")])]
    for c in cases:
        t = c.to(dtype)
        assert cpu_hash.all_finite(t) == _all_finite(t)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13, 100])
def test_trees_are_the_protocols(n):
    hashes = cpu_hash.random_digests(n)
    root = merkle.merkle_root(hashes)
    assert cpu_hash.merkle_root(hashes) == root
    assert cpu_hash.tree_root(hashes) == root
    nodes = cpu_hash.cached_tree(hashes)
    assert nodes[(0, n)] == root
    tree = merkle.MerkleTree(hashes)
    assert len(nodes) == 2 * n - 1
    assert all(tree.node(lo, hi) == h for (lo, hi), h in nodes.items())
    with pytest.raises(ValueError):
        cpu_hash.merkle_root([])


def test_sample_objects_fill_every_thread():
    """A small leaf hashes at least 4 tasks per thread, whatever its count, up to the object
    cap; a leaf of 16 MiB or more is a task of its own, at most its count."""
    for t in (1, 8, 64, 128):
        n, tasks = cpu_hash.sample_objects(32 << 10, 3, t, MB)
        assert tasks >= cpu_hash.TASKS_PER_THREAD * t and n <= cpu_hash.MAX_OBJECTS
        assert n * (32 << 10) >= tasks * cpu_hash.TASK_BYTES - cpu_hash.TASK_BYTES
    assert cpu_hash.sample_objects(1500, 128, 64, MB, max_objects=1000) == (1000, 1)
    assert cpu_hash.sample_objects(1 << 30, 2, 64, 512 * MB) == (2, 2)
    assert cpu_hash.sample_objects(32 * MB, 100, 2, MB) == (8, 8)
    assert cpu_hash.sample_objects(32 * MB, 100, 1, 512 * MB) == (16, 16)


def test_header_bytes_match_the_encoding():
    for shape in [(5,), (3, 4), (2, 3, 4)]:
        t = torch.empty(shape, device="meta")
        assert cpu_hash.header_bytes(shape, "w_t") == len(tensor_leaf_header(0x02, t))
    assert cpu_hash.header_bytes((3, 128), "record") == 6


def test_thread_counts_cap_and_deduplicate():
    assert cpu_hash.thread_counts(18) == [1, 8, 16, 18]
    assert cpu_hash.thread_counts(8) == [1, 8]
    assert cpu_hash.thread_counts(12, [1, "all", 16]) == [1, 12]


def test_time_chunked_on_one_and_several_threads():
    buf = cpu_hash.make_buffer(MB)
    pools = cpu_hash.Pools()
    try:
        for algo in cpu_hash.ALGOS:
            for t in (1, 3):
                assert cpu_hash.time_chunked(buf, algo, 64 << 10, t, pools) > 0
            assert len(cpu_hash.time_chunk_tree(16, algo, 2)) == 2
    finally:
        pools.close()


LIBRESSL_MR = """\
+R:1233512:sha256:3.001021
Forked child 1
Got: +H:16:64:256:1024:8192 from 0
Got: +F:21:sha256:239772430.78:841597430.88:2070951201.14:2887941933.98:3361246910.56 from 0
Got: +H:16:64:256:1024:8192 from 1
Got: +F:21:sha256:238577296.52:839879937.40:2058850809.05:2915857468.75:3367164143.14 from 1
+F:21:evp:478349727.30:1681477368.28:4129802010.19:5803799402.73:6728411053.70
"""

OPENSSL3_MR = """\
+H:16:64:256:1024:8192:16384
+F:22:sha256:1.0:2.0:3.0:4.0:5.0:6.0
"""


def test_parse_openssl_mr():
    rates = cpu_hash.parse_openssl_mr(LIBRESSL_MR)
    assert list(rates) == [16, 64, 256, 1024, 8192]
    assert rates[8192] == pytest.approx(6728411053.70)
    assert cpu_hash.parse_openssl_mr(OPENSSL3_MR)[16384] == 6.0
    with pytest.raises(ValueError):
        cpu_hash.parse_openssl_mr("nothing here")


def run_tiny(s: Step) -> dict[str, list[dict]]:
    rows = cpu_hash.run_e1(buffer_bytes=2 * MB, chunks=[64 << 10], threads=[1, 2], reps=1,
                           size_steps=[s], direct_steps=[s], direct_reps=1, sample_bytes=MB,
                           object_cap=None, native_threads=[1], native=False, copy_bytes=MB,
                           scratch_dir=None, max_objects=1024, log=lambda _: None)
    by_name: dict[str, list[dict]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)
    return by_name


def test_run_e1_tiny():
    s = tiny_step()
    by_name = run_tiny(s)
    assert len(by_name["chunked"]) == 2 * 2  # two algorithms × two thread counts
    for r in by_name["chunked"]:
        v = r["values"]
        assert all(np.isfinite(v[k]) for k in ("gb_per_s_best", "gb_per_s_median",
                                               "node_seconds", "gb_per_s_end_to_end"))
        assert v["gb_per_s_end_to_end"] < v["gb_per_s_median"]
    variants = {(r["params"]["algo"], r["params"]["variant"]) for r in by_name["whole_objects"]}
    assert variants == set(cpu_hash.VARIANTS)
    assert len(by_name["step_from_sizes"]) == len(cpu_hash.VARIANTS) * 2
    assert {r["status"] for r in by_name["step_from_sizes"]} == {"estimated"}
    assert {r["status"] for r in by_name["whole_objects"]} == {"measured"}
    assert {r["status"] for r in by_name["step_direct"]} == {"measured"}
    assert by_name["step_direct"][0]["params"]["step_bytes"] == s.total_bytes
    sizes = {r["params"]["leaf_bytes"] for r in by_name["whole_objects"]}
    assert sizes == {b for b, _ in s.histogram()}
    for r in by_name["whole_objects"]:
        p = r["params"]
        assert p["objects_hashed"] <= 1024 and p["tasks_per_thread"] == p["tasks"] / p["threads"]
        assert len(r["values"]["seconds"]) == p["reps"]
    trees = {r["params"]["builder"]: r for r in by_name["outer_tree"]}
    assert set(trees) == {"prover", "verifier"}
    assert trees["prover"]["values"]["node_hashes"] == len(s.leaves) - 1
    floats = sum(1 for leaf in s.leaves if leaf.dtype.is_floating_point)
    per_step = by_name["finiteness_per_step"]
    assert [r["params"]["leaves_checked"] for r in per_step] == [floats, floats]
    assert {r["params"]["path"] for r in by_name["finiteness"]} == {"numpy max and min"}
    assert by_name["openssl_speed"][0]["status"] == "skipped"
    under = by_name["chunked_under_h2d_copy"][0]
    assert under["status"] == ("skipped" if not torch.cuda.is_available() else "measured")


def test_object_cap_makes_rows_estimated():
    s = tiny_step()
    big = max(b for b, _ in s.histogram())
    pools = cpu_hash.Pools()
    try:
        rows = cpu_hash.per_size(s, cpu_hash.make_buffer(MB), [1], pools, reps=1,
                                 sample_bytes=1 << 16, object_cap=big // 2, max_objects=1024)
    finally:
        pools.close()
    capped = [r for r in rows if r["name"] == "whole_objects"
              and r["params"]["measured_object_bytes"] < r["params"]["leaf_bytes"]]
    assert capped and {r["status"] for r in capped} == {"estimated"}
    assert all(r["note"] for r in capped)


def test_a_failing_part_keeps_the_others(monkeypatch):
    def broken(*_, **__):
        raise RuntimeError("tree broke")

    monkeypatch.setattr(cpu_hash, "outer_tree", broken)
    by_name = run_tiny(tiny_step())
    (row,) = by_name["outer_tree"]
    assert row["status"] == "skipped" and row["reason"] == "error: RuntimeError: tree broke"
    assert row["params"]["model"] == "tiny"
    assert by_name["finiteness"] and by_name["step_direct"] and by_name["whole_objects"]
