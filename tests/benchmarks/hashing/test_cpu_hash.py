"""E1's hashing replicas against the protocol's, and its helpers."""

import numpy as np
import pytest
import torch

from benchmarks.hashing import cpu_hash
from benchmarks.hashing.sizes import Step
from tests.verification.decoder_helpers import tiny_config
from verification.commitment import merkle
from verification.commitment.encoding import tensor_leaf_header
from verification.computation.leaf_sizes import leaf_sizes

MB = 1 << 20


@pytest.mark.parametrize("nbytes", [0, 1, 1000, 16 * MB - 1, 16 * MB, 17 * MB])
def test_hash_leaf_is_the_protocols(nbytes):
    """Below and above the 16 MiB multithreading threshold."""
    buf = cpu_hash.make_buffer(max(nbytes, 1))[:nbytes]
    parts = [b"\x02\x01\x01", memoryview(buf)]
    assert cpu_hash.hash_leaf("blake3", parts) == merkle.hash_leaf(*parts)


def test_hash_leaves_keep_order_and_values():
    buf = cpu_hash.make_buffer(64 * MB)
    views = cpu_hash.Views(buf)
    leaves = [views.leaf(n, 11) for n in (100, 20 * MB, 5, 9 * MB, 9 * MB, 1)]
    expected = merkle.hash_leaves(leaves)
    pools = cpu_hash.Pools()
    try:
        for t in (1, 4):
            assert cpu_hash.hash_leaves("blake3", leaves, t, pools) == expected
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


def test_run_e1_tiny():
    config = tiny_config()
    s = Step("tiny", 2, 12, torch.float32, torch.bfloat16, tuple(leaf_sizes(config, n_s=2, n=12)))
    rows = cpu_hash.run_e1(buffer_bytes=2 * MB, chunks=[64 << 10], threads=[1, 2], reps=1,
                           size_steps=[s], direct_steps=[s], direct_reps=1, sample_bytes=MB,
                           object_cap=None, native_threads=[1], native=False, copy_bytes=MB,
                           scratch_dir=None, log=lambda _: None)
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)
    assert len(by_name["chunked"]) == 2 * 2  # two algorithms × two thread counts
    assert {r["status"] for r in by_name["step_from_sizes"]} == {"estimated"}
    assert {r["status"] for r in by_name["step_direct"]} == {"measured"}
    assert by_name["step_direct"][0]["params"]["step_bytes"] == s.total_bytes
    sizes = {r["params"]["leaf_bytes"] for r in by_name["whole_objects"]}
    assert sizes == {b for b, _ in s.histogram()}
    assert by_name["openssl_speed"][0]["status"] == "skipped"
    under = by_name["chunked_under_h2d_copy"][0]
    assert under["status"] == ("skipped" if not torch.cuda.is_available() else "measured")
    assert all(np.isfinite(v) for r in by_name["chunked"]
               for v in (r["values"]["gb_per_s_best"], r["values"]["gb_per_s_median"]))
