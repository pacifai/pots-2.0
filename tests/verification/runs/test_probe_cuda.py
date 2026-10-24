"""The probe on real GPUs: every visible device measured, the peer pairs, ``hash_kernel``.
Skips cleanly on a machine with no CUDA (the Mac); the four-H100 pod runs it."""

import pytest
import torch

from verification.runs import probe

N = torch.cuda.device_count() if torch.cuda.is_available() else 0
DEVICES = ",".join(f"cuda:{i}" for i in range(N))
SECONDS = 8.0  # small: the measures take their minimum repeats

pytestmark = pytest.mark.skipif(N < 1, reason="needs at least one CUDA device")


@pytest.fixture(scope="module")
def doc():
    return probe.run_probe(SECONDS, devices=DEVICES)


def test_every_visible_device_is_measured(doc):
    assert doc["version"] == 2 and doc["devices"] == DEVICES.split(",")
    assert list(doc["per_device"]) == doc["devices"]
    uuids = [d["uuid"] for d in doc["per_device"].values()]
    assert len(set(uuids)) == N and all(u.startswith("GPU-") for u in uuids)
    for d in doc["per_device"].values():
        assert d["rates"]["gemm_bf16_flops"] > 0 and d["rates"]["hbm_copy_bytes"] > 0
        assert d["costs"]["sync_s"] > 0
        assert not set(d["rates"]) & set(probe.HOST_RATE_KEYS)
    first = doc["per_device"][doc["devices"][0]]
    assert doc["rates"]["gemm_bf16_flops"] == first["rates"]["gemm_bf16_flops"]


def test_hash_kernel_is_measured_on_every_device(doc):
    for name, d in doc["per_device"].items():
        h = d["rates"]["hash_kernel"]
        assert h is not None, d["errors"].get("hash_kernel")
        assert h["per_byte_s"] > 0 and "threads" not in h and len(h["mixes"]) == 7


def test_every_ordered_pair_has_a_peer_rate(doc):
    if N < 2:
        pytest.skip("a peer copy needs two CUDA devices")
    want = {f"{a}->{b}" for a in doc["devices"] for b in doc["devices"] if a != b}
    assert set(doc["peer"]) == want
    assert all(v is None or v > 0 for v in doc["peer"].values())
    assert set(doc["peer_errors"]) == {k for k, v in doc["peer"].items() if v is None}


def test_one_device_has_no_peer_pairs():
    one = probe.run_probe(SECONDS, devices="cuda:0")
    assert one["peer"] == {} and list(one["per_device"]) == ["cuda:0"]
