"""FS31 part B: one card's identity report, on a tiny random decoder on the CPU."""

import json
from types import SimpleNamespace

import pytest
import torch

from setup import data
from tests.verification.computation.test_decoder import make_records, tiny_config
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.prover.step import prove_step
from verification.runs import card_identity as ci
from verification.transcript.store import InMemoryStore

N = 12
SEED = 1


@pytest.fixture(scope="module")
def tiny():
    c = DecoderComputation(tiny_config(), n_s=2, n=N, eta=1e-3)
    D = make_records([N, N - 3, N - 5, N, N - 1, N - 4, N - 2, N], seed=1)
    return c, D


def test_root_matches_a_direct_prove_and_commit(tiny, tmp_path):
    c, D = tiny
    out = tmp_path / "card.json"
    rep = ci.run_card(c, D, c.build_model, seed=SEED, out=out, card={"device": "cpu"},
                      model={"name": "tiny", "revision": "r0"})
    w0 = snapshot_weights(c, c.build_model())
    records = [D[i] for i in data.schedule(1, c.n_s, len(D), SEED)]
    step = prove_step(c, c.build_model(), w0, records)
    _, tree = InMemoryStore.commit_step(c, step)
    assert rep["proved"]["root"] == tree.root.hex()
    assert rep["same_w_next_as_plain"]
    assert rep["plain"]["w_next_root"] == rep["proved"]["w_next_root"]
    assert rep["proved"]["loss"] == rep["plain"]["loss"]
    assert json.loads(out.read_text()) == rep


def test_digest_file_has_every_leaf_hash(tiny, tmp_path):
    c, D = tiny
    out = tmp_path / "card.json"
    rep = ci.run_card(c, D, c.build_model, seed=SEED, out=out, card={"device": "cpu"},
                      model={"name": "tiny", "revision": "r0"})
    digests = out.with_suffix(".digests")
    assert rep["digests_file"] == "card.digests"
    raw = digests.read_bytes()
    assert len(raw) == c.n_leaves * 32 == rep["n_leaves"] * 32
    lay = rep["layout"]
    assert lay["n_leaves"] == lay["n_s"] + 2 * lay["n_w"] + lay["K"]
    assert len(lay["weight_names"]) == lay["n_w"] and len(lay["product_leaf_names"]) == lay["K"]


def test_two_runs_give_the_same_root(tiny, tmp_path):
    c, D = tiny
    a = ci.run_card(c, D, c.build_model, seed=SEED, out=tmp_path / "a.json", card={}, model={})
    b = ci.run_card(c, D, c.build_model, seed=SEED, out=tmp_path / "b.json", card={}, model={})
    assert a["proved"]["root"] == b["proved"]["root"]
    assert (tmp_path / "a.digests").read_bytes() == (tmp_path / "b.digests").read_bytes()


def test_cpu_card_facts_have_no_ids():
    assert ci.card_facts("cpu") == {"device": "cpu", "name": None, "uuid": None,
                                    "pci_bus_id": None}


def test_cuda_card_facts_read_nvidia_smi_for_the_visible_card(monkeypatch):
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="NVIDIA H100 80GB HBM3, GPU-abc, 00000000:8A:00.0\n")

    props = SimpleNamespace(name="NVIDIA H100 80GB HBM3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    f = ci.card_facts("cuda", run)
    assert seen[0][:3] == ["nvidia-smi", "-i", "2"]
    assert (f["uuid"], f["pci_bus_id"]) == ("GPU-abc", "00000000:8A:00.0")


def test_main_exits_1_when_capture_changes_the_bits(tiny, tmp_path, monkeypatch):
    c, D = tiny
    cfg = SimpleNamespace(data_dir=tmp_path, device="cpu", model="tiny", model_revision="r0",
                          seed=SEED)
    (tmp_path / "D.bin").write_bytes(b"")
    monkeypatch.setattr(ci, "load_config", lambda: cfg)
    monkeypatch.setattr(ci, "setup_determinism", lambda cfg: None)
    monkeypatch.setattr(DecoderComputation, "from_config", classmethod(lambda cls, cfg: c))
    import verification.runs.llama_step as ls
    monkeypatch.setattr(ls, "load_committed_dataset", lambda c, d: (D, b""))
    monkeypatch.setattr(ci, "section_root", lambda c, w: (b"\0" * 32, []))
    out = tmp_path / "card.json"
    assert ci.main(["--out", str(out)]) == 1
    assert out.exists()


def test_card_facts_take_torchs_uuid_and_record_both(monkeypatch):
    def run(argv, **kw):
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="NVIDIA H100, GPU-1111, 00000000:8A:00.0\n")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name="NVIDIA H100", uuid="2222"))
    f = ci.card_facts("cuda", run)
    assert (f["uuid"], f["uuid_torch"], f["uuid_smi"]) == ("GPU-2222", "GPU-2222", "GPU-1111")
    assert f["uuids_agree"] is False and f["device_order"] == "PCI_BUS_ID"
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name="NVIDIA H100", uuid="GPU-1111"))
    f = ci.card_facts("cuda", run)
    assert f["uuid"] == "GPU-1111" and f["uuids_agree"] is True


def test_card_facts_fall_back_to_the_smi_uuid_when_torch_has_none(monkeypatch):
    def run(argv, **kw):
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="NVIDIA H100, GPU-1111, 00000000:8A:00.0\n")

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name="NVIDIA H100"))
    f = ci.card_facts("cuda", run)
    assert f["uuid"] == "GPU-1111" and f["uuid_torch"] is None and f["uuids_agree"] is None


def test_card_facts_ignore_the_text_of_a_failed_nvidia_smi(monkeypatch):
    def run(argv, **kw):
        return SimpleNamespace(returncode=9, stderr="",
                               stdout="NVIDIA H100, GPU-1111, 00000000:8A:00.0\n")

    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(name="NVIDIA H100"))
    f = ci.card_facts("cuda", run)
    assert f["uuid"] is None and "exit code 9" in f["nvidia_smi_error"]
