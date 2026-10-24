"""``setup/machine.py`` on a machine with CUDA: ``gpus`` against what ``nvidia-smi`` prints.
Skips cleanly without CUDA or ``nvidia-smi`` (the Mac); the four-H100 pod runs it."""

import shutil
import subprocess

import pytest
import torch

from setup.machine import machine_facts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or shutil.which("nvidia-smi") is None,
    reason="needs a CUDA device and nvidia-smi")


def _smi_uuids():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def test_gpus_has_one_entry_per_device_with_distinct_smi_uuids():
    facts = machine_facts()
    gpus = facts["gpus"]
    n = torch.cuda.device_count()
    assert [g["index"] for g in gpus] == list(range(n))
    uuids = [g["uuid"] for g in gpus]
    assert len(set(uuids)) == n
    assert set(uuids) <= _smi_uuids()
    for g in gpus:
        assert g["name"] and g["pci_bus_id"] and g["memory_bytes"] > 0
        assert g["name"] == torch.cuda.get_device_name(g["index"])


def test_gpu_zero_keys_are_the_first_entry():
    facts = machine_facts()
    assert facts["gpu_count"] == len(facts["gpus"])
    assert facts["gpus"][0]["pci_bus_id"] == facts["gpu_pci_bus_id"]


def test_every_devices_role_and_probe_uuid_is_the_machine_records_by_index():
    from setup.machine import torch_gpu_uuid
    from verification.runs import metrics

    smi = {g["index"]: g["uuid"] for g in machine_facts()["gpus"]}
    devs = [f"cuda:{k}" for k in range(torch.cuda.device_count())]
    roles = metrics.gpu_roles(devs[0], devs[1:])
    assert [r["device"] for r in roles] == devs
    assert {r["index"]: r["uuid"] for r in roles} == smi
    assert {k: torch_gpu_uuid(dev) for k, dev in enumerate(devs)} == smi
    assert all(u.startswith("GPU-") and not u.startswith("GPU-GPU-") for u in smi.values())
