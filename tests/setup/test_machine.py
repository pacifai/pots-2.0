"""``setup/machine.py``: the moved machine record's keys and the new ``machine_facts``."""

import sys
import types

from benchmarks.hashing.machine import machine_record
from setup import machine as setup_machine

# machine_record()'s keys as they stood before the move out of benchmarks/hashing/machine.py.
RECORD_KEYS = {
    "absent", "arm_sha2", "avx2", "avx512f", "avx512vl", "b3sum", "cgroup_cpu_quota",
    "cgroup_memory_limit_bytes", "core_types", "cores_for_threads", "cores_for_threads_basis",
    "cpu_arch", "cpu_freq_mhz", "cpu_governor", "cpu_max_freq_mhz", "cpu_model",
    "cuda_driver_supports", "cuda_runtime", "driver_version", "gil_enabled", "gpu_count",
    "gpu_form_factor", "gpu_memory_bytes", "gpu_name", "gpu_numa_node", "gpu_pci_bus_id",
    "gpu_power_max_w", "gpu_topology", "hashlib_openssl", "hashlib_sha256", "host_ram_bytes",
    "hostname", "logical_cpus", "notes", "numa_nodes", "openssl_cli", "os", "packages",
    "pcie_gen_current", "pcie_gen_max", "pcie_width_current", "pcie_width_max",
    "physical_cores", "python", "sha_ni", "ssl_openssl", "torch_threads", "usable_cpus"}


def test_machine_record_keys_are_unchanged():
    assert set(machine_record()) == RECORD_KEYS


def test_machine_facts_has_ram_and_no_nvml_on_the_mac():
    facts = setup_machine.machine_facts()
    assert facts["ram_total_bytes"] > 0
    assert "cpu_model" in facts and "cpu_affinity" in facts
    if sys.platform == "darwin":
        assert facts["nvml"] is None


class _FakeNvml(types.ModuleType):
    """A pynvml whose one call, the power limit, raises."""

    NVML_CLOCK_SM, NVML_CLOCK_MEM = 1, 2

    def __init__(self) -> None:
        super().__init__("pynvml")

    def nvmlInit(self): pass
    def nvmlShutdown(self): pass
    def nvmlDeviceGetHandleByIndex(self, i): return object()
    def nvmlSystemGetDriverVersion(self): return "550.1"
    def nvmlDeviceGetUUID(self, h): return "GPU-abc"
    def nvmlDeviceGetEnforcedPowerLimit(self, h): raise RuntimeError("no permission")
    def nvmlDeviceGetMaxClockInfo(self, h, kind): return {1: 1980, 2: 2619}[kind]
    def nvmlDeviceGetMaxPcieLinkGeneration(self, h): return 5
    def nvmlDeviceGetCurrPcieLinkGeneration(self, h): return 4
    def nvmlDeviceGetMaxPcieLinkWidth(self, h): return 16
    def nvmlDeviceGetCurrPcieLinkWidth(self, h): return 16


def test_one_failed_nvml_read_nulls_that_field_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _FakeNvml())
    block = setup_machine.nvml_facts()
    assert block["power_limit_w"] is None
    assert block["driver"] == "550.1" and block["uuid"] == "GPU-abc"
    assert block["max_sm_clock_mhz"] == 1980 and block["max_mem_clock_mhz"] == 2619
    assert (block["pcie_gen_max"], block["pcie_gen_current"]) == (5, 4)
    assert (block["pcie_width_max"], block["pcie_width_current"]) == (16, 16)


def test_nvml_block_is_none_without_pynvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError
    assert setup_machine.nvml_facts() is None


class _BytesNvml(_FakeNvml):
    """A pynvml that returns bytes for text and milliwatts for the power limit."""

    def nvmlSystemGetDriverVersion(self): return b"550.1"
    def nvmlDeviceGetUUID(self, h): return b"GPU-abc"
    def nvmlDeviceGetEnforcedPowerLimit(self, h): return 250000


class _NoInitNvml(_FakeNvml):
    def nvmlInit(self): raise RuntimeError("driver not loaded")


def test_nvml_decodes_bytes_and_converts_milliwatts(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _BytesNvml())
    block = setup_machine.nvml_facts()
    assert (block["driver"], block["uuid"], block["power_limit_w"]) == ("550.1", "GPU-abc", 250.0)


def test_nvml_block_is_none_when_init_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _NoInitNvml())
    assert setup_machine.nvml_facts() is None


def test_python_loop_ns_is_a_positive_per_iteration_time():
    from setup.machine import python_loop_ns
    ns = python_loop_ns(1_000_000)
    print(f"python_loop_ns on this machine: {ns:.1f}")
    assert 0 < ns < 1e5


def test_affinity_freq_parses_khz_and_no_ops_without_sysfs(tmp_path):
    from setup.machine import affinity_freq_mhz, parse_khz_mean_mhz
    assert parse_khz_mean_mhz(["1000000", None, "junk", "3000000\n"]) == 2000.0
    assert parse_khz_mean_mhz([None]) is None
    assert affinity_freq_mhz([0, 1], root=str(tmp_path)) is None


def test_machine_facts_carries_the_loop_benchmark():
    from setup.machine import machine_facts
    facts = machine_facts()
    assert facts["python_loop_ns"] > 0 and "affinity_freq_mhz" in facts


# ---- gpus: one entry per CUDA device, joined to nvidia-smi by bus id ----

def _props(name, bus, uuid, mem=80 * 2**30):
    return types.SimpleNamespace(name=name, total_memory=mem, pci_domain_id=0, pci_bus_id=bus,
                                 pci_device_id=0, uuid=uuid)


def _smi(rows):
    """nvidia-smi's csv: index, uuid, name, driver, mem MiB, power, bus id, 4 PCIe numbers."""
    return "\n".join(
        f"{i}, GPU-{u}, {n}, 550.1, 81559, 700.00, 00000000:{b:02X}:00.0, 5, 5, 16, 16"
        for i, (u, n, b) in enumerate(rows)) + "\n"


def _fake_cuda(monkeypatch, torch_order, smi_text):
    """torch sees ``torch_order`` (a list of (name, bus, uuid)); nvidia-smi prints ``smi_text``."""
    cuda = setup_machine.torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "device_count", lambda: len(torch_order))
    monkeypatch.setattr(cuda, "get_device_properties",
                        lambda k: _props(torch_order[k][0], torch_order[k][1], torch_order[k][2]))
    calls = []

    def run(*cmd, timeout=30):
        calls.append(cmd)
        return smi_text if "--query-gpu" in " ".join(cmd) else None
    monkeypatch.setattr(setup_machine, "_run", run)
    return calls


def _record():
    r = setup_machine._Record()
    setup_machine._gpus(r)
    return r


def test_gpus_join_by_bus_id_when_torch_orders_differently_from_nvidia_smi(monkeypatch):
    smi_rows = [("aaaa", "H100", 0x19), ("bbbb", "H100", 0x3B), ("cccc", "H100", 0x5D),
                ("dddd", "H100", 0x7F)]
    # torch's cuda:0 is the device at bus 0x5D, cuda:1 is at 0x19, ...
    torch_order = [("H100", 0x5D, "cccc"), ("H100", 0x19, "aaaa"), ("H100", 0x7F, "dddd"),
                   ("H100", 0x3B, "bbbb")]
    calls = _fake_cuda(monkeypatch, torch_order, _smi(smi_rows))
    r = _record()
    gpus = r.data["gpus"]
    assert [g["index"] for g in gpus] == [0, 1, 2, 3]
    assert [g["uuid"] for g in gpus] == ["GPU-cccc", "GPU-aaaa", "GPU-dddd", "GPU-bbbb"]
    assert [g["pci_bus_id"] for g in gpus] == [f"00000000:{b:02X}:00.0"
                                               for b in (0x5D, 0x19, 0x7F, 0x3B)]
    assert gpus[0]["power_max_w"] == 700 and gpus[0]["pcie_gen_max"] == 5
    assert gpus[0]["pcie_width_current"] == 16 and gpus[0]["form_factor"].startswith("SXM")
    assert gpus[0]["memory_bytes"] == 80 * 2**30
    assert len([c for c in calls if "--query-gpu" in " ".join(c)]) == 1  # one smi call


def test_gpus_two_devices_in_another_order_match_by_bus_not_position(monkeypatch):
    # nvidia-smi rows differ in name by bus, so a positional join would swap the names
    smi_rows = [("aaaa", "NVIDIA A100 PCIe", 0x19), ("bbbb", "NVIDIA H100 NVL", 0x3B)]
    torch_order = [("H100", 0x3B, "bbbb"), ("A100", 0x19, "aaaa")]
    _fake_cuda(monkeypatch, torch_order, _smi(smi_rows))
    gpus = _record().data["gpus"]
    assert [(g["name"], g["uuid"], g["form_factor"]) for g in gpus] == [
        ("NVIDIA H100 NVL", "GPU-bbbb", "NVL"), ("NVIDIA A100 PCIe", "GPU-aaaa", "PCIe")]


def test_gpus_without_nvidia_smi_keep_torchs_facts(monkeypatch):
    _fake_cuda(monkeypatch, [("H100", 0x19, "aaaa"), ("H100", 0x3B, "bbbb")], None)
    gpus = _record().data["gpus"]
    assert [g["uuid"] for g in gpus] == ["GPU-aaaa", "GPU-bbbb"]
    assert gpus[1]["name"] == "H100" and gpus[1]["pci_bus_id"] == "00000000:3B:00.0"
    assert gpus[0]["power_max_w"] is None and gpus[0]["pcie_gen_max"] is None


def test_gpus_is_none_with_a_reason_without_cuda(monkeypatch):
    monkeypatch.setattr(setup_machine.torch.cuda, "is_available", lambda: False)
    r = _record()
    assert r.data["gpus"] is None and "gpus" in r.absent


def test_machine_facts_carries_gpus_and_keeps_the_gpu_zero_keys():
    facts = setup_machine.machine_facts()
    assert "gpus" in facts and "gpu_name" in facts and "gpu_count" in facts


def test_nvml_facts_takes_an_index(monkeypatch):
    seen = []

    class Nvml(_FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, i):
            seen.append(i)
            return object()
    monkeypatch.setitem(sys.modules, "pynvml", Nvml())
    setup_machine.nvml_facts()
    setup_machine.nvml_facts(index=3)
    assert seen == [0, 3]


def test_gpu_zero_keys_are_torchs_device_zero_not_the_first_smi_row(monkeypatch):
    smi_rows = [("aaaa", "NVIDIA A100 PCIe", 0x19), ("bbbb", "NVIDIA H100 NVL", 0x3B)]
    calls = _fake_cuda(monkeypatch, [("H100", 0x3B, "bbbb"), ("A100", 0x19, "aaaa")],
                       _smi(smi_rows))
    r = setup_machine._Record()
    setup_machine._gpu(r)
    assert r.data["gpu_name"] == "NVIDIA H100 NVL" and r.data["gpu_form_factor"] == "NVL"
    assert r.data["gpu_pci_bus_id"] == "00000000:3B:00.0"
    assert r.data["gpu_memory_bytes"] == 80 * 2**30
    assert len([c for c in calls if "--query-gpu" in " ".join(c)]) == 1


def test_gpu_falls_back_to_the_first_row_when_no_row_matches(monkeypatch):
    _fake_cuda(monkeypatch, [("H100", 0x5D, "zzzz")], _smi([("aaaa", "NVIDIA H100 SXM", 0x19)]))
    r = setup_machine._Record()
    setup_machine._gpu(r)
    assert r.data["gpu_pci_bus_id"] == "00000000:19:00.0"
