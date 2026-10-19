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
