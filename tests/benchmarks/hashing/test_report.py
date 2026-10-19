"""The report's shape: results, slots, and the JSON file."""

import json
import math

import pytest

from benchmarks.hashing import machine, report
from benchmarks.hashing.machine import cores_basis, cores_for_threads, machine_record
from setup import machine as setup_machine


def _doc(**slots):
    full = {key: report.not_implemented_slot(key, "not built") for key in report.SLOTS}
    full.update(slots)
    return {"schema": report.SCHEMA, "slots": full}


def test_result_needs_values_with_units_or_a_reason():
    r = report.result("x", "measured", values={"a": 1.0}, units={"a": "s"}, params={"t": 1})
    assert r == {"name": "x", "status": "measured", "params": {"t": 1}, "values": {"a": 1.0},
                 "units": {"a": "s"}, "reason": None, "note": None}
    with pytest.raises(ValueError, match="needs values"):
        report.result("x", "estimated")
    with pytest.raises(ValueError, match="differ"):
        report.result("x", "measured", values={"a": 1.0, "b": 2.0}, units={"a": "s"})
    for status in ("skipped", "not_implemented"):
        with pytest.raises(ValueError, match="needs a reason"):
            report.result("x", status)
    with pytest.raises(ValueError, match="status"):
        report.result("x", "guessed", reason="no")


def test_slot_status_is_the_strongest_result():
    skip = report.result("s", "skipped", reason="r")
    est = report.result("e", "estimated", values={"a": 1}, units={"a": "s"})
    meas = report.result("m", "measured", values={"a": 1}, units={"a": "s"})
    assert report.slot("E1", [skip, est])["status"] == "estimated"
    assert report.slot("E1", [skip, est, meas])["status"] == "measured"
    assert report.slot("E1", [skip])["status"] == "skipped"
    assert report.not_implemented_slot("E2", "GPU")["status"] == "not_implemented"
    with pytest.raises(ValueError):
        report.slot("E9", [skip])
    with pytest.raises(ValueError):
        report.slot("E1", [])


def test_check_report_wants_every_slot_in_order():
    report.check_report(_doc())
    doc = _doc()
    del doc["slots"]["E3"]
    with pytest.raises(ValueError, match="slots"):
        report.check_report(doc)
    doc = _doc()
    doc["slots"] = dict(reversed(doc["slots"].items()))
    with pytest.raises(ValueError, match="slots"):
        report.check_report(doc)
    doc = _doc()
    doc["slots"]["E1"]["results"][0]["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        report.check_report(doc)
    with pytest.raises(ValueError, match="schema"):
        report.check_report({**_doc(), "schema": "other"})


def test_write_report_is_strict_json(tmp_path):
    r = report.result("x", "measured", values={"a": math.nan, "b": [1.0, math.inf]},
                      units={"a": "s", "b": "s"})
    path = report.write_report(_doc(E1=report.slot("E1", [r])), tmp_path / "sub" / "r.json")
    doc = json.loads(path.read_text())
    assert doc["slots"]["E1"]["results"][0]["values"] == {"a": None, "b": [1.0, None]}
    assert not list(tmp_path.glob("sub/*.tmp"))


def test_machine_record_names_every_field():
    m = machine_record()
    for key in ("cpu_model", "physical_cores", "host_ram_bytes", "gpu_name", "driver_version",
                "cuda_runtime", "pcie_gen_current", "pcie_width_current", "gpu_numa_node",
                "numa_nodes", "cores_for_threads", "packages", "hashlib_sha256"):
        assert key in m, key
    for key in ("avx2", "avx512vl", "cpu_freq_mhz", "cpu_max_freq_mhz", "cpu_governor",
                "hashlib_openssl", "ssl_openssl", "cores_for_threads_basis", "notes"):
        assert key in m, key
    assert set(m["absent"]) == {k for k, v in m.items() if v is None}
    assert m["cores_for_threads"] >= 1
    assert m["cores_for_threads_basis"].startswith(("physical cores", "logical CPUs"))
    assert {"pcie_gen_current", "cores_for_threads"} <= set(m["notes"])
    if m["hashlib_openssl"] is not None:
        assert m["hashlib_openssl"].startswith(("OpenSSL", "LibreSSL", "BoringSSL"))
    json.dumps(m)


def test_cores_basis_names_its_caps():
    m = {"physical_cores": 32, "logical_cpus": 64, "usable_cpus": 16, "cgroup_cpu_quota": None}
    assert cores_for_threads(m) == 16
    assert cores_basis(m) == "physical cores, capped by CPU affinity"
    assert cores_basis({"physical_cores": None, "logical_cpus": 8}) == \
        "logical CPUs (no physical core count)"


def test_linux_frequency_reads_sysfs_then_cpuinfo(monkeypatch):
    """The H100 host runs Linux, which this test fakes through its files."""
    base = "/sys/devices/system/cpu/cpu0/cpufreq/"
    files = {base + "scaling_cur_freq": "2100000\n", base + "cpuinfo_max_freq": "3800000\n",
             base + "scaling_governor": "performance\n"}
    cpuinfo = "processor\t: 0\ncpu MHz\t\t: 1999.998\n"
    monkeypatch.setattr(setup_machine, "_read", files.get)
    r = machine._Record()
    machine._frequency(r, "Linux", cpuinfo)
    assert r.data == {"cpu_freq_mhz": 2100.0, "cpu_max_freq_mhz": 3800.0,
                      "cpu_governor": "performance"}
    monkeypatch.setattr(setup_machine, "_read", {}.get)  # a VM without cpufreq
    r = machine._Record()
    machine._frequency(r, "Linux", cpuinfo)
    assert r.data == {"cpu_freq_mhz": 1999.998, "cpu_max_freq_mhz": None, "cpu_governor": None}
    assert set(r.absent) == {"cpu_max_freq_mhz", "cpu_governor"}
