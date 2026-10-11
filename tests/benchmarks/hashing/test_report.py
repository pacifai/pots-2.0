"""The report's shape: results, slots, and the JSON file."""

import json
import math

import pytest

from benchmarks.hashing import report
from benchmarks.hashing.machine import machine_record


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
    assert set(m["absent"]) == {k for k, v in m.items() if v is None}
    assert m["cores_for_threads"] >= 1
    json.dumps(m)
