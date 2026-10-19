"""FS22's probe: the hashing fit, the datasheet lookup, a CPU run's file, and the session part."""

import json

import pytest

from verification.runs import fs11_session as fs
from verification.runs import probe


def test_fit_recovers_exact_rates():
    per_byte, per_leaf = 1.0e-10, 2.0e-6
    mixes = [(b, n, b * per_byte + n * per_leaf)
             for b, n in ((64 << 20, 32768), (64 << 20, 64), (32 << 20, 4096), (16 << 20, 1),
                          (1 << 20, 1000))]
    got_byte, got_leaf = probe.fit_hash_rate(mixes)
    assert got_byte == pytest.approx(per_byte, rel=1e-2)
    assert got_leaf == pytest.approx(per_leaf, rel=1e-2)


def test_datasheet_lookup():
    h100 = probe.datasheet_for("NVIDIA H100 80GB HBM3")
    assert h100 is not None and h100["bf16_flops"] == 9.894e14 and h100["name"] == "H100 SXM"
    assert probe.datasheet_for("Apple M3") is None
    assert probe.datasheet_for(None) is None


def test_linear_shapes_and_fit_m():
    shapes = probe.linear_shapes()
    assert len({(k, n) for _, k, n in shapes}) == len(shapes)
    assert any(n == 128256 for _, _, n in shapes)  # Llama's head
    for _, k, n in shapes:
        m = probe.fit_m(k, n, 4)
        assert (m * k + k * n + m * n) * 4 <= probe.GPU_LIMIT // 2


def test_cpu_run_writes_a_valid_file(tmp_path):
    out = tmp_path / "sub" / "probe.json"
    assert probe.main(["--out", str(out), "--seconds", "5"]) == 0
    doc = json.loads(out.read_text())
    assert doc["version"] == 1 and doc["device"] == "cpu"
    assert set(doc["rates"]) == set(probe.RATE_KEYS) and set(doc["costs"]) == set(probe.COST_KEYS)
    assert doc["determinism"] == {"tf32": False, "deterministic": True}
    assert doc["datasheet"] is None and doc["shapes"] == []
    host = {"host_first_touch_bytes", "hash", "host_memcpy_bytes"}
    for key in (*probe.RATE_KEYS, *probe.COST_KEYS):
        table = doc["rates"] if key in doc["rates"] else doc["costs"]
        if key in host:
            assert table[key] is not None and key not in doc["errors"]
        else:
            assert table[key] is None and doc["errors"][key]
    h = doc["rates"]["hash"]
    assert h["per_byte_s"] > 0 and h["threads"] == doc["threads"]
    assert doc["rates"]["host_memcpy_bytes"] > 0 and doc["wall_s"] > 0


def test_probe_part_and_extras(tmp_path):
    part = fs._probe_part(tmp_path)
    assert part.kind == "probe" and part.name == "probe"
    assert part.argv[1:3] == ["-m", "verification.runs.probe"]
    assert part.env["VERIF_DEVICE"] == "cuda" and part.env["VERIF_K"] == "21"
    report = tmp_path / "probe" / "probe.json"
    assert part.info["report"] == str(report) and part.argv[-1] == str(report)
    assert fs.part_extras(part, tmp_path / "log.txt") == {"probe": None}
    report.parent.mkdir()
    report.write_text(json.dumps({
        "device": "cuda", "threads": 8, "wall_s": 118.0, "errors": {"launch_s": "x"},
        "rates": {"gemm_bf16_flops": 7.1e14, "gemm_fp32_flops": None, "hbm_copy_bytes": 2.9e12,
                  "hash": {"per_byte_s": 1e-10, "per_leaf_s": 2e-6, "threads": 8},
                  "host_memcpy_bytes": 1.2e10}}))
    e = fs.part_extras(part, tmp_path / "log.txt")
    assert e["record"] == str(report) and e["errors"] == {"launch_s": "x"}
    assert "bf16 710.0 TFLOP/s" in e["summary"] and "fp32 n/a" in e["summary"]
    assert "hashing 10.0 GB/s" in e["summary"] and "\n" not in e["summary"]
    report.write_text("{")
    assert "extras_error" in fs.part_extras(part, tmp_path / "log.txt")
