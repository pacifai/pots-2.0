"""FS22's probe: the hashing fit, the datasheet lookup, a CPU run's file, and the session part."""

import json
from types import SimpleNamespace

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


def test_fit_refits_a_negative_leaf_cost_to_zero():
    # times that fall with the leaf count at equal bytes: the free fit's leaf cost is negative
    mixes = [(100, 10, 1.0), (100, 1000, 0.9), (1000, 10, 10.0), (1000, 1000, 9.9)]
    per_byte, per_leaf = probe.fit_hash_rate(mixes)
    assert per_leaf == 0.0 and per_byte > 0
    assert 0 <= probe.fit_residual(mixes, per_byte, per_leaf) < 0.1


def test_alloc_fit_recovers_exact_rates():
    per_call, per_byte = 3.0e-3, 4.0e-10
    points = [(n, per_call + n * per_byte) for n in (16 << 20, 64 << 20, 256 << 20, 1 << 30)]
    got_call, got_byte = probe.fit_alloc_rate(points)
    assert got_call == pytest.approx(per_call, rel=1e-6)
    assert got_byte == pytest.approx(per_byte, rel=1e-6)


def test_alloc_fit_refits_a_negative_per_call_cost_to_zero():
    # times on a line with a negative intercept, as noise can give
    points = [(n, n * 1.0e-9 - 5.0e-4) for n in (1 << 20, 1 << 24, 1 << 28, 1 << 30)]
    per_call, per_byte = probe.fit_alloc_rate(points)
    assert per_call == 0.0 and per_byte > 0


def test_datasheet_lookup():
    h100 = probe.datasheet_for("NVIDIA H100 80GB HBM3")
    assert h100 is not None and h100["bf16_flops"] == 9.894e14 and h100["name"] == "H100 SXM"
    assert probe.datasheet_for("Apple M3") is None
    assert probe.datasheet_for(None) is None


def _cfg(d, d_f, heads, kv, vocab, head_dim=None):
    ns = SimpleNamespace(hidden_size=d, intermediate_size=d_f, num_attention_heads=heads,
                         num_key_value_heads=kv, vocab_size=vocab)
    if head_dim is not None:
        ns.head_dim = head_dim
    return ns


def test_linear_shapes_from_configs():
    cfgs = {"a/x": _cfg(64, 256, 4, 2, 1000), "a/y": _cfg(64, 128, 2, 1, 500, head_dim=32)}

    def loader(repo, revision):
        if repo == "a/bad":
            raise OSError("not in the cache")
        return cfgs[repo]

    errors = {}
    pins = [("a/x", "r1"), ("a/bad", "r2"), ("a/y", "r3")]
    got = probe.linear_shapes(loader, pins, errors)
    # x: head dim 64/4 = 16, so q and o are 64 x 64 and kv 64 x 32; y: head dim 32, q 64 x 64
    assert {(k, n): name for name, k, n in got} == {
        (64, 64): "x.q+x.o+y.q+y.o", (64, 32): "x.kv+y.kv", (64, 256): "x.up",
        (256, 64): "x.down", (64, 1000): "x.head", (64, 128): "y.up", (128, 64): "y.down",
        (64, 500): "y.head"}
    assert list(errors) == ["shapes:a/bad"] and "OSError" in errors["shapes:a/bad"]


def test_gemm_without_shapes_records_errors(monkeypatch):
    monkeypatch.setattr(probe, "linear_shapes", lambda loader=None, pins=None, errors=None: [])
    rates, errors, shapes = {}, {}, []
    probe.measure_gemm(1.0, rates, errors, shapes)
    assert rates == {} and shapes == []
    assert errors == {"gemm_bf16_flops": "no model shapes in the HF cache",
                      "gemm_fp32_flops": "no model shapes in the HF cache"}


def test_machine_facts_failure_is_recorded(monkeypatch):
    def boom():
        raise RuntimeError("no facts")
    monkeypatch.setattr(probe, "machine_facts", boom)
    doc = probe.run_probe(1.0)
    assert doc["machine"] is None and "no facts" in doc["errors"]["machine"]


def test_fit_m():
    for k, n in ((2048, 128256), (8192, 2048), (896, 151936)):
        m = probe.fit_m(k, n, 4)
        assert (m * k + k * n + m * n) * 4 <= probe.GPU_LIMIT // 2
    assert probe.fit_m(2048, 128256, 4) < probe.TOKENS
    assert probe.fit_m(64, 64, 4) == probe.TOKENS


@pytest.mark.parametrize("device, index", [("cuda", 0), ("cuda:1", 1)])
def test_cuda_device_is_set_by_index(monkeypatch, device, index):
    """A bare "cuda" names no index, and ``set_device`` needs one (the first H100 check)."""
    seen = []

    class Stop(Exception):
        pass

    def set_device(d):
        seen.append(d)
        raise Stop

    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe.torch.cuda, "set_device", set_device)
    with pytest.raises(Stop):
        probe.run_probe(1, env={"VERIF_DEVICE": device})
    assert seen == [index]


def test_cpu_run_writes_a_valid_file(tmp_path):
    out = tmp_path / "sub" / "probe.json"
    assert probe.main(["--out", str(out), "--seconds", "5"]) == 0
    doc = json.loads(out.read_text())
    assert doc["version"] == 1 and doc["device"] == "cpu"
    assert set(doc["rates"]) == set(probe.RATE_KEYS) and set(doc["costs"]) == set(probe.COST_KEYS)
    assert doc["determinism"] == {"tf32": False, "deterministic": True}
    assert doc["datasheet"] is None and doc["machine"] is not None
    assert doc["shapes"] and all(r["flops_per_s"] is None for r in doc["shapes"])
    assert {r["dtype"] for r in doc["shapes"]} == {"bf16", "fp32"}
    host = {"host_first_touch_bytes", "hash", "host_memcpy_bytes"}
    for key in (*probe.RATE_KEYS, *probe.COST_KEYS):
        table = doc["rates"] if key in doc["rates"] else doc["costs"]
        if key in host:
            assert table[key] is not None and key not in doc["errors"]
        else:
            assert table[key] is None and doc["errors"][key]
    for key in ("pinned_alloc_bytes", "pinned_alloc_per_call_s"):
        assert doc["rates"][key] is None and "CUDA" in doc["errors"][key]
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
