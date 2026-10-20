"""FS29's kernel-speed run: the sizes file, the table layout, the gate, and the session's plan.
Everything here runs on the CPU; the timed GPU pass is the session's."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.verification.commitment.test_gpu_blake3 import HostKernels, hostlib  # noqa: F401
from tests.verification.decoder_helpers import tiny_config
from tests.verification.runs import test_fs11_session as t11
from verification.commitment import gpu_blake3 as gb
from verification.commitment import native
from verification.computation.leaf_sizes import leaf_sizes
from verification.runs import fs11_session as fs
from verification.runs import kernel_speed as ks

REPO = Path(__file__).resolve().parents[3]


def tiny_model(repo="tiny/model", n_s=2, n=8, dtype=torch.bfloat16):
    return ks.make_sizes(repo, "rev", tiny_config(), n_s=n_s, n=n, compute_dtype=dtype)


# ---- the sizes file ---------------------------------------------------------------------------


def test_sizes_round_trip_keeps_every_leaf(tmp_path):
    m = tiny_model()
    ref = leaf_sizes(tiny_config(), n_s=2, n=8, compute_dtype=torch.bfloat16)
    assert [(r.section, r.dtype, r.shape, r.nbytes) for r in m.rows] == \
           [(l.section, l.dtype, tuple(l.shape), l.nbytes) for l in ref]
    path = tmp_path / "s.json"
    ks.write_sizes(path, [m, tiny_model("tiny/two", n_s=1, n=4)], n_s=2, n=8,
                   compute_dtype="bfloat16")
    back = ks.read_sizes(path)
    assert [b.repo for b in back] == ["tiny/model", "tiny/two"]
    assert back[0] == m
    doc = json.loads(path.read_text())
    assert doc["models"][0]["leaves"] == len(m.rows) and doc["models"][0]["bytes"] == m.nbytes


def test_a_sizes_file_with_wrong_counts_or_lengths_is_refused(tmp_path):
    path = tmp_path / "s.json"
    ks.write_sizes(path, [tiny_model()], n_s=2, n=8, compute_dtype="bfloat16")
    doc = json.loads(path.read_text())
    doc["models"][0]["bytes"] += 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="disagree"):
        ks.read_sizes(path)
    doc["models"][0]["bytes"] -= 1
    doc["models"][0]["rows"][5][3] += 4  # a length that isn't header plus values
    doc["models"][0]["bytes"] += 4
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="header and"):
        ks.read_sizes(path)
    doc["format"] = 99
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="format"):
        ks.read_sizes(path)


def test_headers_match_the_commitments_and_are_shared():
    m = tiny_model()
    hdr_buf, off, ln, idx, payload, kind = ks.leaf_headers(m.rows)
    assert len(off) == len(ln) == len(set(idx)) < len(m.rows)
    for r, i, p, k in zip(m.rows, idx, payload, kind):
        assert len(hdr_buf[int(off[i]):int(off[i]) + int(ln[i])]) + int(p) == r.nbytes
        assert k == native.FINITE_KIND[r.dtype]
    assert {int(k) for r, k in zip(m.rows, kind) if r.section == "record"} == {0}


# ---- the table layout -------------------------------------------------------------------------


def lens(*xs):
    return np.array(xs, dtype=np.uint64)


@pytest.mark.parametrize("seed", range(5))
def test_every_leaf_is_in_one_table_in_order_within_the_pool(seed):
    rng = np.random.default_rng(seed)
    payload = rng.integers(1, 5000, size=300).astype(np.uint64)
    pool = 12_000
    tables = ks.layout(payload, pool)
    assert tables[0].start == 0 and tables[-1].end == len(payload)
    assert all(a.end == b.start for a, b in zip(tables, tables[1:]))
    for t in tables:
        assert 0 < t.used <= pool
        assert int(t.offsets[0]) == 0 and len(t.offsets) == t.end - t.start
        size = ks._aligned(payload[t.start:t.end]).astype(np.int64)
        assert np.array_equal(t.offsets.astype(np.int64), np.cumsum(size) - size)
        assert all(int(o) % ks.ALIGN == 0 for o in t.offsets)
        assert t.used == int(size.sum())


def test_a_table_fills_the_pool_before_the_next_starts():
    tables = ks.layout(lens(256, 256, 256, 256, 256), 768)
    assert [(t.start, t.end) for t in tables] == [(0, 3), (3, 5)]


def test_a_leaf_larger_than_the_pool_is_a_clear_error():
    with pytest.raises(ValueError, match="leaf 1 needs 1024 bytes, more than the 512-byte pool"):
        ks.layout(lens(100, 1000, 100), 512)


def test_pool_size_is_the_step_or_the_free_memory_less_the_margin():
    gib = 2 ** 30
    assert ks.pool_size(10 * gib, 80 * gib) == 10 * gib
    p = ks.pool_size(500 * gib, 80 * gib)
    assert p % ks.ALIGN == 0 and p * (1 + ks.NODE_FRACTION) <= 80 * gib - ks.MARGIN_BYTES
    with pytest.raises(ValueError):
        ks.pool_size(gib, ks.MARGIN_BYTES)


def test_the_bf16_fill_is_finite_in_every_float_view_and_hashes_clean():
    pool = torch.empty(1 << 16, dtype=torch.uint8)
    pool.view(torch.bfloat16).normal_()
    for dt in (torch.bfloat16, torch.float32, torch.float16):
        assert torch.isfinite(pool.view(dt)).all()


# ---- the gate ---------------------------------------------------------------------------------


def models(*rates, mism=0, kernel=None):
    return [{"repo": f"m{i}", "call_gbps": r, "kernel_gbps": kernel or r * 1.5, "mismatches": mism}
            for i, r in enumerate(rates)]


@pytest.mark.parametrize("rate, verdict", [(300.0, "pass"), (299.99, "review"), (150.0, "review"),
                                           (149.99, "fail"), (900.0, "pass"), (0.5, "fail")])
def test_gate_bands_follow_the_slowest_model(rate, verdict):
    g = ks.gate(models(1000.0, rate, 2000.0))
    assert g["verdict"] == verdict and g["slowest"] == "m1" and g["call_gbps"] == rate
    assert g["kernel_gbps"] == rate * 1.5
    assert f"gate: {verdict}" in ks.gate_line(g) and "kernels alone" in ks.gate_line(g)


def test_a_mismatch_error_or_missing_count_fails_whatever_the_speed():
    assert ks.gate(models(1000.0, mism=1))["verdict"] == "fail"
    assert ks.gate([{**models(1000.0)[0], "mismatches": None}])["verdict"] == "fail"
    err = [{"repo": "a", "mismatches": None, "error": "OutOfMemoryError: x"}, *models(1000.0)]
    g = ks.gate(err)
    assert g["verdict"] == "fail" and any("OutOfMemoryError" in r for r in g["reasons"])
    assert ks.gate([])["verdict"] == "fail"
    assert "digest mismatches" in ks.gate_line(ks.gate(models(1000.0, mism=3)))


# ---- run: no CUDA, and the whole path on the host build ---------------------------------------


def test_run_without_cuda_exits_2_with_a_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rc = ks.main(["run", "--sizes", str(tmp_path / "none.json"), "--out", str(tmp_path / "o.json")])
    assert rc == 2 and "needs CUDA" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_the_module_run_from_the_shell_exits_2_without_cuda(tmp_path):
    import subprocess
    code = ("import sys, torch; torch.cuda.is_available = lambda: False; "
            "from verification.runs import kernel_speed as k; "
            f"sys.exit(k.main(['run', '--sizes', 'x', '--out', {str(tmp_path / 'o')!r}]))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2 and "needs CUDA" in r.stderr


def test_run_models_on_the_host_build_checks_every_digest(hostlib, capsys):  # noqa: F811
    m = tiny_model(n_s=2, n=8)
    step = int(ks._aligned(np.array([r.nbytes for r in m.rows], np.int64)).sum())
    pool = torch.empty(step // 3 // ks.ALIGN * ks.ALIGN + 4096, dtype=torch.uint8)
    pool.view(torch.bfloat16).normal_()
    big = max(int(x) for x in ks._aligned(np.array([r.nbytes for r in m.rows], np.int64)))
    assert pool.numel() >= big
    g = gb.GpuBlake3(HostKernels(hostlib))
    res = ks.run_models([m], pool, pool.clone(), g, 1, threads=1, say=lambda s: None)[0]
    assert "error" not in res, res.get("error")
    assert res["mismatches"] == 0 and res["tables"] > 1
    assert res["leaves"] == len(m.rows) and res["bytes"] == m.nbytes
    assert [b["band"] for b in res["bands"]] == [b[0] for b in ks.BANDS]
    assert sum(b["leaves"] for b in res["bands"]) == len(m.rows)
    assert sum(b["bytes"] for b in res["bands"]) == m.nbytes
    assert ks.gate([res])["verdict"] in ("pass", "review", "fail")


def test_a_flipped_digest_is_counted_as_a_mismatch(hostlib):  # noqa: F811
    m = tiny_model()
    pool = torch.empty(1 << 20, dtype=torch.uint8)
    pool.view(torch.bfloat16).normal_()
    host = pool.clone()
    host[:64] ^= 1  # the reference sees different contents
    g = gb.GpuBlake3(HostKernels(hostlib))
    res = ks.run_models([m], pool, host, g, 1, threads=1, say=lambda s: None)[0]
    assert res["mismatches"] > 0
    assert ks.gate([res])["verdict"] == "fail"


# ---- the session ------------------------------------------------------------------------------


def plan(tmp_path):
    return fs.kernel_speed_plan(tmp_path / "up", tmp_path / "out")


def test_the_session_is_registered_and_starts_with_the_native_build(tmp_path):
    label, build = fs.SESSIONS["kernel-speed"]
    assert label == "kernel-speed"
    chosen = fs.choose_plan(build, tmp_path / "up", tmp_path / "out", plan_file=None,
                            selftest=False)
    assert [p.name for p in chosen] == ["native_build", "kernel_tests", "kernel_speed"]


def test_the_plan_runs_the_kernel_tests_then_the_speed_run_on_the_uploads_sizes(tmp_path):
    tests, run = plan(tmp_path)
    assert tests.argv[:3] == ["{python}", "-m", "pytest"]
    assert [a for a in tests.argv if a.startswith("tests/")] == list(fs.KERNEL_TESTS)
    assert all((REPO / a).exists() for a in fs.KERNEL_TESTS)
    assert run.argv[:4] == ["{python}", "-m", "verification.runs.kernel_speed", "run"]
    assert run.argv[run.argv.index("--sizes") + 1] == str(tmp_path / "up" / fs.SIZES_NAME)
    assert run.argv[run.argv.index("--out") + 1] == run.info["report"]
    assert run.kind == "kernel_speed" and fs.SIZES_NAME == ks.SIZES_NAME


def test_the_plan_has_no_models_part_no_secrets_and_every_timeout(tmp_path):
    p = plan(tmp_path)
    assert all(x.timeout_s and x.timeout_s <= 3600 for x in p)
    assert all(x.kind != "models" for x in p)
    for x in p:
        assert not set(x.env) & set(fs.SECRETS)
        assert all("TOKEN" not in k.upper() and not k.upper().endswith("_KEY") for k in x.env)
        assert x.env.get("HF_HUB_OFFLINE", "1") == "1"


def test_the_nominal_parts_fit_half_the_default_deadline(tmp_path):
    parts = sum(x.timeout_s for x in fs.with_native_build(plan(tmp_path)))
    assert 5400 >= 1.2 * parts / 2  # the deadline is about twice the parts' limits' worth


def test_the_other_sessions_have_no_kernel_parts(tmp_path):
    for key in ("fs11", "recheck", "profile-check", "nvlink-probe"):
        for p in fs.SESSIONS[key][1](tmp_path / "up", tmp_path / "out"):
            assert p.kind != "kernel_speed" and "pytest" not in p.argv
    assert sorted(fs.SESSIONS) == ["fs11", "kernel-speed", "nvlink-probe", "profile-check",
                                   "recheck"]


def test_the_extras_carry_the_gate_into_the_summary(tmp_path):
    rep = tmp_path / "r.json"
    assert fs.part_extras(fs.Part("k", [], kind="kernel_speed", info={"report": str(rep)}),
                          tmp_path / "log") == {"gate": None}
    g = ks.gate(models(400.0))
    rep.write_text(json.dumps({"gate": g, "gate_line": ks.gate_line(g)}))
    ex = fs.part_extras(fs.Part("k", [], kind="kernel_speed", info={"report": str(rep)}),
                        tmp_path / "log")
    assert ex["gate"]["verdict"] == "pass" and ex["gate_line"].startswith("FS29 gate: pass")


def test_the_job_accepts_the_session_and_runs_its_selftest(tmp_path):
    r, out = t11._job(tmp_path, plan=t11.OK_PLAN,
                      extra={"FS11_SESSION": "kernel-speed", "FS11_ALLOW_ANY_GPU": "1",
                             "FS11_SELFTEST": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "session kernel-speed" in (out / "job.log").read_text()


def test_the_job_log_ends_with_the_gate_line(tmp_path):
    g = ks.gate(models(250.0))
    ok = {"kind": "kernel_speed", "status": "ok", "extras": {"gate_line": ks.gate_line(g)}}
    assert fs._hw_lines([ok], None) == [ks.gate_line(g)]
    assert "FS29 gate: fail" in fs._hw_lines([{"kind": "kernel_speed", "status": "failed"}],
                                              None)[0]
    assert fs._hw_lines([{"kind": "plain", "status": "ok"}], None) == []
