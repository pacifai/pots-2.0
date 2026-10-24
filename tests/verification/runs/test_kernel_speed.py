"""FS29's kernel-speed run: the sizes file, the table layout, the gate, and the session's plan.
Everything here runs on the CPU; the timed GPU pass is the session's."""

import json
import platform
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
    g = Flipping(hostlib)
    res = ks.run_models([m], pool, torch.empty(1 << 20, dtype=torch.uint8), g, 1, threads=1,
                        say=lambda s: None)[0]
    assert res["mismatches"] > 0 and res["band_mismatches"] > 0
    assert ks.gate([res])["verdict"] == "fail"


class Flipping:
    """The host build of the kernel with the first digest of every call flipped."""

    def __init__(self, lib):
        self.g = gb.GpuBlake3(HostKernels(lib))

    def hash_leaves(self, *a, **k):
        d, bad = self.g.hash_leaves(*a, **k)
        d = d.clone()
        d[0, 0] ^= 1
        return d, bad

    def prepare(self, *a, **k):
        return self.g.prepare(*a, **k)

    def launch(self, *a, **k):
        return self.g.launch(*a, **k)


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


def test_the_parts_limits_sum_inside_the_jobs_work_window(tmp_path):
    parts = sum(x.timeout_s for x in fs.with_native_build(plan(tmp_path)))
    assert parts == 900 + 900 + 1500 <= 5400 - 1200 - 600


def test_the_other_sessions_have_no_kernel_parts(tmp_path):
    for key in ("fs11", "recheck", "profile-check", "nvlink-probe"):
        for p in fs.SESSIONS[key][1](tmp_path / "up", tmp_path / "out"):
            assert p.kind != "kernel_speed" and "pytest" not in p.argv
    # the two-GPU session runs pytest parts of its own, but no kernel part
    for p in fs.SESSIONS["fs32-pair"][1](tmp_path / "up", tmp_path / "out"):
        assert p.kind != "kernel_speed" and not any("test_gpu_blake3" in a for a in p.argv)
    assert sorted(fs.SESSIONS) == ["device-leaves", "fs11", "fs32-pair", "kernel-speed",
                                   "nvlink-probe", "profile-check", "recheck"]


def test_the_extras_carry_the_gate_into_the_summary(tmp_path):
    rep = tmp_path / "r.json"
    assert fs.part_extras(fs.Part("k", [], kind="kernel_speed", info={"report": str(rep)}),
                          tmp_path / "log") == {"gate": None}
    g = ks.gate(models(400.0))
    rep.write_text(json.dumps({"gate": g, "gate_line": ks.gate_line(g)}))
    ex = fs.part_extras(fs.Part("k", [], kind="kernel_speed", info={"report": str(rep)}),
                        tmp_path / "log")
    assert ex["gate"]["verdict"] == "pass" and ex["gate_line"].startswith("FS29 gate: pass")
    rep.write_text("{not json")
    ex = fs.part_extras(fs.Part("k", [], kind="kernel_speed", info={"report": str(rep)}),
                        tmp_path / "log")
    assert ex["gate"] is None and "report_error" in ex


def test_the_job_accepts_the_session_and_runs_its_selftest(tmp_path):
    r, out = t11._job(tmp_path, plan=t11.OK_PLAN,
                      extra={"FS11_SESSION": "kernel-speed", "FS11_ALLOW_ANY_GPU": "1",
                             "FS11_SELFTEST": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "session kernel-speed" in (out / "job.log").read_text()


def test_the_job_log_ends_with_the_gate_line(tmp_path):
    g = ks.gate(models(250.0))
    ok = {"kind": "kernel_speed", "status": "ok", "extras": {"gate_line": ks.gate_line(g)}}
    ok["name"] = "kernel_speed"
    ok["extras"]["gate"] = g
    done = [{"name": "native_build", "status": "ok"}, {"name": "kernel_tests", "status": "ok"}]
    assert fs._hw_lines([*done, ok], None) == [ks.gate_line(g)]
    line = fs._hw_lines([{"kind": "kernel_speed", "status": "failed"}], None)[0]
    assert line.startswith("FS29 gate: no result (") and "fail" not in line.split("(")[0]
    assert fs._hw_lines([{"kind": "plain", "status": "ok"}], None) == []


# ---- round 2: the gate's inputs, the sliced check, the memory check, the prep -----------------


def write_tiny(path):
    ks.write_sizes(path, [tiny_model()], n_s=2, n=8, compute_dtype="bfloat16")


def _parts(native="ok", tests="ok", speed="ok", gate=None, line=None):
    ex = {"gate": gate, "gate_line": line} if gate else {"gate": None}
    return [{"name": "native_build", "status": native}, {"name": "kernel_tests", "status": tests},
            {"name": "kernel_speed", "kind": "kernel_speed", "status": speed, "extras": ex}]


def _ok_gate(rate=400.0):
    g = ks.gate(models(rate))
    return {"gate": g, "line": ks.gate_line(g)}


def test_no_report_is_no_result_never_fail():
    out = fs.kernel_speed_gate(_parts(speed="failed"))
    assert out["verdict"] == "no result" and out["line"].startswith("FS29 gate: no result (")
    assert fs.kernel_speed_gate([{"name": "x", "status": "ok"}]) is None


@pytest.mark.parametrize("bad", ["native", "tests"])
def test_a_part_before_the_speed_run_not_ok_fails_the_gate_and_keeps_the_rates(bad):
    g = _ok_gate()
    out = fs.kernel_speed_gate(_parts(**{bad: "failed"}, gate=g["gate"], line=g["line"]))
    assert out["verdict"] == "fail"
    assert out["native_build" if bad == "native" else "kernel_tests"] == "failed"
    assert out["line"].startswith("FS29 gate: fail.") and "median 400 GB/s" in out["line"]
    ok = fs.kernel_speed_gate(_parts(gate=g["gate"], line=g["line"]))
    assert ok["verdict"] == "pass" and ok["line"] == g["line"]
    assert ok["native_build"] == ok["kernel_tests"] == "ok"


def test_the_median_feeds_the_gate_with_the_best_beside_it():
    rows = [{"repo": "a", "call_gbps": 320.0, "call_best_gbps": 400.0, "kernel_gbps": 500.0,
             "kernel_best_gbps": 600.0, "mismatches": 0, "band_mismatches": 0}]
    g = ks.gate(rows)
    assert g["verdict"] == "pass" and g["call_gbps"] == 320.0 and g["call_best_gbps"] == 400.0
    line = ks.gate_line(g)
    assert "median 320 GB/s (best 400)" in line and "median 500 GB/s (best 600)" in line


def test_a_band_mismatch_fails_the_gate_whatever_the_speed():
    rows = [{**models(1000.0)[0], "band_mismatches": 2}]
    g = ks.gate(rows)
    assert g["verdict"] == "fail" and any("bands" in r for r in g["reasons"])


def test_measure_reports_median_and_best_from_every_repeat(hostlib, monkeypatch):  # noqa: F811
    m = tiny_model()
    pool = torch.empty(1 << 20, dtype=torch.uint8)
    pool.view(torch.bfloat16).normal_()
    grp = ks.Group(*ks.leaf_headers(m.rows), np.array([r.nbytes for r in m.rows], np.int64))
    seq = iter([0.0, 3.0, 10.0, 11.0, 20.0, 22.0])  # three timed calls: 3 s, 1 s, 2 s
    monkeypatch.setattr(ks.time, "perf_counter", lambda: next(seq, 99.0))
    monkeypatch.setattr(ks, "_kernel_seconds", lambda *a: [5.0, 1.0, 3.0])
    g = gb.GpuBlake3(HostKernels(hostlib))
    res = ks.measure(g, grp, pool, torch.empty(1 << 20, dtype=torch.uint8), 3, threads=1,
                     check=False, say=lambda s: None)
    t = res["table_times"][0]
    assert t["call_s"] == [3.0, 1.0, 2.0] and t["kernel_s"] == [5.0, 1.0, 3.0]
    assert res["call_s"] == 2.0 and res["call_best_s"] == 1.0 and res["tables"] == 1
    assert res["kernel_s"] == 3.0 and res["kernel_best_s"] == 1.0 and res["repeats"] == 3
    assert res["call_gbps"] == res["bytes"] / 2.0 / 1e9


def test_the_sliced_check_covers_every_leaf_in_several_slices(hostlib, monkeypatch):  # noqa: F811
    m = tiny_model(n_s=2, n=8)
    pool = torch.empty(1 << 20, dtype=torch.uint8)
    pool.view(torch.bfloat16).normal_()
    grp = ks.Group(*ks.leaf_headers(m.rows), np.array([r.nbytes for r in m.rows], np.int64))
    big = int(ks._aligned(grp.payload_len).max())
    spec = ks.layout(grp.payload_len, pool.numel())[0]
    runs = ks.check_slices(grp, spec, 3 * big)
    assert len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == spec.end - spec.start
    assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    assert all(b1 - b0 <= 3 * big for _, _, b0, b1 in runs)
    small = torch.empty(3 * big, dtype=torch.uint8)
    res = ks.run_models([m], pool, small, gb.GpuBlake3(HostKernels(hostlib)), 1, threads=1,
                        say=lambda s: None)[0]
    assert "error" not in res and res["mismatches"] == 0 and res["band_mismatches"] == 0
    with pytest.raises(ValueError, match="check slice"):
        ks.check_slices(grp, spec, big - 1)


def test_host_available_bytes_reads_meminfo_and_the_cgroup_headroom(tmp_path):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal: 999 kB\nMemAvailable:    2048 kB\n")
    none = (("/nonexistent/a", "/nonexistent/b"),)
    assert ks.host_available_bytes(info, none) == 2048 * 1024
    (tmp_path / "max").write_text("1048576\n")
    (tmp_path / "cur").write_text("1000000\n")
    assert ks.host_available_bytes(info, ((str(tmp_path / "max"), str(tmp_path / "cur")),)) \
        == 48576
    (tmp_path / "max").write_text("max\n")
    assert ks.host_available_bytes(info, ((str(tmp_path / "max"), str(tmp_path / "cur")),)) \
        == 2048 * 1024
    assert ks.host_available_bytes(tmp_path / "none", none) is None


def test_the_run_stops_when_host_memory_is_under_twice_the_slice(tmp_path, monkeypatch, capsys):
    sizes = tmp_path / "s.json"
    write_tiny(sizes)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (6 << 30, 8 << 30))
    monkeypatch.setattr(ks, "host_available_bytes", lambda *a, **k: 1000)
    rc = ks.main(["run", "--sizes", str(sizes), "--out", str(tmp_path / "o.json")])
    assert rc == 1 and "under twice" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_a_sizes_file_with_no_models_or_a_non_step_dtype_is_a_clear_error(tmp_path, monkeypatch,
                                                                         capsys):
    good = tmp_path / "g.json"
    write_tiny(good)
    doc = json.loads(good.read_text())
    empty = tmp_path / "e.json"
    empty.write_text(json.dumps({**doc, "models": []}))
    with pytest.raises(ValueError, match="no models"):
        ks.read_sizes(empty)
    odd = tmp_path / "d.json"
    odd.write_text(json.dumps({**doc, "dtypes": [*doc["dtypes"][:-1], "float64"]}))
    with pytest.raises(ValueError, match="float64"):
        ks.read_sizes(odd)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    rc = ks.main(["run", "--sizes", str(empty), "--out", str(tmp_path / "o.json")])
    assert rc == 1 and "cannot use" in capsys.readouterr().err


def test_packages_list_the_kernel_tests_and_plot_dependencies():
    from verification.runs import fs11_prep
    assert "pytest" in fs11_prep.PACKAGES and "matplotlib" in fs11_prep.PACKAGES


def test_the_kernel_speed_prep_packs_only_repo_native_sizes_requirements_and_manifest(tmp_path):
    import tarfile
    from verification.runs import fs11_prep
    src = tmp_path / "native" / f"blake3-{native.BLAKE3_TAG}"
    src.mkdir(parents=True)
    for n in (*native.SOURCE_SHA256, "ARCHIVE_SHA256"):
        (src / n).write_bytes(b"x")
    sizes = tmp_path / "s.json"
    write_tiny(sizes)

    def repo(root, build):
        return ([fs11_prep.Member("repo/a.py", data=b"print(1)\n")],
                {"commit": "abc", "dirty": False})

    monkey = pytest.MonkeyPatch()
    monkey.setattr(native, "bad_sources", lambda d: [])
    try:
        man = fs11_prep.prepare_kernel_speed(tmp_path / "out", sizes=sizes, repo_root=REPO,
                                             repo=repo, native_src=src, say=lambda s: None)
    finally:
        monkey.undo()
    with tarfile.open(man["tarball"]) as t:
        names = t.getnames()
    tops = {n.split("/")[1] for n in names if "/" in n}
    assert tops <= {"manifest.json", "repo", "native_cache", ks.SIZES_NAME,
                    "requirements-fs11.txt"}
    assert not [n for n in names if n.split("/")[1] in ("hf", "data", "bench_cache")]
    assert f"{fs11_prep.TOP}/{ks.SIZES_NAME}" in names and f"{fs11_prep.TOP}/manifest.json" in names
    assert set(man["files"]) >= {ks.SIZES_NAME, "requirements-fs11.txt", "repo/a.py"}
    with tarfile.open(man["tarball"]) as t:  # the pod's Python is pinned to this one
        packed = json.load(t.extractfile(f"{fs11_prep.TOP}/manifest.json"))
    assert man["python_version"] == packed["python_version"] == platform.python_version()
    assert fs11_prep.main(["--session", "kernel-speed", "--out", str(tmp_path / "o")]) == 2


def test_the_readme_and_docstring_say_what_whole_call_covers():
    for text in ((REPO / "verification" / "README.md").read_text(),
                 (REPO / "verification" / "CLAUDE.md").read_text(), ks.__doc__):
        assert "FS30" in text and "ceiling" in text
