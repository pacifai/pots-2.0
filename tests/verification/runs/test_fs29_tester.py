"""FS29 tester's tests: the sizes file against the protocol's own leaf encoding, the table layout's
edges, the check's failure modes through ``run_command`` on the host kernels, the gate, the rates
and the session. No CUDA: ``run_command`` runs with the device calls stood in for."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from setup.records import Record
from tests.verification.commitment.test_gpu_blake3 import HostKernels, hostlib  # noqa: F401
from tests.verification.decoder_helpers import tiny_config
from verification.commitment import gpu_blake3 as gb
from verification.commitment import leaves as lv
from verification.computation.leaf_sizes import leaf_sizes
from verification.computation.instances.decoder import DecoderComputation
from verification.runs import fs11_prep
from verification.runs import fs11_session as fs
from verification.runs import kernel_speed as ks

REPO = Path(__file__).resolve().parents[3]
MARGIN = ks.MARGIN_BYTES


# ---- sizes ------------------------------------------------------------------------------------


def test_sizes_command_follows_full_scale_env_and_equals_leaf_sizes(tmp_path, monkeypatch):
    import setup.model as sm
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(sm, "load_model_config", lambda repo, rev: tiny_config())
    monkeypatch.setattr(ks, "RUN_PINS", (("a/one", "r1"), ("b/two", "r2")))
    for batch, seq, dt in [("3", "8", "bfloat16"), ("5", "16", "bfloat16"),
                           ("2", "8", "float32")]:
        monkeypatch.setitem(fs.FULL_SCALE_ENV, "VERIF_BATCH", batch)
        monkeypatch.setitem(fs.FULL_SCALE_ENV, "VERIF_SEQ_LEN", seq)
        monkeypatch.setitem(fs.FULL_SCALE_ENV, "VERIF_COMPUTE_DTYPE", dt)
        out = tmp_path / f"s_{batch}_{seq}_{dt}.json"
        assert ks.sizes_command(out) == 0
        doc = json.loads(out.read_text())
        assert (doc["n_s"], doc["n"], doc["compute_dtype"]) == (int(batch), int(seq), dt)
        ref = leaf_sizes(tiny_config(), n_s=int(batch), n=int(seq),
                         compute_dtype=getattr(torch, dt))
        back = ks.read_sizes(out)
        assert [m.repo for m in back] == ["a/one", "b/two"]
        assert [m.revision for m in back] == ["r1", "r2"]
        for m in back:
            assert [(r.section, r.dtype, r.shape, r.nbytes) for r in m.rows] == \
                   [(l.section, l.dtype, tuple(l.shape), l.nbytes) for l in ref]
            assert m.nbytes == sum(l.nbytes for l in ref)
        # the JSON round trip is lossless
        out2 = tmp_path / "again.json"
        ks.write_sizes(out2, back, n_s=int(batch), n=int(seq), compute_dtype=dt)
        assert ks.read_sizes(out2) == back


def test_sizes_products_follow_the_compute_dtype(tmp_path, monkeypatch):
    import setup.model as sm
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(sm, "load_model_config", lambda repo, rev: tiny_config())
    monkeypatch.setattr(ks, "RUN_PINS", (("a/one", "r1"),))
    got = {}
    for dt in ("bfloat16", "float32"):
        monkeypatch.setitem(fs.FULL_SCALE_ENV, "VERIF_COMPUTE_DTYPE", dt)
        ks.sizes_command(tmp_path / f"{dt}.json")
        rows = ks.read_sizes(tmp_path / f"{dt}.json")[0].rows
        got[dt] = {r.dtype for r in rows if r.section == "product"}
        assert {r.dtype for r in rows if r.section in ("w_t", "w_next")} == {torch.float32}
    assert got == {"bfloat16": {torch.bfloat16}, "float32": {torch.float32}}


# ---- headers ----------------------------------------------------------------------------------


@pytest.mark.parametrize("family, tied", [("llama", True), ("llama", False), ("qwen2", True)])
def test_every_leafs_table_header_is_what_the_protocol_encodes(family, tied):
    cfg = tiny_config(family=family, tied=tied)
    n_s, n = 3, 8
    c = DecoderComputation(cfg, n_s=n_s, n=n, eta=1.0, compute_dtype=torch.bfloat16)
    sizes = leaf_sizes(cfg, n_s=n_s, n=n, compute_dtype=torch.bfloat16)
    rows = [ks.Row(l.section, l.dtype, tuple(l.shape), l.nbytes) for l in sizes]
    hdr_buf, off, ln, idx, payload, kind = ks.leaf_headers(rows)
    assert len(rows) == c.n_leaves
    zeros = torch.zeros(n, dtype=torch.int32)
    for i, r in enumerate(rows):
        table_hdr = hdr_buf[int(off[idx[i]]):int(off[idx[i]]) + int(ln[idx[i]])]
        if i < n_s:
            enc = c.encode_record(Record(ids=zeros, targets=zeros, mask=zeros))
        else:
            tag, shape, dtype, _ = lv._tensor_slot(c, i)
            assert (tuple(r.shape), r.dtype) == (shape, dtype)
            hdr, body = lv.leaf_parts(c, i, torch.zeros(shape, dtype=dtype))
            enc = bytes(hdr) + bytes(body)
        assert enc[:len(table_hdr)] == table_hdr, f"leaf {i} ({r.section})"
        assert len(enc) == r.nbytes == len(table_hdr) + int(payload[i])


# ---- the table layout -------------------------------------------------------------------------


def _check_tables(payload, pool):
    tables = ks.layout(payload, pool)
    size = ks._aligned(payload).astype(np.int64)
    seen = []
    for t in tables:
        assert t.used <= pool
        assert int(t.offsets[-1]) + int(size[t.end - 1]) == t.used
        seen += list(range(t.start, t.end))
    assert seen == list(range(len(payload)))
    # greedy: the next leaf would not have fitted in the table before it
    for a, b in zip(tables, tables[1:]):
        assert a.used + int(size[b.start]) > pool
    return tables


def test_a_leaf_exactly_the_pool_gets_a_table_of_its_own_beside_small_ones():
    t = _check_tables(np.array([256, 1024, 256], np.uint64), 1024)
    assert [(x.start, x.end) for x in t] == [(0, 1), (1, 2), (2, 3)]
    t = _check_tables(np.array([1024], np.uint64), 1024)
    assert len(t) == 1 and t[0].used == 1024


def test_an_unaligned_leaf_whose_alignment_pushes_it_over_the_pool_is_refused():
    with pytest.raises(ValueError, match="more than the"):
        ks.layout(np.array([1000], np.uint64), 1000 // 256 * 256 + 100)  # pool 868 < 1024


def test_an_empty_step_has_no_tables():
    assert ks.layout(np.array([], np.uint64), 4096) == []


def test_zero_length_payloads_do_not_loop_or_vanish():
    t = _check_tables(np.array([0, 0, 300, 0, 256], np.uint64), 512)
    assert t[0].start == 0 and t[-1].end == 5


def test_one_huge_leaf_among_small_ones_names_the_leaf():
    with pytest.raises(ValueError, match="leaf 2 needs"):
        ks.layout(np.array([10, 10, 10 ** 6, 10], np.uint64), 4096)


@pytest.mark.parametrize("seed", range(4))
def test_random_layouts_keep_every_invariant(seed):
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 3000, size=500).astype(np.uint64)
    _check_tables(payload, int(ks._aligned(payload).max()) + int(rng.integers(0, 2000)))


def test_pool_size_edges():
    gib = 2 ** 30
    assert ks.pool_size(0, 80 * gib) == 0  # an empty step has a zero pool
    # exactly the margin leaves nothing; a hair more leaves a pool
    with pytest.raises(ValueError):
        ks.pool_size(gib, MARGIN)
    p = ks.pool_size(np.array([gib], np.uint64), 80 * gib)  # array form: aligned payloads
    assert p == gib


# ---- run_command on the host kernels (no CUDA) ------------------------------------------------


class Flaky(gb.GpuBlake3):
    """The host-kernel GpuBlake3 with one fault injected into ``hash_leaves``."""

    fault = None
    calls = 0

    def hash_leaves(self, *a, **k):
        type(self).calls += 1
        digests, bad = super().hash_leaves(*a, **k)
        f = type(self).fault
        if f == "flip":
            digests = digests.clone()
            digests[0, 0] ^= 1
        elif f == "bad_plus":
            bad += 1
        elif f == "bad_minus":
            bad -= 1
        elif f == "raise_first" and type(self).calls == 1:
            raise RuntimeError("the kernel failed")
        return digests, bad


@pytest.fixture
def fake_gpu(monkeypatch, hostlib):  # noqa: F811
    """Run ``run_command`` on the CPU: CUDA looks present, the pool is a host tensor."""
    real_empty = torch.empty

    def empty(*a, device=None, **k):
        return real_empty(*a, **({} if str(device).startswith("cuda") else {"device": device}), **k)

    def make():
        Flaky.fault, Flaky.calls = None, 0
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch, "empty", empty)
        monkeypatch.setattr(gb, "describe", lambda dev: {"fake": True})
        # a small pool: the margin plus 1.08 x 24 KiB, so a tiny step needs several tables
        monkeypatch.setattr(torch.cuda, "mem_get_info",
                            lambda *a: (MARGIN + int(24 * 1024 * 1.08) + 300, 0))
        monkeypatch.setattr(gb, "GpuBlake3", lambda: Flaky(HostKernels(hostlib)))
        return Flaky

    return make


def _sizes_file(tmp_path, *models):
    path = tmp_path / "sizes.json"
    ks.write_sizes(path, models, n_s=2, n=8, compute_dtype="bfloat16")
    return path


def _run(tmp_path, *models):
    out = tmp_path / "rep.json"
    rc = ks.main(["run", "--sizes", str(_sizes_file(tmp_path, *models)), "--out", str(out),
                  "--repeats", "1"])
    return rc, json.loads(out.read_text()) if out.exists() else None


def _tiny(repo="t/a"):
    return ks.make_sizes(repo, "rev", tiny_config(), n_s=2, n=8, compute_dtype=torch.bfloat16)


def test_a_clean_run_on_a_tiny_sizes_file_has_zero_mismatches_and_exit_0(
        tmp_path, fake_gpu, capsys):
    fake_gpu()
    rc, rep = _run(tmp_path, _tiny("t/a"), _tiny("t/b"))
    assert rc == 0
    assert [m["mismatches"] for m in rep["models"]] == [0, 0]
    assert all(m["tables"] > 1 for m in rep["models"])
    assert rep["gate"]["verdict"] in ("pass", "review", "fail") and not rep["gate"]["reasons"]
    assert rep["gate_line"] in capsys.readouterr().out


@pytest.mark.parametrize("fault, why", [("flip", "digests differ"),
                                        ("bad_plus", "first_bad"),
                                        ("bad_minus", "first_bad")])
def test_a_wrong_result_is_a_mismatch_a_failed_gate_and_exit_1(tmp_path, fake_gpu, capsys,
                                                               fault, why):
    fake_gpu().fault = fault
    rc, rep = _run(tmp_path, _tiny("t/a"), _tiny("t/b"))
    assert rc == 1
    assert all(m["mismatches"] > 0 for m in rep["models"])
    assert rep["gate"]["verdict"] == "fail"
    assert "digest mismatches" in rep["gate_line"] and "gate: fail" in rep["gate_line"]
    assert why in capsys.readouterr().out


def test_a_model_that_raises_fails_the_gate_exits_1_and_the_next_model_still_runs(
        tmp_path, fake_gpu):
    fake_gpu().fault = "raise_first"
    rc, rep = _run(tmp_path, _tiny("t/a"), _tiny("t/b"))
    assert rc == 1 and rep["gate"]["verdict"] == "fail"
    first, second = rep["models"]
    assert "RuntimeError: the kernel failed" in first["error"] and first["mismatches"] is None
    assert second["mismatches"] == 0 and "error" not in second
    assert any("the kernel failed" in r for r in rep["gate"]["reasons"])


def test_a_leaf_larger_than_the_pool_is_an_error_row_not_a_crash(tmp_path, fake_gpu):
    fake_gpu()
    big = _tiny("t/big")
    # a 2-d fp32 weight's header length, taken from an existing row, so read_sizes accepts it
    two_d = next(r for r in big.rows if r.section == "w_t" and len(r.shape) == 2)
    head = two_d.nbytes - two_d.shape[0] * two_d.shape[1] * 4
    rows = list(big.rows) + [ks.Row("w_t", torch.float32, (64, 1024), head + 64 * 1024 * 4)]
    model = ks.ModelSizes("t/big", "rev", rows)
    rc, rep = _run(tmp_path, model, _tiny("t/ok"))
    assert rc == 1 and rep["gate"]["verdict"] == "fail"
    assert "more than the" in rep["models"][0]["error"] and rep["models"][1]["mismatches"] == 0


def test_a_bad_sizes_file_exits_1_with_a_message(tmp_path, fake_gpu, capsys):
    fake_gpu()
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert ks.main(["run", "--sizes", str(p), "--out", str(tmp_path / "o.json")]) == 1
    assert "cannot use" in capsys.readouterr().err
    assert ks.main(["run", "--sizes", str(tmp_path / "missing.json"),
                    "--out", str(tmp_path / "o.json")]) == 1
    assert not (tmp_path / "o.json").exists()


# ---- the rates --------------------------------------------------------------------------------


def test_the_rates_divide_header_plus_payload_bytes_by_the_time_measured(
        tmp_path, monkeypatch, hostlib):  # noqa: F811
    m = _tiny()
    grp = ks.Group(*ks.leaf_headers(m.rows), np.array([r.nbytes for r in m.rows], np.int64))
    pool = torch.empty(1 << 20, dtype=torch.uint8)
    pool.view(torch.bfloat16).normal_()
    ticks = iter(range(10 ** 6))
    # every perf_counter() call is 0.5 s after the last, so every timed interval is 0.5 s
    monkeypatch.setattr(ks, "time", SimpleNamespace(perf_counter=lambda: next(ticks) * 0.5))
    g = gb.GpuBlake3(HostKernels(hostlib))
    res = ks.measure(g, grp, pool, pool.clone(), 2, threads=1, check=True, say=lambda s: None)
    assert res["tables"] == 1 and res["mismatches"] == 0
    assert res["call_s"] == res["kernel_s"] == 0.5
    # the bytes are the table's header plus payload, per leaf, not payload alone, no 0x00 prefix
    hashed = int(grp.hdr_len[grp.hdr_idx].astype(np.int64).sum()
                 + grp.payload_len.astype(np.int64).sum())
    assert res["bytes"] == hashed == m.nbytes
    assert res["bytes"] > int(grp.payload_len.sum())
    assert res["call_gbps"] == pytest.approx(hashed / 0.5 / 1e9)
    assert res["kernel_gbps"] == pytest.approx(hashed / 0.5 / 1e9)


# ---- the gate ---------------------------------------------------------------------------------


def _m(repo, call, kernel, mism=0):
    return {"repo": repo, "call_gbps": call, "kernel_gbps": kernel, "mismatches": mism}


def test_the_slowest_models_whole_call_rate_decides_not_kernel_only_not_another_model():
    # m0 is slowest by kernel-only rate, m1 by whole call: the whole call wins
    g = ks.gate([_m("m0", 500.0, 10.0), _m("m1", 200.0, 900.0), _m("m2", 900.0, 5000.0)])
    assert g["verdict"] == "review" and g["slowest"] == "m1"
    assert (g["call_gbps"], g["kernel_gbps"]) == (200.0, 900.0)
    # a fast kernel and one slow model cannot be rescued by the others
    g = ks.gate([_m("a", 10_000.0, 10_000.0), _m("b", 149.99, 10_000.0)])
    assert g["verdict"] == "fail" and g["slowest"] == "b"
    # a low kernel-only rate with a passing whole call still passes (it is shown, not gated)
    assert ks.gate([_m("a", 300.0, 1.0)])["verdict"] == "pass"


@pytest.mark.parametrize("call, verdict", [(300.0, "pass"), (299.999, "review"),
                                           (150.0, "review"), (149.999, "fail"),
                                           (0.0, "fail"), (float("nan"), "fail"),
                                           (float("inf"), "fail")])
def test_gate_edges(call, verdict):
    assert ks.gate([_m("a", call, 1.0)])["verdict"] == verdict


@pytest.mark.parametrize("mism", [1, 3, None, "0", True])
def test_any_mismatch_fails_whatever_the_speed(mism):
    g = ks.gate([_m("a", 1e9, 1e9, mism=mism), _m("b", 1e9, 1e9)])
    assert g["verdict"] == "fail" and g["reasons"]
    assert "gate: fail" in ks.gate_line(g)


# ---- the session ------------------------------------------------------------------------------


def test_every_parts_env_is_stripped_of_secrets_on_the_pod(tmp_path, monkeypatch):
    for k in fs.SECRETS:
        monkeypatch.setenv(k, "x")
    env = fs.base_env(tmp_path)
    assert not set(env) & set(fs.SECRETS)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["POTS_NATIVE_CACHE"] == str(tmp_path / "native_cache")


def test_the_session_on_this_mac_is_refused_and_says_so(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "FS11_ALLOW_ANY_GPU"}
    up, out = tmp_path / "up", tmp_path / "out"
    up.mkdir()
    r = subprocess.run([sys.executable, "-m", "verification.runs.fs11_session", "run", "--upload",
                        str(up), "--out", str(out), "--deadline", str(time.time() + 600),
                        "--session", "kernel-speed"], cwd=REPO, env=env, capture_output=True,
                       text=True, timeout=120)
    if "CUDA" in r.stdout and "H100" in r.stdout and "available: True" in r.stdout:
        pytest.skip("this machine has an H100")
    assert r.returncode == fs.EXIT_REFUSED, r.stdout + r.stderr
    doc = json.loads((out / "summary.json").read_text())
    assert doc["session"] == "kernel-speed" and doc["refused"] and doc["parts"] == []


def test_the_refusal_names_a_machine_with_cuda_but_no_h100():
    env = {"cuda_available": True, "gpus": ["NVIDIA A40"]}
    assert "no H100" in fs.refusal(env, False)
    assert fs.refusal(env, True) is None
    assert fs.refusal({"cuda_available": True, "gpus": ["NVIDIA H100 80GB HBM3"]}, False) is None


def test_no_part_of_the_kernel_session_names_a_model_or_a_secret(tmp_path):
    parts = fs.choose_plan(fs.SESSIONS["kernel-speed"][1], tmp_path / "up", tmp_path / "out",
                           plan_file=None, selftest=False)
    text = json.dumps([(p.argv, p.env) for p in parts])
    for secret in fs.SECRETS:
        assert secret not in text
    assert all(repo not in text for repo, _ in fs.RUN_PINS)
    assert "manifest" not in text and "fs11_prep" not in text


def test_a_pod_following_the_readme_has_pytest_for_the_kernel_tests_part():
    tests = fs.kernel_speed_plan(Path("up"), Path("out"))[0]
    assert tests.argv[1:3] == ["-m", "pytest"]
    assert "pytest" in fs11_prep.PACKAGES


def test_the_readmes_kernel_speed_paragraph_says_where_the_venv_comes_from():
    text = (REPO / "verification/README.md").read_text()
    start = text.index("**The kernel's speed (FS29).**")
    para = text[start:text.index("**Estimated wall time.**", start)]
    assert "requirements" in para


def test_the_parts_limits_fit_the_work_deadline_of_the_default_deadline(tmp_path):
    parts = fs.choose_plan(fs.SESSIONS["kernel-speed"][1], tmp_path / "up", tmp_path / "out",
                           plan_file=None, selftest=False)
    work = 5400 - 1200 - 600
    assert sum(p.timeout_s for p in parts) <= work
