"""FS32 round 4 tester: tries to break the peer-copy byte check, its session stop and the probe's
verdict. The comparison runs on the CPU with injected copies; the side-stream case is held against
``MatmulCapture._route`` with recorded fake streams. Its three former ``xfail(strict=True)``
tests, defects the tester found, pass since round 5."""

import contextlib
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode

from tests.verification.runs.test_nvlink_probe import P2P_4, TOPO_4, pair_rates
from verification.prover.capture import MatmulCapture
from verification.runs import fs11_session as fs
from verification.runs import nvlink_probe as nv
from verification.runs import peer_check as pc

CPU = torch.device("cpu")
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def small(monkeypatch):
    """The 256 MB case at 1 MB: the comparison is the same, the CPU test is quick."""
    monkeypatch.setitem(pc.SIZES, "256MB", 1 << 18)


def right(kind, src, dst_device):
    return src.contiguous().clone()


def control_right(src, dst_device):
    return src.clone()


def run(copy, control=control_right, **kw):
    return pc.check_pair(CPU, CPU, copy, control, **kw)


def flip_last_byte(out):
    out.contiguous().view(torch.uint8).view(-1)[-1] ^= 0x80


def hits(case, kind, src):
    """Whether the copy ``kind`` of ``src`` belongs to ``case``."""
    if case == "strided":
        return kind == "copy_" and src.ndim == 2
    if case == "side_stream":
        return kind == "side_stream"
    if case == "bf16_4MB":
        return src.dtype == torch.bfloat16
    if case == "3d_4MB":
        return src.ndim == 3
    if case == "fp32_4MB":
        return src.dtype == torch.float32 and src.ndim == 1
    if case == "offset_4MB":
        return src.dtype == torch.int32 and src.ndim == 1 and src.storage_offset() != 0
    plain = src.dtype == torch.int32 and src.ndim == 1 and src.storage_offset() == 0
    if not plain or kind == "side_stream":
        return False
    k, label = case.rsplit("_", 1)
    return kind == (k if k == "to" else "copy_") and src.numel() == pc.SIZES[label]


# ---- the comparison ---------------------------------------------------------------------------

ALL_CASES = [*pc.CONTIGUOUS_CASES, *pc.EXTRA_CASES, "strided", "side_stream"]


@pytest.mark.parametrize("case", ALL_CASES)
def test_a_wrong_last_byte_in_each_case_is_caught_and_named_alone(case):
    def copy(kind, src, dst):
        out = right(kind, src, dst)
        if hits(case, kind, src):
            flip_last_byte(out)
        return out

    r = run(copy)
    assert r["cases"][case] is False and not r["ok"] and r["gbps"] is None
    assert [c for c, ok in r["cases"].items() if not ok] == [case]
    report = pc.check([CPU, CPU], copy, control_right)
    assert all(f"wrong bytes in {case}" in x for x in pc.report_lines(report))


@pytest.mark.parametrize("fill", [0, -1])
def test_a_copy_that_writes_a_constant_is_caught_in_every_direct_case(fill):
    r = run(lambda kind, src, dst: torch.full(src.shape, fill, dtype=src.dtype))
    assert r["cases"]["host_staged"] is True
    assert [c for c, ok in r["cases"].items() if ok] == ["host_staged"]


def test_a_strided_copy_that_ignores_the_transpose_is_caught():
    def copy(kind, src, dst):
        # copies the storage order, not the view's order
        return src.t().contiguous().clone() if src.ndim == 2 and not src.is_contiguous() \
            else right(kind, src, dst)

    r = run(copy)
    assert r["cases"]["strided"] is False and r["cases"]["to_4MB"] is True


@pytest.mark.parametrize("case", ["to_4KB", "strided"])
def test_a_right_value_in_another_shape_is_caught(case):
    """``torch.equal`` of the uint8 views compares shapes too, so a reshaped result is wrong."""
    def copy(kind, src, dst):
        out = right(kind, src, dst)
        if hits(case, kind, src):
            return out.reshape(2, -1) if out.ndim == 1 else out.reshape(-1)
        return out

    assert run(copy)["cases"][case] is False


def test_a_right_value_in_another_dtype_is_caught():
    def copy(kind, src, dst):
        out = right(kind, src, dst)
        return out.view(torch.float32) if hits("to_4KB", kind, src) else out

    assert run(copy)["cases"]["to_4KB"] is False


def test_a_copy_that_returns_the_source_object_on_the_wrong_device_is_caught():
    r = pc.check_pair(CPU, torch.device("meta"), lambda kind, src, dst: src,
                      lambda src, dst: src.clone())
    assert not r["ok"]


def test_the_sources_are_random_bits():
    seen = []
    run(lambda kind, src, dst: (seen.append(src.clone()), right(kind, src, dst))[1])
    assert len(seen) >= 8
    for s in seen[:8]:
        b = s.contiguous().view(torch.uint8).view(-1)
        assert b.unique().numel() >= 200
        assert s.dtype in (torch.int32, torch.bfloat16, torch.float32)
        assert (b != 0).float().mean() > 0.9  # not zeros, so a copy that writes zeros shows


def test_each_case_has_its_own_source():
    seen = []
    run(lambda kind, src, dst: (seen.append(src.clone()), right(kind, src, dst))[1])
    direct = seen[:8]  # to/copy_ at three sizes, strided, side stream
    for i, a in enumerate(direct):
        for b in direct[i + 1:]:
            assert not (a.shape == b.shape and torch.equal(a, b))


def test_the_host_staged_source_differs_from_the_4kb_direct_source():
    seen = []

    def control(src, dst):
        seen.append(src.clone())
        return src.clone()

    run(lambda kind, src, dst: (seen.append(src.clone()), right(kind, src, dst))[1], control)
    assert not torch.equal(seen[0], seen[1])


def test_a_control_that_raises_marks_the_card_and_no_direct_copy_runs():
    calls = []

    def control(src, dst):
        raise RuntimeError("CUDA error: unspecified launch failure")

    r = run(lambda kind, src, dst: calls.append(kind), control)
    assert r["card_suspect"] is True and r["ok"] is False and calls == []
    assert "unspecified launch failure" in r["errors"]["host_staged"]
    lines = pc.report_lines({"pairs": {"0->1": r}, "all_ok": False, "seconds": 0})
    assert "could not copy" in lines[0] and "host_staged raised RuntimeError" in lines[0]
    assert "wrong bytes" not in lines[0] and "peer path is" not in lines[0]


def test_a_corrupt_control_runs_no_direct_copy():
    calls = []
    r = run(lambda kind, src, dst: calls.append(kind), lambda s, d: s.clone() ^ 1)
    assert r["card_suspect"] and calls == []


def test_every_case_that_raises_is_one_false_case_and_the_pair_keeps_its_other_cases():
    def copy(kind, src, dst):
        if hits("copy_4MB", kind, src):
            raise MemoryError("out of memory")
        return right(kind, src, dst)

    r = run(copy)
    assert r["cases"]["copy_4MB"] is False and "MemoryError" in r["errors"]["copy_4MB"]
    assert sum(r["cases"].values()) == len(r["cases"]) - 1
    report = pc.check([CPU, CPU, CPU], copy, control_right)
    assert len(report["pairs"]) == 6 and not report["all_ok"]


def test_the_comparison_runs_after_the_synchronization():
    """A copy whose bytes land only at ``sync`` is right only if ``sync`` runs before the read."""
    pending = []

    def copy(kind, src, dst):
        out = torch.zeros(src.shape, dtype=src.dtype)
        pending.append((out, src.clone()))
        return out

    def control(src, dst):
        out = torch.zeros(src.shape, dtype=src.dtype)
        pending.append((out, src.clone()))
        return out

    def sync():
        while pending:
            out, want = pending.pop()
            out.copy_(want)

    assert run(copy, control, sync=sync)["ok"]
    # the same copies without the sync land nothing
    assert not run(copy, control, sync=lambda: None)["ok"]


def test_every_ordered_pair_is_named_by_its_positions():
    report = pc.check([CPU] * 4, right, control_right)
    assert sorted(report["pairs"]) == sorted(f"{i}->{j}" for i in range(4) for j in range(4)
                                             if i != j) and report["all_ok"]


# ---- the side-stream case against capture._route ----------------------------------------------


class Log(TorchFunctionMode):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", str(func))
        if name == "empty":
            self.log.append(("empty", tuple(args[0]) if args else None, str(kwargs.get("dtype")),
                             str(kwargs.get("device"))))
        elif name == "copy_":
            self.log.append(("copy_", tuple(args[0].shape), str(args[1].dtype),
                             bool(kwargs.get("non_blocking", args[2] if len(args) > 2 else False))))
        return func(*args, **kwargs)


def _fake_cuda(monkeypatch, log):
    class Stream:
        def __init__(self, device=None):
            self.device = device
            log.append("stream")

        def wait_stream(self, other):
            log.append(("wait_stream", other))

    class Event:
        def record(self, s):
            log.append("event.record")

        def synchronize(self):
            log.append("event.sync")

        def query(self):
            return True

    @contextlib.contextmanager
    def stream(s):
        log.append("enter")
        try:
            yield
        finally:
            log.append("exit")

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda d=None: "current")
    return Stream


def test_the_side_stream_case_does_what_route_does(monkeypatch):
    """Both: a side stream of the source device waits on its current stream; under it an empty on
    the destination and ``copy_(non_blocking=True)``; an event recorded on it. The differences:
    ``_route`` leaves the event to ``_drain``; the check synchronizes it at once. Both keep one
    stream per source device (``buffers.stream``, ``_side_stream``)."""
    src = torch.randint(0, 100, (64,), dtype=torch.int32)
    dst_device = torch.device("meta")
    monkeypatch.setattr(pc, "_SIDE", {})
    ours, theirs = [], []
    _fake_cuda(monkeypatch, ours)
    with Log(ours):
        pc.cuda_copy("side_stream", src, dst_device)
    _fake_cuda(monkeypatch, theirs)
    cap = SimpleNamespace(buffers=SimpleNamespace(stream=lambda d: torch.cuda.Stream(d)),
                          _pending=deque(), _drain=lambda **kw: None,
                          _hashable=lambda r, o: False)
    rec = SimpleNamespace(out=src, hint=dst_device, versions={})
    with Log(theirs):
        MatmulCapture._route(cap, rec)
    assert ours == theirs + ["event.sync"], (ours, theirs)
    assert ours[:4] == ["stream", ("wait_stream", "current"), "enter", ours[3]]
    assert [x for x in ours if isinstance(x, tuple) and x[0] in ("empty", "copy_")] == [
        ("empty", (64,), "torch.int32", "meta"), ("copy_", (64,), "torch.int32", True)]
    assert rec.out.device == dst_device and len(cap._pending) == 1


# ---- the sessions ----------------------------------------------------------------------------


def test_every_multi_gpu_plan_is_native_build_then_peer_check_then_the_rest(tmp_path):
    for name in fs.MULTI_GPU_SESSIONS:
        plan = fs.choose_plan(fs.SESSIONS[name][1], tmp_path / "up", tmp_path / "out",
                              plan_file=None, selftest=False)
        assert [p.name for p in plan[:2]] == [fs.NATIVE_BUILD_NAME, fs.PEER_CHECK_NAME], name
        assert sum(p.kind == "peer_check" for p in plan) == 1, name


def test_the_multi_gpu_list_is_every_session_that_names_two_or_more_cards(tmp_path):
    """A session that sets up a two-card placement (VERIF_VERIFIER_DEVICES with a comma) or reads
    FS31_CARDS is multi-GPU; none may be left out of ``MULTI_GPU_SESSIONS``."""
    for name, (_, build) in fs.SESSIONS.items():
        plan = build(tmp_path / "up", tmp_path / "out")
        multi = any("," in (p.env.get("VERIF_VERIFIER_DEVICES") or "") for p in plan) \
            or any(p.kind in ("nvlink", "pair_card", "pair_run") for p in plan)
        assert multi == (name in fs.MULTI_GPU_SESSIONS), name


def test_the_session_module_and_every_plan_do_not_import_the_check():
    """The session module imports no torch (``verification.commitment.native`` is imported where
    the upload check uses it), and the session and its plans must not import ``peer_check``,
    which does."""
    code = ("import sys, tempfile, pathlib\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from verification.runs import fs11_session as fs\n"
            "t = pathlib.Path(tempfile.mkdtemp())\n"
            "for n, (_, b) in fs.SESSIONS.items(): b(t / 'up', t / 'out')\n"
            "fs.selftest_plan(); fs.peer_check_part(t)\n"
            "fs.peer_check_stop({'status': 'ok', 'extras': {}})\n"
            "bad = [m for m in sys.modules if m.endswith('peer_check')]\n"
            "print(bad)\n"
            "sys.exit(1 if bad else 0)\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                       timeout=60, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


def _session(tmp_path, argv, *, timeout_s=30, name="nvlink-probe", plan_fn=None):
    """The probe session's plan with the check's command replaced by ``argv``; the others pass."""
    import dataclasses

    plan = fs.with_native_build(
        (plan_fn or fs.nvlink_probe_plan)(tmp_path / "up", tmp_path / "out"))
    plan = [dataclasses.replace(p, argv=argv, timeout_s=timeout_s) if p.kind == "peer_check"
            else dataclasses.replace(p, argv=["{python}", "-c", "pass"], timeout_s=30)
            for p in plan]
    lines = []
    s = fs.run_session(plan, upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=time.time() + 120, cwd=Path.cwd(), say=lines.append, session=name)
    return s, lines, plan


@pytest.fixture
def no_upload_check(monkeypatch):
    monkeypatch.setattr(fs, "check_upload", lambda up: {"s": 0.0, "packed": [], "models": []})


def _stops(s, status):
    by = {p["name"]: p for p in s["parts"]}
    assert by["peer_check"]["status"] == status
    assert by["models"]["status"] == by["nvlink_probe"]["status"] == "not_started"
    assert by["models"]["reason"] == s["stopped"]
    assert not (Path(s["out"]) / "parts" / "models" / "DONE").exists()


def test_a_check_that_times_out_stops_the_session(tmp_path, no_upload_check):
    s, lines, _ = _session(tmp_path, ["{python}", "-c", "import time; time.sleep(60)"],
                           timeout_s=2)
    _stops(s, "killed_timeout")
    assert "left no report" in s["stopped"] and "killed_timeout" in s["stopped"]
    assert any(x.startswith("STOP:") for x in lines)
    assert any(x.startswith("HW2 FAIL:") and "did not run" in x for x in lines)


def test_a_check_killed_by_a_signal_stops_the_session(tmp_path, no_upload_check):
    s, _, _ = _session(tmp_path, ["{python}", "-c", "import os, signal; os.kill(os.getpid(), 9)"])
    _stops(s, "failed")
    assert "ended failed" in s["stopped"]


def test_a_check_that_exits_zero_with_a_report_that_is_not_json_stops_the_session(
        tmp_path, no_upload_check):
    code = ("import sys, os; os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True); "
            "open(sys.argv[1], 'w').write('{')")
    plan = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")
    report = next(p for p in plan if p.kind == "peer_check").info["report"]
    s, _, _ = _session(tmp_path, ["{python}", "-c", code, report])
    _stops(s, "ok")
    assert "left no report" in s["stopped"]


def test_a_check_that_exits_zero_but_reports_not_all_ok_stops_the_session(
        tmp_path, no_upload_check):
    plan = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")
    report = next(p for p in plan if p.kind == "peer_check").info["report"]
    code = ("import sys, os, json; os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True); "
            "json.dump({'all_ok': False, 'pairs': {}, 'reason': '1 CUDA device visible'}, "
            "open(sys.argv[1], 'w'))")
    s, _, _ = _session(tmp_path, ["{python}", "-c", code, report])
    _stops(s, "ok")
    assert "1 CUDA device visible" in s["stopped"]


def test_a_done_check_whose_report_is_gone_stops_the_rerun(tmp_path, no_upload_check):
    plan = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")
    report = Path(next(p for p in plan if p.kind == "peer_check").info["report"])
    code = ("import sys, os, json; os.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True); "
            "json.dump({'all_ok': True, 'pairs': {'0->1': {'ok': True, 'cases': {}}}}, "
            "open(sys.argv[1], 'w'))")
    s, _, argv_plan = _session(tmp_path, ["{python}", "-c", code, str(report)])
    assert "stopped" not in s
    report.unlink()
    s, _, _ = _session(tmp_path, ["{python}", "-c", "pass"])
    by = {p["name"]: p for p in s["parts"]}
    assert by["peer_check"]["status"] == "skipped_done" and "left no report" in s["stopped"]


def test_a_stopped_session_still_writes_summary_and_marks_the_failure(tmp_path, no_upload_check):
    s, _, _ = _session(tmp_path, ["{python}", "-c", "import sys; sys.exit(3)"])
    doc = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert doc["finished"] is True and doc["stopped"] == s["stopped"]
    assert "peer_check" in doc["failed"] and "models" in doc["failed"]


def test_the_pair_session_stop_names_the_peer_copies_in_its_gate_and_fails_it(
        tmp_path, no_upload_check):
    s, lines, _ = _session(tmp_path, ["{python}", "-c", "import sys; sys.exit(1)"],
                           name="fs32-pair", plan_fn=fs.fs32_pair_plan)
    pair = s["fs32_pair"]
    assert pair["pass"] is False and pair["peer_bytes_ok"] is False
    assert [x for x in lines if x.startswith("FS32")][0].startswith("FS32 FAIL: ")
    assert "peer" in [x for x in lines if x.startswith("FS32")][0]
    assert all(p["status"] == "not_started" for p in s["parts"] if p["name"] not in
               (fs.NATIVE_BUILD_NAME, "peer_check"))


# ---- the probe ---------------------------------------------------------------------------------


def _verdict(peer_bytes, n=4, expected=4, topo=TOPO_4, p2p=P2P_4):
    return nv.hw2_verdict(nv.parse_gpu_matrix(topo), nv.parse_gpu_matrix(p2p),
                          pair_rates(n, 380.0), expected, None, peer_bytes)


def test_the_verdict_for_missing_empty_partial_all_right_and_one_wrong_bytes():
    keys = [f"{i}->{j}" for i in range(4) for j in range(4) if i != j]
    good = {k: True for k in keys}
    assert _verdict(good)["pass"] is True
    for bad, word in ((None, "byte check"), ({}, "byte check"),
                      ({k: True for k in keys[:-1]}, keys[-1]),
                      ({**good, "2->1": False}, "2->1")):
        v = _verdict(bad)
        assert v["verdict"] == "fail" and v["pass"] is False, bad
        assert any(word in r for r in v["reasons"]), (bad, v["reasons"])
        assert not all(v["peer_bytes_ok"].values())


def test_a_byte_result_that_is_not_the_boolean_true_is_wrong():
    keys = [f"{i}->{j}" for i in range(4) for j in range(4) if i != j]
    for bad in (1, "yes", None):
        v = _verdict({**{k: True for k in keys}, "0->3": bad})
        assert v["verdict"] == "fail" and v["peer_bytes_ok"]["0->3"] is False


def test_wrong_bytes_beat_a_review_rate_and_a_pass_rate():
    keys = [f"{i}->{j}" for i in range(4) for j in range(4) if i != j]
    v = nv.hw2_verdict(nv.parse_gpu_matrix(TOPO_4), nv.parse_gpu_matrix(P2P_4),
                       pair_rates(4, 380.0, slow=("0->1", 200.0)), 4, None,
                       {**{k: True for k in keys}, "3->0": False})
    assert v["verdict"] == "fail" and v["pass"] is False


@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_checks_pair_keys_are_the_probes_pair_keys(n, monkeypatch):
    """``peer_check.check``'s ``"i->j"`` keys, for ``n`` devices, are the keys the verdict wants
    from a topology of ``n`` GPUs: a rehearsal on two cards and the gate on four."""
    names = [f"GPU{i}" for i in range(n)]
    topo = "\t".join([""] + names) + "\n" + "\n".join(
        f"GPU{i}\t" + "\t".join("X" if i == j else "NV18" for j in range(n)) for i in range(n))
    p2p = "\t".join([""] + names) + "\n" + "\n".join(
        f"GPU{i}\t" + "\t".join("X" if i == j else "OK" for j in range(n)) for i in range(n))
    report = pc.check([CPU] * n, right, control_right)
    peer = {k: p["ok"] for k, p in report["pairs"].items()}
    monkeypatch.setenv("FS31_CARDS", str(n))
    v = nv.hw2_verdict(nv.parse_gpu_matrix(topo), nv.parse_gpu_matrix(p2p), pair_rates(n, 380.0),
                       n, None, peer)
    assert v["peer_bytes_ok"] == {k: True for k in peer} and v["verdict"] == "pass"
    assert not v["reasons"] and v["gate"] is (n == 4)


def test_peer_bytes_by_pair_runs_the_check_on_every_visible_device(monkeypatch):
    seen = {}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(pc, "check", lambda devices: seen.setdefault("d", list(devices)) and {
        "pairs": {"0->1": {"ok": True}, "1->0": {"ok": False}}})
    assert nv.peer_bytes_by_pair() == {"0->1": True, "1->0": False}
    assert [str(d) for d in seen["d"]] == ["cuda:0", "cuda:1", "cuda:2"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="needs two"):
        nv.peer_bytes_by_pair()


def test_hw2_line_names_wrong_pairs_for_review_and_fail_but_not_for_a_rehearsal_pass():
    keys = [f"{i}->{j}" for i in range(4) for j in range(4) if i != j]
    wrong = _verdict({**{k: True for k in keys}, "1->2": False})
    line = nv.hw2_line(wrong)
    assert line.startswith("HW2 FAIL: peer copies were WRONG") and "1->2" in line
    ok = _verdict({k: True for k in keys})
    assert "WRONG" not in nv.hw2_line(ok)
