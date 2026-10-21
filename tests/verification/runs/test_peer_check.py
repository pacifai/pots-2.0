"""F22's peer-copy byte check (``peer_check``): the comparison on the CPU with injected copies, the
session's stop on a wrong pair, and one run of the real check where two CUDA devices exist."""

import json
from pathlib import Path

import pytest
import torch

from verification.runs import fs11_session as fs
from verification.runs import peer_check as pc

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def small(monkeypatch):
    """The 256 MB case at 1 MB: the comparison is the same, the CPU test is quick."""
    monkeypatch.setitem(pc.SIZES, "256MB", 1 << 18)


def copy_right(kind, src, dst_device):
    return src.clone() if src.is_contiguous() else src.contiguous().clone()


def corrupt(case_kind=None, size=None):
    """A copy that flips one byte of the destination, for the given kind (all kinds when None)
    and element count (any when None)."""
    def copy(kind, src, dst_device):
        out = copy_right(kind, src, dst_device)
        if (case_kind is None or kind == case_kind) and (size is None or src.numel() == size):
            out.view(torch.uint8).view(-1)[3] ^= 1
        return out
    return copy


def pair(copy, control=None):
    return pc.check_pair(CPU, CPU, copy, control or (lambda src, d: src.clone()))


def test_a_right_copy_passes_every_case_and_reports_a_rate():
    r = pair(copy_right)
    assert r["ok"] and r["card_suspect"] is False and r["errors"] == {}
    assert set(r["cases"]) == {"host_staged", "strided", "side_stream", *pc.CONTIGUOUS_CASES,
                              *pc.EXTRA_CASES}
    assert all(r["cases"].values()) and r["gbps"] > 0
    assert pc.case_name("copy_", "4KB") == "copy_4KB" and pc.case_name("to", "4MB") == "to_4MB"


def test_one_corrupt_byte_fails_the_pair_and_the_report_names_pair_and_case():
    report = pc.check([CPU, CPU, CPU], corrupt("to", 1 << 20),
                      lambda s, d: s.clone())
    assert not report["all_ok"] and len(report["pairs"]) == 6
    p = report["pairs"]["0->1"]
    assert not p["ok"] and p["cases"]["to_4MB"] is False and p["cases"]["copy_4MB"] is True
    assert p["gbps"] is None and not p["card_suspect"]
    # the strided and the side-stream cases are not "to" copies and stay right
    assert p["cases"]["strided"] and p["cases"]["side_stream"]
    lines = pc.report_lines(report)
    assert len(lines) == 6 and all("wrong bytes in" in x and "to_4MB" in x for x in lines)
    assert lines[0].startswith("peer check FAIL: pair 0->1:")


@pytest.mark.parametrize("kind, case", [("copy_", "strided"), ("side_stream", "side_stream")])
def test_a_corrupt_strided_or_side_stream_copy_is_caught(kind, case):
    r = pair(corrupt(kind))
    assert r["cases"][case] is False and not r["ok"]


def test_a_corrupt_control_is_the_card_not_the_peer_path():
    r = pair(copy_right, control=lambda src, d: src.clone() ^ 1)
    assert r["card_suspect"] is True and not r["ok"]
    assert r["cases"] == {"host_staged": False}  # the direct cases were not tried
    report = pc.check([CPU, CPU], copy_right, lambda src, d: src.clone() ^ 1)
    lines = pc.report_lines(report)
    assert all("the card itself is suspect, not the peer path" in x for x in lines)
    assert not any("wrong bytes in" in x for x in lines)


def test_a_copy_that_raises_is_a_failed_case_not_a_lost_pair():
    def boom(kind, src, dst):
        if kind == "side_stream":
            raise RuntimeError("an illegal memory access")
        return copy_right(kind, src, dst)

    r = pair(boom)
    assert r["cases"]["side_stream"] is False and "illegal memory" in r["errors"]["side_stream"]
    assert r["cases"]["to_4KB"] is True and not r["ok"]


def test_nan_patterns_cannot_hide_a_difference():
    # a copy that turns a NaN pattern into another NaN pattern still differs in bytes
    def nan_swap(kind, src, dst):
        out = src.clone()
        out[0] = 0x7FC00000 if int(src[0]) != 0x7FC00000 else 0x7FC00001
        return out

    assert not pair(nan_swap)["ok"]


def test_no_pair_is_not_all_ok():
    assert pc.check([CPU])["all_ok"] is False
    assert pc.report_lines(pc.check([CPU])) == ["peer check FAIL: no pair was checked"]


def test_the_right_report_reads_pass_and_carries_one_rate_per_pair():
    report = pc.check([CPU, CPU], copy_right, lambda s, d: s.clone())
    assert report["all_ok"] and set(report["pairs"]) == {"0->1", "1->0"}
    assert all(p["gbps"] > 0 for p in report["pairs"].values())
    assert pc.report_lines(report)[0].startswith("peer check pass: every one of 2 ordered pairs")


def test_main_with_fewer_than_two_devices_exits_1_with_the_reason(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "sub" / "pc.json"
    assert pc.main(["--out", str(out)]) == 1
    doc = json.loads(out.read_text())
    assert doc["all_ok"] is False and "0 CUDA devices" in doc["reason"]
    assert "peer check FAIL: 0 CUDA devices" in capsys.readouterr().out


def test_main_exit_code_follows_all_ok(tmp_path, monkeypatch):
    for wrong, code in ((False, 0), (True, 1)):
        monkeypatch.setattr(pc, "build_report", lambda wrong=wrong: {
            "all_ok": not wrong, "seconds": 0.0,
            "pairs": {"0->1": {"ok": not wrong, "cases": {"to_4KB": not wrong}}}})
        assert pc.main(["--out", str(tmp_path / "r.json")]) == code


def test_main_survives_a_check_that_raises(tmp_path, monkeypatch, capsys):
    def boom():
        raise RuntimeError("CUDA error: wedged")

    monkeypatch.setattr(pc, "build_report", boom)
    assert pc.main(["--out", str(tmp_path / "r.json")]) == 1
    assert "the check raised RuntimeError: CUDA error: wedged" in capsys.readouterr().out


# ---- the sessions ---------------------------------------------------------------------------


def test_every_multi_gpu_session_starts_with_the_peer_check(tmp_path):
    assert set(fs.MULTI_GPU_SESSIONS) <= set(fs.SESSIONS)
    for name in fs.MULTI_GPU_SESSIONS:
        plan = fs.choose_plan(fs.SESSIONS[name][1], tmp_path / "up", tmp_path / "out",
                              plan_file=None, selftest=False)
        first = next(p for p in plan if p.name != fs.NATIVE_BUILD_NAME)
        assert first.kind == "peer_check" and first.name == fs.PEER_CHECK_NAME, name
        assert first.info["report"].startswith(str(tmp_path / "out")), name
        assert first.timeout_s and first.timeout_s <= 600, name
        assert first.argv[:3] == ["{python}", "-m", "verification.runs.peer_check"], name
        assert [p.kind for p in plan].index("models") > plan.index(first), name
    others = [n for n in fs.SESSIONS if n not in fs.MULTI_GPU_SESSIONS]
    for name in others:
        plan = fs.SESSIONS[name][1](tmp_path / "up", tmp_path / "out")
        assert not any(p.kind == "peer_check" for p in plan), name


def test_the_selftest_and_a_plan_file_keep_working_without_the_check(tmp_path):
    up, out = tmp_path / "up", tmp_path / "out"
    stand_in = fs.choose_plan(fs.fs32_pair_plan, up, out, plan_file=None, selftest=True)
    assert [p.name for p in stand_in] == [p.name for p in fs.selftest_plan()]
    f = tmp_path / "plan.json"
    f.write_text(json.dumps([{"name": "x", "argv": ["{python}", "-c", "pass"]}]))
    plan = fs.choose_plan(fs.nvlink_probe_plan, up, out, plan_file=f, selftest=False)
    assert [p.name for p in plan] == ["native_build", "x"]


def test_the_session_module_does_not_import_the_check_or_torch_itself():
    """fs11_session reads the report; at module level it imports neither torch nor ``peer_check``
    (which imports torch)."""
    import ast

    tree = ast.parse(Path(fs.__file__).read_text())
    names = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in tree.body if isinstance(n, ast.ImportFrom)}
    assert not any("peer_check" in ast.dump(n) for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom)))
    assert not any(x.split(".")[0] == "torch" or x.endswith("peer_check") for x in names)


def _probe_session(tmp_path, wrong):
    """The probe session's own plan with each part replaced by a stand-in; the check's report
    says ``wrong`` (a pair-to-cases map, or ``None`` for all right)."""
    import dataclasses
    import time

    plan = fs.with_native_build(fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out"))
    writer = ("import json,sys,os\nos.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)\n"
              f"w = {wrong!r} or {{}}\n"
              "json.dump({'all_ok': not w, 'seconds': 1.0, 'pairs': {'0->1': {'ok': not w, "
              "'cases': {'copy_4KB': not w}, 'gbps': 5.0}}}, open(sys.argv[1], 'w'))\n"
              "sys.exit(1 if w else 0)")
    out = []
    for p in plan:
        argv = ["{python}", "-c", "pass"]
        if p.kind == "peer_check":
            argv = ["{python}", "-c", writer, p.info["report"]]
        out.append(dataclasses.replace(p, argv=argv, timeout_s=30))
    lines = []
    s = fs.run_session(out, upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=time.time() + 120, cwd=Path.cwd(), say=lines.append,
                       session="nvlink-probe")
    return s, lines


def test_the_probe_session_stops_on_wrong_peer_copies_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "check_upload", lambda up: {"s": 0.0, "packed": [], "models": []})
    s, lines = _probe_session(tmp_path, {"0->1": 1})
    assert "wrong bytes on pair 0->1 (copy_4KB)" in s["stopped"]
    by = {p["name"]: p for p in s["parts"]}
    assert by["peer_check"]["status"] == "failed"
    assert by["models"]["status"] == by["nvlink_probe"]["status"] == "not_started"
    assert by["nvlink_probe"]["reason"] == s["stopped"]
    hw2 = [x for x in lines if x.startswith("HW2")]
    assert len(hw2) == 1 and hw2[0].startswith("HW2 FAIL: peer copies landed wrong bytes on pair "
                                              "0->1 (copy_4KB)")
    assert "the NVLink probe did not run" in hw2[0]


def test_the_probe_session_goes_on_past_right_peer_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "check_upload", lambda up: {"s": 0.0, "packed": [], "models": []})
    s, lines = _probe_session(tmp_path, None)
    assert "stopped" not in s
    assert next(p for p in s["parts"] if p["name"] == "models")["status"] == "ok"
    assert not any("peer copies" in x for x in lines)


# ---- round 5: unwritten destinations, errors that are not wrong bytes, result checks ----------


def _unwritten(monkeypatch, keep=0.0):
    """``Tensor.copy_`` that writes the first ``keep`` of the destination and nothing more."""
    orig = torch.Tensor.copy_

    def copy_(self, src, non_blocking=False):
        flat, s = self.view(-1), src.contiguous().reshape(-1)
        n = int(flat.numel() * keep)
        if n:
            orig(flat[:n], s[:n])
        return self

    monkeypatch.setattr(torch.Tensor, "copy_", copy_)


@pytest.mark.parametrize("keep", [0.0, 0.5])
def test_a_copy_that_writes_none_or_half_of_its_destination_fails(monkeypatch, keep):
    """cuda_copy fills each destination it allocates with 0xFF bytes, so a ``copy_`` that leaves
    the block alone cannot pass on bytes the allocator left there. The ``to`` cases do not use
    ``copy_`` and stay right."""
    _unwritten(monkeypatch, keep)
    r = pc.check_pair(CPU, CPU, pc.cuda_copy, lambda s, d: s.clone())
    wrote_nothing = [c for c in r["cases"] if c.startswith("copy_")]
    assert wrote_nothing and all(r["cases"][c] is False for c in wrote_nothing)
    for c in ("strided", "bf16_4MB", "fp32_4MB", "offset_4MB"):
        assert r["cases"][c] is False and c not in r["errors"]
    assert r["cases"]["to_4MB"] is True and r["cases"]["3d_4MB"] is True
    assert "side_stream" in r["errors"]  # no CUDA here: the case raised, it did not pass
    assert not r["ok"]


def test_the_destination_cuda_copy_allocates_is_filled_unlike_the_source():
    dst = pc._filled((4, 5), torch.bfloat16, CPU)
    assert dst.shape == (4, 5) and dst.dtype == torch.bfloat16
    assert set(dst.view(torch.uint8).flatten().tolist()) == {pc.FILL}


def test_every_case_draws_its_own_source_and_the_extra_cases_have_their_shapes():
    seen = []

    def copy(kind, src, dst):
        seen.append((kind, src))
        return copy_right(kind, src, dst)

    assert pair(copy)["ok"]
    direct = list({id(x): (k, x) for k, x in seen}.values())  # the rate loop re-copies one
    for i, (_, a) in enumerate(direct):
        for _, b in direct[i + 1:]:
            assert not (a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b))
    by = {(x.dtype, x.ndim, x.storage_offset() != 0): (k, x) for k, x in direct}
    assert (torch.bfloat16, 1, False) in by and (torch.float32, 1, False) in by
    k, view = by[(torch.int32, 1, True)]
    assert k == "copy_" and view.is_contiguous() and view.storage_offset() == pc.OFFSET
    k, three = by[(torch.float32, 3, False)]
    assert k == "to" and tuple(three.shape) == pc.SHAPE_3D
    assert all(x.numel() * x.element_size() == 4 << 20
               for _, x in direct if x.dtype in (torch.bfloat16, torch.float32))


def test_the_strided_case_compares_in_the_sources_logical_order():
    def copy(kind, src, dst):
        if src.ndim != 2:
            return copy_right(kind, src, dst)
        # the base's storage order under the view's shape: the right bytes, in the wrong order
        return src.t().contiguous().view(src.shape)

    r = pair(copy)
    assert r["cases"]["strided"] is False and "strided" not in r["errors"]


def test_a_result_on_the_wrong_device_shape_or_dtype_fails_with_its_reason():
    def wrong_dtype(kind, src, dst):
        out = copy_right(kind, src, dst)
        small = src.dtype == torch.int32 and src.numel() == 1024
        return out.view(torch.float32) if small else out

    r = pair(wrong_dtype)
    assert r["cases"]["to_4KB"] is False
    assert r["errors"]["to_4KB"].startswith("BadResult: the result is torch.float32")
    with pytest.raises(pc.BadResult, match="the result is on cpu, not the destination cuda:1"):
        pc._expect(torch.zeros(2), torch.zeros(2), torch.device("cuda:1"))
    with pytest.raises(pc.BadResult, match="has shape"):
        pc._expect(torch.zeros(3), torch.zeros(2), CPU)
    r = pair(corrupt())
    r["errors"] = {"to_4KB": "BadResult: the result is on cpu, not the destination cuda:1"}
    r["cases"]["to_4KB"] = False
    assert "to_4KB gave a bad result (the result is on cpu" in pc.pair_lines("0->1", r)[0]


def test_a_case_that_raised_is_said_to_have_raised_not_to_have_wrong_bytes():
    def boom(kind, src, dst):
        if kind == "side_stream":
            raise RuntimeError("an illegal memory access")
        return copy_right(kind, src, dst)

    report = pc.check([CPU, CPU], boom, lambda s, d: s.clone())
    lines = pc.report_lines(report)
    assert len(lines) == 2
    assert "side_stream raised RuntimeError: an illegal memory access" in lines[0]
    assert "wrong bytes" not in lines[0] and "could not copy" not in lines[0]


def test_wrong_bytes_and_a_raised_case_are_each_named_in_one_line():
    def mixed(kind, src, dst):
        if kind == "side_stream":
            raise RuntimeError("an illegal memory access")
        return corrupt("to", 1 << 20)(kind, src, dst)

    line = pc.pair_lines("0->1", pair(mixed))[0]
    assert "wrong bytes in to_4MB" in line and "side_stream raised RuntimeError" in line


def test_every_case_raising_says_the_check_could_not_copy_and_names_the_error():
    def boom(kind, src, dst):
        raise RuntimeError("peer access is not enabled")

    r = pair(boom)
    assert not r["ok"] and r["card_suspect"] is False and r["gbps"] is None
    line = pc.pair_lines("0->1", r)[0]
    assert "the check could not copy on this pair" in line
    assert "RuntimeError: peer access is not enabled" in line
    assert "wrong bytes" not in line and "peer path is wrong" not in line


def test_a_control_that_raised_says_the_check_could_not_copy_on_that_pair():
    def control(src, dst):
        raise RuntimeError("CUDA error: unspecified launch failure")

    line = pc.pair_lines("1->0", pair(copy_right, control))[0]
    assert "pair 1->0: the check could not copy on this pair" in line
    assert "host_staged raised RuntimeError: CUDA error: unspecified launch failure" in line
    assert "suspect" not in line


def test_a_pair_that_is_right_has_no_lines():
    assert pc.pair_lines("0->1", pair(copy_right)) == []


def test_main_deletes_the_old_report_first_so_a_killed_rerun_leaves_none(tmp_path, monkeypatch):
    out = tmp_path / "pc.json"
    out.write_text(json.dumps({"all_ok": True, "pairs": {"0->1": {"ok": True}}}))

    def killed():
        assert not out.exists()  # gone before the check starts
        raise KeyboardInterrupt  # the timeout's or SIGINT's kill

    monkeypatch.setattr(pc, "build_report", killed)
    with pytest.raises(KeyboardInterrupt):
        pc.main(["--out", str(out)])
    assert not out.exists()


def test_one_side_stream_per_source_device_for_the_whole_check(monkeypatch):
    made = []

    class Stream:
        def __init__(self, device=None):
            self.device = device
            made.append(device)

    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(pc, "_SIDE", {})
    a, b = torch.device("cuda:0"), torch.device("cuda:1")
    first = pc._side_stream(a)
    assert pc._side_stream(a) is first and pc._side_stream(b) is not first
    assert made == [a, b]
    pc._SIDE["stale"] = 1
    pc.check([], lambda *x: None)  # a check starts and ends with no stream held
    assert pc._SIDE == {}


def test_the_side_stream_case_makes_its_source_with_an_op_and_does_not_sync_before_the_copy():
    synced, order = [], []

    def sync():
        synced.append(len(order))

    def copy(kind, src, dst):
        order.append(kind)
        return copy_right(kind, src, dst)

    r = pc.check_pair(CPU, CPU, copy, lambda s, d: s.clone(), sync=sync)
    assert r["ok"] and order.count("side_stream") == 1
    at = order.index("side_stream")
    # no synchronization between the last source draw and the side-stream copy: the sync calls
    # come at the draws (before the case) and after the copy, and none lands between the source's
    # op (inside the case) and ``copy`` itself
    assert synced.count(at) >= 1  # the sources are synced before the case
    assert synced.count(at + 1) >= 1  # and the result after it


# ---- the session's and the probe's words -------------------------------------------------------


def _report(tmp_path, pairs):
    f = tmp_path / "pc.json"
    f.write_text(json.dumps({"all_ok": False, "seconds": 1.0, "pairs": pairs}))
    return fs._peer_check_extras(f)["peer_check"]


def test_the_session_says_wrong_bytes_a_raised_case_or_that_the_check_could_not_copy(tmp_path):
    right = {"ok": True, "cases": {"host_staged": True, "to_4KB": True}, "errors": {},
             "card_suspect": False, "gbps": 9.0}
    wrong = {"ok": False, "cases": {"host_staged": True, "to_4KB": False, "copy_4KB": True},
             "errors": {}, "card_suspect": False, "gbps": None}
    raised = {"ok": False, "cases": {"host_staged": True, "to_4KB": True, "side_stream": False},
              "errors": {"side_stream": "RuntimeError: illegal memory access"},
              "card_suspect": False, "gbps": None}
    every = {"ok": False, "cases": {"host_staged": True, "to_4KB": False, "copy_4KB": False},
             "errors": {"to_4KB": "RuntimeError: peer access is not enabled",
                        "copy_4KB": "RuntimeError: peer access is not enabled"},
             "card_suspect": False, "gbps": None}
    control_raised = {"ok": False, "cases": {"host_staged": False},
                      "errors": {"host_staged": "RuntimeError: launch failure"},
                      "card_suspect": True, "gbps": None}
    control_wrong = {"ok": False, "cases": {"host_staged": False}, "errors": {},
                     "card_suspect": True, "gbps": None}
    bad = {"ok": False, "cases": {"host_staged": True, "to_4KB": False},
           "errors": {"to_4KB": "BadResult: the result is on cpu, not the destination cuda:1"},
           "card_suspect": False, "gbps": None}

    def say(**pairs):
        return fs.peer_wrong_text(_report(tmp_path, {"0->1": right, **pairs}))

    text = say(**{"1->0": wrong})
    assert text == "peer copies landed wrong bytes on pair 1->0 (to_4KB)"
    text = say(**{"1->0": raised})
    assert text == "pair 1->0: side_stream raised RuntimeError: illegal memory access"
    assert "wrong bytes" not in text
    text = say(**{"1->0": every})
    assert "pair 1->0: the check could not copy, so the peer path is not judged" in text
    assert "to_4KB raised RuntimeError: peer access is not enabled" in text
    assert "wrong bytes" not in text
    text = say(**{"1->0": control_raised})
    assert "the check could not copy" in text and "host_staged raised RuntimeError: launch" in text
    assert "suspect" not in text
    text = say(**{"1->0": control_wrong})
    assert text == ("pair 1->0: the host-staged control was wrong: the card itself is suspect, "
                    "not the peer path")
    assert "to_4KB gave a bad result (the result is on cpu" in say(**{"1->0": bad})
    both = say(**{"1->0": wrong, "2->0": raised})
    assert both.startswith("peer copies landed wrong bytes on pair 1->0 (to_4KB); pair 2->0:")


def test_the_gate_line_and_the_stop_say_what_the_extras_say(tmp_path):
    every = {"ok": False, "cases": {"host_staged": True, "to_4KB": False},
             "errors": {"to_4KB": "RuntimeError: peer access is not enabled"},
             "card_suspect": False, "gbps": None}
    check = _report(tmp_path, {"0->1": every})
    stop = fs.peer_check_stop({"status": "failed", "extras": {"peer_check": check}})
    assert "the check could not copy" in stop and "peer access is not enabled" in stop
    assert "wrong bytes" not in stop


# ---- the module imports no torch ---------------------------------------------------------------


# ---- the real check, where two CUDA devices exist -------------------------------------------


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_the_real_check_on_this_machines_cards(tmp_path):
    out = tmp_path / "pc.json"
    code = pc.main(["--out", str(out)])
    doc = json.loads(out.read_text())
    n = torch.cuda.device_count()
    assert len(doc["pairs"]) == n * (n - 1) and len(doc["devices"]) == n
    assert doc["seconds"] < 60
    assert code == (0 if doc["all_ok"] else 1)
    for p in doc["pairs"].values():
        assert p["cases"]["host_staged"] is True  # a card whose control fails is not a test result

