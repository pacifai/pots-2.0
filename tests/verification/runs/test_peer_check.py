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
    assert set(r["cases"]) == {"host_staged", "strided", "side_stream", *pc.CONTIGUOUS_CASES}
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
