"""FS31 tester's cases: false-pass paths in HW2 and HW3, the plan's secrets and limits, the job
switch. A test marked ``xfail(strict=True)`` is a defect found; the reason says what."""

import ast
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.verification.runs import test_fs11_session as t11
from tests.verification.runs.test_nvlink_probe import (P2P_4, TOPO_4, TOPO_8, pair_rates,
                                                       with_label)
from verification.runs import fs11_session as fs
from verification.runs import nvlink_probe as nv

REPO = Path(__file__).resolve().parents[3]


def parse(text):
    return nv.parse_gpu_matrix(text)


def verdict(topo=TOPO_4, p2p=P2P_4, rates=None, n=4):
    t = parse(topo)
    right = {f"{a[3:]}->{b[3:]}": True for a in t for b in t if a != b} if t else None  # bytes right
    return nv.hw2_verdict(t, parse(p2p), pair_rates(n, 380.0) if rates is None else rates,
                          peer_bytes=right)


# ---- HW2: topology ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["PIX", "PXB", "SYS", "NODE", "PHB", "X", "", "NV", "NV18x",
                                   "nv18", "NVL18", "N/A"])
def test_any_non_nv_label_on_one_pair_fails(label):
    text = with_label(TOPO_4, 2, 3, label) if label else with_label(TOPO_4, 2, 3, "-")
    v = verdict(topo=text)
    assert not v["pass"] and v["reasons"]


@pytest.mark.parametrize("label", ["NV1", "NV4", "NV12", "NV17", "NV19", "NV180"])
def test_a_different_nv_count_fails(label):
    v = verdict(topo=with_label(TOPO_4, 0, 1, label))
    assert not v["pass"] and not v["nv18_everywhere"]


def test_an_asymmetric_matrix_fails():
    # GPU1's view of GPU2 is SYS, GPU2's view of GPU1 stays NV18.
    v = verdict(topo=with_label(TOPO_4, 1, 2, "SYS"))
    assert not v["pass"]


def test_p2p_not_ok_in_the_lower_triangle_fails():
    assert not verdict(p2p=with_label(P2P_4, 3, 0, "NS"))["pass"]


@pytest.mark.parametrize("cell", ["NS", "CNS", "GNS", "TNS", "U", "ok", "OK1", "-"])
def test_p2p_other_statuses_fail(cell):
    assert not verdict(p2p=with_label(P2P_4, 1, 3, cell))["pass"]


def test_extra_columns_do_not_hide_a_bad_pair():
    bad = with_label(TOPO_8, 3, 6, "SYS")
    m = parse(bad)
    v = nv.hw2_verdict(m, parse(P2P_4.replace("GPU3", "GPU3")), pair_rates(4, 380.0))
    assert m["GPU3"]["GPU6"] == "SYS" and not v["pass"]


def test_extra_columns_on_the_4_gpu_matrix_are_ignored():
    m = parse(TOPO_4)
    assert list(m) == ["GPU0", "GPU1", "GPU2", "GPU3"] and len(m["GPU0"]) == 4


@pytest.mark.parametrize("keep", range(1, 12))
def test_truncated_topo_text_never_passes(keep):
    lines = TOPO_4.splitlines(keepends=True)
    text = "".join(lines)[: sum(len(x) for x in lines[:keep // 3]) + keep * 2]
    assert not verdict(topo=text)["pass"]


def test_truncated_in_the_last_row_never_passes():
    cut = TOPO_4.index("GPU3    NV18")
    end = TOPO_4.index("X", cut)  # the row is whole once its diagonal cell is there
    for k in range(0, end - cut):
        assert not verdict(topo=TOPO_4[: cut + k])["pass"]


def test_a_row_missing_never_passes():
    rows = [ln for ln in TOPO_4.splitlines() if not ln.startswith("GPU2")]
    assert not verdict(topo="\n".join(rows))["pass"]


def test_garbled_text_never_passes():
    for junk in ("\x00\x01 GPU0 GPU1\nGPU0 X NV18\nGPU1 NV18 X", "GPU0 GPU1\nfoo bar baz",
                 "GPU0\tGPU1\nGPU0\tX\nGPU1\tNV18",
                 TOPO_4.replace("GPU", "gpu"), "NV18 NV18 NV18"):
        assert not verdict(topo=junk)["pass"]


def test_ansi_coloured_topo_still_parses_and_passes():
    colored = TOPO_4.replace("NV18", "\x1b[32mNV18\x1b[0m")
    assert verdict(topo=colored)["pass"]


def test_two_gpu_matrix_fails_hw2_defect():
    two = """\
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV18    0-47    0
GPU1    NV18     X      0-47    0
"""
    p2p = " \tGPU0\tGPU1\n GPU0\tX\tOK\n GPU1\tOK\tX\n"
    v = nv.hw2_verdict(parse(two), parse(p2p), pair_rates(2, 380.0))
    assert not v["pass"]


def test_eight_gpu_matrix_fails_hw2_defect():
    p2p = " \t" + "\t".join(f"GPU{i}" for i in range(8)) + "\n" + "\n".join(
        f" GPU{i}\t" + "\t".join("X" if i == j else "OK" for j in range(8)) for i in range(8))
    v = nv.hw2_verdict(parse(TOPO_8), parse(p2p), pair_rates(8, 380.0))
    assert not v["pass"]


def test_topo_with_a_non_zero_exit_is_recorded_as_failed_command():
    def run(argv, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    rec = nv.run_command(["nvidia-smi", "topo", "-m"], 5, run)
    assert not rec["ok"]


def _fake_run(topo_code=0, p2p_code=0, topo_text=TOPO_4, p2p_text=P2P_4):
    def run(argv, **kw):
        if argv[:3] == ["nvidia-smi", "topo", "-m"]:
            return SimpleNamespace(returncode=topo_code, stdout=topo_text, stderr="")
        if argv[:4] == ["nvidia-smi", "topo", "-p2p", "r"]:
            return SimpleNamespace(returncode=p2p_code, stdout=p2p_text, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return run


def _four_fast(monkeypatch):
    monkeypatch.setattr(nv, "peer_bytes_by_pair", lambda: {
        f"{a}->{b}": True for a in range(4) for b in range(4) if a != b})
    monkeypatch.setattr(nv, "torch_view", lambda: {"device_count": 4})
    monkeypatch.setattr(nv, "peer_copy_bandwidth", lambda: {"pairs": pair_rates(4, 380.0)})
    monkeypatch.setattr(nv, "fanout_bandwidth", lambda: {})
    monkeypatch.setattr(nv, "pinned_bandwidth", lambda: {})
    monkeypatch.setattr(nv.shutil, "which", lambda n: None)


def test_report_with_clean_commands_passes(monkeypatch):
    _four_fast(monkeypatch)
    assert nv.build_report(_fake_run())["hw2"]["pass"]


def test_nonzero_exit_of_topo_with_good_text_does_not_pass(monkeypatch):
    _four_fast(monkeypatch)
    assert not nv.build_report(_fake_run(topo_code=2))["hw2"]["pass"]


def test_nonzero_exit_of_p2p_with_good_text_does_not_pass(monkeypatch):
    _four_fast(monkeypatch)
    assert not nv.build_report(_fake_run(p2p_code=2))["hw2"]["pass"]


def test_timeout_of_topo_does_not_pass(monkeypatch):
    _four_fast(monkeypatch)

    def run(argv, **kw):
        if argv[:3] == ["nvidia-smi", "topo", "-m"]:
            raise subprocess.TimeoutExpired(argv, 60)
        return _fake_run()(argv, **kw)

    assert not nv.build_report(run)["hw2"]["pass"]


# ---- HW2: bandwidth --------------------------------------------------------------------------


def test_rate_at_the_pass_line_passes_and_just_below_is_a_review():
    t = nv.PEER_COPY_PASS_GBPS
    assert verdict(rates=pair_rates(4, t))["verdict"] == "pass"
    v = verdict(rates=pair_rates(4, t, slow=("1->3", t - 0.01)))
    assert v["verdict"] == "review" and not v["pass"] and any("1->3" in r for r in v["reasons"])


def test_rate_at_the_fail_line_is_a_review_and_just_below_fails():
    t = nv.PEER_COPY_FAIL_GBPS
    assert verdict(rates=pair_rates(4, 380.0, slow=("1->3", t)))["verdict"] == "review"
    v = verdict(rates=pair_rates(4, 380.0, slow=("1->3", t - 0.01)))
    assert v["verdict"] == "fail" and any("1->3" in r for r in v["reasons"])
    assert verdict(rates=pair_rates(4, 380.0, slow=("0->1", 25.0)))["verdict"] == "fail"
    assert not verdict(rates=pair_rates(4, 380.0, slow=("0->1", 0.0)))["pass"]
    assert not verdict(rates=pair_rates(4, 380.0, slow=("0->1", -5.0)))["pass"]


def test_a_pair_recorded_as_none_fails():
    assert not verdict(rates=pair_rates(4, 380.0, slow=("2->3", None)))["pass"]


def test_no_rates_fail():
    assert not verdict(rates={})["pass"]
    assert not nv.hw2_verdict(parse(TOPO_4), parse(P2P_4), None)["pass"]


def test_missing_pairs_do_not_pass():
    assert not verdict(rates={"0->1": 380.0})["pass"]


def test_nan_rate_does_not_pass():
    assert not verdict(rates=pair_rates(4, 380.0, slow=("1->2", math.nan)))["pass"]


def test_inf_rate_fails():
    # a zero-length timing would give inf; only finite rates count
    assert verdict(rates=pair_rates(4, math.inf))["verdict"] == "fail"


def test_without_cuda_nothing_is_measured_and_hw2_fails(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(nv.shutil, "which", lambda n: None)
    r = nv.build_report(_fake_run())
    assert not r["hw2"]["pass"]
    assert any("no peer-copy bandwidth" in x for x in r["hw2"]["reasons"])


@pytest.mark.parametrize("failure", ["missing", "exit", "timeout", "garbage"])
def test_nvbandwidth_trouble_is_recorded_and_never_raises(failure):
    def run(argv, **kw):
        if failure == "exit":
            return SimpleNamespace(returncode=3, stdout="", stderr="cuda error")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 300)
        if failure == "missing":
            raise FileNotFoundError("nvbandwidth")
        return SimpleNamespace(returncode=0, stdout="\x00 not a table", stderr="")

    out = nv.nvbandwidth(run, which=lambda n: "/x/nvbandwidth")
    assert set(out) == set(nv.NVBANDWIDTH_TESTS)
    assert all(isinstance(r["ok"], bool) for r in out.values())
    assert (failure == "garbage") == all(r["ok"] for r in out.values())


def test_nvbandwidth_absent_is_unavailable():
    assert nv.nvbandwidth(which=lambda n: None) == "unavailable"


def test_nvbandwidth_failure_does_not_change_the_verdict(monkeypatch):
    """Recorded fact: the verdict rests on torch's copies; nvbandwidth never gates it."""
    _four_fast(monkeypatch)
    bad = nv.nvbandwidth(lambda argv, **kw: SimpleNamespace(returncode=1, stdout="", stderr="x"),
                         which=lambda n: "/x/nvbandwidth")
    monkeypatch.setattr(nv, "nvbandwidth", lambda run=None: bad)
    r = nv.build_report(_fake_run())
    assert not r["nvbandwidth"]["device_to_device_memcpy_read_ce"]["ok"]
    assert r["hw2"]["pass"]


# ---- HW3: the comparison ---------------------------------------------------------------------

LAYOUT = {"n_s": 2, "n_w": 3, "M": 4, "n_leaves": 12, "weight_names": ["w_a", "w_b", "w_c"],
          "product_names": ["P1", "P2", "P3", "P4"]}


def card_part(tmp_path, model, card, *, root="r", status="ok", leaves=12, changed=(),
              uuid=None, write=True, layout=None):
    d = tmp_path / model.replace("/", "_") / f"c{card}"
    d.mkdir(parents=True, exist_ok=True)
    rep = d / "card.json"
    extras = {"card": None}
    if write:
        hs = [bytes([i]) * 32 for i in range(leaves)]
        for i in changed:
            hs[i] = b"\xff" * 32
        rep.write_text(json.dumps({"layout": layout or LAYOUT, "n_leaves": leaves,
                                   "card": {"uuid": uuid or f"GPU-{card}"}}))
        rep.with_suffix(".digests").write_bytes(b"".join(hs))
        extras = {"root": root, "uuid": uuid or f"GPU-{card}"}
    return {"name": f"c{card}", "kind": "card", "status": status, "extras": extras,
            "info": {"model": model, "card": card, "report": str(rep)}}


def test_hw3_equal_roots_pass(tmp_path):
    assert fs.hw3_cards([card_part(tmp_path, "o/m", k) for k in range(4)])["pass"]


def test_hw3_one_card_differs_in_one_leaf_names_card_and_leaf(tmp_path):
    parts = [card_part(tmp_path, "o/m", k) for k in range(3)]
    parts.append(card_part(tmp_path, "o/m", 3, root="x", changed=(10,)))
    hw3 = fs.hw3_cards(parts)
    d = hw3["models"]["o/m"]["first_difference"]
    assert not hw3["pass"]
    assert (d["card"], d["leaf"], d["section"], d["name"]) == (3, 10, "w_next", "w_b")
    assert d["n_differing_leaves"] == 1
    assert "card 3 differs from card 0 first at leaf 10 (w_next w_b)" in fs._hw_lines([], hw3)[0]


def test_hw3_differing_card_in_the_middle_is_named(tmp_path):
    parts = [card_part(tmp_path, "o/m", 0), card_part(tmp_path, "o/m", 1),
             card_part(tmp_path, "o/m", 2, root="x", changed=(0,)),
             card_part(tmp_path, "o/m", 3)]
    d = fs.hw3_cards(parts)["models"]["o/m"]["first_difference"]
    assert (d["card"], d["leaf"], d["section"]) == (2, 0, "record")


def test_hw3_reference_card_differs_is_still_located(tmp_path):
    parts = [card_part(tmp_path, "o/m", 0, root="x", changed=(5,))]
    parts += [card_part(tmp_path, "o/m", k) for k in (1, 2, 3)]
    hw3 = fs.hw3_cards(parts)
    assert not hw3["pass"] and hw3["models"]["o/m"]["first_difference"]["leaf"] == 5


def test_hw3_card_missing_entirely_from_the_plan_fails_on_the_count(tmp_path):
    """Only three card parts exist for a four-card session: fails on the count (round 3)."""
    parts = [card_part(tmp_path, "o/m", k) for k in range(3)]
    hw3 = fs.hw3_cards(parts, expected=4)
    assert not hw3["pass"]
    assert "3 cards compared, FS31_CARDS is 4" in fs._hw_lines([], hw3)[0]


def test_hw3_card_with_a_failed_run_fails(tmp_path):
    parts = [card_part(tmp_path, "o/m", k) for k in range(3)]
    parts.append(card_part(tmp_path, "o/m", 3, status="failed", write=False))
    assert not fs.hw3_cards(parts)["pass"]


@pytest.mark.parametrize("status", ["failed", "killed_deadline", "timeout", "not_started",
                                    "killed_timeout"])
def test_hw3_failed_status_with_a_report_present_fails(tmp_path, status):
    parts = [card_part(tmp_path, "o/m", k) for k in range(3)]
    parts.append(card_part(tmp_path, "o/m", 3, status=status))
    assert not fs.hw3_cards(parts)["pass"]


def test_hw3_two_cards_both_failed_do_not_agree(tmp_path):
    parts = [card_part(tmp_path, "o/m", k, status="failed", write=False) for k in range(2)]
    hw3 = fs.hw3_cards(parts)
    assert not hw3["pass"] and not hw3["models"]["o/m"]["same_bits"]
    parts = [card_part(tmp_path, "o/m", k, status="failed", root="r") for k in range(4)]
    assert not fs.hw3_cards(parts)["pass"]


def test_hw3_root_none_on_every_card_does_not_pass(tmp_path):
    parts = [card_part(tmp_path, "o/m", k) for k in range(2)]
    for p in parts:
        p["extras"] = {"card": None}
    assert not fs.hw3_cards(parts)["pass"]


def test_hw3_leaf_counts_differ(tmp_path):
    big = {**LAYOUT, "M": 6, "n_leaves": 14, "product_names": [f"P{i}" for i in range(1, 7)]}
    parts = [card_part(tmp_path, "o/m", 0), card_part(tmp_path, "o/m", 1, root="x", leaves=14,
                                                      layout=big)]
    hw3 = fs.hw3_cards(parts)
    d = hw3["models"]["o/m"]["first_difference"]
    assert not hw3["pass"] and d["leaf"] is None and "12" not in d["note"] or d["note"]
    assert d["card"] == 1


def test_hw3_one_failed_model_fails_the_session(tmp_path):
    parts = [card_part(tmp_path, "o/a", k) for k in range(2)]
    parts += [card_part(tmp_path, "o/b", k, root="r" if k == 0 else "x") for k in range(2)]
    assert not fs.hw3_cards(parts)["pass"]


def test_hw3_with_unreadable_digests_raises_so_the_session_records_an_error(tmp_path):
    parts = [card_part(tmp_path, "o/m", 0), card_part(tmp_path, "o/m", 1, root="x")]
    Path(parts[1]["info"]["report"]).with_suffix(".digests").unlink()
    with pytest.raises(OSError):
        fs.hw3_cards(parts)


def test_hw3_error_is_stored_and_the_session_goes_on(tmp_path, monkeypatch):
    def boom(parts):
        raise OSError("gone")

    monkeypatch.setattr(fs, "hw3_cards", boom)
    plan = [fs.Part("c", ["{python}", "-c", "print(1)"], kind="card",
                    info={"model": "o/m", "card": 0, "report": str(tmp_path / "x.json")})]
    s = fs.run_session(plan, upload=tmp_path / "up", out=tmp_path / "out",
                       deadline=__import__("time").time() + 60, cwd=REPO, say=lambda m: None)
    assert s["hw3_cards"] == {"hw3_cards_error": "OSError: gone"} and s["finished"]


def test_hw3_four_runs_on_the_same_physical_card_do_not_pass(tmp_path):
    parts = [card_part(tmp_path, "o/m", k, uuid="GPU-same") for k in range(4)]
    assert not fs.hw3_cards(parts)["pass"]


def test_hw3_single_card_is_not_a_pass(tmp_path):
    assert not fs.hw3_cards([card_part(tmp_path, "o/m", 0)])["pass"]


# ---- the session plan ------------------------------------------------------------------------


def test_every_part_has_a_timeout_and_the_nominal_total_fits(tmp_path):
    plan = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")
    assert all(p.timeout_s for p in plan)
    assert all(p.timeout_s <= 3600 for p in plan)


def test_the_models_part_is_the_only_one_that_goes_online(tmp_path):
    plan = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")
    online = [p.name for p in plan if p.env.get("HF_HUB_OFFLINE", "1") is None]
    assert online == ["models"]
    for p in (p for p in plan if p.name != "models"):
        assert p.env.get("HF_HUB_OFFLINE", "1") == "1"
        assert p.env.get("TRANSFORMERS_OFFLINE", "1") == "1"


def test_no_part_env_names_a_secret_token_or_key(tmp_path):
    plan = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")
    for p in plan:
        for k in p.env:
            assert k not in fs.SECRETS
            assert "TOKEN" not in k.upper() and not k.upper().endswith("_KEY")


def test_the_parts_run_through_base_env_with_secrets_stripped(tmp_path, monkeypatch):
    for k in (*fs.SECRETS, "HF_TOKEN"):
        monkeypatch.setenv(k, "secret-value")
    env = fs.base_env(tmp_path / "up")
    assert not set(env) & set(fs.SECRETS)
    assert "secret-value" not in env.values()


@pytest.mark.xfail(strict=True, reason="pre-existing: base_env strips only SECRETS; outside "
                   "FS31's plan (orchestrator, 2026-10-20)")
def test_base_env_strips_other_token_and_key_names(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "v1")
    monkeypatch.setenv("OTHER_API_KEY", "v2")
    env = fs.base_env(tmp_path / "up")
    assert not [k for k in env if k.endswith("_TOKEN") or k.endswith("_KEY")]


def test_a_running_part_never_sees_the_secrets(tmp_path, monkeypatch):
    for k in fs.SECRETS:
        monkeypatch.setenv(k, "secret-value")
    dump = tmp_path / "env.txt"
    code = f"import os; open({str(dump)!r}, 'w').write(repr(sorted(os.environ.items())))"
    plan = [fs.Part("p", ["{python}", "-c", code], timeout_s=30)]
    fs.run_session(plan, upload=tmp_path / "up", out=tmp_path / "out",
                   deadline=__import__("time").time() + 60, cwd=REPO, say=lambda m: None)
    text = dump.read_text()
    assert "secret-value" not in text
    for k in fs.SECRETS:
        assert f"'{k}'" not in text


def test_card_parts_pick_distinct_cards_and_run_dirs_in_model_major_order(tmp_path):
    plan = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")
    cards = [p for p in plan if p.kind == "card"]
    models = [p.info["model"] for p in cards]
    assert models == sorted(models, key=[r for r, _ in fs.RUN_PINS].index)
    for model in dict.fromkeys(models):
        mine = [p for p in cards if p.info["model"] == model]
        assert [p.env["CUDA_VISIBLE_DEVICES"] for p in mine] == ["0", "1", "2", "3"]
    for p in cards:
        d = Path(p.info["run_dir"])
        assert d.name.endswith(f"_card{p.info['card']}") and (d / "data").is_symlink()
        assert d.parent == tmp_path / "out" / "runs"
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        assert len({p.env.get(key) for p in cards}) == 1


def test_the_probe_part_runs_with_no_card_pinned(tmp_path):
    probe = fs.nvlink_probe_plan(tmp_path / "up", tmp_path / "out")[1]
    assert "CUDA_VISIBLE_DEVICES" not in probe.env


def test_models_part_fetch_never_downloads_a_gated_model(tmp_path):
    gated = sorted(fs.GATED)[0]
    up = tmp_path / "up"
    up.mkdir()
    (up / "manifest.json").write_text(json.dumps({
        "files": {}, "missing": [],
        "models": [{"repo": gated, "revision": "r", "gated": True,
                    "files": [{"path": "model.safetensors", "packed": False, "sha256": "0" * 64,
                               "bytes": 1}]}]}))
    calls = []
    rc = fs.fetch_models(up, tmp_path / "out", download=lambda *a: calls.append(a),
                         say=lambda m: None)
    assert rc == 1 and not calls
    doc = json.loads((tmp_path / "out" / "models.json").read_text())
    assert any("gated, never fetched" in p for p in doc["problems"])


def test_fs11_session_imports_only_the_standard_library_at_module_level():
    tree = ast.parse((REPO / "verification/runs/fs11_session.py").read_text())
    std = set(sys.stdlib_module_names)
    bad = []
    for node in tree.body:
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [("." * node.level) + (node.module or "")]
        for m in mods:
            top = m.split(".")[0]
            if top not in std and top != "__future__" and not m.startswith(
                    "verification.commitment"):
                bad.append(m)
    assert not bad


# ---- the job script --------------------------------------------------------------------------


@pytest.mark.parametrize("session, expect", [("fs11", 12600), ("recheck", 7500),
                                             ("profile-check", 19800), ("nvlink-probe", 5400),
                                             ("kernel-speed", 5400)])
def test_default_deadline_per_session(tmp_path, session, expect):
    proc, out = t11._job(tmp_path, python=sys.executable, popen=True,
                         extra={"FS11_SESSION": session, "FS11_DEADLINE_S": "",
                                "FS11_GRACE_S": "", "FS11_PACK_RESERVE_S": ""})
    try:
        seen = []
        for raw in proc.stdout:
            seen.append(raw.decode())
            if "deadline in" in seen[-1]:
                break
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        # the job's timer subshells; their command line holds the upload path
        subprocess.run(["pkill", "-9", "-f", str(tmp_path)], check=False)
    assert f"deadline in {expect} s" in "".join(seen)


def test_job_passes_fs31_cards_to_the_session_and_ends_the_pod(tmp_path):
    dump = tmp_path / "cards.txt"
    code = f"import os; open({str(dump)!r}, 'w').write(os.environ.get('FS31_CARDS', 'unset'))"
    r, out = t11._job(tmp_path, plan=[{"name": "p", "argv": t11._py(code)}],
                      extra={"FS31_CARDS": "2", "FS11_SESSION": "nvlink-probe",
                             "FS11_ALLOW_ANY_GPU": "1", "FS11_SELFTEST": "1"})
    assert dump.read_text() == "2"
    assert Path(f"{out}.TERMINATED").exists()
    assert "DRY_RUN: terminate" in r.stdout


def test_job_terminates_on_a_failing_part_in_the_probe_session(tmp_path):
    r, out = t11._job(tmp_path, plan=[{"name": "bad", "argv": t11._py("import sys; sys.exit(7)")}],
                      extra={"FS11_SESSION": "nvlink-probe", "FS11_ALLOW_ANY_GPU": "1", "FS11_SELFTEST": "1"})
    assert r.returncode == 1 and Path(f"{out}.TERMINATED").exists()


def test_job_terminates_on_a_hanging_part_in_the_probe_session(tmp_path):
    r, out = t11._job(tmp_path, plan=[{"name": "hang", "argv": ["sleep", "120"]}],
                      deadline=10, reserve=3,
                      extra={"FS11_SESSION": "nvlink-probe", "FS11_ALLOW_ANY_GPU": "1", "FS11_SELFTEST": "1"})
    s = json.loads((out / "summary.json").read_text())
    assert s["parts"][-1]["status"] == "killed_deadline"
    assert Path(f"{out}.TERMINATED").exists()


def test_job_unknown_session_still_terminates(tmp_path):
    r, out = t11._job(tmp_path, plan=t11.OK_PLAN, extra={"FS11_SESSION": "nvlink_probe"})
    assert "it must be fs11, recheck, profile-check, nvlink-probe or kernel-speed" in r.stdout + r.stderr
    assert Path(f"{out}.TERMINATED").exists()
