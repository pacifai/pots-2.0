"""FS30 tester: the device-leaves gate fed by what the real code writes. The bring-up lines come
from ``bring_up.device_leaves_line`` over a real step's leaves and a real computation's ``M`` and
``dK`` count, and the tests part's cases from a real ``pytest --junitxml`` run, so a change of
either format shows here and not on the card."""

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from tests.verification.runs.test_device_leaves_session import (  # noqa: F401
    GOOD_TESTS, _tests_part)
from verification.commitment.leaves import commit_leaves
from verification.runs import bring_up as bu
from verification.runs import fs11_session as fs

M0 = fs.RUN_PINS[0][0]
MEM = "test_gpu_memory_does_not_hold_the_transcript"


@pytest.fixture(scope="module")
def real():
    from setup.data import schedule
    from tests.verification.decoder_helpers import make_records, tiny_config
    from verification.computation.instances import DecoderComputation
    from verification.computation.interface import snapshot_weights
    from verification.prover.step import prove_step

    c = DecoderComputation(tiny_config(), n_s=2, n=12, eta=1e-3, compute_dtype=torch.float32,
                           device="cpu")
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    torch.manual_seed(0)
    model = c.build_model()
    w0 = snapshot_weights(c, model)
    out = prove_step(c, model, w0, [D[i] for i in schedule(1, c.n_s, len(D), 1)])
    leaves = out.leaves()
    return c, leaves, commit_leaves(c, leaves).root


def written_line(real, *, used=None, hashed=None, known=None, accepted=True, root="same",
                 capture=True):
    c, leaves, root_ = real
    dk = sum(1 for cl in c.product_classes if cl == "dK")
    ah = {"used": c.M - dk if used is None else used,
          "leaves": c.M if hashed is None else hashed}
    step = SimpleNamespace(
        t=1, rejection=None if accepted else object(),
        root=None if root is None else (root_ if root == "same" else bytes(32)),
        known=c.M - dk if known is None else known,
        capture={"arrival_hash": ah} if capture else {"n_calls": 1})
    b = SimpleNamespace(kept=SimpleNamespace(leaf=lambda i: leaves[i]),
                        result=SimpleNamespace(loop=SimpleNamespace(steps=[step])))
    return bu.device_leaves_line(b, c, T=1)


def bring_part(model, text, tmp_path):
    log = tmp_path / (model.replace("/", "_") + ".log")
    log.write_text("some earlier output\n" + text + "\ntrailing output\n")
    part = fs.Part("bring_up_" + model, [], kind="bring_up", info={"model": model})
    return {"name": part.name, "kind": "bring_up", "status": "ok", "info": part.info,
            "extras": fs.part_extras(part, log)}


def parts_with(real, tmp_path, tests=None, **bad_for_first):
    out = [{"name": "native_build", "status": "ok"}, tests or _tests_part()]
    for k, (repo, _) in enumerate(fs.RUN_PINS):
        out.append(bring_part(repo, written_line(real, **(bad_for_first if k == 0 else {})),
                              tmp_path))
    return out


def test_the_computation_has_dK_leaves_so_the_two_counts_differ(real):
    c, _, _ = real
    assert sum(1 for cl in c.product_classes if cl == "dK") > 0


def test_an_honest_run_written_by_the_real_line_passes(real, tmp_path):
    g = fs.device_leaves_gate(parts_with(real, tmp_path))
    assert g["pass"] is True, g
    c, _, _ = real
    row = g["models"][M0]
    assert row["M"] == c.M and row["hashed"] == c.M and row["arrival"] == c.M - row["dK"]
    assert row["known"] == c.M - row["dK"]
    assert row["prover_root"] == row["cpu_root"] and row["accepted"] is True


@pytest.mark.parametrize("kw,needle", [
    ({"accepted": False}, "the step was not accepted"),
    ({"root": "other"}, "the roots differ"),
    ({"root": None}, "the roots differ"),
    ({"used": "M"}, "arrival digests, expected"),  # the count before the fix round
    ({"used": 0}, "arrival digests, expected"),
    ({"known": "M"}, "the commit took"),  # the commit hashed everything itself
    ({"known": 0}, "the commit took"),  # the commit stopped passing ``known``
    ({"hashed": "dk"}, "products hashed on arrival, expected"),
    ({"capture": False}, "arrival digests, expected"),
])
def test_each_failure_of_a_real_line_fails_the_gate_with_its_reason(real, tmp_path, kw, needle):
    c, _, _ = real
    dk = sum(1 for cl in c.product_classes if cl == "dK")
    kw = {k: ({"M": c.M, "dk": c.M - dk}[v] if v in ("M", "dk") else v) for k, v in kw.items()}
    g = fs.device_leaves_gate(parts_with(real, tmp_path, **kw))
    assert g["pass"] is False and needle in " | ".join(g["reasons"]), g
    assert g["models"][M0]["ok"] is False
    assert all(g["models"][r]["ok"] for r, _ in fs.RUN_PINS[1:])  # only the bad model fails


def test_a_bring_up_that_printed_no_line_fails_even_with_status_ok(real, tmp_path):
    parts = parts_with(real, tmp_path)
    bad = [p for p in parts if (p.get("info") or {}).get("model") == fs.RUN_PINS[2][0]][0]
    bad["extras"] = fs.part_extras(
        fs.Part("b", [], kind="bring_up"), tmp_path / "no_such_log")
    g = fs.device_leaves_gate(parts)
    assert g["pass"] is False and "no device leaves line" in " | ".join(g["reasons"])


def test_the_last_line_of_a_log_is_the_one_read(real, tmp_path):
    first = written_line(real, accepted=False)
    last = written_line(real)
    log = tmp_path / "two.log"
    log.write_text(first + "\n" + last + "\n")
    got = fs.part_extras(fs.Part("b", [], kind="bring_up"), log)["device_leaves"]
    assert got["accepted"] is True


# ---- the tests part, from a real pytest junit report -------------------------------------------


def run_pytest(tmp_path, body_cuda, body_mem):
    d = tmp_path / "t"
    d.mkdir()
    (d / "test_device_leaves_cuda.py").write_text(textwrap.dedent(body_cuda))
    (d / "test_capture_cuda.py").write_text(textwrap.dedent(body_mem))
    (d / "test_gpu_blake3.py").write_text("def test_k():\n    assert True\n")
    junit = tmp_path / "junit.xml"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        f"--junitxml={junit}", "--rootdir", str(d), str(d)],
                       capture_output=True, text=True, cwd=d)
    assert junit.exists(), r.stdout + r.stderr
    part = fs.Part("device_tests", [], kind="device_tests", info={"junit": str(junit)})
    return {"name": "device_tests", "kind": "device_tests", "status": "ok" if r.returncode == 0
            else "failed", "extras": fs.part_extras(part, tmp_path / "log")}


def gate_with(real, tmp_path, tests):
    parts = parts_with(real, tmp_path, tests=tests)
    return fs.device_leaves_gate(parts, expected=["test_device_leaves_cuda::test_a",
                                                  "test_device_leaves_cuda::test_b"])


OK_CUDA = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"
OK_MEM = f"def {MEM}():\n    assert True\n"


def test_a_real_junit_report_with_every_test_passing_passes_the_gate(real, tmp_path):
    tests = run_pytest(tmp_path, OK_CUDA, OK_MEM)
    assert set(tests["extras"]["cases"].values()) == {"passed"}
    assert gate_with(real, tmp_path, tests)["pass"] is True


def test_a_real_junit_skip_of_an_fs30_test_fails_the_gate(real, tmp_path):
    cuda = "import pytest\npytestmark = pytest.mark.skipif(True, reason='no cuda')\n" + OK_CUDA
    tests = run_pytest(tmp_path, cuda, OK_MEM)
    g = gate_with(real, tmp_path, tests)
    assert g["pass"] is False and "skipped" in " | ".join(g["reasons"])


def test_a_real_junit_skip_of_the_memory_test_fails_the_gate(real, tmp_path):
    mem = f"import pytest\n\n\n@pytest.mark.skip(reason='x')\ndef {MEM}():\n    pass\n"
    tests = run_pytest(tmp_path, OK_CUDA, mem)
    g = gate_with(real, tmp_path, tests)
    assert g["pass"] is False and MEM in " | ".join(g["reasons"])


def test_a_real_junit_failure_fails_the_gate(real, tmp_path):
    tests = run_pytest(tmp_path, OK_CUDA + "\n\ndef test_c():\n    assert False\n", OK_MEM)
    g = gate_with(real, tmp_path, tests)
    assert g["pass"] is False and "failures" in " | ".join(g["reasons"])


def test_a_real_junit_without_the_memory_test_fails_the_gate(real, tmp_path):
    tests = run_pytest(tmp_path, OK_CUDA, "def test_other():\n    assert True\n")
    g = gate_with(real, tmp_path, tests)
    assert g["pass"] is False and "GPU-memory test did not run" in " | ".join(g["reasons"])


def test_a_real_xfail_in_an_fs30_file_reads_as_skipped_and_fails_the_gate(real, tmp_path):
    """pytest writes an xfail as ``skipped`` in junit: the gate refuses it, so an expected failure
    must never be left in ``test_device_leaves_cuda.py``."""
    cuda = OK_CUDA + "\n\nimport pytest\n\n\n@pytest.mark.xfail\ndef test_x():\n    assert False\n"
    tests = run_pytest(tmp_path, cuda, OK_MEM)
    assert "skipped" in tests["extras"]["cases"].values()
    assert gate_with(real, tmp_path, tests)["pass"] is False


def test_a_real_junit_missing_one_expected_case_fails_the_gate_naming_it(real, tmp_path):
    tests = run_pytest(tmp_path, "def test_a():\n    assert True\n", OK_MEM)
    g = gate_with(real, tmp_path, tests)
    assert g["pass"] is False
    assert ("test_device_leaves_cuda::test_b: expected but not in the junit report"
            in g["reasons"])


def test_a_real_junit_with_an_unknown_extra_passing_case_still_passes(real, tmp_path):
    tests = run_pytest(tmp_path, OK_CUDA + "\n\ndef test_extra():\n    assert True\n", OK_MEM)
    assert gate_with(real, tmp_path, tests)["pass"] is True


def test_the_cpu_root_is_the_c_librarys_even_when_a_kept_leaf_is_routed_to_a_kernel(
        real, monkeypatch):
    """No CUDA here: the routing's test (``_kernel_leaf``) is patched to take the first product
    leaf as a device leaf, and the kernel's table to answer with a wrong digest. The routed
    ``commit_leaves`` then stops with ``KernelMismatchError`` (FS34's per-shape check) and the
    line's cpu root, taken with ``kernel=False``, stays the C library's."""
    from verification.commitment import kernel_check
    from verification.commitment import leaves as L

    c, leaves, honest = real
    first = c.n_s + c.n_w
    monkeypatch.setattr(L, "_kernel_leaf", lambda t: t is leaves[first])
    monkeypatch.setattr(L, "_enqueue_table", lambda idx, ts, tags: SimpleNamespace(idx=idx))
    monkeypatch.setattr(L, "_table_outcome", lambda tb: ([b"\x00" * 32] * len(tb.idx), None))
    kernel_check.new_run()
    with pytest.raises(kernel_check.KernelMismatchError):
        commit_leaves(c, leaves)  # the fake kernel is in the routed path
    got = fs._DEVICE_LEAVES_LINE.match(written_line(real))
    assert got is not None and got[4] == honest.hex()
