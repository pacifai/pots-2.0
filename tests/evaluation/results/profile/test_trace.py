"""FS24: the trace summarizer on a hand-built trace.

Times are microseconds. CPU events sit on pid 1, tid 1 and GPU events on pid 0, tid 7. The
sections, with the numbers worked out by hand:

Numbers are inclusive, like the time rows: ``A`` (0–1000) holds ``A.inner`` (400–600), so
``A``'s wall is 1000 and ``A.inner``'s 200, and ``A`` counts ``A.inner``'s events too.

- ``A`` itself launches a GEMM (30), an elementwise (10), a reduction (6) and an unknown kernel (4)
  and has three copies, a memset and one pageable device-to-host ``cudaMemcpyAsync`` (12 of
  sync wait). ``A.inner`` launches a flash kernel (50), syncs for 80 and does a blocking
  ``cudaMemcpy`` of 64 bytes (30 more of sync);
- ``5`` (2000–3000) launches a cutlass GEMM (40) and a 4096-byte copy (60) and holds the
  check-5 stamps. Class ``Q.fwd`` has runs 0–4 of one product each, runs 0 and 3 traced, so
  its scale is 2.5. Class ``attn`` has runs 5 (four products) and 6 (two), run 5 traced, so
  its scale is 6 ÷ 4.
- Three of the ten GPU events under a ``gpu_user_annotation`` sit in the wrong one.
"""

import gzip
import json

import pytest

from evaluation.results.profile import trace

CPU = {"pid": 1, "tid": 1}
GPU = {"pid": 0, "tid": 7}


def x(cat, name, ts, dur, args=None):
    where = GPU if cat.startswith("gpu_") or cat == "kernel" else CPU
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, **where,
            "args": args or {}}


def launch(ts, corr, name="cudaLaunchKernel", dur=5):
    return x("cuda_runtime", name, ts, dur, {"correlation": corr})


def kernel(name, ts, dur, corr):
    return x("kernel", name, ts, dur, {"correlation": corr})


def memcpy(kind, ts, dur, nbytes, corr):
    args = {"correlation": corr}
    if nbytes is not None:
        args["bytes"] = nbytes
    return x("gpu_memcpy", f"Memcpy {kind}", ts, dur, args)


def ann(name, ts, dur):
    return x("user_annotation", name, ts, dur)


def events():
    return [
        ann("A", 0, 1000), ann("section:A.inner", 400, 200), ann("5", 2000, 1000),
        ann("ProfilerStep#1", 0, 3000),
        # Section A
        x("cpu_op", "aten::mm", 100, 50, {"Input Dims": [[4, 8], [8, 2]],
                                           "Input type": ["float", "float"], "flops": 128}),
        x("cpu_op", "aten::add", 200, 20),
        launch(110, 1), kernel("sm80_xmma_gemm_f32", 120, 30, 1),
        launch(200, 2), kernel("void at::native::vectorized_elementwise_kernel", 160, 10, 2),
        launch(250, 3, "cudaMemcpyAsync", 4),
        memcpy("HtoD (Pageable -> Device)", 180, 20, 1000, 3),
        launch(700, 6, "cudaMemcpyAsync", 12),
        memcpy("DtoH (Device -> Pageable)", 500, 5, 32, 6),
        launch(800, 7, "cudaMemcpyAsync", 3),
        memcpy("DtoD (Device -> Device)", 520, 4, 16, 7),
        launch(810, 8, "cudaMemsetAsync", 2),
        x("gpu_memset", "Memset (Device)", 530, 2, {"correlation": 8}),
        launch(900, 9), kernel("void at::native::reduce_kernel<512>", 540, 6, 9),
        launch(910, 10), kernel("mystery_kernel", 560, 4, 10),
        # Section A.inner
        launch(410, 4), kernel("flash_fwd_kernel", 400, 50, 4),
        launch(500, 0, "cudaStreamSynchronize", 80),
        launch(560, 5, "cudaMemcpy", 30), memcpy("DtoH (Device -> Pageable)", 450, 10, 64, 5),
        # Section 5, with its stamps
        x("cpu_op", "aten::to", 2150, 40), x("cpu_op", "aten::copy_", 2155, 20),
        launch(2120, 11), kernel("cutlass_gemm", 2125, 40, 11),
        launch(2160, 12, "cudaMemcpyAsync", 10),
        memcpy("HtoD (Pageable -> Device)", 2170, 60, 4096, 12),
        ann("c5.run|run=0", 2000, 500), ann("c5.glue|run=0|group=0", 2000, 100),
        ann("c5.serve|run=0|group=0", 2100, 200), ann("c5.copy|run=0|group=0", 2150, 100),
        ann("c5.measure|run=0|group=0", 2300, 150), ann("c5.judge|window=0", 2600, 50),
        ann("c5.run|run=3", 2700, 200), ann("c5.glue|run=3|group=3", 2700, 30),
        ann("c5.serve|run=3|group=3", 2730, 100), ann("c5.copy|run=3|group=3", 2750, 40),
        ann("c5.measure|run=3|group=3", 2830, 60), ann("c5.run|run=5", 2900, 80),
        # GPU annotations: A.inner's wrongly holds A's copy at 500, copy at 520 and memset at 530.
        x("gpu_user_annotation", "A", 100, 300), x("gpu_user_annotation", "A.inner", 400, 140),
        x("gpu_user_annotation", "5", 2100, 400),
    ]


def write(path, evs):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt") as f:
        json.dump({"traceEvents": evs}, f)
    return path


def layout(path):
    ents = [{"run": r, "group": r, "window": 0, "class": "Q.fwd", "members": 1}
            for r in range(5)]
    ents += [{"run": 5, "group": 5, "window": 0, "class": "attn", "members": 4},
             {"run": 6, "group": 6, "window": 0, "class": "attn", "members": 2}]
    path.write_text(json.dumps({"version": 1, "entries": ents, "trace_slice": [0, 3, 5]}))
    return path


@pytest.fixture
def summary(tmp_path):
    t = write(tmp_path / "step_00005.json.gz", events())
    return trace.summarize(t, layout(tmp_path / "c5_layout.json"))


def test_top_level(summary):
    assert set(summary) == {"version", "sections", "c5_slice", "warnings"}
    assert summary["version"] == 1
    assert set(summary["sections"]) == {"A", "A.inner", "5"}


def test_section_a(summary):
    a = summary["sections"]["A"]
    assert a["parent"] is None and summary["sections"]["A.inner"]["parent"] == "A"
    assert a["wall_us"] == 1000 and a["kernel_us"] == 50 + 50 and a["kernel_span_us"] == 100
    assert a["gpu_idle_us"] == 900 and a["kernels"] == 5 and a["launches"] == 5
    assert a["categories_us"] == {
        "gemm": 30, "attention": 50, "elementwise": 10, "reduction": 6, "memcpy_h2d": 20,
        "memcpy_d2h": 5 + 10, "memcpy_d2d": 4, "memset": 2, "other": 4}
    assert a["copy_bytes"] == {"h2d": 1000, "d2h": 32 + 64, "d2d": 16}
    assert a["gemm_flops"] == 128 and a["gemm_bytes"] == 128 + 64 + 32
    assert a["sync_wait_us"] == 12 + 80 + 30
    assert a["top_cpu_ops"] == [{"name": "aten::mm", "self_us": 50, "calls": 1},
                                {"name": "aten::add", "self_us": 20, "calls": 1}]


def test_parent_is_own_plus_child(summary):
    secs = summary["sections"]
    assert secs["A"]["kernel_us"] == 50 + secs["A.inner"]["kernel_us"]
    top = [s for s in secs.values() if s["parent"] is None]
    assert sum(s["kernel_us"] for s in top) == 50 + 50 + 40  # every attributed kernel once


def test_nested_section(summary):
    inner = summary["sections"]["A.inner"]
    assert inner["wall_us"] == 200 and inner["kernels"] == 1 and inner["launches"] == 1
    assert inner["categories_us"]["attention"] == 50
    assert inner["categories_us"]["memcpy_d2h"] == 10
    assert inner["copy_bytes"] == {"h2d": 0, "d2h": 64, "d2d": 0}
    assert inner["sync_wait_us"] == 80 + 30
    assert inner["gpu_idle_us"] == 150
    assert inner["gemm_flops"] == 0 and inner["gemm_bytes"] == 0


def test_section_with_stamps(summary):
    s = summary["sections"]["5"]
    assert s["parent"] is None and s["wall_us"] == 1000 and s["gpu_idle_us"] == 960
    assert s["categories_us"]["gemm"] == 40 and s["categories_us"]["memcpy_h2d"] == 60
    assert s["copy_bytes"]["h2d"] == 4096 and s["launches"] == 1
    assert s["top_cpu_ops"] == [{"name": "aten::copy_", "self_us": 20, "calls": 1},
                                {"name": "aten::to", "self_us": 20, "calls": 1}]


def test_class_scale_two_runs_of_five(summary):
    q = summary["c5_slice"]["Q.fwd"]
    assert (q["runs_traced"], q["runs_total"], q["scale"]) == (2, 5, 2.5)
    assert q["phases_us"] == {"run": 700, "glue": 130, "serve": 300, "copy": 140,
                              "stack": 160, "measure": 210}
    assert q["scaled"] == {"run": 1750, "glue": 325, "serve": 750, "copy": 350, "stack": 400,
                           "measure": 525}


def test_scale_weighs_products(summary):
    a = summary["c5_slice"]["attn"]
    assert (a["runs_traced"], a["runs_total"], a["scale"]) == (1, 2, 1.5)
    assert a["phases_us"]["run"] == 80 and a["scaled"]["run"] == 120


def test_annotation_disagreement_warned(summary):
    assert summary["warnings"] == [
        "30.0% of GPU events (3 of 10) sit in a different gpu_user_annotation than their "
        "launch's section"]


def test_markers_cross_check(tmp_path):
    t = write(tmp_path / "t.json.gz", events())
    m = tmp_path / "markers.jsonl"

    def line(names):
        return json.dumps({"kind": "step", "traced": True,
                           "sections": [{"name": n} for n in names]})

    m.write_text(json.dumps({"kind": "header"}) + "\n" + line(["A", "A.inner", "5"]) + "\n")
    assert len(trace.summarize(t, markers_path=m)["warnings"]) == 1
    m.write_text(line(["A", "5", "6"]) + "\n")
    w = trace.summarize(t, markers_path=m)["warnings"]
    assert any("'6' is in the markers" in s for s in w)
    assert any("'A.inner' is in the trace but not the markers" in s for s in w)


def test_missing_bytes_and_flops(tmp_path):
    evs = [ann("S", 0, 100),
           x("cpu_op", "aten::mm", 10, 20, {"Input Dims": [[2, 2], [2, 2]],
                                             "Input type": ["float", "float"]}),
           launch(50, 1, "cudaMemcpyAsync"),
           memcpy("DtoH (Device -> Pageable)", 60, 5, None, 1)]
    s = trace.summarize(write(tmp_path / "t.json.gz", evs))
    sec = s["sections"]["S"]
    assert sec["copy_bytes"]["d2h"] is None and sec["copy_bytes"]["h2d"] == 0
    assert sec["gemm_flops"] is None and sec["gemm_bytes"] == 48
    assert any("no args.bytes" in w for w in s["warnings"])
    assert any("no flops" in w for w in s["warnings"])


def test_matmul_counts_its_mm_once(tmp_path):
    dims = {"Input Dims": [[4, 8], [8, 2]], "Input type": ["float", "float"]}
    evs = [ann("S", 0, 100), x("cpu_op", "aten::matmul", 10, 30, dims),
           x("cpu_op", "aten::mm", 12, 20, {**dims, "flops": 128})]
    sec = trace.summarize(write(tmp_path / "t.json", evs))["sections"]["S"]
    assert sec["gemm_bytes"] == 224 and sec["gemm_flops"] == 128


def test_kernel_span_is_a_union(tmp_path):
    evs = [ann("S", 0, 100), launch(1, 1), launch(2, 2),
           kernel("gemm_a", 10, 20, 1), kernel("gemm_b", 20, 20, 2)]
    sec = trace.summarize(write(tmp_path / "t.json", evs))["sections"]["S"]
    assert sec["kernel_us"] == 40 and sec["kernel_span_us"] == 30 and sec["gpu_idle_us"] == 70


def test_cli(tmp_path):
    t = write(tmp_path / "t.json.gz", events())
    out = tmp_path / "trace_summary.json"
    assert trace.main(["--trace", str(t), "--layout", str(layout(tmp_path / "l.json")),
                       "--out", str(out)]) == 0
    assert json.loads(out.read_text())["version"] == 1
