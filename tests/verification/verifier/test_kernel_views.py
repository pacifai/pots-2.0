"""Check 5's kernel path (FS37, ``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.3) on the Mac.

``freivalds.kernel_device`` is switched on and ``measure_groups_rows`` replaced by a stand-in
that measures each member alone on contiguous fp32 copies (``measure_product_row``, under
``freivalds.reference()``), so its bits can't depend on grouping, strides or widening, as the
kernel's can't. ``drain`` records its calls. What is under test is the serving: views from
``Replay.operand_blocks``, ``P`` as stored, rows mapped back to the run's positions.
"""

import weakref

import pytest
import torch

from setup.data import schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.instances.decoder import DecoderReplay
from verification.computation.interface import Replay, ReplayError
from verification.prover.step import prove_step
from verification.transcript.store import InMemoryStore
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check import freivalds

from tests.verification.computation.test_decoder import SEED, _bump
from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config,
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


class Kernel:
    """The stand-in kernel: records every call and every drain."""

    def __init__(self):
        self.calls = []  # per measure_groups_rows call: [(ms, bias given, a/b/p dtypes)]
        self.events = []  # "measure", "drain", "read" in order
        self.views = []  # every served MeasureGroup

    def measure(self, groups, *, h, k, guard=True, share=False, plan_out=None):
        self.events.append("measure")
        self.calls.append([(list(g.ms), g.bias is not None) for g in groups])
        rows = []
        for g in groups:
            self.views.append(g)
            for t, m in enumerate(g.ms):
                f = lambda x: x.detach().to(torch.float32).contiguous()  # noqa: E731
                with freivalds.reference():
                    rows.append(freivalds.measure_product_row(
                        f(g.a[t]), f(g.b[t]), f(g.p[t]), h=h, m=m, k=k, guard=guard,
                        bias=None if g.bias is None else f(g.bias)))
        return torch.stack(rows)

    def drain(self, device=None):
        self.events.append("drain")


@pytest.fixture
def canonical_order(monkeypatch):
    """Pins check 5 to its canonical-order loop (FS38a): the decoder offers no unit schedule."""
    monkeypatch.setattr(DecoderReplay, "unit_schedule", lambda self: None)


@pytest.fixture
def kernel(monkeypatch):
    kern = Kernel()
    # `reference()` still forces the torch path, as the real `kernel_device` does: the stand-in
    # measures through `measure_product_row` under it
    monkeypatch.setattr(freivalds, "kernel_device", lambda device: freivalds._reference_depth == 0)
    monkeypatch.setattr(freivalds, "measure_groups_rows", kern.measure)
    monkeypatch.setattr(freivalds, "drain", kern.drain)
    real_read = checks._read_rows

    def read(rows):
        kern.events.append("read")
        return real_read(rows)

    monkeypatch.setattr(checks, "_read_rows", read)
    return kern


def _setup(switch, dtype, *, biases=True):
    family, tied = switch
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                           compute_dtype=DTYPES[dtype])
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    tree = dataset_tree(c, D)
    w0 = {n: p.detach().clone() for n, p in c.build_model().named_parameters()}
    if biases:
        randomize_biases(w0)
    return c, D, tree, w0


def _verify(c, D, tree, w0, **kw):
    idx = schedule(1, c.n_s, len(D), SEED)
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1, bands=Bands.provisional(),
                 allow_provisional=True, w0=w0, seed=SEED)
    assert v.start_run(D) is None
    out = prove_step(c, c.build_model(), w0, [D[i] for i in idx], **kw)
    store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
    return v, v.verify_step(1, store)


def _numbers(v):
    return [(s.m, s.name, s.kappa, s.normalized) for s in v.stats[1].products]


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=map(switch_id, SWITCHES))
def test_honest_step_is_accepted_and_measured_from_views(switch, dtype, kernel):
    c, D, tree, w0 = _setup(switch, dtype)
    v, rejection = _verify(c, D, tree, w0)
    assert rejection is None
    assert sorted(m for call in kernel.calls for grp in call for m in grp[0]) \
        == list(range(1, c.M + 1))  # every product measured once
    assert len(v.stats[1].products) == c.M
    op = c.operand_dtype
    for g in kernel.views:
        assert g.a.dtype == g.b.dtype == op  # no fp32 widening
        assert g.p.dtype == c.product_dtype
        assert g.a.dim() == g.b.dim() == g.p.dim() == 3
    # attention members arrive as blocks of views over the whole operands, not as copies
    assert any(len(g.ms) > 1 for g in kernel.views)
    # biased products are runs of one with their bias
    biased = [g for g in kernel.views if g.bias is not None]
    assert bool(biased) == bool(c.qkv_bias)
    assert all(len(g.ms) == 1 and g.bias.dtype == op for g in biased)


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("switch", SWITCHES, ids=map(switch_id, SWITCHES))
def test_views_give_the_numbers_of_the_fp32_serving(switch, dtype, kernel, monkeypatch):
    """Every product's numbers equal those of the same stand-in fed by the replay's default
    blocks (one member each, from ``operands``): the block mapping changes nothing."""
    c, D, tree, w0 = _setup(switch, dtype)
    v, rejection = _verify(c, D, tree, w0)
    assert rejection is None
    got = _numbers(v)
    assert any(len(grp[0]) > 1 for call in kernel.calls for grp in call)
    kernel.calls.clear()
    monkeypatch.setattr(DecoderReplay, "operand_blocks", Replay.operand_blocks)
    v2, rejection = _verify(c, D, tree, w0)
    assert rejection is None
    assert all(len(grp[0]) == 1 for call in kernel.calls for grp in call)
    assert _numbers(v2) == got


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_kernel_path_equals_the_torch_path_to_float_error(dtype, kernel, monkeypatch):
    """Against today's path, which measures members batched: the same products, the same
    verdict, and numbers equal to fp32 rounding (the two measure by different code)."""
    c, D, tree, w0 = _setup(("qwen2", False), dtype)
    v, rejection = _verify(c, D, tree, w0)
    assert rejection is None
    monkeypatch.setattr(freivalds, "kernel_device", lambda device: False)
    v2, rejection = _verify(c, D, tree, w0)
    assert rejection is None
    for x, y in zip(_numbers(v), _numbers(v2), strict=True):
        assert x[:2] == y[:2]
        assert x[2] == pytest.approx(y[2], rel=1e-4, abs=1e-6)
        assert x[3] == pytest.approx(y[3], rel=1e-3, abs=1e-5)


@pytest.mark.parametrize("name", ["L1.S", "L2.O", "L2.dK", "L1.dV", "L1.Y_q", "Lambda"])
def test_a_flipped_product_is_rejected_where_today_rejects_it(name, kernel, monkeypatch):
    c, D, tree, w0 = _setup(("qwen2", False), "fp32")
    ms = [p.m for p in c.products if p.name == name or p.name.startswith(name + "[")
          or (p.member is not None and f"L{p.layer}.{c.product_class(p)}" == name)]
    if name == "Lambda":
        ms = [c.m_of(c.head_name("Lambda", 0))] if hasattr(c, "head_name") else ms
    assert ms, name
    m = ms[len(ms) // 2]
    kw = dict(perturb={m: lambda p: _bump(p)})
    _, kern = _verify(c, D, tree, w0, **kw)
    monkeypatch.setattr(freivalds, "kernel_device", lambda device: False)
    _, torch_path = _verify(c, D, tree, w0, **kw)
    assert kern is not None and torch_path is not None
    assert (kern.step, kern.check_id) == (torch_path.step, torch_path.check_id) == (1, "5")
    assert kern.detail == torch_path.detail
    assert f"P_{m} " in kern.detail


@pytest.mark.parametrize("role", ["O", "dK", "dV"])
def test_a_serving_error_in_a_group_finds_the_same_member_and_error(role, kernel, monkeypatch):
    """A held operand of one role changes before its group is served: the group fails whole,
    the run is finished member by member, and the error and the products judged before it are
    those of today's path (for a backward role, after earlier groups of the run were measured
    from blocks)."""
    c, D, tree, w0 = _setup(("llama", True), "fp32", biases=False)
    layer = 1 if role == "O" else c.L
    # Each replay is tampered once, at its first group of the role: the unit pass's replay
    # and, after its fallback, the fresh one the canonical rerun serves.
    done, seen = [], weakref.WeakSet()

    def tamper(replay, ms):
        spec = c.product(ms[0])
        mine = [m for m in ms if c.product(m).member is not None
                and c.product_class(c.product(m)) == role and c.product(m).layer == layer]
        if replay not in seen and mine:
            seen.add(replay)
            done.append(mine[0])
            first = c.product(mine[0])
            if first.kind.name == "FORWARD":
                replay._forward_unit(first)
                held = replay._units[layer].fwd
            else:
                replay._backward_unit(first)
                held = replay._bops
            with torch.no_grad():
                held[f"L{layer}.{role}"][0].add_(0)  # bumps _version only

    for fn in ("operand_group", "operand_blocks"):
        real = getattr(DecoderReplay, fn)
        monkeypatch.setattr(DecoderReplay, fn, lambda self, ms, *a, _r=real: (
            tamper(self, ms), _r(self, ms, *a))[1])
    runs = []
    for on in (True, False):
        monkeypatch.setattr(freivalds, "kernel_device", lambda device, on=on: on and freivalds._reference_depth == 0)
        done.clear()
        kernel.calls.clear()
        with pytest.raises(ReplayError) as e:
            _verify(c, D, tree, w0)
        assert done
        assert bool(kernel.calls) == on
        runs.append(str(e.value))
    assert runs[0] == runs[1] and "changed after its product read it" in runs[0]


def test_drain_runs_before_each_window_read_and_at_the_end(kernel, canonical_order,
                                                           monkeypatch):
    c, D, tree, w0 = _setup(("llama", True), "fp32")
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", 20)  # several windows in a step
    v, rejection = _verify(c, D, tree, w0)
    assert rejection is None
    ev = kernel.events
    reads = [i for i, e in enumerate(ev) if e == "read"]
    assert len(reads) >= 2
    # `read` is stamped before _read_rows runs; the drain is its first act, so a drain
    # precedes the measures after it and the events show drain right after each read.
    assert ev.count("drain") >= len(reads) + 1
    for i in reads:
        assert ev[i + 1] == "drain"
    assert ev[-1] == "drain"  # check 5's finally, after the last read


def test_drain_runs_before_the_one_read_of_the_unit_pass_and_at_the_end(kernel, monkeypatch):
    """The unit pass measures every run, then one read brings every window's rows (a drain
    right before it and none needed between windows), then check 5's ``finally`` drains."""
    c, D, tree, w0 = _setup(("llama", True), "fp32")
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", 20)  # several windows in a step
    v, rejection = _verify(c, D, tree, w0)
    assert rejection is None and len(v.stats[1].products) == c.M
    ev = kernel.events
    assert ev.count("read") == 1
    read = ev.index("read")
    assert "measure" not in ev[read:] and ev[:read].count("measure") > 0
    assert ev[read + 1] == "drain" and ev[-1] == "drain"
    assert len(ev) == read + 3  # read, the drain _read_rows starts with, check 5's drain


def test_check_5_drains_when_a_product_is_rejected(kernel):
    c, D, tree, w0 = _setup(("llama", True), "fp32")
    m = c.m_of("L1.Y_q")
    _, rejection = _verify(c, D, tree, w0, perturb={m: lambda p: _bump(p)})
    assert rejection is not None and rejection.check_id == "5"
    assert kernel.events[-1] == "drain" or "drain" in kernel.events[-3:]


# ---- the two row reads outside check 5's window loop (FS37 fix 1) ----

def test_record_pass_drains_when_a_later_run_fails_to_serve(kernel, monkeypatch):
    from types import SimpleNamespace

    state = SimpleNamespace(claimed_root=b"r", replay=object(), run_rest=[], run_rest_rows=[],
                            run_rest_error=None, committed=lambda: None)
    ctx = SimpleNamespace(state=state, stats=SimpleNamespace(products=[]), k=7, kappa_guard=True)
    spec = SimpleNamespace(m=1)
    boom = RuntimeError("serving failed")
    monkeypatch.setattr(checks, "TranscriptView", lambda c, leaves: None)
    monkeypatch.setattr(checks, "_member_runs", lambda products: [[spec], [spec]])
    calls = []

    def served(*a, **kw):
        calls.append(1)
        kernel.events.append("measure")
        return (None, boom) if len(calls) == 2 else (object(), None)

    monkeypatch.setattr(checks, "_measure_served", served)
    monkeypatch.setattr(checks, "_finish", lambda batch, ctx: [])
    with pytest.raises(RuntimeError) as e:
        checks._record_rest_5(SimpleNamespace(products=[spec]), ctx, Bands.provisional())
    assert e.value is boom and len(calls) == 2
    assert kernel.events == ["measure", "measure", "drain"]
    # a judged product that raises leaves no hold either
    kernel.events.clear()
    calls.clear()
    monkeypatch.setattr(checks, "_measure_served", lambda *a, **kw: (object(), None))
    monkeypatch.setattr(checks, "_finish", lambda batch, ctx: [(spec, None)])

    def judge(*a, **kw):
        raise ValueError("judging failed")

    monkeypatch.setattr(checks, "_judge_product", judge)
    with pytest.raises(ValueError):
        checks._record_rest_5(SimpleNamespace(products=[spec]), ctx, Bands.provisional())
    assert kernel.events == ["drain"]


def test_measure_run_drains_before_reading_its_rows(kernel, monkeypatch):
    from types import SimpleNamespace

    class Rows:
        def tolist(self):
            kernel.events.append("read")
            return []

    monkeypatch.setattr(checks, "_stacked", lambda specs, served: None)
    monkeypatch.setattr(checks, "_measure_rows", lambda *a, **kw: SimpleNamespace(rows=Rows()))
    monkeypatch.setattr(checks, "_finished", lambda *a, **kw: [])
    assert checks._measure_run([], [], h=b"h", k=7, eps_in=1.0, eps_acc=1.0) == []
    assert kernel.events == ["drain", "read"]


@pytest.mark.parametrize("switch", SWITCHES, ids=map(switch_id, SWITCHES))
def test_one_copy_stamp_per_group_on_the_kernel_path(switch, kernel):
    """Every group is stamped once by each of ``c5.glue``, ``c5.serve`` and ``c5.copy``
    (``c5.measure`` is once per schedule step's table), also a group whose roles interleave and so is served as several blocks."""
    import contextlib
    from collections import Counter

    from verification.verifier.context import CommittedLeaves, StepContext

    c, D, tree, w0 = _setup(switch, "fp32")
    idx = schedule(1, c.n_s, len(D), SEED)
    out = prove_step(c, c.build_model(), w0, [D[i] for i in idx])
    committed = CommittedLeaves()
    for leaf in out.leaves():
        committed.add(leaf)
    stamps = []

    class Section:
        def __call__(self, name):
            return contextlib.nullcontext()

        @contextlib.contextmanager
        def stamp(self, name, run=-1, group=-1, window=-1):
            stamps.append((name, run, group))
            yield

    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=7, judge=True,
                                      section=Section())
    ctx.state.leaves, ctx.state.claimed_root = committed, b"\x07" * 32
    assert checks.check_5_matmuls(None, c, ctx, Bands.provisional()) is None
    n_groups = sum(len(checks._measure_groups(run)) for run in checks._member_runs(c.products))
    count = Counter(name for name, _, _ in stamps)
    for name in ("c5.glue", "c5.serve", "c5.copy"):
        assert count[name] == n_groups, name
        assert len({(r, g) for n, r, g in stamps if n == name}) == n_groups, name
    # FS38b: the unit path measures one table per schedule step, stamped (its first run, -1)
    runs = list(checks._member_runs(c.products))
    steps = checks._schedule_runs(runs, ctx.state.replay.unit_schedule())
    assert sorted((r, g) for n, r, g in stamps if n == "c5.measure") == sorted(
        (step[0], -1) for step in steps)
    # at least one group was served as several blocks, so the stamp is per group, not per block
    assert len(kernel.views) > n_groups


def test_a_run_served_on_two_devices_falls_back_to_the_canonical_rerun(kernel, monkeypatch,
                                                                       caplog):
    """FS38b: the step table splits rows back per device, which is serving order only for a run
    on one device, so a run whose groups sit on two (a meta tensor stands in for the second)
    ends the unit pass, and the canonical rerun gives the numbers of a clean verification."""
    import logging

    c, D, tree, w0 = _setup(("llama", True), "fp32", biases=False)
    clean, rejection = _verify(c, D, tree, w0)
    assert rejection is None
    kernel.calls.clear()
    real, hit = checks._served_blocks, []

    def two_devices(*args, **kw):
        order, groups = real(*args, **kw)
        if not hit:  # the unit pass's first member run only; the rerun is served as is
            hit.append(1)
            groups = [*groups, groups[0]._replace(a=groups[0].a.to("meta"))]
        return order, groups

    monkeypatch.setattr(checks, "_served_blocks", two_devices)
    with caplog.at_level(logging.DEBUG, logger=checks.__name__):
        v, rejection = _verify(c, D, tree, w0)
    assert rejection is None and hit
    assert any("unit pass raised RuntimeError" in r.getMessage() for r in caplog.records)
    assert _numbers(v) == _numbers(clean)
