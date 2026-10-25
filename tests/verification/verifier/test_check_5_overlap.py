"""Check 5's overlap with the glue on the Mac (FS39, ``MEASURE_KERNEL_PLAN_DRAFT.md``
§2.5–§2.7).

The stand-in kernel of ``test_kernel_views`` measures on the host, ``freivalds.kernel_device``
is switched on, and every seam the overlap uses is spied: ``freivalds.drain``,
``measure_groups_rows``, ``measure_kernel.prefetch`` and the replay's serving. What is under
test is the order of the calls on the unit path: a device is drained before each step's call so
one step is in flight per device, each step's challenges are submitted before the step before
it is served and handed to that step's call, and nothing prefetches on the canonical path.
"""

import contextlib

import pytest
import torch

from verification.computation.instances.decoder import DecoderReplay
from verification.computation.interface import ReplayError
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.context import CommittedLeaves, StepContext
from verification.verifier.matmul_check import freivalds, measure_kernel

from tests.verification.verifier.test_kernel_views import Kernel
from tests.verification.verifier.test_unit_schedule import CHEAP, _flipped, _layer_target, proved


@pytest.fixture
def spied(monkeypatch):
    """The stand-in kernel with every overlap seam recorded in ``events``: ``("prefetch", ms)``,
    ``("serve",)``, ``("drain", device)``, ``("measure", prefetch, in flight before)``."""
    kern = Kernel()
    events, flight = [], {}
    monkeypatch.setattr(freivalds, "kernel_device", lambda d: freivalds._reference_depth == 0)

    def drain(device=None):
        events.append(("drain", device))
        for d in list(flight) if device is None else [torch.device(device)]:
            flight[d] = 0
        kern.drain(device)

    def measure(groups, *, h, k, guard=True, share=False, plan_out=None, prefetched=None):
        dev = groups[0].a.device
        events.append(("measure", prefetched, flight.get(dev, 0)))
        if prefetched is not None:  # what enqueue would draw is in the prefetch
            ms = [int(m) for g in groups for m in g.ms]
            widths = [g.b.shape[2] for g in groups for _ in g.ms]
            assert prefetched.covers(h, ms, k, widths)
        flight[dev] = flight.get(dev, 0) + 1
        return kern.measure(groups, h=h, k=k, guard=guard, share=share, plan_out=plan_out)

    real_prefetch = measure_kernel.prefetch

    def prefetch(h, ms, k, widths, challenges=None):
        events.append(("prefetch", tuple(ms)))
        return real_prefetch(h, ms, k, widths, challenges)

    monkeypatch.setattr(freivalds, "drain", drain)
    monkeypatch.setattr(freivalds, "measure_groups_rows", measure)
    monkeypatch.setattr(measure_kernel, "prefetch", prefetch)
    for name in ("operands", "operand_blocks"):
        real = getattr(DecoderReplay, name)

        def serve(self, *a, _real=real, **kw):
            events.append(("serve",))
            return _real(self, *a, **kw)

        monkeypatch.setattr(DecoderReplay, name, serve)
    return events


def _run(setup, *, leaves=None, section=None):
    c = setup.c
    committed = CommittedLeaves()
    for leaf in (setup.leaves if leaves is None else leaves):
        committed.add(leaf)
    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=7, judge=True,
                                      section=section)
    ctx.state.leaves, ctx.state.claimed_root = committed, b"\x07" * 32
    rej = checks.check_5_matmuls(None, c, ctx, Bands.provisional())
    return rej, ctx


def _steps(ctx):
    c = ctx.state.replay.c if hasattr(ctx.state.replay, "c") else None
    runs = list(checks._member_runs(c.products))
    return runs, checks._schedule_runs(runs, ctx.state.replay.unit_schedule())


def test_a_device_is_drained_before_each_steps_call_and_one_step_is_in_flight(spied):
    setup = proved(CHEAP, 2)
    rej, ctx = _run(setup)
    assert rej is None
    measures = [i for i, e in enumerate(spied) if e[0] == "measure"]
    assert len(measures) > 2
    for i in measures:
        assert spied[i - 1][0] == "drain" and spied[i - 1][1] == torch.device("cpu")
        assert spied[i][2] == 0  # nothing of the step before was still in flight
    _, steps = _steps(ctx)
    assert len(measures) == len(steps)  # one call per step on the one device


def test_the_next_steps_challenges_are_submitted_before_this_step_is_served(spied):
    setup = proved(CHEAP, 2)
    rej, ctx = _run(setup)
    assert rej is None
    runs, steps = _steps(ctx)
    want = [tuple(spec.m for i in step for spec in runs[i]) for step in steps]
    prefetches = [e for e in spied if e[0] == "prefetch"]
    assert [e[1] for e in prefetches] == want
    # the stream of events is P(0), then per step s: P(s + 1), serves, the call of step s
    seen, first_serve, calls = 0, [], []
    start = True
    for e in spied:
        if e[0] == "prefetch":
            seen += 1
        elif e[0] == "serve" and start:
            first_serve.append(seen)  # prefetches submitted before this step's first serve
            start = False
        elif e[0] == "measure":
            calls.append(e)
            start = True
    assert len(first_serve) == len(steps) == len(calls)
    for s, n in enumerate(first_serve):
        assert n == min(s + 2, len(steps)), (s, n)
    handed = [e[1] for e in calls]
    assert all(p is not None for p in handed) and len({id(p) for p in handed}) == len(handed)
    for s, p in enumerate(handed):  # each call got its own step's prefetch
        assert tuple(sorted(m for m, _ in p.offsets)) == tuple(sorted(set(want[s])))


def test_the_canonical_path_prefetches_nothing(spied, monkeypatch):
    monkeypatch.setattr(DecoderReplay, "unit_schedule", lambda self: None)
    setup = proved(CHEAP, 2)
    rej, ctx = _run(setup)
    assert rej is None
    assert not [e for e in spied if e[0] == "prefetch"]
    assert [e[1] for e in spied if e[0] == "measure"] and all(
        e[1] is None for e in spied if e[0] == "measure")


def test_the_record_pass_prefetches_nothing(spied):
    setup = proved(CHEAP, 2)
    m = _layer_target(setup.c, 1, "dV")
    rej, ctx = _run(setup, leaves=_flipped(setup, m))
    assert rej is not None and rej.kind == "failed"
    # drop what the unit pass saved, so the record pass measures the later runs itself
    ctx.state.run_rest, ctx.state.run_rest_rows = [], []
    before = len(spied)
    status = checks.record_rest(None, setup.c, ctx, Bands.provisional(), "5")
    assert status != checks.RECORD_ERROR
    after = [e for e in spied[before:] if e[0] in ("measure", "prefetch")]
    assert after and all(e[0] == "measure" and e[1] is None for e in after)


def test_a_held_input_that_moved_in_the_drain_falls_back_to_the_canonical_loop(spied,
                                                                              monkeypatch):
    setup = proved(CHEAP, 2)
    clean, clean_ctx = _run(setup)
    spied.clear()
    real = freivalds.drain
    calls = []

    def drain(device=None):
        calls.append(device)
        if device is not None and len(calls) == 3:  # a step's own drain
            real(device)
            raise ReplayError("a held input changed")
        return real(device)

    monkeypatch.setattr(freivalds, "drain", drain)
    rej, ctx = _run(setup)
    assert rej is None  # the canonical rerun judged the honest step
    assert [(s.m, s.normalized) for s in ctx.stats.products] == \
        [(s.m, s.normalized) for s in clean_ctx.stats.products]
    measures = [e for e in spied if e[0] == "measure"]
    assert any(e[1] is None for e in measures)  # the rerun's calls
    assert measures[0][1] is not None


def test_timed_host_asks_the_seam_for_its_host_section_when_it_has_one():
    names = []

    class Seam:
        def __call__(self, name):
            names.append(("plain", name))
            return contextlib.nullcontext()

        def host(self, name):
            names.append(("host", name))
            return contextlib.nullcontext()

    ctx = StepContext.for_computation(proved(CHEAP, 1).c, step=1, indices=(), h_D=b"",
                                      n_records=0, prev_w_hashes=(), chain_check_id="0", k=7,
                                      section=Seam())
    with ctx.timed("x", host=True), ctx.timed("y"), ctx.timed("z", host=False):
        pass
    assert names == [("host", "x"), ("plain", "y"), ("plain", "z")]
    plain = []
    ctx.section = lambda name: plain.append(name) or contextlib.nullcontext()
    with ctx.timed("x", host=True), ctx.timed("y"):
        pass
    assert plain == ["x", "y"]
    ctx.section = None
    with ctx.timed("x", host=True):
        pass


def test_kernel_path_sections_are_host_timed(spied):
    names = []

    class Seam:
        def __call__(self, name):
            names.append(("plain", name))
            return contextlib.nullcontext()

        def host(self, name):
            names.append(("host", name))
            return contextlib.nullcontext()

    rej, _ = _run(proved(CHEAP, 2), section=Seam())
    assert rej is None
    assert ("host", "5.measure") in names and ("plain", "5.measure") not in names
    assert ("plain", "5.glue") in names


def test_overlap_context_nests_and_a_cpu_device_has_no_stream():
    assert measure_kernel.overlap_on()
    cpu = torch.device("cpu")
    assert measure_kernel.measure_stream(cpu) is None
    with measure_kernel.overlap():  # False by default
        assert not measure_kernel.overlap_on()
        with measure_kernel.overlap(True):
            assert measure_kernel.overlap_on()
        assert not measure_kernel.overlap_on()
        assert measure_kernel.prefetch(b"\0" * 32, [1], 3, [6]) is None
    assert measure_kernel.overlap_on()
    with pytest.raises(RuntimeError):
        with measure_kernel.overlap(False):
            raise RuntimeError("x")
    assert measure_kernel.overlap_on()
    with measure_kernel.stream_for(cpu):
        pass  # nothing happens


def test_prefetch_is_none_for_a_source_that_cannot_prefetch():
    class Source:
        def draw(self, *a, **kw):
            raise AssertionError

    assert measure_kernel.prefetch(b"\0" * 32, [1], 3, [6], Source()) is None
    assert measure_kernel.prefetch(b"\0" * 32, [1], 3, [6]) is not None


def test_the_pinned_read_equals_the_plain_read_off_cuda(monkeypatch):
    rows = [torch.rand(n, 5, dtype=torch.float64) for n in (3, 1, 4, 2)]
    monkeypatch.setattr(checks, "_rows_device", lambda r: int(r.shape[0]) % 2)  # two "devices"
    plain = checks._read_rows(rows)
    pinned = checks._read_rows(rows, pinned=True)
    assert all(torch.equal(torch.from_numpy(a), torch.from_numpy(b)) for a, b in zip(plain, pinned))
    assert [len(x) for x in pinned] == [3, 1, 4, 2]
