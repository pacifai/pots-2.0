"""FS38a: check 5 in the replay's unit order against canonical order
(``MEASURE_KERNEL_PLAN_DRAFT.md`` §2.4 and §2.4a).

"The canonical path" is a step with the decoder's schedule patched to ``None``
(``DecoderReplay.unit_schedule``); "the unit path" is the same step unpatched. Every test runs
check 5 on both and compares: every ``ProductStat`` field (the ``compare=False`` ones too, bit
for bit, NaN equal), the rejection or the error that ended the step, ``StepState.run_rest``,
the record pass's status and the stats after it.

The harness is check 5 alone on a proved step's leaves (``test_kernel_step._check_5_alone``'s
way: a forged weight gradient would be caught by check 6a first in a whole verification), then
``checks.record_rest`` as the driver calls it. Flips are made at store level, on the committed
leaf, so each (switch, head blocks) setting is proved once. A few tests go through the whole
``Verifier`` as well.

Tests marked "needs the code" fail until FS38a's code is in: they look at the unit path itself
(``Replay.unit_schedule``, ``StepState.run_rest_rows``, ``checks._check_5_canonical``).
"""

import gc
import weakref
from collections import Counter
from dataclasses import dataclass

import pytest
import torch

from setup.data import schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.computation.instances.decoder import DecoderReplay
from verification.computation.interface import ReplayError
from verification.prover.step import prove_step
from verification.transcript.errors import StoreMutationError
from verification.transcript.store import InMemoryStore
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.context import CommittedLeaves, StepContext
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check import freivalds

from tests.verification.computation.test_decoder import SEED
from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config,
)
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    bits, host_measure, hostlib,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_host import phased  # noqa: F401
from tests.verification.verifier.test_kernel_step import _flip, _prove, install
from tests.verification.verifier.test_kernel_views import Kernel

CHEAP = ("llama", True)  # the switch the flip sweeps use
BLOCKS = (1, 2)  # head blocks B (n_s = 2)
LINEAR_ROLES = ["q", "k", "v", "o", "gate", "up", "down"]
LAYER_CLASSES = ([f"{k}_{r}" for k in ("Y", "dX", "G") for r in LINEAR_ROLES]
                 + ["S", "O", "dA", "dV", "dQ", "dK"])
HEAD_KINDS = ("Lambda", "dF", "G")


# ---- the proved steps, one per (switch, head blocks, devices) ---------------------------

@dataclass
class Setup:
    c: DecoderComputation
    D: list
    tree: object
    w0: dict
    out: object
    leaves: list  # the step's leaves, detached; never written


_SETUPS: dict = {}


def _build(switch, blocks, devices=None):
    family, tied = switch
    torch.manual_seed(0)
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                           compute_dtype=torch.float32, device="cpu", head_blocks=blocks,
                           verifier_devices=devices)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    tree = dataset_tree(c, D)
    torch.manual_seed(0)
    w0 = {n: p.detach().cpu().clone() for n, p in c.build_model().named_parameters()}
    randomize_biases(w0)
    return c, D, tree, w0


def proved(switch, blocks, devices=None) -> Setup:
    key = (switch, blocks, devices)
    if key not in _SETUPS:
        c, D, tree, w0 = _build(switch, blocks, devices)
        out, _ = _prove(c, D, w0)
        leaves = [x.detach() if isinstance(x, torch.Tensor) else x for x in out.leaves()]
        _SETUPS[key] = Setup(c, D, tree, w0, out, leaves)
    return _SETUPS[key]


def _m(c, name):
    return next(s.m for s in c.products if s.name == name)


def _layer_target(c, layer, cls):
    ms = [s.m for s in c.products if s.name.startswith(f"L{layer}.")
          and c.product_class(s) == cls]
    assert ms, (layer, cls)
    return ms[len(ms) // 2]


def _head_target(c, kind, b):
    return c.m_of(c.head_name(kind, b))


# ---- running check 5 and what is compared ------------------------------------------------

def _f(x):
    return float(x).hex()


def _stat_sig(s):
    return (s.m, s.name, s.cls, s.q, _f(s.kappa), tuple(map(_f, s.normalized)), _f(s.p_norm),
            _f(s.nu), _f(s.p_abs1), _f(s.u))


def _measure_sig(mp):
    return (_f(mp.nu), _f(mp.p_abs1), _f(mp.kappa), tuple(map(_f, mp.residuals)),
            _f(mp.p_norm), _f(mp.unit), tuple(map(_f, mp.normalized)), _f(mp.u))


def _err(e):
    return None if e is None else (type(e).__name__, str(e))


@dataclass
class Result:
    rejection: object  # check 5's return
    error: tuple | None  # the exception check 5 raised: (type name, message)
    stats: list  # signatures, after check 5
    run_rest: list  # (m, measure signature), as the record pass finds it
    run_rest_error: tuple | None
    record: str  # the record pass's status
    record_stats: list  # signatures, after the record pass
    ctx: StepContext
    rows_rest: int | None  # len(StepState.run_rest_rows) before the record pass; None: no such field


def _rej(r):
    return None if r is None else (r.step, r.check_id, r.detail, r.kind)


def run_check5(setup, *, leaves=None, bump=(), bands=None) -> Result:
    """Check 5 alone on ``leaves`` (default: the step's), then the record pass as the driver
    runs it. ``bump`` lists leaf indices whose ``_version`` is bumped after they are committed
    (as a write through an alias after check 2's hashing)."""
    c = setup.c
    leaves = list(setup.leaves if leaves is None else leaves)
    for i in bump:
        leaves[i] = leaves[i].clone()
    committed = CommittedLeaves()
    for leaf in leaves:
        committed.add(leaf)
    for i in bump:
        leaves[i].add_(0.0)
    ctx = StepContext.for_computation(c, step=1, indices=(), h_D=b"", n_records=0,
                                      prev_w_hashes=(), chain_check_id="0", k=7, judge=True)
    ctx.state.leaves, ctx.state.claimed_root = committed, b"\x07" * 32
    bands = bands or Bands.provisional()
    rej = err = None
    try:
        rej = checks.check_5_matmuls(None, c, ctx, bands)
    except Exception as e:  # noqa: BLE001 (it is what is compared)
        err = e
    stats = [_stat_sig(s) for s in ctx.stats.products]
    rest = [(spec.m, _measure_sig(mp)) for spec, mp in ctx.state.run_rest]
    rest_err = _err(ctx.state.run_rest_error)
    later = getattr(ctx.state, "run_rest_rows", None)
    rows_rest = None if later is None else len(later)
    status = checks.RECORD_NOT_RUN
    if rej is not None and rej.kind == "failed":  # the driver's rule and its error handling
        try:
            status = checks.record_rest(None, c, ctx, bands, "5")
        except Exception as e:  # noqa: BLE001
            status = f"{checks.RECORD_ERROR}: {type(e).__name__}: {e}"
    return Result(_rej(rej), _err(err), stats, rest, rest_err, status,
                  [_stat_sig(s) for s in ctx.stats.products], ctx, rows_rest)


def both(setup, mp, **kw):
    """``(unit, canonical)`` results of the same step. ``mp`` is the test's monkeypatch; the
    canonical run patches the schedule away for its own duration only."""
    unit = run_check5(setup, **kw)
    with pytest.MonkeyPatch.context() as cm:
        cm.setattr(DecoderReplay, "unit_schedule", lambda self: None, raising=False)
        canon = run_check5(setup, **kw)
    return unit, canon


def assert_same(unit, canon):
    assert unit.rejection == canon.rejection
    assert unit.error == canon.error
    assert unit.stats == canon.stats
    assert unit.run_rest == canon.run_rest
    assert unit.run_rest_error == canon.run_rest_error
    assert unit.record == canon.record
    assert unit.record_stats == canon.record_stats


def _flipped(setup, *ms):
    leaves = list(setup.leaves)
    for m in ms:
        g, i = setup.c.leaf_of(m)  # O18: a group's members are one 3-D leaf; flip member i
        k = setup.c.product_leaf_index(g)
        if i is None:
            leaves[k] = _flip(leaves[k])
        else:
            leaf = leaves[k].clone()
            leaf[i] = _flip(leaf[i])
            leaves[k] = leaf
    return leaves


# ---- an honest step ------------------------------------------------------------------------

@pytest.mark.parametrize("blocks", BLOCKS)
@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_honest_step_is_the_same_on_both_paths(switch, blocks, monkeypatch):
    setup = proved(switch, blocks)
    unit, canon = both(setup, monkeypatch)
    assert canon.rejection is None and canon.error is None
    assert len(canon.stats) == setup.c.M
    assert [s[0] for s in canon.stats] == list(range(1, setup.c.M + 1))  # canonical order
    assert_same(unit, canon)


# ---- a flip in every class ----------------------------------------------------------------

def _flip_case(setup, monkeypatch, m, **kw):
    unit, canon = both(setup, monkeypatch, leaves=_flipped(setup, m), **kw)
    name = setup.c.product(m).name
    assert canon.rejection is not None and canon.rejection[:2] == (1, "5"), (name, canon)
    assert canon.rejection[2].startswith(f"P_{m} ({name})"), canon.rejection  # the first failing
    assert_same(unit, canon)
    return unit, canon


@pytest.mark.parametrize("cls", LAYER_CLASSES)
@pytest.mark.parametrize("layer", [1, 2])
def test_a_flip_in_each_layer_class_is_judged_alike(layer, cls, monkeypatch):
    setup = proved(CHEAP, 2)
    _flip_case(setup, monkeypatch, _layer_target(setup.c, layer, cls))


def test_a_flip_at_l1_y_q_the_first_product_is_judged_alike(monkeypatch):
    setup = proved(CHEAP, 1)
    m = _m(setup.c, "L1.Y_q")
    assert m == 1
    _, canon = _flip_case(setup, monkeypatch, m)
    assert canon.rejection[2].startswith("P_1 (L1.Y_q)")


@pytest.mark.parametrize("b", [0, 1])
@pytest.mark.parametrize("kind", HEAD_KINDS)
@pytest.mark.parametrize("blocks", BLOCKS)
def test_a_flip_in_a_head_class_is_judged_alike(blocks, kind, b, monkeypatch):
    if b >= blocks:
        pytest.skip("no such head block")
    setup = proved(CHEAP, blocks)
    _flip_case(setup, monkeypatch, _head_target(setup.c, kind, b))


@pytest.mark.parametrize("blocks", BLOCKS)
@pytest.mark.parametrize("layer", [1, 2])
def test_flips_in_layers_at_each_head_block_setting(layer, blocks, monkeypatch):
    """A forward, a backward and a member class of each layer, at both head-block settings
    (the schedule's head steps differ; the layers' do not)."""
    setup = proved(CHEAP, blocks)
    for cls in ("Y_down", "dX_gate", "G_o", "dQ"):
        _flip_case(setup, monkeypatch, _layer_target(setup.c, layer, cls))


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_a_flip_on_every_switch(switch, monkeypatch):
    setup = proved(switch, 2)
    _flip_case(setup, monkeypatch, _layer_target(setup.c, 2, "dX_down"))
    _flip_case(setup, monkeypatch, _head_target(setup.c, "G", 1))


# ---- windows --------------------------------------------------------------------------------

def _windows(c, size):
    """The judging windows as ``(first m, last m)`` pairs, by the rule check 5 applies: a
    window ends after the run that fills it."""
    out, first, pending = [], None, 0
    for run in checks._member_runs(c.products):
        first = run[0].m if first is None else first
        pending += len(run)
        if pending >= size:
            out.append((first, run[-1].m))
            first, pending = None, 0
    if first is not None:
        out.append((first, c.products[-1].m))
    return out


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", 8)
    return 8


def test_the_windows_are_several_and_cover_the_products(small_window):
    c = proved(CHEAP, 2).c
    w = _windows(c, small_window)
    assert len(w) > 5 and w[0][0] == 1 and w[-1][1] == c.M
    assert all(a[1] + 1 == b[0] for a, b in zip(w, w[1:]))


def test_the_honest_step_with_small_windows(small_window, monkeypatch):
    setup = proved(CHEAP, 2)
    unit, canon = both(setup, monkeypatch)
    assert canon.rejection is None
    assert_same(unit, canon)


@pytest.mark.parametrize("where", ["w0-first", "w0-last", "w1-first", "w1-last", "mid-first",
                                   "mid-last", "last-first", "m=M"])
@pytest.mark.parametrize("blocks", BLOCKS)
def test_a_flip_at_a_windows_first_and_last_product(where, blocks, small_window, monkeypatch):
    setup = proved(CHEAP, blocks)
    w = _windows(setup.c, small_window)
    idx, end = where.split("-") if "-" in where else ("m=M", "last")
    if where == "m=M":
        m = setup.c.M
        assert m == w[-1][1]
    else:
        wi = {"w0": 0, "w1": 1, "mid": len(w) // 2, "last": len(w) - 1}[idx]
        m = w[wi][0 if end == "first" else 1]
    _flip_case(setup, monkeypatch, m)


def test_two_flips_in_different_windows(small_window, monkeypatch):
    setup = proved(CHEAP, 2)
    w = _windows(setup.c, small_window)
    unit, canon = both(setup, monkeypatch, leaves=_flipped(setup, w[1][1], w[-2][0]))
    assert canon.rejection is not None
    assert_same(unit, canon)


# ---- serving errors -------------------------------------------------------------------------

class ServingFault:
    """Patch ``operands``, ``operand_group`` and ``operand_blocks`` of ``DecoderReplay`` so
    that serving product ``m`` raises ``make()``, on every replay (a fresh one too), or only on
    the first replay that serves (``first_only``, keyed on the replay object; bound once, so
    if it dies no later replay matches)."""

    def __init__(self, mp, m, make, first_only=False):
        self.m, self.make, self.first_only = m, make, first_only
        self.first = None  # weakref to the first replay that served; bound once, never re-bound
        self.fired = 0
        for name, arity in (("operands", "one"), ("operand_group", "many"),
                            ("operand_blocks", "many")):
            mp.setattr(DecoderReplay, name, self._wrap(getattr(DecoderReplay, name), arity))

    def _wrap(self, real, arity):
        def serve(replay, ms, *args, **kw):
            if self.first is None:
                self.first = weakref.ref(replay)
            hit = self.m == ms if arity == "one" else self.m in ms
            if hit and (not self.first_only or self.first() is replay):
                self.fired += 1
                raise self.make()
            return real(replay, ms, *args, **kw)
        return serve


def _targets(c, flip_first):
    """``(serving target, an earlier canonical product to flip)``."""
    B = c.head_blocks
    y_q, lam, g_v = 1, _head_target(c, "Lambda", B - 1), c.M
    assert c.product(g_v).name == "L1.G_v"
    return {"L1.Y_q": (y_q, None), "Lambda_B-1": (lam, _m(c, "L1.Y_k")),
            "L1.G_v": (g_v, _m(c, "L2.Y_q"))}


ERRORS = {
    "ReplayError": lambda: ReplayError("serving fault: the replay's model run does not match C"),
    "StoreMutationError": lambda: StoreMutationError("serving fault: leaf changed in place"),
}
SERVING_CASES = [(t, flip) for t in ("L1.Y_q", "Lambda_B-1", "L1.G_v") for flip in (False, True)
                 if not (t == "L1.Y_q" and flip)]


@pytest.mark.parametrize("error", list(ERRORS))
@pytest.mark.parametrize("target,with_flip", SERVING_CASES,
                         ids=[f"{t}{'+flip' if f else ''}" for t, f in SERVING_CASES])
@pytest.mark.parametrize("blocks", BLOCKS)
def test_a_serving_error_ends_both_paths_alike(blocks, target, with_flip, error, monkeypatch):
    setup = proved(CHEAP, blocks)
    m, earlier = _targets(setup.c, True)[target]
    fault = ServingFault(monkeypatch, m, ERRORS[error])
    leaves = _flipped(setup, earlier) if with_flip else None
    unit, canon = both(setup, monkeypatch, leaves=leaves)
    assert fault.fired >= 2  # both paths served it
    if with_flip:
        assert canon.rejection is not None and canon.rejection[1] == "5"
        # the error is left to the record pass, which raises it (a malformed rejection for a
        # TranscriptFormatError: the driver would end the pass as "ended: malformed")
        if error == "ReplayError":
            assert canon.run_rest_error == _err(ERRORS[error]())
            assert canon.record.startswith(checks.RECORD_ERROR)
        else:
            assert canon.run_rest_error[0] == "_Rejected"
            assert "StoreMutationError" in canon.run_rest_error[1]
            assert canon.record == checks.RECORD_MALFORMED
    elif error == "ReplayError":
        assert canon.error == _err(ERRORS[error]())  # the step raises, on both paths alike
    else:
        assert canon.rejection is not None and canon.rejection[3] == "malformed"
        assert "StoreMutationError" in canon.rejection[2]
    assert_same(unit, canon)


# ---- an error on the first replay only: the unit path falls back ---------------------------

@pytest.mark.parametrize("target", ["L1.Y_q", "Lambda_B-1", "L1.G_v"])
def test_an_error_on_the_first_replay_only_ends_as_the_honest_canonical_step(target,
                                                                              monkeypatch):
    """Needs the code: the unit path's first replay raises, check 5 falls back to a fresh
    replay and the canonical loop, and ends exactly as the honest canonical step."""
    setup = proved(CHEAP, 2)
    m, _ = _targets(setup.c, False)[target]
    honest_canon = both(setup, monkeypatch)[1]
    fault = ServingFault(monkeypatch, m, ERRORS["ReplayError"], first_only=True)
    unit = run_check5(setup)
    assert fault.fired == 1
    assert unit.rejection is None and unit.error is None
    assert_same(unit, honest_canon)
    unit_hex = unit.ctx.state.replay
    assert unit_hex is not None and fault.first() is not unit_hex  # a fresh replay is kept


def test_a_first_replay_error_calls_the_canonical_loop_once(monkeypatch):
    """Needs the code: ``_check_5_canonical`` runs once, after the fallback, and not on an
    honest step."""
    setup = proved(CHEAP, 2)
    calls = []
    real = checks._check_5_canonical  # AttributeError until the code lands
    monkeypatch.setattr(checks, "_check_5_canonical",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    assert run_check5(setup).rejection is None and calls == []
    ServingFault(monkeypatch, _m(setup.c, "L1.Y_q"), ERRORS["ReplayError"], first_only=True)
    assert run_check5(setup).rejection is None and calls == [1]


@pytest.mark.parametrize("target", ["L1.Y_q", "L1.G_v"])
def test_after_a_fallback_the_first_replay_is_unreachable(target, monkeypatch):
    """Needs the code: once check 5 returns, nothing keeps the replay that raised alive (its
    parameter storage is replaced on the fresh one, and no hook or traceback holds it)."""
    setup = proved(CHEAP, 2)
    m, _ = _targets(setup.c, False)[target]
    fault = ServingFault(monkeypatch, m, ERRORS["ReplayError"], first_only=True)
    result = run_check5(setup)
    assert result.error is None and fault.fired == 1 and fault.first is not None
    gc.collect()
    assert fault.first() is None, "the first replay is still alive after the fallback"


# ---- a measuring error, a malformed leaf, a bumped version ----------------------------------

@pytest.mark.parametrize("with_flip", [False, True])
@pytest.mark.parametrize("target", ["L1.Y_q", "Lambda_B-1", "L1.G_v"])
def test_a_measuring_error_ends_both_paths_alike(target, with_flip, monkeypatch):
    setup = proved(CHEAP, 2)
    c = setup.c
    m, earlier = _targets(c, True)[target]
    if with_flip and earlier is None:
        pytest.skip("nothing earlier than the first product")
    real = checks._measure_rows
    fired = []

    def measure_rows(specs, *a, **kw):
        if any(s.m == m for s in specs):
            fired.append(1)
            raise RuntimeError("measuring fault")
        return real(specs, *a, **kw)

    monkeypatch.setattr(checks, "_measure_rows", measure_rows)
    unit, canon = both(setup, monkeypatch, leaves=_flipped(setup, earlier) if with_flip else None)
    assert len(fired) >= 2
    if with_flip:
        assert canon.rejection is not None and canon.run_rest_error == ("RuntimeError",
                                                                        "measuring fault")
    else:
        assert canon.error == ("RuntimeError", "measuring fault")
    assert_same(unit, canon)


def _malformed(leaf):
    """A leaf of the right kind and a wrong shape (its hash would differ; check 2 is not here)."""
    return leaf.reshape(-1)[:-1].clone()


@pytest.mark.parametrize("name", ["L1.Y_q", "L1.S[1,3]", "L2.dX_down", "Lambda[1]", "L1.G_v"])
def test_a_malformed_leaf_is_rejected_alike(name, monkeypatch):
    setup = proved(CHEAP, 2)
    m = _m(setup.c, name)
    g = setup.c.leaf_of(m)[0]
    # a whole leaf (every attention member too: one leaf, O18) loses its last entry
    leaves = list(setup.leaves)
    j = setup.c.product_leaf_index(g)
    leaves[j] = _malformed(leaves[j])
    unit, canon = both(setup, monkeypatch, leaves=leaves)
    assert canon.rejection is not None or canon.error is not None
    assert_same(unit, canon)


@pytest.mark.parametrize("name", ["L1.Y_q", "L2.O[0,1]", "L2.G_down", "Lambda[0]", "L1.G_v"])
def test_a_leaf_whose_version_is_bumped_after_check_2_is_rejected_alike(name, monkeypatch):
    setup = proved(CHEAP, 2)
    i = setup.c.product_leaf_index(setup.c.leaf_of(_m(setup.c, name))[0])
    unit, canon = both(setup, monkeypatch, bump=(i,))
    assert canon.rejection is not None and canon.rejection[3] == "malformed"
    assert "StoreMutationError" in canon.rejection[2]
    assert_same(unit, canon)


def test_a_bumped_weight_leaf_is_rejected_alike(monkeypatch):
    """A weight leaf the replay reads for many products (here ``W_q`` of layer 1)."""
    setup = proved(CHEAP, 2)
    c = setup.c
    name = next(n for n in c.linear_weights if "layers.0." in n and "q_proj" in n)
    unit, canon = both(setup, monkeypatch, bump=(c.w_t_index(name),))
    assert canon.rejection is not None or canon.error is not None
    assert_same(unit, canon)


# ---- the unit path itself (needs the code) -------------------------------------------------

def _count(mp, cls, name):
    seen = Counter()
    real = getattr(cls, name)

    def wrapped(self, arg=None, *a, **k):
        seen[arg] += 1
        return real(self, arg, *a, **k) if arg is not None else real(self)

    mp.setattr(cls, name, wrapped)
    return seen


@pytest.mark.parametrize("blocks", BLOCKS)
def test_the_unit_path_is_taken_and_builds_and_runs_each_unit_once(blocks, monkeypatch):
    """Needs the code. On an honest step the replay builds its forward pass once, runs each
    layer's and each head block's backward once, check 5 measures in the schedule's order, and
    the canonical loop is not entered."""
    setup = proved(CHEAP, blocks)
    c = setup.c
    builds = []
    real_build = DecoderReplay._build
    monkeypatch.setattr(DecoderReplay, "_build",
                        lambda self: (builds.append(self), real_build(self))[1])
    units = _count(monkeypatch, DecoderReplay, "_run_unit")
    heads = _count(monkeypatch, DecoderReplay, "_run_head_block")
    canonical = []
    real_canon = getattr(checks, "_check_5_canonical", None)
    monkeypatch.setattr(checks, "_check_5_canonical",
                        lambda *a, **k: canonical.append(1) or real_canon(*a, **k), raising=False)
    order = []
    real_measure = checks._measure_served

    def measure_served(c_, specs, *a, **k):
        order.extend(s.m for s in specs)
        return real_measure(c_, specs, *a, **k)

    monkeypatch.setattr(checks, "_measure_served", measure_served)
    result = run_check5(setup)
    assert result.rejection is None and result.error is None
    assert len(builds) == 1, "the forward pass is built once"
    assert dict(units) == {1: 1, 2: 1} and dict(heads) == {b: 1 for b in range(blocks)}
    assert canonical == []
    sched = builds[0].unit_schedule()  # AttributeError until the code lands
    assert sched is not None
    flat = [m for step in sched for m in step]
    assert sorted(flat) == list(range(1, c.M + 1))
    assert order == flat, "check 5 measured in the schedule's order"
    assert order != sorted(order)


def test_the_decoders_schedule_is_the_documented_one():
    """Needs the code: head steps ``(Λ_b, δF_b, G_b)`` first, then each layer ``L … 1``, its
    forward products then its backward products, each canonical."""
    setup = proved(CHEAP, 2)
    c = setup.c
    sched = run_check5(setup).ctx.state.replay.unit_schedule()  # AttributeError before the code
    assert sched is not None
    heads = [tuple(_head_target(c, k, b) for k in HEAD_KINDS) for b in range(c.head_blocks)]
    assert list(sched[:c.head_blocks]) == heads
    rest = sched[c.head_blocks:]
    assert len(rest) == 2
    for step, layer in zip(rest, (2, 1)):
        own = [s for s in c.products if s.layer == layer]
        fwd = [s.m for s in own if s.kind.name == "FORWARD"]
        bwd = [s.m for s in own if s.kind.name != "FORWARD"]
        assert list(step) == fwd + bwd


@pytest.mark.parametrize("window", [8, 1024])
def test_a_rejection_leaves_the_later_windows_rows_to_the_record_pass(window, monkeypatch):
    """Needs the code: after a rejection in window 0, the unit path holds the later windows'
    measured rows in ``run_rest_rows`` (none with one window), the record pass finishes them
    and empties it, and ends with the canonical path's stats."""
    monkeypatch.setattr(checks, "_JUDGE_WINDOW", window)
    setup = proved(CHEAP, 2)
    unit = run_check5(setup, leaves=_flipped(setup, 1))
    assert unit.rows_rest is not None, "StepState has no run_rest_rows"  # until the code lands
    assert (unit.rows_rest > 0) == (window == 8)
    assert unit.ctx.state.run_rest_rows is not None and not unit.ctx.state.run_rest_rows
    assert unit.record == checks.RECORD_COMPLETE
    assert [s[0] for s in unit.record_stats] == list(range(1, setup.c.M + 1))


def test_the_record_pass_serves_nothing_again_after_a_unit_rejection(small_window, monkeypatch):
    """Needs the code: the unit path measures every run in check 5, so after a rejection the
    record pass calls ``_measure_served`` zero times."""
    setup = proved(CHEAP, 2)
    unit_runs = []
    real = checks._measure_served

    def spy(c_, specs, ctx, replay, view, *a, **k):
        unit_runs.append(k.get("run"))
        return real(c_, specs, ctx, replay, view, *a, **k)

    monkeypatch.setattr(checks, "_measure_served", spy)
    run_check5(setup, leaves=_flipped(setup, 1))
    assert None not in unit_runs, "the record pass measured a run again"
    assert len(unit_runs) == len(list(checks._member_runs(setup.c.products)))


# ---- sharded CPU placement ------------------------------------------------------------------

THREE = ("cpu", "cpu", "cpu")


@pytest.mark.parametrize("flip", [None, ("layer", 2, "dX_down"), ("layer", 1, "G_o"),
                                  ("head", "Lambda", 1)])
def test_sharded_cpu_placement_equals_unsharded_on_the_unit_path(flip, monkeypatch):
    plain, three = proved(CHEAP, 2), proved(CHEAP, 2, THREE)
    assert three.c.placement.sharded and not plain.c.placement.sharded

    def leaves(s):
        if flip is None:
            return None
        m = (_layer_target(s.c, flip[1], flip[2]) if flip[0] == "layer"
             else _head_target(s.c, flip[1], flip[2]))
        return _flipped(s, m)

    a = run_check5(plain, leaves=leaves(plain))
    b = run_check5(three, leaves=leaves(three))
    assert_same(a, b)
    if flip is not None:
        assert a.rejection is not None
    with pytest.MonkeyPatch.context() as cm:
        cm.setattr(DecoderReplay, "unit_schedule", lambda self: None, raising=False)
        assert_same(run_check5(three, leaves=leaves(three)), b)  # and on the canonical path


# ---- the whole verifier ----------------------------------------------------------------------

def _whole(setup, path_unit, **kw):
    c, D, tree, w0 = setup.c, setup.D, setup.tree, setup.w0
    idx = schedule(1, c.n_s, len(D), SEED)
    v = Verifier(c, h_D=tree.root, n_records=len(D), k=7, n_steps=1, bands=Bands.provisional(),
                 allow_provisional=True, w0=w0, seed=SEED)
    assert v.start_run(D) is None
    out, _ = _prove(c, D, w0, **kw)
    store = InMemoryStore.from_step(c, out, dataset_paths=[tree.path(i) for i in idx])
    with pytest.MonkeyPatch.context() as cm:
        if not path_unit:
            cm.setattr(DecoderReplay, "unit_schedule", lambda self: None, raising=False)
        rej = v.verify_step(1, store)
    return (_rej(rej), v.record_pass.get(1), [_stat_sig(s) for s in v.stats[1].products],
            [(t.weight, t.check_id, _f(t.rho_max)) for t in v.stats[1].tensors])


@pytest.mark.parametrize("flip", [None, "L2.dX_down", "L1.dQ[1,3]", "Lambda[1]"])
def test_the_whole_verifier_agrees_on_both_paths(flip):
    setup = proved(CHEAP, 2)
    kw = {} if flip is None else dict(perturb={_m(setup.c, flip): _flip})
    unit, canon = _whole(setup, True, **kw), _whole(setup, False, **kw)
    assert (unit[0] is None) == (flip is None)
    assert unit == canon


# ---- the kernel path through the Mac emulation ----------------------------------------------

def _rows_by_m(rec):
    rows, out = rec.rows(), {}
    for mem, row in zip(rec.members, rows):
        key = row.tobytes()
        assert out.setdefault(mem["m"], key) == key, "a product measured twice with other bits"
    return out


def _kernel_pair(setup, install_engine, flip_m=None):
    """``(unit, canonical)`` results and rows by product, with the kernel path on."""
    got = []
    for unit in (True, False):
        with pytest.MonkeyPatch.context() as cm:
            rec = install_engine(cm)
            if not unit:
                cm.setattr(DecoderReplay, "unit_schedule", lambda self: None, raising=False)
            res = run_check5(setup, leaves=None if flip_m is None else _flipped(setup, flip_m))
            assert rec.calls, "the kernel path measured nothing"
            got.append((res, _rows_by_m(rec)))
    return got


@pytest.mark.parametrize("flip", [None, ("layer", 2, "dX_down"), ("head", "G", -1)])
@pytest.mark.parametrize("blocks", BLOCKS)
@pytest.mark.parametrize("switch", [("llama", True), ("qwen2", False)], ids=switch_id)
def test_kernel_path_mac_emulation_rows_and_stats_equal(switch, blocks, flip, hostlib,
                                                         phased):  # noqa: F811
    setup = proved(switch, blocks)
    m = None
    if flip is not None:
        m = (_layer_target(setup.c, flip[1], flip[2]) if flip[0] == "layer"
             else _head_target(setup.c, flip[1], flip[2] % setup.c.head_blocks))
    (unit, urows), (canon, crows) = _kernel_pair(setup, lambda mp: install(mp, "mac", hostlib), m)
    assert sorted(urows) == sorted(crows) == list(range(1, setup.c.M + 1))
    assert urows == crows  # every row, bit for bit
    assert (canon.rejection is None) == (flip is None)
    assert_same(unit, canon)


@pytest.mark.parametrize("flip", [None, ("layer", 1, "dV")])
def test_kernel_path_stand_in_rows_and_stats_equal(flip):
    """The stand-in kernel of ``test_kernel_views`` (each member measured alone)."""
    from tests.verification.verifier.test_kernel_step import Recorder

    setup = proved(CHEAP, 2)
    m = None if flip is None else _layer_target(setup.c, flip[1], flip[2])

    def stand_in(mp):
        kern = Kernel()
        mp.setattr(freivalds, "kernel_device", lambda d: freivalds._reference_depth == 0)
        mp.setattr(freivalds, "measure_groups_rows", kern.measure)
        mp.setattr(freivalds, "drain", kern.drain)
        rec = Recorder()
        mp.setattr(freivalds, "measure_groups_rows", rec.wrap(kern.measure))
        return rec

    (unit, urows), (canon, crows) = _kernel_pair(setup, stand_in, m)
    assert urows == crows and sorted(urows) == list(range(1, setup.c.M + 1))
    assert_same(unit, canon)
