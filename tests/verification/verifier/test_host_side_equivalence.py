"""FS50t: check 5's per-run cache (fix A) changes no number.

Written from ``docs/verification/HOST_SIDE_PLAN_DRAFT.md`` §2.1–§2.3 alone. Every test that needs
a coder's code looks it up by name and skips with its reason until that code is merged:
``checks.check5_static_cache(enabled)`` (a context manager, like ``measure_kernel.overlap``;
FS50), ``measure_kernel.table_cache(enabled)`` and a plan-rebuild counter (FS51). The tests of
today's behaviour run now.

- an honest decoder run (two steps in one verifier, so step 2 hits the cache) gives the same
  ``verifier.stats`` (every column, bit for bit), the same verdict, with the cache on and off;
- a perturbed product (the prover's ``perturb`` hook) gives the same ``Rejection`` and the same
  ``StepState.run_rest`` and ``run_rest_rows``, with the cache on and off;
- the measuring tables' rows (the host simulation) are the same with ``table_cache`` on and off,
  also after a shared view moved.
"""

import contextlib
import dataclasses
from unittest import mock

import numpy as np
import pytest
import torch

from setup.data import schedule
from verification.commitment.leaves import dataset_tree
from verification.computation.instances import DecoderComputation
from verification.prover.step import prove_step
from verification.transcript.store import InMemoryStore
from verification.verifier import checks
from verification.verifier.bands import Bands
from verification.verifier.checks import RECORD_NOT_RUN, check_2_commitment, check_5_matmuls
from verification.verifier.context import StepContext
from verification.verifier.driver import Verifier
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check.measure_table import MeasureInput, pack  # noqa: F401

from tests.verification.decoder_helpers import (
    SWITCHES, make_records, randomize_biases, switch_id, tiny_config)
from tests.verification.verifier.matmul_check.sets_tables import shared_view
from tests.verification.verifier.matmul_check.shared_tables import DTYPES as TABLE_DTYPES
from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    bits,
    hostlib,
    host_measure,
)
from tests.verification.verifier.matmul_check.test_measure_kernel_host import levels, phased  # noqa: F401

SEED = 1
K = 7
ETA = 1e-3
COMPUTE = {"fp32": torch.float32, "bf16": torch.bfloat16}


# ---- the toggles of the coders' code, looked up by name -------------------------------------


@contextlib.contextmanager
def check5_cache(enabled: bool):
    """FS50's bypass (``checks.check5_static_cache(enabled)``): skips the test while absent."""
    fn = getattr(checks, "check5_static_cache", None)
    if fn is None:
        pytest.skip("FS50 not merged: checks.check5_static_cache is absent")
    cm = fn(enabled)
    if not hasattr(cm, "__enter__"):
        pytest.skip("checks.check5_static_cache(enabled) is not a context manager")
    with cm:
        yield


@contextlib.contextmanager
def table_cache(enabled: bool):
    """FS51's ``measure_kernel.table_cache(enabled)``: skips the test while absent."""
    fn = getattr(mk, "table_cache", None)
    if fn is None:
        pytest.skip("FS51 not merged: measure_kernel.table_cache is absent")
    with fn(enabled):
        yield


_REBUILD_NAMES = ("plan_rebuilds", "plan_rebuild_count", "rebuilds", "PLAN_REBUILDS")


def plan_rebuilds() -> int:
    """FS51's plan-rebuild counter, by ``getattr`` (a number or a callable); skips if absent."""
    for name in _REBUILD_NAMES:
        v = getattr(mk, name, None)
        if v is not None:
            return int(v() if callable(v) else v)
    pytest.skip("FS51 not merged: no plan-rebuild counter in measure_kernel "
                f"(looked for {', '.join(_REBUILD_NAMES)})")


# ---- a tiny decoder run -----------------------------------------------------------------------


def bump(t, index=-1):
    """``t`` with one entry raised by 1, by default the last, which the causal mask keeps."""
    t = t.clone()
    t.view(-1)[index] += 1.0
    return t


@dataclasses.dataclass
class Env:
    c: DecoderComputation
    D: list
    tree: object
    w0: dict


def make_env(switch, dtype="fp32", *, head_blocks=1) -> Env:
    family, tied = switch
    torch.manual_seed(0)  # the weights and biases, whatever tests ran before
    c = DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=ETA,
                           compute_dtype=COMPUTE[dtype], head_blocks=head_blocks)
    w0 = {n: p.detach().clone() for n, p in c.build_model().named_parameters()}
    randomize_biases(w0)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7), seed=1)
    return Env(c, D, dataset_tree(c, D), w0)


def prove(env: Env, n_steps: int, perturb: dict[int, dict] | None = None):
    """``[(StepOutput, indices)]`` for ``n_steps`` consecutive steps; ``perturb`` maps a step to
    the prover's ``{m: fn}`` hook for it."""
    outs, w = [], env.w0
    for t in range(1, n_steps + 1):
        idx = schedule(t, env.c.n_s, len(env.D), SEED)
        out = prove_step(env.c, env.c.build_model(), w, [env.D[i] for i in idx],
                         perturb=(perturb or {}).get(t))
        outs.append((out, idx))
        w = out.w_next
    return outs


def verify(env: Env, outs, *, record: bool = True):
    """One verifier over the steps: ``(verifier, [rejection per step run])``; it stops at the
    first rejection. ``record=False`` skips the record pass after a rejection."""
    v = Verifier(env.c, h_D=env.tree.root, n_records=len(env.D), k=K, n_steps=len(outs),
                 bands=Bands.provisional(), allow_provisional=True, w0=env.w0, seed=SEED)
    assert v.start_run(env.D) is None
    rejs = []
    patch = (contextlib.nullcontext() if record else
             mock.patch.object(Verifier, "_record_rest", return_value=RECORD_NOT_RUN))
    with patch:
        for t, (out, idx) in enumerate(outs, 1):
            store = InMemoryStore.from_step(env.c, out, dataset_paths=[env.tree.path(i) for i in idx])
            rejs.append(v.verify_step(t, store))
            if rejs[-1] is not None:
                break
    return v, rejs


def f64_bits(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).view(np.int64)


def stat_arrays(stats) -> dict:
    """Every column of a step's stats as arrays (floats as their int64 bits, so a NaN payload
    and a signed zero count)."""
    cols = stats.products.columns()
    return {"m": np.asarray(cols.m), "q": np.asarray(cols.q),
            "name": list(cols.name), "cls": list(cols.cls),
            **{f: f64_bits(getattr(cols, f)) for f in
               ("kappa", "normalized", "p_norm", "nu", "p_abs1", "u")},
            "t_weight": [s.weight for s in stats.tensors],
            "t_check": [s.check_id for s in stats.tensors],
            "t_rho": f64_bits([s.rho_max for s in stats.tensors])}


def same_stats(a, b):
    got, want = stat_arrays(a), stat_arrays(b)
    assert got.keys() == want.keys()
    for key in got:
        if isinstance(got[key], np.ndarray):
            assert got[key].shape == want[key].shape, key
            assert np.array_equal(got[key], want[key]), key
        else:
            assert got[key] == want[key], key


def same_verifiers(a, ra, b, rb):
    assert ra == rb  # step, check, detail, kind
    assert a.stats.keys() == b.stats.keys()
    for t in a.stats:
        same_stats(a.stats[t], b.stats[t])
    assert a.record_pass == b.record_pass


def ids(sw, dtype):
    return f"{switch_id(sw)}-{dtype}"


CASES = [pytest.param(sw, dt, id=ids(sw, dt)) for sw in SWITCHES for dt in COMPUTE]


# ---- 1. fix A: judging, the whole verifier ------------------------------------------------------


@pytest.mark.parametrize("switch,dtype", CASES)
def test_two_honest_steps_are_accepted_and_deterministic(switch, dtype):
    """Today's behaviour: two honest steps in one verifier are accepted, and a second verifier
    over the same steps records the same stats bit for bit."""
    env = make_env(switch, dtype)
    outs = prove(env, 2)
    a, ra = verify(env, outs)
    b, rb = verify(env, outs)
    assert ra == [None, None] and rb == [None, None]
    assert [p.m for p in a.stats[2].products] == list(range(1, env.c.M + 1))
    same_verifiers(a, ra, b, rb)


@pytest.mark.parametrize("switch,dtype", CASES)
def test_honest_run_cache_on_and_off_give_the_same_stats(switch, dtype):
    env = make_env(switch, dtype)
    outs = prove(env, 2)
    with check5_cache(True):
        on, r_on = verify(env, outs)  # step 1 fills the cache, step 2 hits it
    with check5_cache(False):
        off, r_off = verify(env, outs)
    assert r_on == [None, None]
    same_verifiers(on, r_on, off, r_off)


@pytest.mark.parametrize("record", [True, False], ids=["record-pass", "no-record-pass"])
@pytest.mark.parametrize("switch,dtype", CASES)
def test_a_perturbed_product_is_rejected_the_same_with_the_cache_on_and_off(switch, dtype, record):
    """Step 1 honest (fills the cache), step 2 with a perturbed product: the same rejection
    (step, check, detail, kind) and the same recorded stats, whole record pass or not."""
    env = make_env(switch, dtype)
    m = env.c.m_of("L1.Y_q")
    outs = prove(env, 2, perturb={2: {m: bump}})
    with check5_cache(True):
        on, r_on = verify(env, outs, record=record)
    with check5_cache(False):
        off, r_off = verify(env, outs, record=record)
    assert r_on[0] is None and r_on[1] is not None
    assert (r_on[1].step, r_on[1].check_id, r_on[1].kind) == (2, "5", "failed")
    # At bf16 the first product to fail can be one fed by the bumped one (a later product reads
    # the bumped value through the glue); which one is the same with the cache on and off.
    same_verifiers(on, r_on, off, r_off)


def _direct(env: Env, out, *, judge=True):
    """Check 5 alone on one step's output, as ``test_check_5_unit_pass`` does: the rejection and
    the context (``stats``, ``state.run_rest``, ``state.run_rest_rows``), no record pass."""
    store = InMemoryStore.from_step(env.c, out, copy=True)
    ctx = StepContext.for_computation(
        env.c, step=2, indices=tuple(range(env.c.n_s)), h_D=bytes(32), n_records=env.c.n_s,
        prev_w_hashes=(), chain_check_id="0", k=K, judge=judge)
    assert check_2_commitment(store, env.c, ctx, Bands.provisional()) is None
    return check_5_matmuls(store, env.c, ctx, Bands.provisional()), ctx


def _held(ctx):
    """What a rejecting check 5 leaves for the record pass, as comparable values."""
    rest = [(s.m, repr(dataclasses.astuple(mp))) for s, mp in ctx.state.run_rest]
    rows = [(r.order if hasattr(r, "order") else None, np.asarray(a).tobytes())
            for r, a in ctx.state.run_rest_rows]
    return rest, [(None if o is None else list(o), b) for o, b in rows], \
        repr(ctx.state.run_rest_error)


# first, middle and last of the step's products, a head product, a weight-free and a bias one
PLANTS = ["L1.Y_q", "Lambda", "L2.G_down", "L1.O[1,2]", "L1.G_v", "L1.S[0,1]", "L2.dK[1,3]"]


@pytest.mark.parametrize("window", [1, 20, None], ids=["w1", "w20", "wdefault"])
@pytest.mark.parametrize("plant", PLANTS)
def test_a_planted_product_leaves_the_same_run_rest_with_the_cache_on_and_off(plant, window,
                                                                              monkeypatch):
    if window is not None:
        monkeypatch.setattr(checks, "_JUDGE_WINDOW", window)
    env = make_env(("llama", True))
    m = env.c.m_of(plant)
    out = prove(env, 1, perturb={1: {m: bump}})[0][0]
    with check5_cache(True):
        _direct(env, prove(env, 1)[0][0])  # an honest pass first: the cache is warm
        rej_on, ctx_on = _direct(env, out)
    with check5_cache(False):
        rej_off, ctx_off = _direct(env, out)
    assert rej_on is not None and rej_on == rej_off
    assert f"P_{m} " in rej_on.detail
    assert _held(ctx_on) == _held(ctx_off)
    same_stats(ctx_on.stats, ctx_off.stats)


@pytest.mark.parametrize("switch,dtype", [CASES[0], CASES[3], CASES[7]])
def test_run_rest_after_a_rejection_equals_with_the_cache_on_and_off(switch, dtype):
    """The same on the other families: a backward product's rejection and what it holds."""
    env = make_env(switch, dtype)
    m = env.c.m_of("L1.dX_q")
    out = prove(env, 1, perturb={1: {m: bump}})[0][0]
    with check5_cache(True):
        rej_on, ctx_on = _direct(env, out)
    with check5_cache(False):
        rej_off, ctx_off = _direct(env, out)
    assert rej_on is not None and rej_on == rej_off
    assert _held(ctx_on) == _held(ctx_off)
    same_stats(ctx_on.stats, ctx_off.stats)


def test_head_blocks_and_a_second_computation_object_have_their_own_entry():
    """Two equal-shaped computations with different head blocking, and a second object of the
    same declaration, each give the numbers of a cache-off run."""
    envs = [make_env(("llama", True), head_blocks=1), make_env(("llama", True), head_blocks=2),
            make_env(("llama", True), head_blocks=1)]
    runs = []
    for env in envs:
        outs = prove(env, 2)
        with check5_cache(True):
            on = verify(env, outs)
        with check5_cache(False):
            off = verify(env, outs)
        same_verifiers(on[0], on[1], off[0], off[1])
        runs.append(on[0])
    # the two head-blocking choices are distinct declarations; the third equals the first
    same_stats(runs[0].stats[2], runs[2].stats[2])


# ---- 2. fix A: the measuring tables (the host simulation) --------------------------------------


def rows_with(levels, mis, reverse=False):
    return levels(mis, reverse=reverse)


def random_group(rng: np.random.Generator, dtype: torch.dtype, seed: int):
    """``(spec, build)``: the structure of one random group and ``build(seed)``, its tensors
    for a seed (fresh storages, same structure)."""
    e = int(rng.integers(1, 65))
    mode = ("A", "B", "mix")[int(rng.integers(0, 3))]
    n, q = int(rng.integers(1, 41)), int(rng.integers(1, 41))
    layout = ("row", "col")[int(rng.integers(0, 2))]
    guard = bool(rng.integers(0, 2))
    bias_at = None if mode == "B" or rng.integers(0, 2) else 0
    spec = dict(e=e, mode=mode, n=n, q=q, layout=layout, guard=guard, bias_at=bias_at)

    def build(s):
        return shared_view(e, mode, n=n, q=q, dtype=dtype, guard=guard, k=3, layout=layout,
                           seed=s, bias_at=bias_at)
    return spec, build


def move_one_view(mis, x, j):
    """``mis`` with product ``j``'s read of the shared ``x`` replaced by a clone of it: same
    values and strides, another address, so the share no longer holds."""
    out = []
    for i, mi in enumerate(mis):
        if i == j:
            xc = x.clone()
            assert xc.stride() == x.stride()
            mi = MeasureInput(xc if mi.a is x else mi.a, xc if mi.b is x else mi.b, mi.p, mi.r,
                              mi.bias, mi.guard)
        out.append(mi)
    return out


@pytest.mark.parametrize("dtype", TABLE_DTYPES)
@pytest.mark.parametrize("seed", range(6))
def test_rows_are_the_same_with_the_table_cache_on_and_off(levels, phased, dtype, seed):
    """Random groups (1-64 members, shapes 1-40, row-major and transposed views, a biased
    single product, share on and off): two calls with fresh tensors of the same structure, then
    one shared view moved; every rows array equals the cache-off build's bit for bit."""
    rng = np.random.default_rng(1000 + seed)
    spec, build = random_group(rng, dtype, seed)
    with table_cache(False):
        mis1, x1 = build(1)
        mis2, x2 = build(2)
        want1, want2 = bits(levels(mis1)), bits(levels(mis2))
        plain1, plain2 = bits(phased(mis1)), bits(phased(mis2))  # share off (the plain plan)
        j = int(rng.integers(0, len(mis2)))
        want_moved = bits(levels(move_one_view(mis2, x2, j)))
    with table_cache(True):
        mis1, x1 = build(1)
        mis2, x2 = build(2)
        before = plan_rebuilds() if any(hasattr(mk, n) for n in _REBUILD_NAMES) else None
        got1 = bits(levels(mis1))     # the cache's first call
        got2 = bits(levels(mis2))     # a hit: the same structure, other addresses
        assert np.array_equal(got1, want1) and np.array_equal(got2, want2), spec
        assert np.array_equal(bits(phased(mis1)), plain1) and np.array_equal(bits(phased(mis2)), plain2)
        moved = move_one_view(mis2, x2, j)
        got_moved = bits(levels(moved))
        assert np.array_equal(got_moved, want_moved), (spec, j)
        # the moved view carries the same values, so the rows are also the unmoved build's
        assert np.array_equal(got_moved, got2), (spec, j)
        if before is not None:
            assert plan_rebuilds() >= before  # counted, never uncounted (strictly: A40, §5.2)


@pytest.mark.parametrize("reverse", [False, True])
def test_a_permuted_group_order_gives_the_same_rows(levels, phased, reverse):
    """FS38b's rule the cached plan relies on: the order of the groups inside a level changes no
    row. The members' order is permuted and the rows put back."""
    rng = np.random.default_rng(7)
    _, build = random_group(rng, torch.float32, 0)
    mis, _ = build(3)
    perm = rng.permutation(len(mis))
    with table_cache(True):
        base = bits(levels(mis, reverse=reverse))
        shuffled = bits(levels([mis[i] for i in perm], reverse=reverse))
    back = np.empty_like(shuffled)
    back[perm] = shuffled
    assert np.array_equal(back, base)
