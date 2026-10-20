"""Check 5's vectorized judging of a window against the one-product-at-a-time path.

``_judge_window`` judges a window's passing prefix by masks and runs ``_judge_product`` from
the first failing product on. Every case runs both and compares the rejection, the recorded
stats and ``StepState.run_rest``.
"""

import math

import numpy as np
import pytest
import torch

from verification.computation.interface import ProductKind, ProductSpec
from verification.verifier import checks
from verification.verifier.bands import Bands, product_class
from verification.verifier.context import StepContext
from verification.verifier.matmul_check.freivalds import ROW

EPS = 2.0 ** -24
K = 3
BANDS = Bands(tau=8.0, kappa_max=100.0, kappa_classes={"forward:W_hot": 1.5})
GOOD = [1.0, 1.0, 1.0, 1.0, 0.0, 1e-9, 2e-9, 0.0]  # a row that passes every test


class _C:
    """Just the part of a computation ``product_class`` reads."""


def _spec(m, bias=None, weight="W"):
    return ProductSpec(m, f"P{m}", ProductKind.FORWARD, (5 + m, 40), (40, 6), weight=weight,
                       bias=bias)


def _ctx(**kw):
    return StepContext(step=4, indices=(), h_D=b"", n_records=0, prev_w_hashes=(),
                       chain_check_id="7", k=K, eps_in=EPS, eps_acc=EPS, eps_w=EPS, **kw)


def _window(n, edits, orders=None, specs=None):
    """``n`` products in canonical order split over two runs (``orders``: each run's row
    order, as run positions), with ``edits[i]`` replacing entries of product ``i``'s row: a
    :data:`ROW` name or a column index for the key."""
    specs = specs or [_spec(i + 1) for i in range(n)]
    rows = np.array([GOOD] * n)
    for i, edit in edits.items():
        for col, v in edit.items():
            rows[i, ROW.index(col) if isinstance(col, str) else col] = v
    cut = n // 2
    runs = [(specs[:cut], rows[:cut]), (specs[cut:], rows[cut:])]
    orders = orders or [list(reversed(range(cut))), list(range(n - cut))]
    batch = []
    for (sp, rw), order in zip(runs, orders):
        stored = np.empty_like(rw)
        for j, pos in enumerate(order):  # row j holds the product at run position pos
            stored[j] = rw[pos]
        batch.append(checks._Rows(sp, order, torch.from_numpy(stored)))
    return batch


def _scalar(batch, ctx, bands):
    """The window as judged before the vectorization."""
    done = checks._finish(batch, ctx)
    for i, (spec, mp) in enumerate(done):
        rej = checks._judge_product(_C(), spec, mp, ctx, bands)
        if rej is not None:
            ctx.state.run_rest = done[i + 1:]
            return rej
    return None


def _both(batch, **kw):
    a, b = _ctx(**kw), _ctx(**kw)
    got = checks._judge_window(_C(), batch, a, BANDS)
    want = _scalar(batch, b, BANDS)
    assert got == want
    assert repr(a.state.run_rest) == repr(b.state.run_rest)
    assert repr(a.stats.products) == repr(b.stats.products)  # every field, NaN included
    return got, a


def test_a_window_that_passes_is_recorded_in_bulk():
    rej, ctx = _both(_window(9, {}))
    assert rej is None and [p.m for p in ctx.stats.products] == list(range(1, 10))
    assert not ctx.state.run_rest
    scalar = _ctx()
    _scalar(_window(9, {}), scalar, BANDS)
    assert list(ctx.stats.products) == list(scalar.stats.products)


def test_the_window_comes_out_in_canonical_order_whatever_the_runs_row_order():
    edits = {i: {"p_norm": 1.0 + i} for i in range(7)}
    batch = _window(7, edits, orders=[[2, 0, 1], [3, 1, 0, 2]])
    rej, ctx = _both(batch)
    assert rej is None
    assert [p.p_norm for p in ctx.stats.products] == [1.0 + i for i in range(7)]


@pytest.mark.parametrize("at", [0, 1, 5, 8])
@pytest.mark.parametrize("edit, text", [
    ({"nu": math.nan}, "non-finite ν"),
    ({"p_abs1": math.inf}, "non-finite ‖|P|·1‖"),
    ({"p_norm": math.inf}, "non-finite ‖P‖_F"),
    ({"u2_terms": math.inf}, "non-finite U"),
    ({7: math.nan}, "non-finite residual j=3"),
    ({5: math.inf}, "non-finite residual j=1"),
    ({"nu": 1000.0}, "cancellation factor"),
    ({6: 1.0}, "challenge j=2"),
    ({7: 1.0}, "challenge j=3"),
])
def test_a_failing_product_is_rejected_as_before(edit, text, at):
    """Product ``at + 1`` fails, and the next one would fail differently."""
    edits = {at: edit, **({at + 1: {"nu": math.nan}} if at < 8 else {})}
    rej, ctx = _both(_window(9, edits))
    assert rej is not None and rej.check_id == "5" and text in rej.detail
    assert f"P_{at + 1} " in rej.detail
    assert [p.m for p in ctx.stats.products] == list(range(1, at + 2))
    assert [s.m for s, _ in ctx.state.run_rest] == list(range(at + 2, 10))


def test_the_first_failure_wins_when_one_product_fails_several_ways():
    batch = _window(6, {3: {"nu": 1000.0, 7: 1.0, 6: math.nan}, 4: {"p_norm": math.nan}})
    rej, _ = _both(batch)
    assert rej is not None and "P_4 " in rej.detail and "non-finite residual j=2" in rej.detail


def test_a_class_with_its_own_kappa_ceiling():
    specs = [_spec(i + 1, weight="W_hot" if i == 3 else "W") for i in range(6)]
    rows = {i: {"nu": 2.0, "p_abs1": 1.0} for i in (3, 4)}  # κ = 2: over W_hot's 1.5, not W's
    rej, _ = _both(_window(6, rows, specs=specs))
    assert rej is not None and "P_4 " in rej.detail and "κ_max = 1.5" in rej.detail
    assert product_class(_C(), specs[3]) == "forward:W_hot"


def test_calibration_mode_records_everything_and_judges_nothing_but_finiteness():
    rej, ctx = _both(_window(6, {2: {"nu": 1000.0, 7: 1.0}}), judge=False)
    assert rej is None and len(ctx.stats.products) == 6
    rej, ctx = _both(_window(6, {2: {"nu": 1000.0}, 4: {"p_norm": math.nan}}), judge=False)
    assert rej is not None and "non-finite" in rej.detail and "P_5 " in rej.detail


def test_with_the_kappa_guard_off_nu_is_neither_recorded_nor_judged():
    batch = _window(6, {2: {"nu": math.nan, "p_abs1": math.inf}, 4: {7: 1.0}})
    rej, ctx = _both(batch, kappa_guard=False)
    assert rej is not None and "P_5 " in rej.detail and "challenge j=3" in rej.detail
    assert all(math.isnan(p.kappa) for p in ctx.stats.products)


def test_a_biased_product_adds_its_bias_to_the_term_scale():
    specs = [_spec(i + 1, bias="b" if i in (1, 3) else None) for i in range(6)]
    rows = {1: {"bias_sq": 4.0}, 3: {"bias_sq": 9e30}}
    rej, ctx = _both(_window(6, rows, specs=specs))
    assert rej is None
    assert ctx.stats.products[1].u == math.sqrt(1.0 + 7 * 4.0)  # A has 5 + m = 7 rows
    assert ctx.stats.products[3].u == math.sqrt(1.0 + 9 * 9e30)


def test_a_bias_that_widens_the_band_keeps_a_residual_in_that_would_otherwise_fail():
    """The bias raises ``U``, so the band; the same row without a bias is rejected."""
    wide = {2: {"bias_sq": 1e18, 6: 1e-3}}
    biased = [_spec(i + 1, bias="b" if i == 2 else None) for i in range(5)]
    rej, _ = _both(_window(5, wide, specs=biased))
    assert rej is None
    rej, _ = _both(_window(5, wide, specs=[_spec(i + 1) for i in range(5)]))
    assert rej is not None and "P_3 " in rej.detail and "challenge j=2" in rej.detail


def test_an_empty_window_judges_nothing():
    ctx = _ctx()
    assert checks._judge_window(_C(), [], ctx, BANDS) is None
    assert checks._judge_window(_C(), [checks._Rows([], [], None)], ctx, BANDS) is None
    assert not ctx.stats.products
