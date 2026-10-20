"""``ProductStats.columns()`` and the consumers that read it: each gives the numbers (bit for
bit, and the same types) the loop over ``ProductStat`` objects it replaced gave, for stats built
by ``append``, by ``extend_columns`` and by both, on NaN, inf, ±0 and empty input."""

import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from verification.computation.interface import (
    DeclaredComputation,
    ProductKind,
    ProductSpec,
    default_product_class,
)
from verification.runs import metrics, run_verified as rv
from verification.verifier import calibration
from verification.verifier.bands import Bands, product_class, product_classes
from verification.verifier.context import ProductStat, ProductStats, StepStats, TensorStat

NAN, INF = math.nan, math.inf
BANDS = Bands(tau=20.0, kappa_max=4.0, kappa_classes={"c0": 12.0, "c1": 2.5}, tau_w=4.0)


def _stat(i, k=3, **kw):
    f = dict(m=i, name=f"P{i}", cls=f"c{i % 3}", kappa=1.0 + i / 7,
             normalized=tuple(i / 3 + j for j in range(k)), q=10 + i, p_norm=2.0 + i,
             nu=3.0 + i, p_abs1=4.0 + i, u=5.0 + i / 9)
    f.update(kw)
    return ProductStat(**f)


def _cols(stats):
    return dict(
        m=[s.m for s in stats], name=[s.name for s in stats], cls=[s.cls for s in stats],
        kappa=np.array([s.kappa for s in stats]),
        normalized=np.array([s.normalized for s in stats]).reshape(
            len(stats), len(stats[0].normalized) if stats else 3),
        q=[s.q for s in stats], p_norm=np.array([s.p_norm for s in stats]),
        nu=np.array([s.nu for s in stats]), p_abs1=np.array([s.p_abs1 for s in stats]),
        u=np.array([s.u for s in stats]))


def _built(want):
    """The same records stored three ways: singles, one block, and singles around blocks."""
    one = ProductStats(want)
    two = ProductStats()
    if want:
        two.extend_columns(**_cols(want))
    mix = ProductStats()
    h = len(want) // 3
    mix.extend(want[:h])
    if want[h:2 * h]:
        mix.extend_columns(**_cols(want[h:2 * h]))
    mix.extend(want[2 * h:2 * h + 1])
    if want[2 * h + 1:]:
        mix.extend_columns(**_cols(want[2 * h + 1:]))
    return {"append": one, "block": two, "mixed": mix}


def _same(a, b):
    """Equal as values, as types, and bit for bit (NaN equals NaN, -0.0 differs from 0.0)."""
    assert type(a) is type(b), (a, b)
    if isinstance(a, dict):
        assert list(a) == list(b), (a, b)
        for key in a:
            _same(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    else:
        assert repr(a) == repr(b), (a, b)


def _data():
    """Named record lists, each an edge of the old generator expressions' behaviour."""
    plain = [_stat(i) for i in range(1, 11)]
    return {
        "plain": plain,
        "single": [_stat(1)],
        "nan_first_of_a_row": [_stat(1), _stat(2, normalized=(NAN, 99.0, 1.0)), _stat(3)],
        "nan_later_in_a_row": [_stat(1), _stat(2, normalized=(1.0, NAN, 99.0)), _stat(3)],
        "nan_in_first_row_first": [_stat(1, normalized=(NAN, 5.0, 6.0)), _stat(2), _stat(3)],
        "nan_in_first_row_later": [_stat(1, normalized=(5.0, NAN, 6.0)), _stat(2)],
        "nan_kappa_first": [_stat(1, kappa=NAN)] + plain[1:],
        "nan_kappa_later": plain[:3] + [_stat(4, kappa=NAN, cls="c0")] + plain[4:],
        "all_nan_class": [_stat(1, normalized=(NAN,) * 3, kappa=NAN, cls="c1"), _stat(2, cls="c0"),
                          _stat(4, normalized=(NAN,) * 3, kappa=NAN, cls="c1"), _stat(5)],
        "inf": [_stat(1, normalized=(1.0, INF, 2.0)), _stat(2, kappa=INF), _stat(3, u=INF),
                _stat(4, normalized=(-INF, -INF, -INF)), _stat(5, p_norm=INF)],
        "zeros": [_stat(1, normalized=(0.0, 0.0, 0.0), kappa=0.0),
                  _stat(2, normalized=(0.0, 0.0, 0.0), kappa=0.0)],
        "unscored": [_stat(1, q=0), _stat(2, p_norm=NAN), _stat(3, p_norm=0.0), _stat(4, u=NAN),
                     _stat(5, p_norm=INF), _stat(6, nu=NAN, p_abs1=NAN), _stat(7)],
        "wide": [_stat(i, k=7) for i in range(1, 9)],
        "empty": [],
    }


CASES = list(_data())


@pytest.fixture(params=CASES)
def case(request):
    return _data()[request.param]


@pytest.fixture(params=["append", "block", "mixed"])
def stats(request, case):
    return case, _built(case)[request.param]


# ---- the column view ---------------------------------------------------------------------


def test_columns_hold_every_field_in_order(stats):
    want, got = stats
    cols = got.columns()
    assert len(cols) == len(want)
    assert cols.m.tolist() == [s.m for s in want] and cols.m.dtype == np.int64
    assert cols.q.tolist() == [s.q for s in want] and cols.q.dtype == np.int64
    assert cols.name == [s.name for s in want] and cols.cls == [s.cls for s in want]
    for key in ("kappa", "p_norm", "nu", "p_abs1", "u"):
        a = getattr(cols, key)
        assert a.dtype == np.float64 and a.shape == (len(want),)
        assert a.tobytes() == np.array([getattr(s, key) for s in want]).tobytes()
    k = len(want[0].normalized) if want else 0
    assert cols.normalized.dtype == np.float64 and cols.normalized.shape == (len(want), k)
    assert cols.normalized.tobytes() == np.array([s.normalized for s in want]).tobytes()
    assert not cols.kappa.flags.writeable and not cols.normalized.flags.writeable


def test_columns_build_no_product_stat(monkeypatch):
    made = []
    real = ProductStat.__init__
    monkeypatch.setattr(ProductStat, "__init__",
                        lambda self, *a, **kw: (made.append(1), real(self, *a, **kw))[1])
    st = ProductStats()
    want = [_stat(i) for i in range(6)]
    n = len(made)
    st.extend_columns(**_cols(want))
    cols = st.columns()
    assert len(cols) == 6 and len(made) == n
    StepStats(products=st).max_kappa()
    assert len(made) == n


def test_class_codes_follow_first_appearance(stats):
    want, got = stats
    codes, classes = got.columns().class_codes()
    assert classes == tuple(dict.fromkeys(s.cls for s in want))
    assert [classes[c] for c in codes.tolist()] == [s.cls for s in want]
    assert codes.dtype == np.int64


def test_the_cache_lasts_until_the_next_change():
    st = ProductStats([_stat(i) for i in range(4)])
    a = st.columns()
    assert st.columns() is a and st.copy().columns() is a
    list(st)  # iterating flattens blocks but changes no record
    st.extend_columns(**_cols([_stat(9)]))
    b = st.columns()
    assert b is not a and len(b) == 5
    blocks = ProductStats()
    blocks.extend_columns(**_cols([_stat(i) for i in range(3)]))
    c = blocks.columns()
    list(blocks)
    assert blocks.columns() is c
    steps = [
        lambda s: s.append(_stat(20)),
        lambda s: s.extend([_stat(21), _stat(22)]),
        lambda s: s.extend_columns(**_cols([_stat(23)])),
        lambda s: s.__setitem__(1, _stat(24, kappa=77.0)),
        lambda s: s.__setitem__(-1, _stat(25, kappa=78.0)),
        lambda s: s.__iadd__([_stat(26)]),
        lambda s: s.clear(),
    ]
    for step in steps:
        before = st.columns()
        step(st)
        after = st.columns()
        assert after is not before
        assert after.m.tolist() == [s.m for s in st] and after.kappa.tolist() == \
            [s.kappa for s in st]
    assert len(st.columns()) == 0 and st.columns().normalized.shape == (0, 0)


def test_an_empty_extend_columns_keeps_the_cache():
    st = ProductStats([_stat(1)])
    a = st.columns()
    st.extend_columns(**_cols([]))
    assert st.columns() is a


def test_a_copy_has_its_own_cache_and_a_pickle_none():
    st = ProductStats([_stat(i) for i in range(3)])
    cols = st.columns()
    dup = st.copy()
    dup.append(_stat(8))
    assert len(st.columns()) == 3 and len(dup.columns()) == 4 and st.columns() is cols
    back = pickle.loads(pickle.dumps(st))
    assert back == st and back._cols is None and len(back.columns()) == 3


def test_ragged_normalized_widths_raise():
    st = ProductStats([_stat(1, k=2), _stat(2, k=3)])
    with pytest.raises(ValueError):
        st.columns()
    mixed = ProductStats([_stat(1, k=2)])
    mixed.extend_columns(**_cols([_stat(2, k=3)]))
    with pytest.raises(ValueError):
        mixed.columns()


# ---- the consumers, old against new ------------------------------------------------------


def _old_max_normalized(ps):
    return max((max(p.normalized) for p in ps), default=0.0)


def _old_max_kappa(ps):
    return max((p.kappa for p in ps), default=0.0)


def test_max_normalized_and_max_kappa(stats):
    want, got = stats
    s = StepStats(products=got)
    _same(s.max_normalized(), _old_max_normalized(want))
    _same(s.max_kappa(), _old_max_kappa(want))


@pytest.mark.parametrize("kappas", [(-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (0.0, 0.0, -0.0)])
def test_max_kappa_keeps_the_first_of_equal_zeros(kappas):
    want = [_stat(i, kappa=x) for i, x in enumerate(kappas, start=1)]
    for got in _built(want).values():
        _same(StepStats(products=got).max_kappa(), _old_max_kappa(want))


def _old_summary(ps, bands, watch):
    cmax, kmax = {}, {}
    for p in ps:
        cmax[p.cls] = max(cmax.get(p.cls, -math.inf), *p.normalized)
        kmax[p.cls] = max(kmax.get(p.cls, -math.inf), p.kappa)
    ratio = {cls: x / bands.kappa_for(cls) for cls, x in kmax.items()}
    seen = {p.name: tuple(p.normalized) for p in ps if p.cls == watch}
    return cmax, ratio, seen


def test_step_summary(stats):
    want, got = stats
    for watch in ("c1", "absent"):
        s = rv.step_summary(1, StepStats(products=got, tensors=[TensorStat("w", "6a", 1.5)]),
                            BANDS, in_sample=False, watch=watch)
        cmax, ratio, seen = _old_summary(want, BANDS, watch)
        _same(s.class_max, cmax)
        _same(s.kappa_ratio, ratio)
        _same(s.watch, seen)
        assert s.rho_max == {"6a": 1.5}


def _old_step_floor(ps, *, tau, k, N, eps_in, eps_acc, c=calibration.C_ANTI):
    best = None
    for p in ps:
        if p.q <= 0 or not math.isfinite(p.p_norm) or p.p_norm == 0 or math.isnan(p.u):
            continue
        f = calibration.f_achieved(
            c, tau, calibration.product_e_m(p.q, eps_in, eps_acc, p.p_norm, p.u), N, k)
        if best is None or f > best.f_achieved:
            best = calibration.StepFloor(f, p.name)
    return best


def test_step_floor(stats):
    want, got = stats
    kw = dict(tau=8.0, k=3, N=40.0, eps_in=2.0 ** -24, eps_acc=2.0 ** -24)
    new = calibration.step_floor(StepStats(products=got), **kw)
    old = _old_step_floor(want, **kw)
    assert (new is None) == (old is None)
    if new is not None:
        _same(new.f_achieved, old.f_achieved)
        _same(new.name, old.name)


def _old_rho_arrays(stats, bands, *, k, kappa_guard=True):
    ps, ts = stats.products, stats.tensors
    ps = list(ps)
    width = len(ps[0].normalized) if ps else (k or 0)
    norm = np.array([p.normalized for p in ps], dtype=np.float64).reshape(len(ps), width)
    if bands is None:
        return {"rho_5": np.full_like(norm, np.nan), "rho_kappa": np.full(len(ps), np.nan),
                "rho_6": np.full(len(ts), np.nan)}
    rho_5 = norm / bands.tau
    if kappa_guard:
        nu = np.array([p.nu for p in ps], dtype=np.float64)
        ceiling = np.array([bands.kappa_for(p.cls) * p.p_abs1 for p in ps], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_kappa = np.where(nu == 0, 0.0, nu / ceiling)
    else:
        rho_kappa = np.full(len(ps), np.nan)
    rho_6 = np.array([s.rho_max / bands.tau_w_for(s.weight) for s in ts], dtype=np.float64)
    return {"rho_5": rho_5, "rho_kappa": rho_kappa.astype(np.float64), "rho_6": rho_6}


@pytest.mark.parametrize("bands", [BANDS, None])
@pytest.mark.parametrize("guard", [True, False])
def test_rho_arrays(stats, bands, guard):
    want, got = stats
    ts = [TensorStat("w.a", "6a", 1.5), TensorStat("w.b", "6b", 0.25)]
    new = metrics.rho_arrays(StepStats(products=got, tensors=ts), bands, k=3, kappa_guard=guard)
    old = _old_rho_arrays(StepStats(products=list(want), tensors=ts), bands, k=3,
                          kappa_guard=guard)
    _same(new, old)


def _old_residual_arrays(run, scenario, step, stats, c, *, bands=None, k=None, kappa_guard=True):
    ps = list(stats.products)
    width = len(ps[0].normalized) if ps else (k or 0)
    layers, classes = [], []
    n_layers = 0 if c is None else metrics.layer_count(c)
    for p in ps:
        spec = c.product(p.m) if c is not None else None
        layers.append(-1 if spec is None else metrics.layer_index(spec, n_layers))
        classes.append("" if spec is None else metrics.spec_class(c, spec))
    rho = _old_rho_arrays(StepStats(products=ps, tensors=stats.tensors), bands, k=k,
                          kappa_guard=kappa_guard)
    return {
        "run": np.array(run), "scenario": np.array(scenario), "step": np.array(step),
        "p_m": np.array([p.m for p in ps], dtype=np.int32),
        "p_name": np.array([p.name for p in ps], dtype=str),
        "p_cls": np.array([p.cls for p in ps], dtype=str),
        "p_spec_class": np.array(classes, dtype=str),
        "p_layer": np.array(layers, dtype=np.int32),
        "p_kappa": np.array([p.kappa for p in ps], dtype=np.float64),
        "p_q": np.array([p.q for p in ps], dtype=np.int64),
        "p_norm": np.array([p.p_norm for p in ps], dtype=np.float64),
        "p_nu": np.array([p.nu for p in ps], dtype=np.float64),
        "p_abs1": np.array([p.p_abs1 for p in ps], dtype=np.float64),
        "p_u": np.array([p.u for p in ps], dtype=np.float64),
        "p_normalized": np.array([p.normalized for p in ps],
                                 dtype=np.float64).reshape(len(ps), width),
        "w_check": np.array([s.check_id for s in stats.tensors], dtype=str),
        "w_name": np.array([s.weight for s in stats.tensors], dtype=str),
        "w_rho": np.array([s.rho_max for s in stats.tensors], dtype=np.float64),
        "band_source": np.array("" if bands is None else str(bands.source)),
        **rho,
    }


class _Comp:
    """A stub computation with an inventory of 16 products: ``product_class`` and ``products``,
    ``M`` and ``product`` as ``DeclaredComputation`` has them."""

    def __init__(self, table=False, own=True):
        kinds = [ProductKind.FORWARD, ProductKind.INPUT_GRAD, ProductKind.WEIGHT_GRAD]
        self.products = tuple(
            ProductSpec(m, f"L{m % 4}.P{m}", kinds[m % 3], (4, 4), (4, 4),
                        weight=None if m % 5 == 0 else f"W{m % 2}", layer=None if m == 16 else
                        1 + m % 4)
            for m in range(1, 17))
        self.M = 16
        if own:
            self.product_class = self._class
        if table:
            self.product_classes = tuple(product_class(self, s) for s in self.products)

    def product(self, m):
        if not 1 <= m <= self.M:
            raise IndexError(f"product m={m} outside 1..{self.M}")
        return self.products[m - 1]

    def _class(self, spec):
        return "S" if spec.weight is None else default_product_class(spec)


@pytest.mark.parametrize("table", [False, True])
@pytest.mark.parametrize("use_c", [False, True])
def test_residual_arrays(stats, table, use_c):
    want, got = stats
    c = _Comp(table) if use_c else None
    ts = [TensorStat("w.a", "6a", 1.5)]
    new = metrics._residual_arrays("r", "s", 3, StepStats(products=got, tensors=ts), c,
                                   bands=BANDS, k=3)
    old = _old_residual_arrays("r", "s", 3, StepStats(products=list(want), tensors=ts), c,
                               bands=BANDS, k=3)
    _same(new, old)
    for a in new.values():
        assert a.flags.writeable


def test_residual_arrays_reject_a_product_outside_the_inventory():
    st = StepStats(products=[_stat(1), _stat(99)])
    with pytest.raises(IndexError, match="outside 1..16"):
        metrics._residual_arrays("r", "s", 1, st, _Comp(True), bands=BANDS, k=3)


# ---- the class table ---------------------------------------------------------------------


@pytest.mark.parametrize("own", [False, True])
def test_the_default_class_table_is_each_specs_class(own):
    c = _Comp(False, own=own)
    table = DeclaredComputation.product_classes.func(c)
    assert table == tuple(product_class(c, s) for s in c.products)
    assert isinstance(table, tuple) and ("S" in table) == own


def test_product_classes_is_none_for_a_computation_without_a_table():
    assert product_classes(SimpleNamespace()) is None
    assert product_classes(_Comp(True)) == _Comp(True).product_classes


def test_default_product_class_replaces_digit_runs():
    spec = ProductSpec(1, "L12.S3", ProductKind.FORWARD, (2, 2), (2, 2))
    assert default_product_class(spec) == "forward:L*.S*"
    spec = ProductSpec(1, "x", ProductKind.INPUT_GRAD, (2, 2), (2, 2), weight="layers.7.w1")
    assert default_product_class(spec) == f"{spec.kind.value}:layers.*.w*"
