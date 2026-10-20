"""``ProductStats``: the list of a step's ``ProductStat`` records, stored by columns when added in
bulk."""

import copy
import json
import math
import pickle
from dataclasses import asdict

import numpy as np
import pytest

from verification.verifier.context import ProductStat, ProductStats, StepStats


def _stat(i, k=3):
    return ProductStat(i, f"P{i}", f"cls{i % 3}", 1.0 + i / 7, tuple(i / 3 + j for j in range(k)),
                       q=10 + i, p_norm=2.0 + i, nu=3.0 + i, p_abs1=4.0 + i, u=5.0 + i / 9)


def _columns(stats):
    return dict(
        m=[s.m for s in stats], name=[s.name for s in stats], cls=[s.cls for s in stats],
        kappa=np.array([s.kappa for s in stats]),
        normalized=np.array([s.normalized for s in stats]).reshape(
            len(stats), 0 if not stats else len(stats[0].normalized)),
        q=[s.q for s in stats], p_norm=np.array([s.p_norm for s in stats]),
        nu=np.array([s.nu for s in stats]), p_abs1=np.array([s.p_abs1 for s in stats]),
        u=np.array([s.u for s in stats]))


def _mixed():
    """Singles, a block and singles again, in order, beside the list they must equal."""
    want = [_stat(i) for i in range(10)]
    got = ProductStats()
    got.append(want[0])
    got.append(want[1])
    got.extend_columns(**_columns(want[2:7]))
    got.extend_columns(**_columns(want[7:8]))
    got.append(want[8])
    got.extend_columns(**_columns(want[9:]))
    return got, want


def test_a_stat_built_from_columns_equals_the_one_appended_field_for_field():
    got, want = _mixed()
    for g, w in zip(got, want):
        assert g == w and repr(g) == repr(w)
        assert asdict(g) == asdict(w)
        assert [type(x) for x in asdict(g).values()] == [type(x) for x in asdict(w).values()]
        assert type(g.normalized) is tuple and all(type(x) is float for x in g.normalized)
        assert type(g.m) is int and type(g.q) is int and type(g.kappa) is float
    assert json.dumps(asdict(got[4])) == json.dumps(asdict(want[4]))


def test_indexing_negative_and_slices_without_building_the_rest():
    got, want = _mixed()
    assert len(got) == 10
    for i in range(-10, 10):
        assert got[i] == want[i]
    assert got[3:8] == want[3:8] and got[::-3] == want[::-3] and got[20:] == []
    assert isinstance(got[2:4], list)
    for bad in (10, -11):
        with pytest.raises(IndexError):
            got[bad]
    assert any(type(g).__name__ == "_Block" for g in got._segs)  # indexing built nothing


def test_it_acts_as_a_list():
    got, want = _mixed()
    assert got == want and want == list(got) and got == ProductStats(want)
    assert got != want[:-1] and got != 3
    assert bool(got) and not ProductStats() and len(ProductStats()) == 0
    assert list(got) == want and [p.m for p in reversed(got)] == list(range(9, -1, -1))
    assert want[4] in got and _stat(40) not in got
    assert next(p for p in got if p.m == 6) == want[6]
    assert got + [1] == want + [1] and [1] + got == [1] + want
    assert repr(got) == repr(want)
    assert max(p.kappa for p in got) == max(p.kappa for p in want)


def test_extend_with_a_list_and_in_place_add():
    got, want = _mixed()
    got += [_stat(10), _stat(11)]
    got.extend([_stat(12)])
    assert got == want + [_stat(10), _stat(11), _stat(12)]


def test_item_assignment_replaces_one_stat_wherever_it_lives():
    for i in (0, 3, -1, -8):
        got, want = _mixed()
        new = _stat(99)
        got[i] = new
        want[i] = new
        assert got == want and len(got) == 10
    got, _ = _mixed()
    with pytest.raises(IndexError):
        got[10] = _stat(1)


def test_copy_copy_gives_an_independent_container():
    got, want = _mixed()
    dup = copy.copy(got)
    assert dup == want and dup is not got
    dup.append(_stat(50))
    got.clear()
    assert len(dup) == 11 and len(got) == 0
    got, _ = _mixed()
    dup = got.copy()
    got[0] = _stat(60)
    assert dup[0] == _stat(0)


def test_clear_then_reuse():
    got, _ = _mixed()
    got.clear()
    assert not got and len(got) == 0 and list(got) == []
    got.extend_columns(**_columns([_stat(1), _stat(2)]))
    assert got == [_stat(1), _stat(2)]


def test_an_empty_bulk_add_adds_nothing_and_widths_may_differ_between_adds():
    got = ProductStats()
    got.extend_columns(**_columns([]))
    assert not got
    got.append(_stat(1, k=2))
    got.extend_columns(**_columns([_stat(2, k=4), _stat(3, k=4)]))
    assert [len(p.normalized) for p in got] == [2, 4, 4]


def test_nan_columns_come_back_as_nan_and_stay_out_of_equality():
    s = ProductStat(1, "P", "c", math.nan, (math.inf, 0.0))
    got = ProductStats()
    got.extend_columns(**_columns([s]))
    assert math.isnan(got[0].kappa) and math.isnan(got[0].nu) and got[0].q == 0
    assert repr(got) == repr([s])


def test_copies_and_pickles_keep_the_records():  # pickle of our own bytes, nothing foreign
    got, want = _mixed()
    assert copy.deepcopy(got) == want and pickle.loads(pickle.dumps(got)) == want


def test_step_stats_takes_a_list_and_keeps_its_container():
    want = [_stat(i) for i in range(4)]
    st = StepStats(products=want)
    assert isinstance(st.products, ProductStats) and st.products == want
    assert isinstance(StepStats().products, ProductStats)
    got, _ = _mixed()
    assert StepStats(products=got).products is got
    a, b = StepStats(products=want[:2]), StepStats(products=list(want[:2]))
    assert a == b and a != StepStats(products=want[:3])
    assert st.max_kappa() == max(p.kappa for p in want)
    assert st.max_normalized() == max(max(p.normalized) for p in want)
    assert list(st.by_class()) == ["cls0", "cls1", "cls2"]
