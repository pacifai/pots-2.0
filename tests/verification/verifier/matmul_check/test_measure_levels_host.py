"""FS38b's level run on the Mac: ``pots_host_measure_levels`` (the level launches in loops)
against ``pots_host_measure_phased``, bit for bit.

The ``levels`` fixture (in ``test_measure_kernel_host.py``) packs a list of ``MeasureInput`` into
a table, builds its ``measure_kernel.read_plan`` and runs ``pots_host_measure_levels`` with
``levels(mis, reverse=False)``, returning the rows as ``phased`` does. The tables here hold real
shares: tensors that are views of one storage, so the planner finds them by address."""

import numpy as np
import pytest
import torch

from tests.verification.verifier.matmul_check.test_measure_core import (  # noqa: F401
    H,
    bits,
    hostlib,
    host_measure,
)
from tests.verification.verifier.matmul_check.test_measure_core_spec import fs36_cases, same_bits
from tests.verification.verifier.matmul_check.test_measure_kernel_host import levels, phased  # noqa: F401
from tests.verification.verifier.matmul_check.test_read_plan import groups_of, shares_of
from tests.verification.verifier.matmul_check.shared_tables import (
    DTYPES,
    EDITS,
    fwd_and_grad,
    member_stack,
    mixed_table,
    plan_for,
    shared_special,
    three_forward,
    two_sided,
)


def same_rows(levels, phased, mis, *, special=False):
    """Levels' rows equal phased's, forwards and reversed (``special``: NaN equals NaN)."""
    want = phased(mis)
    for reverse in (False, True):
        got = levels(mis, reverse=reverse)
        assert got.shape == want.shape
        if special:
            assert same_bits(got, want), reverse
        else:
            assert np.array_equal(bits(got), bits(want)), reverse
    return want


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_x_by_a_forward_and_its_gradient(levels, phased, dtype, guard):
    mis = fwd_and_grad(dtype, guard)
    plan = plan_for(mis)
    assert shares_of(plan) == {frozenset({(0, 1), (1, 0)})}
    assert plan.n_levels == 4 and plan.shared_groups == 1
    assert plan.bytes_saved == 9 * 21 * (4 if dtype == torch.float32 else 2)
    same_rows(levels, phased, mis)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_biased_products_beside_a_shared_x(levels, phased, dtype, guard):
    mis = fwd_and_grad(dtype, guard, biased_beside=True)
    plan = plan_for(mis)
    # product 2 (biased, A = X) joins phase 1's group; product 3 (biased, B = X) keeps phase 0
    # alone, and its A (δYᵀ, the gradient's A too) is shared with the gradient's phase 1
    assert shares_of(plan) == {frozenset({(0, 1), (1, 0), (2, 1)}), frozenset({(1, 1), (3, 1)})}
    assert (3, 0) in [e for _, es, _, _ in groups_of(plan) for e in es if len(es) == 1]
    same_rows(levels, phased, mis)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_one_x_by_three_forward_products(levels, phased, dtype, guard):
    mis = three_forward(dtype, guard)
    plan = plan_for(mis)
    assert shares_of(plan) == {frozenset({(0, 1), (1, 1), (2, 1)})}
    assert plan.n_levels == 3
    same_rows(levels, phased, mis)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_member_slices_of_one_stack(levels, phased, dtype, guard):
    """Slices ``stack[t]`` of one tensor, each read as ``A`` by the first member set (Q̃ by S, or
    δO by dA) and as ``B`` by the second (δK̃, or dV): one group per member pair."""
    T = 3
    mis = member_stack(dtype, guard, T=T)
    plan = plan_for(mis)
    assert shares_of(plan) == {frozenset({(t, 1), (T + t, 0)}) for t in range(T)}
    assert plan.shared_groups == T and plan.n_levels == 4
    same_rows(levels, phased, mis)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_two_sided_use_runs_unshared(levels, phased, dtype):
    mis = two_sided(dtype)
    plan = plan_for(mis)
    assert shares_of(plan) == set() and plan.shared_groups == 0 and plan.n_levels == 3
    same_rows(levels, phased, mis)


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_mixed_table_with_shares_in_every_order(levels, phased, dtype):
    mis = mixed_table(dtype)
    # the first pair with its biased neighbours (see above), the second pair, and nothing else
    assert shares_of(plan_for(mis)) == {frozenset({(0, 1), (1, 0), (2, 1)}),
                                        frozenset({(1, 1), (3, 1)}), frozenset({(4, 1), (5, 0)})}
    want = same_rows(levels, phased, mis)
    rng = np.random.default_rng(3)
    for _ in range(3):
        order = rng.permutation(len(mis))
        got = levels([mis[i] for i in order])
        for pos, i in enumerate(order):
            assert np.array_equal(bits(got[pos]), bits(want[i]))


@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("biased", [False, True])
def test_nan_inf_and_zero_in_a_shared_view(levels, phased, edit, dtype, guard, biased):
    mis, x = shared_special(dtype, guard, biased)
    EDITS[edit](x)  # the one storage both products read
    assert shares_of(plan_for(mis)) == {frozenset({(0, 1), (1, 0)})}
    same_rows(levels, phased, mis, special=True)


@pytest.mark.parametrize("k", [1, 2])
def test_levels_equal_phased_on_fs36_inputs_as_one_table(levels, phased, k):
    """FS36's 204 cases make one table per challenge width (the cases' ``k`` is 1 or 2; a table has
    one); each product keeps its own guard flag. The cases share no storage, so the plan holds no
    group, and the run must still be phased's, bit for bit."""
    cases = [mi for _, mi in fs36_cases() if mi.r.shape[1] == k]
    assert len(cases) >= (150 if k == 2 else 1)
    assert shares_of(plan_for(cases)) == set()
    want = phased(cases)
    for reverse in (False, True):
        got = levels(cases, reverse=reverse)
        assert same_bits(got, want), reverse
        assert np.array_equal(np.isnan(got), np.isnan(want))
