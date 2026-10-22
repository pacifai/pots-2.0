"""FS38b's shared reads on a GPU: ``freivalds.measure_groups_rows(..., share=True)`` against
``share=False`` (FS37's three launches per table), bit for bit. Skipped without CUDA; they run
on a rented card. Each table's levels, shares and bytes saved are printed (``-s``)."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tests.verification.verifier.matmul_check.shared_tables import (  # noqa: E402
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
from tests.verification.verifier.matmul_check.test_measure_core import H, bits  # noqa: E402
from tests.verification.verifier.matmul_check.test_measure_core_spec import (  # noqa: E402
    fs36_cases,
    same_bits,
)
from tests.verification.verifier.matmul_check.test_read_plan import shares_of  # noqa: E402
from verification.computation.interface import ReplayError  # noqa: E402
from verification.verifier.matmul_check import freivalds as fv  # noqa: E402
from verification.verifier.matmul_check import measure_kernel as mk  # noqa: E402
from verification.verifier.matmul_check.freivalds import MeasureGroup  # noqa: E402

DEV = "cuda"


# ---- tables on the device ----

def to_device(mis):
    """CUDA copies of the tables' tensors: every CPU storage is copied once, so views of one
    storage stay views of one device storage (the shares survive)."""
    stores = {}

    def move(t):
        if t is None:
            return None
        st = t.untyped_storage()
        key = st.data_ptr()
        if key not in stores:
            raw = torch.empty(0, dtype=torch.uint8).set_(st).clone().to(DEV)
            stores[key] = raw.untyped_storage()
        return torch.empty(0, dtype=t.dtype, device=DEV).set_(
            stores[key], t.storage_offset(), t.shape, t.stride())

    return [(move(mi.a), move(mi.b), move(mi.p), move(mi.bias)) for mi in mis]


def groups_of_table(mis):
    """One group per product (labels 1, 2, ...), the device views sharing storage as the CPU
    ones did; ``k`` from the table."""
    return [MeasureGroup(a[None], b[None], p[None], [i + 1], bias)
            for i, (a, b, p, bias) in enumerate(to_device(mis))], mis[0].r.shape[1]


def rows_of(groups, k, *, guard=True, share=False, plan_out=None):
    rows = fv.measure_groups_rows(groups, h=H, k=k, guard=guard, share=share, plan_out=plan_out)
    fv.drain()
    return rows.cpu().numpy()


def shared_vs_plain(groups, k, *, guard=True, label="", special=False):
    plans = []
    got = rows_of(groups, k, guard=guard, share=True, plan_out=plans)
    want = rows_of(groups, k, guard=guard, share=False)
    assert len(plans) == 1 and isinstance(plans[0], mk.ReadPlan)
    p = plans[0]
    print(f"[FS38b] {label}: products {len(groups)}, n_levels {p.n_levels}, "
          f"shared_groups {p.shared_groups}, bytes_saved/bytes_fs37 {p.bytes_saved}/{p.bytes_fs37}"
          f" = {p.bytes_saved / max(p.bytes_fs37, 1):.3%}, cycle {p.cycle}")
    if special:
        assert same_bits(got, want), label
    else:
        assert np.array_equal(bits(got), bits(want)), label
    return p


# ---- FS36's inputs as one table ----

@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("guard", [True, False])
def test_share_equals_no_share_on_fs36_inputs_as_one_table(k, guard):
    """FS36's cases as one table per challenge width and guard (a call has one of each); FS36's
    cases share no storage, so the plan is FS37's three levels and the rows must be FS37's."""
    cases = [mi for _, mi in fs36_cases() if mi.r.shape[1] == k and mi.guard == guard]
    assert cases
    groups, kk = groups_of_table(cases)
    p = shared_vs_plain(groups, kk, guard=guard, label=f"FS36 k={k} guard={guard}", special=True)
    assert p.shared_groups == 0 and p.n_levels == 3 and p.bytes_saved == 0


# ---- the shared tables of the host file, on CUDA copies ----

def check(mis, expect, *, guard=True, label):
    groups, k = groups_of_table(mis)
    p = shared_vs_plain(groups, k, guard=guard, label=label)
    assert shares_of(p) == expect, label
    assert shares_of(p) == shares_of(plan_for(mis)), label   # the copies keep the CPU's shares
    return p


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_x_by_a_forward_and_its_gradient(dtype, guard):
    p = check(fwd_and_grad(dtype, guard), {frozenset({(0, 1), (1, 0)})}, guard=guard,
              label=f"X by Y and G {dtype}")
    assert p.n_levels == 4 and p.shared_groups == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_biased_products_beside_a_shared_x(dtype, guard):
    check(fwd_and_grad(dtype, guard, biased_beside=True),
          {frozenset({(0, 1), (1, 0), (2, 1)}), frozenset({(1, 1), (3, 1)})}, guard=guard,
          label=f"biased beside X {dtype}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_one_x_by_three_forward_products(dtype, guard):
    check(three_forward(dtype, guard), {frozenset({(0, 1), (1, 1), (2, 1)})}, guard=guard,
          label=f"X by q, k, v {dtype}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_member_slices_of_one_stack(dtype, guard):
    T = 3
    check(member_stack(dtype, guard, T=T), {frozenset({(t, 1), (T + t, 0)}) for t in range(T)},
          guard=guard, label=f"member slices {dtype}")


@pytest.mark.parametrize("dtype", DTYPES)
def test_a_two_sided_use_runs_unshared(dtype):
    check(two_sided(dtype), set(), label=f"two-sided {dtype}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_a_mixed_table_with_shares_in_every_order(dtype, guard):
    mis = mixed_table(dtype, guard=guard)
    groups, k = groups_of_table(mis)
    shared_vs_plain(groups, k, guard=guard, label=f"mixed {dtype}")
    rng = np.random.default_rng(3)
    want = rows_of(groups, k, guard=guard, share=False)
    for _ in range(2):
        order = rng.permutation(len(groups))
        got = rows_of([groups[i] for i in order], k, guard=guard, share=True)
        for pos, i in enumerate(order):
            assert np.array_equal(bits(got[pos]), bits(want[i]))


@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("guard", [True, False])
def test_nan_inf_and_zero_in_a_shared_view(edit, dtype, guard):
    mis, x = shared_special(dtype, guard)
    EDITS[edit](x)
    groups, k = groups_of_table(mis)
    p = shared_vs_plain(groups, k, guard=guard, label=f"special {edit}", special=True)
    assert p.shared_groups == 1


# ---- plan_out, the holds and drain ----

def test_plan_out_gets_one_read_plan_per_call_and_none_without_share():
    groups, k = groups_of_table(fwd_and_grad(torch.float32, True))
    plans = []
    rows_of(groups, k, share=False, plan_out=plans)
    assert plans == []
    rows_of(groups, k, share=True, plan_out=plans)
    assert len(plans) == 1 and isinstance(plans[0], mk.ReadPlan)
    rows_of(groups, k, share=True, plan_out=plans)
    rows_of(groups[:1], k, share=True, plan_out=plans)
    assert len(plans) == 3 and [p.shared_groups for p in plans] == [1, 1, 0]
    assert [len(p.node_level) for p in plans] == [2, 2, 1]
    rows_of(groups, k, share=True)  # no plan_out: nothing to append to, nothing breaks


def test_the_default_is_fs37s_call():
    groups, k = groups_of_table(fwd_and_grad(torch.float32, True))
    a = fv.measure_groups_rows(groups, h=H, k=k)
    b = fv.measure_groups_rows(groups, h=H, k=k, share=False)
    fv.drain()
    assert np.array_equal(bits(a.cpu().numpy()), bits(b.cpu().numpy()))


def test_a_held_input_written_before_the_drain_raises_with_share():
    groups, k = groups_of_table(fwd_and_grad(torch.float32, True))
    fv.measure_groups_rows(groups, h=H, k=k, share=True)
    groups[0].a.add_(1.0)  # X: the shared view, held by the table
    with pytest.raises(ReplayError):
        fv.drain()
    fv.drain()  # the list is empty afterwards


def test_an_input_written_after_the_drain_is_not_held_with_share():
    groups, k = groups_of_table(fwd_and_grad(torch.float32, True))
    fv.measure_groups_rows(groups, h=H, k=k, share=True)
    fv.drain()
    groups[0].a.add_(1.0)
    fv.drain()  # nothing held, nothing raised


def test_the_table_holds_every_view_of_the_step_until_the_drain():
    """Each input is held with its ``_version`` as enqueued: a write to any one of the table's
    inputs, not only the shared one, raises at the drain."""
    for which in ("a", "b", "p"):
        groups, k = groups_of_table(fwd_and_grad(torch.float32, True))
        fv.measure_groups_rows(groups, h=H, k=k, share=True)
        getattr(groups[1], which).add_(1.0)
        with pytest.raises(ReplayError):
            fv.drain()
        fv.drain()


def test_a_second_call_releases_the_completed_first_and_still_drains_clean():
    groups, k = groups_of_table(fwd_and_grad(torch.bfloat16, True))
    for _ in range(3):
        fv.measure_groups_rows(groups, h=H, k=k, share=True)
    torch.cuda.synchronize()
    fv.measure_groups_rows(groups, h=H, k=k, share=True)
    fv.drain()


def test_a_refused_group_raises_before_anything_is_launched_with_share():
    groups, k = groups_of_table(fwd_and_grad(torch.float32, True))
    bad = MeasureGroup(groups[0].a.double(), groups[0].b.double(), groups[0].p.double(), [1])
    plans = []
    with pytest.raises(ValueError):
        fv.measure_groups_rows([groups[0], bad], h=H, k=k, share=True, plan_out=plans)
    fv.drain()  # nothing was held
    assert plans == []
