"""FS51's table cache through ``enqueue`` on a GPU (``HOST_SIDE_PLAN_DRAFT.md`` §2.3, §5.2).
Skipped without CUDA; they run on the A40 (FS55).

Two steps of one structure at other addresses go through ``measure_groups_rows``, with the cache
on and off: every row is the same bits, at ``k = 9`` and ``k = 21``, share on and off, and the
second call hits the cache without a plan rebuild."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from tests.verification.verifier.matmul_check.test_measure_core import H, bits  # noqa: E402
from tests.verification.verifier.matmul_check.test_table_cache_host import groups_of  # noqa: E402
from verification.verifier.matmul_check import challenge_source as cs  # noqa: E402
from verification.verifier.matmul_check import freivalds as fv  # noqa: E402
from verification.verifier.matmul_check import measure_kernel as mk  # noqa: E402
from verification.verifier.matmul_check.freivalds import MeasureGroup  # noqa: E402

DEV = "cuda"


@pytest.fixture(autouse=True)
def fresh_cache():
    mk.table_cache_clear()
    mk.table_cache_stats(reset=True)
    yield
    mk.table_cache_clear()


def on_device(groups: list[MeasureGroup]) -> list[MeasureGroup]:
    """The groups on the GPU, one copy per tensor (so a view two groups read stays shared)."""
    moved: dict[int, torch.Tensor] = {}

    def to(t):
        if t is None:
            return None
        if id(t) not in moved:
            moved[id(t)] = t.to(DEV)
        return moved[id(t)]

    return [MeasureGroup(to(g.a), to(g.b), to(g.p), g.ms, to(g.bias)) for g in groups]


def rows_of(groups, k, share, guard=True):
    out = fv.measure_groups_rows(groups, h=H, k=k, guard=guard, share=share)
    fv.drain()
    return out.cpu().numpy()


@pytest.mark.parametrize("k", [9, 21])
@pytest.mark.parametrize("share", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_steps_give_the_same_rows_with_the_cache_on_and_off(k, share, dtype):
    step1 = on_device(groups_of(1, dtype))
    step2 = on_device(groups_of(2, dtype))  # the same structure, other tensors and addresses
    first = rows_of(step1, k, share)
    second = rows_of(step2, k, share)
    stats = mk.table_cache_stats()
    assert stats == {"hits": 1, "misses": 1, "plan_rebuilds": 0}
    with mk.table_cache(False):
        off1 = rows_of(step1, k, share)
        off2 = rows_of(step2, k, share)
    assert np.array_equal(bits(first), bits(off1))
    assert np.array_equal(bits(second), bits(off2))
    assert mk.table_cache_stats() == stats  # nothing counted with the cache off


def test_a_moved_shared_view_is_rebuilt_counted_and_the_same_bits():
    k = 9
    rows_of(on_device(groups_of(1, torch.float32)), k, True)
    moved = on_device(groups_of(2, torch.float32, x_moved=True))
    got = rows_of(moved, k, True)
    assert mk.table_cache_stats() == {"hits": 1, "misses": 1, "plan_rebuilds": 1}
    with mk.table_cache(False):
        want = rows_of(moved, k, True)
    assert np.array_equal(bits(got), bits(want))


@pytest.fixture
def host_source():
    """The host source as the default while a test runs: the default is the GPU source (FS41),
    whose ``prefetch`` returns ``None``, so a test of the host source's prefetch needs this."""
    before = mk.set_default_source(cs.HostChallengeSource())
    try:
        yield
    finally:
        mk.set_default_source(before)


def test_prefetched_challenges_with_the_cache_give_the_same_rows(host_source):
    """Pinned to the host source (``host_source``): it is the one that prefetches."""
    k = 9
    step = on_device(groups_of(1, torch.float32))
    ms = [m for g in step for m in g.ms]
    widths = [g.b.shape[2] for g in step for _ in g.ms]
    rows_of(step, k, True)  # fills both caches
    pf = mk.prefetch(H, ms, k, widths)
    assert pf is not None
    got = fv.measure_groups_rows(step, h=H, k=k, share=True, prefetched=pf)
    fv.drain()
    assert np.array_equal(bits(got.cpu().numpy()), bits(rows_of(step, k, True)))
