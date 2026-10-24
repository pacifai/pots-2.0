"""Tables for FS37b's read groups of several entries (``test_measure_sets_sim_host.py``,
``test_measure_sets_cuda.py``): one stored tensor ``X`` of shape ``[n, q]`` read as one view by
``e`` entries of a read group, each an ``A``-kind entry (``X`` is ``A`` of a product, phase 1) or
a ``B``-kind entry (``X`` is ``B`` of a product, phase 0), with the same address, shape, strides
and dtype. ``group_of`` reads the group back from ``mk.read_plan`` so a test can assert that the
table really makes the group it names."""

import numpy as np
import torch

from tests.verification.verifier.matmul_check.shared_tables import prod_of, rnd
from verification.verifier.matmul_check import measure_kernel as mk
from verification.verifier.matmul_check.measure_table import MeasureInput, pack

def rb_of(rows_per_block, e):
    """The rows a block takes of a group of ``e`` entries: one entry per batch, so all
    ``rows_per_block`` whatever ``e``."""
    return rows_per_block


def kinds_of(e, mode):
    """``e`` entry kinds: ``"A"`` all phase 1, ``"B"`` all phase 0, ``"mix"`` alternating from
    ``A`` (so a group of 2 or more holds both)."""
    if mode == "mix":
        return ["A" if i % 2 == 0 else "B" for i in range(e)]
    return [mode] * e


def shared_view(e, mode, *, n, q, dtype, guard=True, k=3, layout="row", seed=0, guards=None,
                bias_at=None):
    """``(mis, x)``: ``e`` products reading ``x`` (``n x q``), kinds by ``mode``. ``guards`` is a
    per-product guard list (else ``guard`` for all); ``bias_at`` the index of an ``A``-kind
    product that carries a bias (its phase-0 node is alone, its phase-1 node still joins)."""
    g = torch.Generator().manual_seed(seed + 100)
    base = rnd(g, n, q, dtype=dtype)
    x = base if layout == "row" else base.t().contiguous().t()
    mis = []
    for i, kind in enumerate(kinds_of(e, mode)):
        gd = guard if guards is None else guards[i]
        if kind == "A":
            w = 9 + 7 * (i % 3)
            bias = rnd(g, w, dtype=dtype) if bias_at == i else None
            mis.append(prod_of(x, rnd(g, w, q, dtype=dtype).t(), k=k, m=i + 1, guard=gd,
                               bias=bias, seed=seed + i))
        else:
            m = 6 + 2 * (i % 3)
            mis.append(prod_of(rnd(g, m, n, dtype=dtype), x, k=k, m=i + 1, guard=gd,
                               seed=seed + i))
    return mis, x


def separate_biased(*, dtype, guard=True, k=3, seed=50, n=11, q=19, w=23):
    """A biased product sharing no view with anything (its own storages)."""
    g = torch.Generator().manual_seed(seed)
    a, b = rnd(g, n, q, dtype=dtype), rnd(g, q, w, dtype=dtype)
    return prod_of(a, b, k=k, m=99, bias=rnd(g, w, dtype=dtype), guard=guard, seed=seed)


def plan_of(mis):
    return mk.read_plan(np.frombuffer(pack(mis).array, dtype=mk.MPROD_DTYPE).copy())


def groups_of_plan(plan):
    """``[(n_entries, count, phases)]`` for every group of every level (sentinels skipped)."""
    out = []
    off = plan.level_off.astype(np.int64)
    for lv in range(plan.n_levels):
        for r in plan.reads[off[lv]:off[lv + 1] - 1]:
            e0, ne = int(r["entry0"]), int(r["n_entries"])
            phases = tuple(int(p) for p in plan.entries["phase"][e0:e0 + ne])
            out.append((ne, int(r["count"]), phases))
    return out


def group_of(mis, e, *, n=None, phases=None):
    """The one group of ``e`` entries of ``mis``' read plan (asserted unique), checked against
    ``n`` rows and, when given, the sorted phase tuple ``phases``."""
    found = [gr for gr in groups_of_plan(plan_of(mis)) if gr[0] == e]
    assert len(found) == 1, f"expected one group of {e} entries, plan has {groups_of_plan(plan_of(mis))}"
    ne, count, ph = found[0]
    if n is not None:
        assert count == n, (count, n)
    if phases is not None:
        assert sorted(ph) == sorted(phases), (ph, phases)
    return found[0]


def phases_of(e, mode):
    return [1 if kd == "A" else 0 for kd in kinds_of(e, mode)]


def with_guard_each(mis, guards):
    return [MeasureInput(m.a, m.b, m.p, m.r, m.bias, g) for m, g in zip(mis, guards)]
