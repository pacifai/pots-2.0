"""Which device owns each replay unit and each leaf (F22).

At full scale the transcript lives on the verifier's GPUs, not in host memory, and the replay is
split over them by layers. The placement is the one object that says where: capture routing, the
split replay and check 5 all ask it, so a product's leaf and the unit that consumes it sit on
the same device and an honest run copies nothing between them. Unsharded
(``VERIF_VERIFIER_DEVICES`` unset), every leaf stays on ``HOST`` and every unit runs on the
prover's device, today's path.

The rule. A unit's bytes are its leaves' bytes: each weight twice (``W_t`` and ``W_{t+1}``, at
``weight_dtype``) and each product once (``p_shape`` at ``product_dtype``). The pinned units
(``pinned_last_units``) go to the last slot. The others are cut, in order, into one contiguous
range per slot so that the largest slot's total is smallest, the pinned bytes counted in the
last slot's; a slot may get no unit. Among the optimal cuts, slot 0 takes as few units as it
can, then slot 1, and so on, so the later slots fill first and the last, which holds the pinned
units, is the fullest. A computation with one unit and nothing pinned lands on the last slot.
Records stay on ``HOST``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from verification.computation.interface import HOST, DeclaredComputation

__all__ = ["Placement", "placement_for", "split_units"]


@dataclass(frozen=True)
class Placement:
    """Build it with :func:`placement_for`; the fields after ``devices`` are derived."""

    prover: torch.device  # VERIF_DEVICE
    devices: tuple[torch.device, ...] | None  # VERIF_VERIFIER_DEVICES; None: unsharded
    _unit_slot: dict[int, int] = field(default_factory=dict, repr=False, compare=False)
    _weight_slot: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _product_slot: tuple[int, ...] = field(default=(), repr=False, compare=False)
    _leaf_device: tuple[torch.device, ...] = field(default=(), repr=False, compare=False)
    _loads: tuple[int, ...] = field(default=(), repr=False, compare=False)
    _product_leaf_slot: tuple[int, ...] = field(default=(), repr=False, compare=False)

    _units: frozenset[int] = field(default=frozenset(), repr=False, compare=False)
    _weights: frozenset[str] = field(default=frozenset(), repr=False, compare=False)
    _n_products: int = field(default=0, repr=False, compare=False)
    _n_product_leaves: int = field(default=0, repr=False, compare=False)
    _n_leaves: int = field(default=0, repr=False, compare=False)

    @property
    def sharded(self) -> bool:
        return self.devices is not None

    def _check_unit(self, j: int) -> None:
        if j not in self._units:
            raise IndexError(f"unit {j} is not one of the units {sorted(self._units)}")

    def unit_slot(self, j: int) -> int:
        """Index into ``devices`` of unit ``j``'s device; 0 unsharded."""
        self._check_unit(j)
        return self._unit_slot[j] if self.devices is not None else 0

    def unit_device(self, j: int) -> torch.device:
        """The device unit ``j`` runs on; the prover's unsharded."""
        self._check_unit(j)
        if self.devices is None:
            return self.prover
        return self.devices[self._unit_slot[j]]

    def weight_device(self, name: str) -> torch.device:
        """Where weight ``name``'s ``W_t`` and ``W_{t+1}`` leaves live; ``HOST`` unsharded."""
        if name not in self._weights:
            raise KeyError(f"{name!r} is not a declared weight")
        if self.devices is None:
            return HOST
        return self.devices[self._weight_slot[name]]

    def product_device(self, m: int) -> torch.device:
        """Where product ``m`` (1-based) has its leaf; ``HOST`` unsharded."""
        if not 1 <= m <= self._n_products:
            raise IndexError(f"product m={m} outside 1..{self._n_products}")
        if self.devices is None:
            return HOST
        return self.devices[self._product_slot[m - 1]]

    def product_leaf_device(self, g: int) -> torch.device:
        """Where product leaf ``g`` (0-based among the product leaves) lives: the device of its
        first product, since a group's members share a layer and so a unit (O18); ``HOST``
        unsharded."""
        if not 0 <= g < self._n_product_leaves:
            raise IndexError(f"product leaf {g} outside 0..{self._n_product_leaves - 1}")
        if self.devices is None:
            return HOST
        return self.devices[self._product_leaf_slot[g]]

    def leaf_device(self, i: int) -> torch.device:
        """Where leaf ``i`` (the transcript index) lives; records are always on ``HOST``."""
        if not 0 <= i < self._n_leaves:
            raise IndexError(f"leaf {i} outside 0..{self._n_leaves - 1}")
        return self._leaf_device[i] if self.devices is not None else HOST

    def loads(self) -> tuple[int, ...]:
        """Leaf bytes per slot (sharded only)."""
        if self.devices is None:
            raise ValueError("an unsharded placement has no slots")
        return self._loads


def split_units(sizes: list[int], pinned: int, slots: int) -> list[int]:
    """Cut ``sizes`` in order into ``slots`` contiguous ranges, minimizing the largest range's
    total with ``pinned`` added to the last one. Returns the ``slots - 1`` cut positions
    ``c`` (range ``s`` is ``sizes[c[s-1]:c[s]]``, ``c[-1] = len(sizes)`` implied). Among the
    optimal cuts it takes the lexicographically smallest, so earlier ranges are as short as the
    optimum allows."""
    n = len(sizes)
    prefix = [0]
    for x in sizes:
        prefix.append(prefix[-1] + x)
    # best[k][i]: the least largest total of sizes[i:] cut into k ranges, pinned in the last.
    best = [[0] * (n + 1) for _ in range(slots + 1)]
    for i in range(n + 1):
        best[1][i] = prefix[n] - prefix[i] + pinned
    for k in range(2, slots + 1):
        for i in range(n + 1):
            best[k][i] = min(max(prefix[j] - prefix[i], best[k - 1][j]) for j in range(i, n + 1))
    bound = best[slots][0]
    cuts, i = [], 0
    for k in range(slots, 1, -1):
        i = next(j for j in range(i, n + 1) if best[k - 1][j] <= bound)
        cuts.append(i)
    return cuts


def placement_for(c: DeclaredComputation, prover: torch.device | str,
                  devices: tuple[torch.device | str, ...] | None) -> Placement:
    """The placement of ``c``'s units and leaves over ``devices`` (``None``: unsharded)."""
    prover = torch.device(prover)
    bounds = dict(_units=frozenset(c.units), _weights=frozenset(c.weight_names),
                  _n_products=c.M, _n_product_leaves=c.n_product_leaves,
                  _n_leaves=c.n_leaves)
    if devices is None:
        return Placement(prover, None, **bounds)
    devs = tuple(torch.device(d) for d in devices)
    if not devs:
        raise ValueError("a sharded placement needs at least one device")
    w_item = torch.empty((), dtype=c.weight_dtype).element_size()
    p_item = torch.empty((), dtype=c.product_dtype).element_size()
    shapes = c.weight_shapes
    size = {j: 0 for j in c.units}
    for name in c.weight_names:
        size[c.weight_unit(name)] += 2 * math.prod(shapes[name]) * w_item
    product_unit = tuple(c.product_unit(p.m) for p in c.products)
    for p, j in zip(c.products, product_unit):
        size[j] += math.prod(p.p_shape) * p_item

    last = len(devs) - 1
    pinned = c.pinned_last_units
    free = [j for j in c.units if j not in pinned]
    cuts = split_units([size[j] for j in free], sum(size[j] for j in pinned), len(devs))
    slot = {j: last for j in pinned}
    lo = 0
    for s, hi in enumerate([*cuts, len(free)]):
        slot.update({j: s for j in free[lo:hi]})
        lo = hi

    weight_slot = {name: slot[c.weight_unit(name)] for name in c.weight_names}
    product_slot = tuple(slot[j] for j in product_unit)
    # Leaf order: records, W_t, product leaves, W_{t+1} (ref block §6, O18).
    w_slots = tuple(weight_slot[name] for name in c.weight_names)
    product_leaf_slot = tuple(product_slot[pl.m0 - 1] for pl in c.product_leaves)
    leaf_slots = w_slots + product_leaf_slot + w_slots
    loads = [0] * len(devs)
    for j, s in slot.items():
        loads[s] += size[j]
    return Placement(prover, devs, slot, weight_slot, product_slot,
                     (HOST,) * c.n_s + tuple(devs[s] for s in leaf_slots), tuple(loads),
                     _product_leaf_slot=product_leaf_slot, **bounds)
