"""The placement rule (FS32a, F22): units, slots and the device of every leaf."""

import itertools
import math
import random

import pytest
import torch

from setup.config import load_config
from verification.computation.instances import DecoderComputation
from verification.computation.instances.mlp import DEFAULT_WIDTHS, MLPComputation
from verification.computation.interface import HOST
from verification.computation.placement import placement_for, split_units

from tests.verification.decoder_helpers import SWITCHES, switch_id, tiny_config

CPU = torch.device("cpu")
THREE = ("cpu", "cpu", "cpu")
FP32 = 4  # bytes: the tiny decoders run at (fp32, fp32)


def tiny(switch, *, layers=2, B=1, devices=None):
    family, tied = switch
    config = tiny_config(family=family, tied=tied)
    config.num_hidden_layers = layers
    return DecoderComputation(config, n_s=2, n=8, eta=1e-3, head_blocks=B,
                              verifier_devices=devices)


def unit_bytes(c):
    """Each unit's leaf bytes: its weights twice, its products once."""
    size = {j: 0 for j in c.units}
    for n in c.weight_names:
        size[c.weight_unit(n)] += 2 * FP32 * math.prod(c.weight_shapes[n])
    for s in c.products:
        size[c.product_unit(s.m)] += FP32 * math.prod(s.p_shape)
    return size


def brute_force(sizes, pinned, slots):
    """The lexicographically smallest cut vector among all that minimize the largest total."""
    best = None
    for cuts in itertools.combinations_with_replacement(range(len(sizes) + 1), slots - 1):
        edges = (0, *cuts, len(sizes))
        loads = [sum(sizes[a:b]) for a, b in zip(edges, edges[1:])]
        loads[-1] += pinned
        key = (max(loads), cuts)
        if best is None or key < best:
            best = key
    return list(best[1])


def test_unsharded_is_the_prover_and_the_host():
    c = tiny(("llama", True))
    p = c.placement
    assert not p.sharded and p.prover == CPU
    for j in c.units:
        assert p.unit_slot(j) == 0 and p.unit_device(j) == CPU
    assert p.weight_device(c.weight_names[0]) == HOST
    assert p.product_device(1) == HOST
    assert all(p.leaf_device(i) == HOST for i in range(c.n_leaves))
    with pytest.raises(ValueError):
        p.loads()


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("layers", [2, 5])
def test_sharded_decoder(switch, B, layers):
    c = tiny(switch, layers=layers, B=B, devices=THREE)
    p, L = c.placement, c.L
    assert p.sharded and c.units == tuple(range(L + 2))
    assert c.pinned_last_units == {0, L + 1}
    slots = [p.unit_slot(j) for j in c.units]
    assert slots[0] == slots[L + 1] == 2
    assert slots[1:L + 1] == sorted(slots[1:L + 1])  # contiguous and in order
    for j in c.units:
        assert p.unit_device(j) == CPU
    for m in range(1, c.M + 1):
        spec = c.product(m)
        j = L + 1 if spec.layer is None else spec.layer
        assert c.product_unit(m) == j
        assert p.product_device(m) == p.unit_device(j)
        assert p._product_slot[m - 1] == p.unit_slot(j)
    for l in range(1, L + 1):
        names = [c.gamma_attn(l), c.gamma_mlp(l),
                 *(c.w(l, x) for x in ("q", "k", "v", "o", "gate", "up", "down"))]
        names += [b for x in ("q", "k", "v") if (b := c.bias(l, x))]
        for n in names:
            assert c.weight_unit(n) == l and p._weight_slot[n] == p.unit_slot(l)
    assert c.weight_unit(c.w_e) == 0 and c.weight_unit(c.gamma_final) == L + 1
    if not c.tied:
        assert c.weight_unit(c.w_out) == L + 1
    # every leaf index is mapped: records on the host, weights and products on their unit's
    assert all(p.leaf_device(i) == HOST for i in range(c.n_s))
    assert [p.leaf_device(i) for i in range(c.n_leaves)] == list(p._leaf_device)
    assert sum(p.loads()) == sum(unit_bytes(c).values())
    with pytest.raises(KeyError):
        c.weight_unit("nope")
    with pytest.raises(IndexError):
        c.product_unit(c.M + 1)


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
@pytest.mark.parametrize("slots", [1, 2, 3, 4])
def test_dp_matches_brute_force_on_decoders(switch, slots):
    c = tiny(switch, layers=5, B=2, devices=("cpu",) * slots)
    size = unit_bytes(c)
    free = [size[j] for j in range(1, c.L + 1)]
    cuts = brute_force(free, size[0] + size[c.L + 1], slots)
    edges = (0, *cuts, len(free))
    want = [s for s, (a, b) in enumerate(zip(edges, edges[1:])) for _ in range(b - a)]
    assert [c.placement.unit_slot(j) for j in range(1, c.L + 1)] == want


def test_split_units_matches_brute_force_on_random_sizes():
    rng = random.Random(0)
    for _ in range(300):
        sizes = [rng.randint(0, 9) for _ in range(rng.randint(0, 7))]
        slots, pinned = rng.randint(1, 5), rng.randint(0, 20)
        assert split_units(sizes, pinned, slots) == brute_force(sizes, pinned, slots)


def test_one_device_holds_everything():
    c = tiny(("qwen2", False), layers=3, devices=("cpu",))
    assert {c.placement.unit_slot(j) for j in c.units} == {0}
    assert c.placement.loads() == (sum(unit_bytes(c).values()),)


def test_more_slots_than_layers():
    c = tiny(("llama", True), layers=2, devices=("cpu",) * 6)
    p = c.placement
    assert p.unit_slot(0) == p.unit_slot(3) == 5
    assert 0 <= p.unit_slot(1) <= p.unit_slot(2) <= 5
    assert len(p.loads()) == 6 and len(p._leaf_device) == c.n_leaves


def test_mlp_is_one_unit_in_the_last_slot():
    c = MLPComputation(DEFAULT_WIDTHS, n_s=4, eta=0.05)
    assert c.units == (0,) and c.pinned_last_units == frozenset()
    assert not c.placement.sharded
    p = placement_for(c, CPU, (CPU, CPU, CPU))
    assert p.unit_slot(0) == 2
    assert {p._weight_slot[n] for n in c.weight_names} == {2}
    assert set(p._product_slot) == {2}
    assert p.loads()[:2] == (0, 0)
    with pytest.raises(KeyError):
        c.weight_unit("nope")
    with pytest.raises(IndexError):
        c.product_unit(0)


def test_from_config_passes_the_setting_through():
    base = {"VERIF_BATCH": "2", "VERIF_SEQ_LEN": "8"}
    off = DecoderComputation.from_config(load_config(base))
    assert off.verifier_devices is None and not off.placement.sharded
    on = DecoderComputation.from_config(
        load_config({**base, "VERIF_VERIFIER_DEVICES": "cpu, cpu"}))
    assert on.verifier_devices == (CPU, CPU) and on.placement.sharded
