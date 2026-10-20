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


# ---- FS32a tester additions ------------------------------------------------------------


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("slots", [1, 2, 3, 4])
def test_leaf_map_matches_an_independent_derivation(switch, B, slots):
    c = tiny(switch, layers=4, B=B, devices=("cpu",) * slots)
    p = c.placement
    # use distinguishable device objects: cpu:0, cpu:1, ... would not exist; compare slots
    # through the placement's own slot arrays, derived independently from the index helpers
    expect = {}
    for n in c.weight_names:
        s = p.unit_slot(c.weight_unit(n))
        expect[c.w_t_index(n)] = s
        expect[c.w_next_index(n)] = s
    for s_ in c.products:
        expect[c.product_index(s_.m)] = p.unit_slot(c.product_unit(s_.m))
    assert sorted(expect) == list(range(c.n_s, c.n_leaves))  # exactly the non-record leaves
    assert len(p._leaf_device) == c.n_leaves
    for i in range(c.n_s):
        assert p.leaf_device(i) == HOST
    for i, s in expect.items():
        assert p.leaf_device(i) == p.devices[s]
    # W_t and W_{t+1} of one weight sit together; each leaf's role matches its unit
    for n in c.weight_names:
        assert p.leaf_device(c.w_t_index(n)) == p.leaf_device(c.w_next_index(n)) \
            == p.weight_device(n)
    for s_ in c.products:
        assert p.leaf_device(c.product_index(s_.m)) == p.product_device(s_.m)
    with pytest.raises(IndexError):
        p.leaf_device(c.n_leaves)


def test_leaf_map_distinguishes_slots():
    """Same as above with real slot identities, via a stand-in device list."""
    c = tiny(("qwen2", False), layers=6, B=2, devices=("cpu", "cpu", "cpu"))
    p = c.placement
    fake = placement_for(c, CPU, tuple(torch.device("cuda", i) for i in range(3)))
    assert fake.devices[2].index == 2
    assert all(fake.leaf_device(i) == HOST for i in range(c.n_s))
    for n in c.weight_names:
        s = fake.unit_slot(c.weight_unit(n))
        assert fake.leaf_device(c.w_t_index(n)) == torch.device("cuda", s)
        assert fake.leaf_device(c.w_next_index(n)) == torch.device("cuda", s)
    for s_ in c.products:
        assert fake.leaf_device(c.product_index(s_.m)) == \
            torch.device("cuda", fake.unit_slot(c.product_unit(s_.m)))
    assert [fake.unit_slot(j) for j in c.units] == [p.unit_slot(j) for j in c.units]


def test_split_units_matches_brute_force_many_seeds():
    for seed in range(40):
        rng = random.Random(seed)
        for _ in range(60):
            sizes = [rng.choice([0, 1, 2, 5, 17, 100]) for _ in range(rng.randint(0, 8))]
            slots, pinned = rng.randint(1, 6), rng.choice([0, 0, 3, 40, 500])
            got = split_units(sizes, pinned, slots)
            assert got == brute_force(sizes, pinned, slots), (seed, sizes, pinned, slots)
            assert len(got) == slots - 1 and got == sorted(got)


def test_split_units_more_slots_than_units():
    assert split_units([], 5, 3) == [0, 0]
    assert split_units([7], 0, 4) == brute_force([7], 0, 4)
    got = split_units([3, 3], 0, 5)
    assert got == brute_force([3, 3], 0, 5) and len(got) == 4
    # the largest total is the best possible
    edges = (0, *got, 2)
    assert max(sum([3, 3][a:b]) for a, b in zip(edges, edges[1:])) == 3


def test_split_units_pinned_bytes_push_work_off_the_last_slot():
    # without a pin the last slot would take 50; with 1000 pinned it takes none
    got = split_units([50, 50, 50, 50], 1000, 2)
    assert got == [4]
    assert split_units([50, 50, 50, 50], 0, 2) == [2]


def _optimal(sizes, pinned, slots):
    best = min(max([sum(sizes[a:b]) for a, b in zip((0, *c), (*c, len(sizes)))][:-1] +
                   [sum(sizes[([0, *c][-1]):]) + pinned], default=0)
               for c in itertools.combinations_with_replacement(range(len(sizes) + 1), slots - 1))
    return best


def test_split_units_reaches_the_optimal_maximum():
    rng = random.Random(5)
    for _ in range(200):
        sizes = [rng.randint(0, 30) for _ in range(rng.randint(1, 8))]
        slots, pinned = rng.randint(1, 5), rng.randint(0, 60)
        cuts = split_units(sizes, pinned, slots)
        edges = (0, *cuts, len(sizes))
        loads = [sum(sizes[a:b]) for a, b in zip(edges, edges[1:])]
        loads[-1] += pinned
        assert max(loads) == _optimal(sizes, pinned, slots)


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_pinned_bytes_are_counted_in_the_last_slot(switch):
    c = tiny(switch, layers=6, B=2, devices=THREE)
    p = c.placement
    size = unit_bytes(c)
    want = [0, 0, 0]
    for j in c.units:
        want[p.unit_slot(j)] += size[j]
    assert list(p.loads()) == want
    assert p.loads()[2] >= size[0] + size[c.L + 1]


def _cached_config(repo, revision):
    from setup.model import load_model_config
    try:
        return load_model_config(repo, revision)
    except Exception as e:  # not in the local cache
        pytest.skip(f"{repo}@{revision[:8]} not cached: {e}")


def _full_scale_cases():
    from setup.config import load_config
    from tests.verification.full_scale_helpers import PINS
    smol = load_config({})
    return [(smol.model, smol.model_revision), *PINS]


@pytest.mark.parametrize("repo,revision", _full_scale_cases(),
                         ids=lambda v: v.split("/")[-1][:20])
def test_cached_configs_place_contiguously_with_exact_loads(repo, revision):
    config = _cached_config(repo, revision)
    c = DecoderComputation(config, n_s=128, n=128, eta=1e-3, source=(repo, revision),
                           master_dtype=torch.float32, compute_dtype=torch.bfloat16,
                           head_blocks=8, verifier_devices=("cpu", "cpu", "cpu"))
    p, L = c.placement, c.L
    slots = [p.unit_slot(j) for j in range(1, L + 1)]
    assert slots == sorted(slots) and set(slots) <= {0, 1, 2}
    assert p.unit_slot(0) == p.unit_slot(L + 1) == 2  # pinned last
    # each slot's load is the sum of its leaves' bytes, counted from the leaf side
    w_item, p_item = 4, 2
    want = [0, 0, 0]
    for n in c.weight_names:
        want[p.unit_slot(c.weight_unit(n))] += 2 * w_item * math.prod(c.weight_shapes[n])
    for s in c.products:
        want[p.unit_slot(c.product_unit(s.m))] += p_item * math.prod(s.p_shape)
    assert list(p.loads()) == want
    assert sum(want) > 0
    loads = list(p.loads())
    lb = {j: 0 for j in c.units}
    for n in c.weight_names:
        lb[c.weight_unit(n)] += 2 * w_item * math.prod(c.weight_shapes[n])
    for s_ in c.products:
        lb[c.product_unit(s_.m)] += p_item * math.prod(s_.p_shape)
    # leaf side: every leaf's device agrees with its unit
    for n in c.weight_names:
        assert p.leaf_device(c.w_t_index(n)) == p.weight_device(n)
    # the transcript's total bytes
    assert sum(loads) == sum(lb.values())


def test_unsharded_matches_the_base_class_for_every_public_method():
    for switch in SWITCHES:
        c = tiny(switch)
        p = c.placement
        assert p.devices is None and not p.sharded
        assert c.verifier_devices is None
        assert c.units == tuple(range(c.L + 2))
        for j in c.units:
            assert p.unit_slot(j) == 0 and p.unit_device(j) == CPU
        for n in c.weight_names:
            assert p.weight_device(n) == HOST
        for m in range(1, c.M + 1):
            assert p.product_device(m) == HOST
        assert all(p.leaf_device(i) == HOST for i in range(c.n_leaves))
    # the empty list is the same as None
    c = tiny(("llama", True), devices=())
    assert c.verifier_devices is None and not c.placement.sharded


def test_placement_does_not_change_other_instance_state():
    a = tiny(("qwen2", True), layers=3)
    b = tiny(("qwen2", True), layers=3, devices=THREE)
    assert a.weight_names == b.weight_names and a.weight_shapes == b.weight_shapes
    assert a.n_leaves == b.n_leaves and a.M == b.M
    assert [s.p_shape for s in a.products] == [s.p_shape for s in b.products]


def test_placement_is_built_once_and_survives_deepcopy():
    import copy
    c = tiny(("llama", True), devices=THREE)
    assert c.placement is c.placement
    d = copy.deepcopy(c)
    assert d.placement.devices == c.placement.devices
    assert list(d.placement.loads()) == list(c.placement.loads())


def test_base_defaults_on_the_mlp():
    c = MLPComputation(DEFAULT_WIDTHS, n_s=4, eta=0.05)
    assert c.placement.prover == HOST and c.units == (0,)
    for n in c.weight_names:
        assert c.weight_unit(n) == 0
    for m in range(1, c.M + 1):
        assert c.product_unit(m) == 0
    for bad in (0, c.M + 1):
        with pytest.raises(IndexError):
            c.product_unit(bad)


def test_empty_device_tuple_is_refused_by_placement_for():
    c = tiny(("llama", True))
    with pytest.raises(ValueError):
        placement_for(c, CPU, ())
