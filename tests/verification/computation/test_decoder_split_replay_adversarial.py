"""The split replay (FS32c, F22), attacked: aliasing on every serving path, placement edge cases,
the pass's kwargs cache, model reuse, the tie, the prover's untouched builders and the real
SmolLM2 step over three host devices.

The Mac runs every sharded path on ``("cpu", "cpu", "cpu")``. Where a move must actually happen
to be counted, ``_FakePlacement`` maps units to ``cpu:0`` (``torch.device("cpu", 0) !=
torch.device("cpu")``, so a ``.to`` call there makes a new tensor) or to ``meta`` (the device
of a parameter shows where a module was placed).
"""

import contextlib
import copy
import importlib
import subprocess
import sys
import types
import weakref
from functools import lru_cache

import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.computation.instances import decoder as decoder_module
from verification.computation.instances.decoder import DecoderReplay
from verification.computation.interface import HOST, ReplayError, snapshot_weights
from verification.computation.placement import placement_for
from verification.prover.step import prove_step
from verification.transcript import reader as reader_module

from tests.verification.computation.test_decoder_split_replay import (
    CASE_IDS,
    CASES,
    DTYPES,
    THREE,
    ListReader,
    digest,
    prover_leaves,
    replay_all,
    tiny,
)
from tests.verification.decoder_helpers import (
    SWITCHES,
    make_records,
    randomize_biases,
    switch_id,
    tiny_config,
)

CPU0 = torch.device("cpu", 0)
RECORDS = make_records((12, 7))


# ---- helpers ------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _step(switch, dtype, B):
    """One prover step's leaves and the unsharded replay's digests, once per case."""
    c = tiny(switch, dtype, B)
    leaves = prover_leaves(c, RECORDS)
    _, ops, glue = replay_all(c, leaves)
    return leaves, ops, glue


def _noisy_leaves(c, records, seed):
    """A prover step from weights perturbed by ``seed``: leaves that differ from other steps'."""
    model = c.build_model()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return prove_step(c, model, snapshot_weights(c, model), records).leaves()


class _FakePlacement:
    """The real placement, except that ``unit_device(j)`` is ``devices[j]``."""

    sharded = True

    def __init__(self, real, devices):
        self._real, self._devices = real, devices

    def unit_device(self, j):
        return self._devices[j]

    def __getattr__(self, name):
        return getattr(self._real, name)


def _fake(monkeypatch, c, devices):
    monkeypatch.setitem(c.__dict__, "placement", _FakePlacement(c.placement, devices))
    return c.placement


def _leaf_state(leaves):
    ts = [t for t in leaves if isinstance(t, torch.Tensor)]
    return ts, [t.clone() for t in ts], [t._version for t in ts]


def _assert_untouched(state, what):
    ts, copies, versions = state
    bad = [i for i, (t, k) in enumerate(zip(ts, copies))
           if not torch.equal(t, k) and not (t.is_floating_point() and torch.equal(
               torch.nan_to_num(t), torch.nan_to_num(k)))]
    assert bad == [], f"{what}: leaf bytes changed at {bad[:5]}"
    assert [t._version for t in ts] == versions, f"{what}: a leaf's _version changed"


def _same_bytes(state):
    ts, copies, _ = state
    return [i for i, (t, k) in enumerate(zip(ts, copies)) if digest(t) != digest(k)]


def _spans(leaves):
    return [(t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            for t in leaves if isinstance(t, torch.Tensor) and t.numel()]


def _overlaps(t, spans):
    if not t.numel():
        return []
    lo = t.untyped_storage().data_ptr()
    hi = lo + t.untyped_storage().nbytes()
    return [i for i, (p, n) in enumerate(spans) if lo < p + n and p < hi]


def _runs(c):
    """Every attention role's members, as one group of ``m``."""
    return [[m0 + st * i for i in range(c.n_s * c.n_h)]
            for (_, _), (m0, st) in c.member_runs.items()]


def _write_all(*ts):
    with torch.no_grad():
        for t in ts:
            t.fill_(float("nan"))


_PATHS = [pytest.param(None, id="unsharded"), pytest.param(THREE, id="cpu-cpu-cpu")]
_ALIAS_CASES = [(("llama", True), "fp32", 2), (("qwen2", False), "bf16", 1)]
_ALIAS_IDS = ["llama-tied-fp32-B2", "qwen2-untied-bf16-B1"]


def _alias_replay(switch, dtype, B, devices):
    c = tiny(switch, dtype, B, devices=devices)
    leaves, _, _ = _step(switch, dtype, B)
    return c, leaves


# ---- operands never alias committed leaves --------------------------------------------------


@pytest.mark.parametrize("devices", _PATHS)
@pytest.mark.parametrize("case", _ALIAS_CASES, ids=_ALIAS_IDS)
def test_every_served_tensor_is_off_the_committed_storage(case, devices, monkeypatch):
    """Over one full canonical pass, and over every attention run through ``operand_group``
    (whole progressions, a reversed order that takes the member path, and a converting dtype):
    no tensor either supply, ``_dk_owned``, ``operands`` or ``operand_group`` hands out shares a
    storage with a committed leaf."""
    c, leaves = _alias_replay(*case, devices)
    spans, served, state = _spans(leaves), [], _leaf_state(leaves)
    orig_dk = decoder_module._dk_owned

    def dk(block, device):
        out = orig_dk(block, device)
        served.append(out)
        return out

    monkeypatch.setattr(decoder_module, "_dk_owned", dk)
    replay = c.replay(ListReader(leaves))
    for name in ("_supply", "_supply_backward"):
        orig = getattr(replay, name)
        setattr(replay, name, lambda *a, _o=orig: served.append(_o(*a)) or served[-1])
    for s in c.products:
        served.extend(replay.operands(s.m))
    replay.glue_gradients()
    for ms in _runs(c):
        for use in (ms, ms[::-1], ms[::2]):
            served.extend(replay.operand_group(use))
            served.extend(replay.operand_group(use, dtype=torch.float32))
    assert len(served) > 2 * c.M
    bad = [(tuple(t.shape), _overlaps(t, spans)[:3]) for t in served if _overlaps(t, spans)]
    assert bad == [], bad[:3]
    _write_all(*served)  # every supplied block, wherever the pass keeps it
    assert _same_bytes(state) == []
    _assert_untouched(state, "the supplies")


@pytest.mark.parametrize("devices", _PATHS)
@pytest.mark.parametrize("case", _ALIAS_CASES, ids=_ALIAS_IDS)
def test_writing_every_served_operand_leaves_the_leaves_alone(case, devices):
    """For each product, on a fresh replay: an in-place write to both operands changes no
    committed leaf's bytes or ``_version`` (single products: every forward and backward path,
    S/O and backward member blocks, and the weights' views)."""
    c, leaves = _alias_replay(*case, devices)
    state = _leaf_state(leaves)
    for s in c.products:
        a, b = c.replay(ListReader(leaves)).operands(s.m)
        _write_all(a, b)
        assert _same_bytes(state) == [], s.name
    _assert_untouched(state, "operands")


@pytest.mark.parametrize("devices", _PATHS)
@pytest.mark.parametrize("case", _ALIAS_CASES, ids=_ALIAS_IDS)
def test_writing_every_served_group_leaves_the_leaves_alone(case, devices):
    """The same through ``operand_group``: whole progressions, member-path orders, bf16 and fp32
    outputs."""
    c, leaves = _alias_replay(*case, devices)
    state = _leaf_state(leaves)
    for ms in _runs(c):
        for use in (ms, ms[::-1], ms[::2]):
            for dtype in (None, torch.float32, torch.bfloat16):
                a, b = c.replay(ListReader(leaves)).operand_group(use, dtype)
                _write_all(a, b)
                assert _same_bytes(state) == [], (use[:3], dtype)
    _assert_untouched(state, "operand_group")


@pytest.mark.parametrize("devices", _PATHS)
def test_the_replays_own_pass_does_not_write_the_leaves(devices):
    """A whole replay (every operand, the glue) leaves every leaf's bytes and ``_version``."""
    c, leaves = _alias_replay(("qwen2", True), "bf16", 2, devices)
    state = _leaf_state(leaves)
    replay_all(c, leaves)
    _assert_untouched(state, "the replay")


@pytest.mark.parametrize("shape", [(1, 4, 5), (3, 1, 5), (3, 4, 1), (1, 1, 1), (2, 3, 4)])
@pytest.mark.parametrize("strided", [False, True])
def test_dk_owned_on_its_own_device_is_fresh_for_any_layout(shape, strided):
    """``_dk_owned`` on the block's device clones whenever ``contiguous()`` is a no-op, also
    for a strided view of a bigger block."""
    block = torch.randn(*shape)
    if strided:
        big = torch.randn(shape[0] * 2, shape[1], shape[2] * 2)
        block = big[::2, :, ::2]
        block.copy_(torch.randn(*shape))
    snap = block.clone()
    out = decoder_module._dk_owned(block, block.device)
    assert torch.equal(out, snap.transpose(1, 2)) and out.is_contiguous()
    assert out.untyped_storage().data_ptr() != block.untyped_storage().data_ptr()
    out.fill_(7.0)
    assert torch.equal(block, snap)


def test_dk_owned_moved_to_another_device_is_the_copy():
    """A block moved to ``cpu:0`` (a different device value) is copied by the move; the result
    shares no storage with the source."""
    block = torch.randn(3, 1, 5)
    snap = block.clone()
    out = decoder_module._dk_owned(block, CPU0)
    assert out.untyped_storage().data_ptr() != block.untyped_storage().data_ptr()
    out.fill_(1.0)
    assert torch.equal(block, snap)


def test_zero_copy_leaves_really_are_in_play():
    """The aliasing tests bite: this prover's leaves give ``stack_block`` a zero-copy view of a
    committed storage when ``own`` is off (so a served alias would be caught)."""
    c, leaves = _alias_replay(("llama", True), "fp32", 2, None)
    spans = _spans(leaves)
    replay = c.replay(ListReader(leaves))
    (m0, st) = c.member_runs[(1, "S")]
    block = replay.view.product_run(m0, st, c.n_s * c.n_h, own=False)
    assert _overlaps(block, spans)
    owned = replay.view.product_run(m0, st, c.n_s * c.n_h, own=True)
    assert not _overlaps(owned, spans)


# ---- placement edge cases -------------------------------------------------------------------


@pytest.mark.parametrize("devices", [("cpu",), ("cpu",) * 4, ("cpu",) * 5],
                         ids=["one", "four", "five"])
@pytest.mark.parametrize("switch,dtype,B", CASES, ids=CASE_IDS)
def test_odd_slot_counts_replay_as_the_unsharded_one_does(switch, dtype, B, devices):
    """A 2-layer decoder over one, four or five slots (some left empty): every operand and glue
    gradient equal by digest to the unsharded replay, and no leaf moved."""
    leaves, ops, glue = _step(switch, dtype, B)
    c = tiny(switch, dtype, B, devices=devices)
    slots = {c.placement.unit_slot(j) for j in c.units}
    if len(devices) > 1:
        assert len(slots) < len(devices) and c.placement.sharded
    replay, ops_s, glue_s = replay_all(c, leaves)
    assert [s.name for s, a, b in zip(c.products, ops, ops_s) if a != b] == []
    assert [n for n in glue if glue[n] != glue_s[n]] == []
    assert replay.leaf_moves == 0


# ---- the pass's kwargs cache ----------------------------------------------------------------


def _three_layer(switch=("llama", True), dtype="fp32", B=1, devices=THREE):
    cfg = tiny_config(family=switch[0], tied=switch[1])
    cfg.num_hidden_layers = 3
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = ["full_attention"] * 3
    return DecoderComputation(cfg, n_s=2, n=12, eta=1e-3, compute_dtype=DTYPES[dtype],
                              head_blocks=B, verifier_devices=devices)


def _route_kwargs_to_units(monkeypatch):
    """CPU tensors report ``cpu`` whatever device they were moved to, so the pre-hook's
    ``x.device`` can never name the fake ``cpu:0``. This asks ``_on`` for the entering unit's
    own (fake) device instead, so the hook, the cache and the pass all run for real."""
    enter, on = DecoderReplay._enter_unit, DecoderReplay._on
    now = []

    def entering(self, j, x):
        now[:] = [j]
        return enter(self, j, x)

    def routed(self, v, device):
        return on(self, v, self.pl.unit_device(now[0]) if now else device)

    monkeypatch.setattr(DecoderReplay, "_enter_unit", entering)
    monkeypatch.setattr(DecoderReplay, "_on", routed)


class _Moves:
    """Counts ``Tensor.to`` calls made inside ``DecoderReplay._on``, by (tensor id, device)."""

    def __init__(self, monkeypatch):
        self.counts, self.depth, self.held = {}, 0, []
        orig_on, orig_to = DecoderReplay._on, torch.Tensor.to
        me = self

        def on(replay, v, device):
            me.depth += 1
            try:
                return orig_on(replay, v, device)
            finally:
                me.depth -= 1

        def to(t, *a, **k):
            if me.depth:
                key = (id(t), str(a[0] if a else k.get("device")))
                me.counts[key] = me.counts.get(key, 0) + 1
                me.held.append(t)  # keep ids unique
            return orig_to(t, *a, **k)

        monkeypatch.setattr(DecoderReplay, "_on", on)
        monkeypatch.setattr(torch.Tensor, "to", to)


def test_each_shared_tensor_moves_once_per_device_per_pass(monkeypatch):
    """Three layers, layers 1 and 3 on one non-host device: every kwarg tensor (RoPE tables,
    mask, ids) moves to that device once, though two layers read it."""
    c = _three_layer()
    leaves = prover_leaves(_three_layer(devices=None), RECORDS)
    _fake(monkeypatch, c, {0: HOST, 1: CPU0, 2: HOST, 3: CPU0, 4: HOST})
    _route_kwargs_to_units(monkeypatch)
    moves = _Moves(monkeypatch)
    replay = c.replay(ListReader(leaves))
    replay.operands(c.M)
    assert moves.counts, "no kwargs tensor was moved: the test is not exercising the cache"
    assert set(moves.counts.values()) == {1}, moves.counts
    assert {d for _, d in moves.counts} == {str(CPU0)}
    # RoPE cos and sin, the mask and the position ids: the pass's shared kwargs tensors
    assert len(moves.counts) >= 3


def test_the_cache_is_empty_after_the_pass_and_after_an_exception(monkeypatch):
    c = _three_layer()
    leaves = prover_leaves(_three_layer(devices=None), RECORDS)
    _fake(monkeypatch, c, {0: HOST, 1: CPU0, 2: HOST, 3: CPU0, 4: HOST})
    _route_kwargs_to_units(monkeypatch)
    replay = c.replay(ListReader(leaves))
    seen = []
    orig = DecoderReplay._on

    def on(s, v, d):
        out = orig(s, v, d)
        seen.append(len(s._moved))
        return out

    monkeypatch.setattr(DecoderReplay, "_on", on)
    replay.operands(1)
    assert replay._built and replay._moved == {} and max(seen) > 0
    boom = DecoderReplay._leave_unit

    def leave(self, j, out):
        if j == 2:
            raise RuntimeError("boom")
        return boom(self, j, out)

    monkeypatch.setattr(DecoderReplay, "_leave_unit", leave)
    failing = c.replay(ListReader(leaves))
    with pytest.raises(RuntimeError, match="boom"):
        failing.operands(1)
    assert failing._moved == {} and not failing._built and failing._units == {}


def test_a_second_steps_replay_does_not_reuse_the_first_steps_moved_tensors(monkeypatch):
    """Two steps on one computation: the second pass builds its own moved tensors (none of the
    first's), and serves the values a fresh computation serves."""
    c = _three_layer()
    other = _three_layer(devices=None)
    la, lb = _noisy_leaves(other, RECORDS, 1), _noisy_leaves(other, make_records((9, 12)), 2)
    _fake(monkeypatch, c, {0: HOST, 1: CPU0, 2: HOST, 3: CPU0, 4: HOST})
    _route_kwargs_to_units(monkeypatch)
    produced = []
    orig = DecoderReplay._on

    def on(self, v, device):
        out = orig(self, v, device)
        produced.append((self, out))
        return out

    monkeypatch.setattr(DecoderReplay, "_on", on)
    first = c.replay(ListReader(la))
    first.operands(c.M)
    mark = len(produced)
    second = c.replay(ListReader(lb))
    got = [tuple(digest(x) for x in second.operands(s.m)) for s in c.products]
    flat = lambda o: [o] if isinstance(o, torch.Tensor) else [x for y in (o or ()) for x in flat(y)]
    moved_first = [t for r, o in produced[:mark] if r is first for t in flat(o)]
    moved_second = [t for r, o in produced[mark:] if r is second for t in flat(o)]
    assert moved_first and moved_second
    assert not [1 for t in moved_second for u in moved_first if t is u]
    assert second._moved == {} and first._moved == {}
    monkeypatch.undo()
    fresh = _three_layer()
    want = [tuple(digest(x) for x in fresh.replay(ListReader(lb)).operands(s.m))
            for s in fresh.products]
    assert got == want


# ---- model reuse across steps -------------------------------------------------------------------


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
def test_two_steps_on_a_reused_placed_model_equal_two_fresh_replays(switch):
    """Over ``cpu,cpu,cpu``: step 1 then step 2 on one computation (the model reused, placed)
    equals each on a fresh computation, operand by operand and glue gradient by glue gradient;
    the previous owner has lost the model, and the buffers sit with their module."""
    c = tiny(switch, "fp32", 2, devices=THREE)
    plain = tiny(switch, "fp32", 2)
    la, lb = _noisy_leaves(plain, RECORDS, 1), _noisy_leaves(plain, make_records((9, 12)), 2)
    r1, ops1, g1 = replay_all(c, la)
    model = r1.model
    r2, ops2, g2 = replay_all(c, lb)
    assert r2.model is model and r1.model is None
    f1 = replay_all(tiny(switch, "fp32", 2, devices=THREE), la)
    f2 = replay_all(tiny(switch, "fp32", 2, devices=THREE), lb)
    assert (ops1, g1) == f1[1:] and (ops2, g2) == f2[1:]
    assert ops1 != ops2
    for name, buf in model.named_buffers():
        assert buf.device == model.model.embed_tokens.weight.device, name
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight or \
        not c.tied


class _Owner:
    model = None


@pytest.mark.parametrize("tied,first", [(True, HOST), (False, HOST), (False, torch.device("meta"))],
                         ids=["tied-host-ends", "untied-host-ends", "untied-meta-ends"])
def test_the_placed_model_keeps_every_tensor_on_its_module_through_reuse(tied, first,
                                                                          monkeypatch):
    """Embedding and rotary table on unit 0's device, layer ``j`` on unit ``j``'s, the final norm
    and the output projection on unit ``L+1``'s, before and after a reuse (fresh parameter
    storage, restored buffers); the tie is one ``Parameter`` when tied."""
    other = torch.device("meta") if first == HOST else HOST
    c = tiny(("llama", tied), "fp32", 1, devices=THREE)
    units = {0: first, 1: other, 2: first, 3: first}
    pl = _fake(monkeypatch, c, units)
    a, b = _Owner(), _Owner()
    for owner in (a, b):
        model = c._replay_model(owner)
        inner = model.model
        assert {p.device for p in inner.embed_tokens.parameters()} == {pl.unit_device(0)}
        assert {x.device for x in inner.rotary_emb.buffers()} == {pl.unit_device(0)}
        assert {p.device for p in inner.layers[0].parameters()} == {pl.unit_device(1)}
        assert {p.device for p in inner.layers[1].parameters()} == {pl.unit_device(2)}
        assert {p.device for p in inner.norm.parameters()} == {pl.unit_device(3)}
        assert {p.device for p in model.lm_head.parameters()} == {pl.unit_device(3)}
        assert (model.lm_head.weight is inner.embed_tokens.weight) == tied
        for n, buf in model.named_buffers():
            assert buf.device == pl.unit_device(0), n
        assert all(p.requires_grad and p.grad is None for p in model.parameters())
        assert a.model is None or owner is a
    assert c._vowner() is b


def test_an_untied_head_goes_to_unit_l_plus_1_and_a_tied_one_refuses_a_split(monkeypatch):
    """``lm_head`` follows unit ``L+1``, the embedding unit 0. Were they on two devices, a tied
    model would have to untie: the placement refuses."""
    meta = torch.device("meta")
    c = tiny(("llama", False), "fp32", 1, devices=THREE)
    _fake(monkeypatch, c, {0: HOST, 1: HOST, 2: HOST, 3: meta})
    model = c._place_model(c.build_model(HOST))
    assert model.lm_head.weight.device == meta
    assert model.model.norm.weight.device == meta
    assert model.model.embed_tokens.weight.device == HOST
    t = tiny(("llama", True), "fp32", 1, devices=THREE)
    _fake(monkeypatch, t, {0: HOST, 1: HOST, 2: HOST, 3: meta})
    with pytest.raises(ReplayError, match="tie"):
        t._place_model(t.build_model(HOST))


def test_a_sharded_model_is_never_built_on_the_provers_device(monkeypatch):
    """Sharded, the verifier's model is built on the host; unsharded, on the run's device."""
    built = []
    orig = DecoderComputation.build_model

    def build(self, device=None):
        built.append(device)
        return orig(self, device)

    monkeypatch.setattr(DecoderComputation, "build_model", build)
    leaves, _, _ = _step(("llama", True), "fp32", 1)
    tiny(("llama", True), "fp32", 1, devices=THREE).replay(ListReader(leaves))
    tiny(("llama", True), "fp32", 1).replay(ListReader(leaves))
    assert built == [HOST, None]


# ---- the exact compare after the move ------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("fake", [False, True], ids=["cpu-cpu-cpu", "moving"])
def test_the_unit_boundary_compare_stays_exact_after_the_move(dtype, fake, monkeypatch):
    """One ulp on one element of layer 2's input is refused; the same input unchanged passes."""
    c = tiny(("llama", True), dtype, 1, devices=THREE)
    leaves, _, _ = _step(("llama", True), dtype, 1)
    if fake:
        _fake(monkeypatch, c, {0: HOST, 1: CPU0, 2: HOST, 3: HOST})
    replay = c.replay(ListReader(leaves))
    layer2 = replay.model.model.layers[1]
    h = layer2.register_forward_pre_hook(lambda mod, args: (args[0].clone(),))
    replay.operands(c.m_of("L1.Y_q"))
    h.remove()

    def ulp(mod, args):
        x = args[0].clone()
        x.view(-1)[0] = torch.nextafter(x.view(-1)[0], torch.tensor(float("inf"), dtype=x.dtype))
        return (x,)

    replay = c.replay(ListReader(leaves))
    replay.model.model.layers[1].register_forward_pre_hook(ulp)
    with pytest.raises(ReplayError, match="layer 1's output is not X_2"):
        replay.operands(c.m_of("L1.Y_q"))


# ---- the prover is untouched by the optional ``device`` -----------------------------------------


def _old_decoder():
    """The decoder module as it was before the coder's first commit, loaded under another name."""
    root = decoder_module.__file__
    try:
        top = subprocess.run(["git", "-C", str(root.rsplit("/verification/", 1)[0]), "rev-parse",
                              "--show-toplevel"], capture_output=True, text=True,
                             check=True).stdout.strip()
        src = subprocess.run(["git", "-C", top, "show",
                              "b307061~1:verification/computation/instances/decoder.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("the pre-FS32c decoder is not in git history here")
    mod = types.ModuleType("decoder_before_fs32c")
    mod.__file__ = root
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src, "decoder_before_fs32c.py", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    # The old decoder declares attention members but no product groups (O18); this test only
    # builds models and batches, so it skips the layout check.
    mod.DecoderComputation.validate = lambda self: None
    return mod


@pytest.mark.parametrize("switch", SWITCHES, ids=switch_id)
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_build_model_and_assemble_without_device_are_what_they_were(switch, dtype):
    """Called as the prover calls them, the shared builders return the same model (bytes,
    dtypes, device, tie) and batch as the pre-FS32c code, and the new ``device`` argument
    defaulting to the run's is the same thing."""
    old = _old_decoder()
    family, tied = switch
    kw = dict(n_s=2, n=12, eta=1e-3, compute_dtype=DTYPES[dtype], head_blocks=2)
    cfg = tiny_config(family=family, tied=tied)
    c_old = old.DecoderComputation(cfg, **kw)
    c_new = DecoderComputation(cfg, **kw)
    torch.manual_seed(5)
    ref = c_old.build_model()
    torch.manual_seed(5)
    new = c_new.build_model()
    torch.manual_seed(5)
    explicit = c_new.build_model(c_new.device)
    assert ref.state_dict().keys() == new.state_dict().keys()
    for n, t in ref.state_dict().items():
        for other in (new, explicit):
            o = other.state_dict()[n]
            assert (t.dtype, t.device, t.shape) == (o.dtype, o.device, o.shape), n
            assert digest(t) == digest(o), n
    for m in (ref, new, explicit):
        assert (m.lm_head.weight is m.model.embed_tokens.weight) == tied
        assert not m.training
    recs = make_records((12, 7))
    b_old, b_new = c_old.assemble(recs), c_new.assemble(recs)
    fields = ("ids", "targets", "mask", "rho")
    for x, y in ((getattr(b_old, f), getattr(b_new, f)) for f in fields):
        assert x.dtype == y.dtype and x.device == y.device and torch.equal(x, y)
    b_dev = c_new.assemble(recs, c_new.device)
    assert all(torch.equal(getattr(b_new, f), getattr(b_dev, f)) for f in fields)


# ---- the real step -----------------------------------------------------------------------------


@pytest.mark.slow
def test_smollm2_real_step_split_over_three_host_devices_equals_unsharded():
    """The real SmolLM2 step at test scale from ``W_0`` on ``π(1)``: replayed unsharded and over
    ``cpu,cpu,cpu``; every operand and glue gradient equal by digest, no leaf moved."""
    import dataclasses
    import os

    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records, schedule

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}; set VERIF_OUTPUT_DIR")
    setup_determinism(cfg)
    base = dataclasses.replace(cfg, device="cpu", verifier_devices=None)
    plain = DecoderComputation.from_config(base)
    D = load_dataset_records(cfg.data_dir / "D.bin", plain.n)
    records = [D[i] for i in schedule(1, plain.n_s, len(D), cfg.seed)]
    leaves = prover_leaves(plain, records)
    split = DecoderComputation.from_config(dataclasses.replace(base, verifier_devices=THREE))
    assert split.placement.sharded and not plain.placement.sharded
    r0, ops, glue = replay_all(plain, leaves)
    r1, ops_s, glue_s = replay_all(split, leaves)
    assert [s.name for s, x, y in zip(plain.products, ops, ops_s) if x != y] == []
    assert [n for n in glue if glue[n] != glue_s[n]] == []
    assert len(ops) == plain.M and r0.leaf_moves == r1.leaf_moves == 0
    assert r1.model.lm_head.weight is r1.model.model.embed_tokens.weight
