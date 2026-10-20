"""The split replay (FS32c, F22): the verifier's replay run over several devices.

Each unit (embedding, layers, head) runs on ``c.placement.unit_device(j)``; only ``X_j`` going
up and ``δX_{j+1}`` going down cross units. Sharded changes where bytes sit, never what they are,
so every operand and glue gradient must equal the unsharded replay's bit for bit. The Mac runs
the sharded paths on ``("cpu", "cpu", "cpu")`` (moves are no-ops, the code runs), a smoke test
over ``(cpu, mps)`` where MPS exists, and the CUDA tests (two devices, skipped otherwise) compare
by digest.
"""

import hashlib

import pytest
import torch

from verification.computation.instances import DecoderComputation
from verification.computation.instances.decoder import DecoderReplay
from verification.computation.interface import snapshot_weights
from verification.computation.placement import placement_for
from verification.computation.substitution import SubstitutionError
from verification.prover.step import prove_step

from tests.verification.decoder_helpers import (
    SWITCHES,
    make_records,
    randomize_biases,
    switch_id,
    tiny_config,
    verifier_devices,
)

THREE = ("cpu", "cpu", "cpu")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CASES = [(s, d, b) for s in SWITCHES for d in DTYPES for b in (1, 2)]
CASE_IDS = [f"{switch_id(s)}-{d}-B{b}" for s, d, b in CASES]
two_cuda = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")


class ListReader:
    def __init__(self, leaves):
        self.leaves = list(leaves)

    def leaf(self, index):
        return self.leaves[index]


def digest(t):
    """SHA-256 over a tensor's dtype, shape and bytes, wherever it sits."""
    t = t.detach().contiguous().cpu()
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.digest()


def tiny(switch, dtype, B, device="cpu", devices=None):
    family, tied = switch
    return DecoderComputation(tiny_config(family=family, tied=tied), n_s=2, n=12, eta=1e-3,
                              compute_dtype=DTYPES[dtype], device=device, head_blocks=B,
                              verifier_devices=devices)


def prover_leaves(c, records, seed=0):
    """One prover step on ``c``'s device: the step's leaves (all on the host)."""
    model = c.build_model()
    randomize_biases(dict(model.named_parameters()), seed)
    return prove_step(c, model, snapshot_weights(c, model), records).leaves()


def replay_all(c, leaves):
    """The replay, every operand's digests in canonical order, then every glue gradient's."""
    replay = c.replay(ListReader(leaves))
    ops = [tuple(digest(x) for x in replay.operands(s.m)) for s in c.products]
    got = replay.glue_gradients()
    return replay, ops, {n: digest(g) for n, g in got.items()}


@pytest.mark.parametrize("switch,dtype,B", CASES, ids=CASE_IDS)
def test_split_replay_equals_unsharded_on_the_host(switch, dtype, B):
    """Two replays of one step's leaves, unsharded and over three (host) devices: every operand
    and every glue gradient bit-equal, and no committed leaf moved."""
    c = tiny(switch, dtype, B)
    leaves = prover_leaves(c, make_records((12, 7)))
    cs = tiny(switch, dtype, B, devices=THREE)
    assert cs.placement.sharded and not c.placement.sharded
    plain, ops, glue = replay_all(c, leaves)
    split, ops_s, glue_s = replay_all(cs, leaves)
    assert len(ops) == c.M and list(glue) == list(c.glue_gradient_weights)
    assert [s.name for s, a, b in zip(c.products, ops, ops_s) if a != b] == []
    assert [n for n in glue if glue[n] != glue_s[n]] == []
    assert plain.leaf_moves == split.leaf_moves == 0


@pytest.mark.parametrize("dtype", list(DTYPES))
def test_unit_boundary_moves_are_requested(dtype, monkeypatch):
    """One move per unit going in (forward, units 1..L+1), one per layer's backward (L..1), and
    ``δX_1`` to unit 0 in the glue; autocast's weight copies are keyed by device and storage."""
    c = tiny(("llama", True), dtype, 2, devices=THREE)
    leaves = prover_leaves(tiny(("llama", True), dtype, 2), make_records((12, 7)))
    calls, copies = [], []
    orig_move, orig_copied = DecoderReplay._to_unit, DecoderReplay._copied

    def to_unit(self, x, j):
        calls.append(j)
        return orig_move(self, x, j)

    def copied(self, src, out):
        orig_copied(self, src, out)
        copies.extend(self._copies)

    monkeypatch.setattr(DecoderReplay, "_to_unit", to_unit)
    monkeypatch.setattr(DecoderReplay, "_copied", copied)
    replay = c.replay(ListReader(leaves))
    replay.operands(1)
    assert calls == list(range(1, c.L + 2))
    for s in c.products[1:]:
        replay.operands(s.m)
    assert calls[c.L + 1:] == [*range(c.L, 0, -1)]
    replay.glue_gradients()
    assert calls[c.L + 1:] == [*range(c.L, 0, -1), 0]
    if dtype == "bf16":
        assert copies and all(isinstance(k, tuple) and isinstance(k[0], torch.device)
                              and isinstance(k[1], int) for k in copies)


def test_model_is_placed_by_unit_and_stays_tied():
    """Sharded, the verifier's model is built on the host and placed by units; reuse keeps it
    placed and the tie intact."""
    c = tiny(("llama", True), "fp32", 1, devices=THREE)
    leaves = prover_leaves(tiny(("llama", True), "fp32", 1), make_records((12, 7)))
    replay = c.replay(ListReader(leaves))
    model = replay.model
    assert model.lm_head.weight is model.model.embed_tokens.weight
    again = c.replay(ListReader(leaves)).model
    assert again is model and replay.model is None
    assert again.lm_head.weight is again.model.embed_tokens.weight


def test_moved_tensors_are_cached_once_per_device():
    """The pre-hook's move of a shared tensor (position embeddings, mask) happens once per
    device per pass; a tuple of tensors moves tensor by tensor."""
    c = tiny(("llama", True), "fp32", 1, devices=THREE)
    leaves = prover_leaves(tiny(("llama", True), "fp32", 1), make_records((12, 7)))
    replay = c.replay(ListReader(leaves))
    meta = torch.device("meta")
    cos, sin = torch.ones(2, 3), torch.zeros(2, 3)
    one = replay._on(cos, meta)
    assert one.device == meta and replay._on(cos, meta) is one
    pair = replay._on((cos, sin), meta)
    assert isinstance(pair, tuple) and pair[0] is one and pair[1].device == meta
    assert replay._on(None, meta) is None and replay._on(cos, cos.device) is cos


# ---- MPS ----------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_split_replay_over_cpu_and_mps(monkeypatch):
    """A tiny fp32 Llama, prover on the host, replay over ``(cpu, mps)``: no device mismatch,
    every operand on its unit's device, values equal the unsharded replay's within fp32
    tolerance (MPS rounds differently)."""
    c = tiny(("llama", True), "fp32", 1)
    leaves = prover_leaves(c, make_records((12, 7)))
    cs = tiny(("llama", True), "fp32", 1)
    # The config refuses mixed device types; the test builds the placement itself.
    monkeypatch.setitem(cs.__dict__, "placement", placement_for(cs, "cpu", ("cpu", "mps")))
    pl = cs.placement
    assert {pl.unit_device(j).type for j in cs.units} == {"cpu", "mps"}
    plain = c.replay(ListReader(leaves))
    split = cs.replay(ListReader(leaves))
    try:
        for s in cs.products:
            a, b = split.operands(s.m)
            ua, ub = plain.operands(s.m)
            dev = pl.unit_device(cs.product_unit(s.m))
            assert a.device.type == b.device.type == dev.type, s.name
            for x, y in ((a, ua), (b, ub)):
                torch.testing.assert_close(x.cpu(), y, rtol=1e-4, atol=1e-5, msg=s.name)
        got, want = split.glue_gradients(), plain.glue_gradients()
    except (NotImplementedError, SubstitutionError) as e:
        if isinstance(e, SubstitutionError) and "aten.linear" not in str(e):
            raise
        # MPS runs nn.Linear as one native aten.linear, so the substitution never sees an mm.
        pytest.skip(f"MPS lacks an op the replay needs: {e}")
    for n, g in got.items():
        assert g.device.type == pl.weight_device(n).type, n
        torch.testing.assert_close(g.cpu(), want[n], rtol=1e-4, atol=1e-5, msg=n)


# ---- CUDA, two devices ----------------------------------------------------------------------


def assert_kept_off_the_prover(replay, c, prover_device):
    """Every parameter and buffer of the replay's model sits on one of the verifier devices,
    and on none of them when the prover's device is not among them (the check bites then)."""
    allowed = set(c.placement.devices)
    seen = {t.device for t in (*replay.model.parameters(), *replay.model.buffers())}
    assert seen <= allowed, (seen, allowed)
    if torch.device(prover_device) not in allowed:
        assert torch.device(prover_device) not in seen


def assert_operand_devices(c, replay, prover_device):
    """Each served operand pair sits on its unit's device, never on an excluded prover device."""
    pl = c.placement
    for s in c.products:
        a, b = replay.operands(s.m)
        assert a.device == b.device == pl.unit_device(c.product_unit(s.m)), s.name
        if torch.device(prover_device) not in set(pl.devices):
            assert a.device != torch.device(prover_device) and b.device != torch.device(prover_device)


@two_cuda
@pytest.mark.parametrize("switch,dtype,B", CASES, ids=CASE_IDS)
def test_split_replay_equals_unsharded_over_two_cuda_devices(switch, dtype, B):
    """The Mac comparison on two GPUs: prover on ``cuda:0``, replay over ``(cuda:0, cuda:1)``
    against the unsharded replay on ``cuda:1``; every operand and glue gradient equal by digest,
    each operand on its unit's device and each glue gradient on its weight's."""
    prover = tiny(switch, dtype, B, device="cuda:0")
    leaves = prover_leaves(prover, make_records((12, 7)))
    devices = verifier_devices()
    ref = tiny(switch, dtype, B, device=devices[0])
    split_c = tiny(switch, dtype, B, device="cuda:0", devices=devices)
    _, ops, glue = replay_all(ref, leaves)
    replay = split_c.replay(ListReader(leaves))
    pl = split_c.placement
    assert_kept_off_the_prover(replay, split_c, "cuda:0")
    assert_operand_devices(split_c, replay, "cuda:0")
    ops_s = []
    for s in split_c.products:
        a, b = replay.operands(s.m)
        ops_s.append((digest(a), digest(b)))
    got = replay.glue_gradients()
    assert [s.name for s, x, y in zip(ref.products, ops, ops_s) if x != y] == []
    assert [n for n in glue if glue[n] != digest(got[n])] == []
    assert all(g.device == pl.weight_device(n) for n, g in got.items())


@two_cuda
@pytest.mark.slow
def test_smollm2_split_replay_over_two_cuda_devices():
    """The same comparison on the real SmolLM2 step at test scale: from ``W_0`` on ``π(1)``,
    prover on ``cuda:0``, replay over ``(cuda:0, cuda:1)`` against the unsharded replay on
    ``cuda:1``, every operand and glue gradient equal by digest."""
    import dataclasses
    import os

    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records, schedule

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}; run runs/materialize_data.py or set "
                    f"VERIF_OUTPUT_DIR")
    setup_determinism(cfg)
    base = dataclasses.replace(cfg, device="cuda:0", verifier_devices=None)
    prover = DecoderComputation.from_config(base)
    D = load_dataset_records(cfg.data_dir / "D.bin", prover.n)
    records = [D[i] for i in schedule(1, prover.n_s, len(D), cfg.seed)]
    leaves = prover_leaves(prover, records)
    devices = verifier_devices()
    ref = DecoderComputation.from_config(dataclasses.replace(base, device=devices[0]))
    split_c = DecoderComputation.from_config(
        dataclasses.replace(base, verifier_devices=devices))
    _, ops, glue = replay_all(ref, leaves)
    replay = split_c.replay(ListReader(leaves))
    assert_kept_off_the_prover(replay, split_c, "cuda:0")
    assert_operand_devices(split_c, replay, "cuda:0")
    got = replay.glue_gradients()
    assert all(g.device == split_c.placement.weight_device(n) for n, g in got.items())
    _, ops_s, glue_s = replay_all(split_c, leaves)
    assert [s.name for s, x, y in zip(ref.products, ops, ops_s) if x != y] == []
    assert [n for n in glue if glue[n] != glue_s[n]] == []
