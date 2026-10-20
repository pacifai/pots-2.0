"""A sharded run end to end through ``run_loop`` (FS32f, F22).

``verifier_devices=("cpu", "cpu", "cpu")`` runs the sharded code paths with every move a no-op,
so a run over the tiny decoders must equal the unsharded run bit for bit: roots, loss,
verdicts, ``stats.products`` and ``stats.tensors``. A forged leaf in a later unit than the
first gives the same first failing product and rejection, and after an early rejection the
record pass of 6a, 6b and 5 records the unsharded numbers. The slow test does the same on
SmolLM2 for 2 steps.
"""

import pytest
import torch

from tests.verification.decoder_helpers import make_records, randomize_biases, tiny_config
from verification.commitment.leaves import dataset_tree, leaf_hash, leaf_hashes_of
from verification.computation.instances import DecoderComputation
from verification.computation.interface import snapshot_weights
from verification.runs.loop import run_loop
from verification.runs.scenarios import honest_final
from verification.transcript.store import IN_MEMORY, StoreHandoff
from verification.verifier.bands import Bands
from verification.verifier.driver import Verifier

SEED, K, T = 1, 5, 2
THREE = ("cpu", "cpu", "cpu")


class Recording(StoreHandoff):
    """In-memory hand-off that notes each step's claimed root, and forges one leaf of step 1
    (leaf replaced, tree re-rooted) when ``edit`` is given."""

    def __init__(self, edit=None):
        self.edit, self.roots = edit, []

    def hold(self, c, t, leaves, tree, dataset_paths=None):
        if self.edit is not None and t == 1:
            i, obj = self.edit(c, leaves)
            tree.update_leaf(i, leaf_hash(c, i, obj))
            leaves[i] = obj
        store = IN_MEMORY.hold(c, t, leaves, tree, dataset_paths)
        self.roots.append(store.root)
        return store


def _bumped(t):
    t = t.detach().clone()
    t.view(-1)[1] += 0.25 * (1.0 + abs(float(t.view(-1)[1])))
    return t


def _last_layer(names):
    return next(n for n in names if "layers.1." in n)


def forge(which):
    """An ``edit`` for :class:`Recording`: one leaf in the last layer's unit changed."""
    def edit(c, leaves):
        if which == "product":
            m = next(s.m for s in c.products
                     if s.name.startswith("L2.") and s.kind.name == "FORWARD")
            i = c.product_index(m)
        else:
            names = c.linear_weights if which == "linear" else c.glue_gradient_weights
            i = c.w_next_index(_last_layer(names))
        return i, _bumped(leaves[i])
    return edit


def _tiny(devices, device="cpu", layers=None):
    torch.manual_seed(0)
    config = tiny_config()
    if layers is not None:
        config.num_hidden_layers = layers
        if getattr(config, "layer_types", None):
            config.layer_types = [config.layer_types[0]] * layers
    c = DecoderComputation(config, n_s=2, n=12, eta=1e-3, device=device,
                           verifier_devices=devices)
    D = make_records((c.n, c.n - 5, c.n - 2, c.n - 7, c.n - 1), seed=1)
    return c, D, c.build_model


def run(devices, edit=None, device="cpu", layers=None):
    """A verified run under provisional bands: ``(c, loop result, verifier, hand-off)``."""
    c, D, build = _tiny(devices, device, layers)
    model = build()
    randomize_biases(dict(model.named_parameters()))
    w0 = snapshot_weights(c, model)
    w_T = honest_final(c, D, w0, T, seed=SEED, build_model=lambda: model)
    final = leaf_hashes_of(c, ((c.w_t_index(n), w_T[n]) for n in c.weight_names))
    v = Verifier(c, h_D=dataset_tree(c, D).root, n_records=len(D), k=K, n_steps=T,
                 bands=Bands.provisional(), w0=w0, allow_provisional=True, seed=SEED)
    hand = Recording(edit)
    res = run_loop(c, model, D, w0, v, final=final, handoff=hand)
    return c, res, v, hand


def _numbers(v):
    return {t: (repr(s.products), repr(s.tensors)) for t, s in v.stats.items()}


def _same(a, b):
    ca, ra, va, ha = a
    cb, rb, vb, hb = b
    assert cb.placement.sharded and not ca.placement.sharded
    assert ha.roots == hb.roots
    step = lambda r: [(s.t, s.loss, s.rejection, s.root) for s in r.steps]  # noqa: E731
    assert step(ra) == step(rb) and all(s.root for s in ra.steps)
    assert ra.verdict == rb.verdict
    assert _numbers(va) == _numbers(vb)
    assert va.record_pass == vb.record_pass


def test_honest_sharded_run_equals_unsharded():
    one, three = run(None), run(THREE)
    assert one[1].verdict.accepted and len(one[3].roots) == T
    _same(one, three)


@pytest.mark.parametrize("which", ["linear", "glue", "product"])
def test_forged_leaf_in_a_later_unit_is_rejected_alike(which):
    """The first failing product and the rejection, and the record pass's numbers for 6a, 6b
    and 5 after the early rejection, are the unsharded run's."""
    one, three = run(None, forge(which)), run(THREE, forge(which))
    rej = one[1].rejection
    assert rej is not None and rej.step == 1 and len(one[1].steps) == 1
    assert three[1].rejection == rej
    _same(one, three)
    assert one[2].record_pass == {1: "complete"}


def smollm2_run(device, devices, steps=T):
    """SmolLM2 at test scale for ``steps`` steps through ``bring_up`` (calibrating, so every
    step is judged on its own fit): ``(losses, kept step's root, bands json, stats)``; skips
    without the committed ``D.bin``."""
    import dataclasses
    import os

    from setup.config import load_config, setup_determinism
    from setup.data import load_dataset_records
    from verification.runs.bring_up import bring_up

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cfg = load_config()
    if not (cfg.data_dir / "D.bin").exists():
        pytest.skip(f"no D.bin under {cfg.data_dir}; run runs/materialize_data.py or set "
                    f"VERIF_OUTPUT_DIR")
    setup_determinism(cfg)
    cfg = dataclasses.replace(cfg, device=device, verifier_devices=devices)
    c = DecoderComputation.from_config(cfg)
    D = load_dataset_records(cfg.data_dir / "D.bin", c.n)
    b = bring_up(c, D, dataset_tree(c, D).root, T=steps, k=K, seed=cfg.seed, tamper=False,
                 device=device, keep_last=True)
    assert b.honest_accepted and b.kept is not None
    losses = [s.loss for s in b.result.loop.steps]
    return c, losses, b.kept.root, b.bands, _numbers(b.result.verifier)


@pytest.mark.slow
def test_smollm2_sharded_run_on_the_host_equals_unsharded():
    """SmolLM2, 2 steps, ``("cpu", "cpu", "cpu")`` against unsharded: losses, the last step's
    root, the fitted bands and every recorded number equal."""
    one, three = smollm2_run("cpu", None), smollm2_run("cpu", THREE)
    assert three[0].placement.sharded
    assert one[1:4] == three[1:4] and one[4] == three[4]
