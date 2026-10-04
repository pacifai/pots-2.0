"""The SmolLM2 instance of ``C`` (ref block §1–§6): the declaration, the prover-side labeling
and the verifier-side replay.

The model is an unmodified Hugging Face ``LlamaForCausalLM`` with eager attention, in fp32,
with dropout zero (§8.A, S4d). The same class serves a tiny randomly initialised
``LlamaConfig`` for the fast tests; the real instance is :meth:`LlamaComputation.from_pretrained`.

Implementation pins (not fixed by the reference block):

- **Weight names** are the HF parameter names (``model.embed_tokens.weight``,
  ``model.layers.{ℓ−1}.self_attn.q_proj.weight``, ``model.norm.weight`` …), listed in the ref
  block §6 order, which differs from ``named_parameters()`` order. The tied output projection
  is ``W_E``; HF lists it once, under the embedding's name.
- **Product names** are ``L{ℓ}.Y_q`` … for the linear products, ``L{ℓ}.S[s,h]``,
  ``L{ℓ}.O[s,h]``, ``L{ℓ}.dA[s,h]``, ``L{ℓ}.dV[s,h]``, ``L{ℓ}.dQ[s,h]`` and ``L{ℓ}.dK[s,h]``
  for the attention members, and ``Lambda``, ``dF`` and ``G_E_head`` outside the stack.
  ``member`` is the 0-based ``(s, h)``, the bmm batch index ``s·n_h + h`` of HF's eager
  attention. Each product's κ class (:meth:`LlamaComputation.product_class`) is its role
  without the layer and member.
- **Batch assembly** is ``data.assemble_batch``: right padding with ``PAD_ID``, ``ρ`` from each
  record's ``ℓ``. ``ρ`` is the HF ``attention_mask``, so ``Ω`` is HF's causal mask over ``ρ``.
  Positions are ``0 … n−1`` in every sequence.
- **Loss** is ref block §3's ``ℒ = Σ_i μ_i·CE_i / Σ_i μ_i`` over the logits, with
  ``F.cross_entropy(reduction="none")`` per token. A batch with ``Σ μ = 0`` raises
  ``ValueError`` instead of producing a 0/0 NaN. ``data.scan`` never builds such a record
  (every record has at least the EOS target), so only a faulted batch can reach it.

Labeling (:meth:`LlamaComputation.label`) maps each captured matmul to its slot by operand
identity (storage), never by call order alone:

- forward ``Y_x`` and ``Λ``: ``B`` is a view of the weight's storage; backward ``δX_x`` and
  ``δF``: ``B`` is the weight itself;
- weight gradients ``G_x`` and ``G_E^head``: ``B`` is the saved forward input (the storage of
  ``Y_x``'s ``A``, shared by q/k/v and by gate/up), and ``A`` is ``δYᵀ``, which shares storage
  with the ``A`` of the already-labeled ``δX_x``. For linear weights the output must also be
  the storage AccumulateGrad adopted as ``W_x.grad``;
- the attention backward bmms, by the saved forward operands: ``δQ̃ = δS·K̃`` has the ``B``
  storage of ``S``, ``Q̃ᵀ·δS`` has its ``A`` storage, ``δA = δO·Vᵀ`` has the ``B`` storage of
  ``O`` and ``δV = Aᵀ·δO`` has its ``A`` storage;
- the forward bmms have no storage link to any weight: RoPE, ``repeat_kv``, softmax and the head
  merge all write fresh tensors. Their layer is the bracket between that layer's identity-labeled
  ``Y_v`` and ``Y_o`` calls, which must hold exactly two bmms; ``S`` is the first, because ``O``
  consumes ``softmax(S)``. That one data-dependence order is checked by the declared shapes,
  by requiring ``O``'s ``A`` to be row-stochastic, and by exact value checks: ``O``'s ``B``
  member ``(s, h)`` equals ``Y_v``'s output at sequence ``s``, kv head ``h // g``
  (``repeat_kv`` copies it), and ``S``'s ``B`` members are equal across the ``g`` query heads
  that share a kv head;
- the captured ``Q̃ᵀ·δS`` is ``δK̃ᵀ`` (``prover/capture.py`` hand-off note); its contiguous
  transpose is committed as the ``δK̃`` leaf, and its operands are checked against the spec
  transposed.

Replay (:class:`LlamaReplay`) is the verifier's side. It runs its own model, loaded from the
committed ``W_t``, under :class:`~verification.computation.substitution.ProductSubstitution`:
every checked product is replaced by its committed leaf, so all glue (embedding, RMSNorm, RoPE,
the causal mask over ``ρ``, softmax, SiLU, residual adds) runs through the model's own modules on
committed values (S4b, check 3). A8 covers the forward products. A9 covers the backward
products and :meth:`LlamaReplay.glue_gradients`, by autograd on the same modules under the
same substitution; no backward is written by hand. As in training, the replay runs one forward
pass with grad on and one backward pass per unit (the head, then layers ``L…1``), with no
reruns; the class docstring has the details.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from setup.config import RunConfig, assert_no_dropout
from setup.data import Batch, assemble_batch
from setup.model import load_model_config, load_pretrained
from setup.records import Record, RecordError, encode_record
from verification.computation.interface import (
    DeclaredComputation,
    LabelingError,
    ProductKind,
    ProductSpec,
    Replay,
    ReplayError,
    load_weights,
)
from verification.computation.matmul_ops import param_storage_map
from verification.computation.substitution import ProductSubstitution
from verification.transcript.reader import LeafReader, TranscriptView

if TYPE_CHECKING:  # labeling's types only; nothing here imports the prover at run time
    from verification.prover.capture import MatmulCapture, MatmulRecord

__all__ = ["LINEARS", "LlamaComputation", "LlamaReplay", "matmul_count_llama"]

# Linear weights of one layer, as ref block §2 names them, with their HF module paths.
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")
_LINEAR_PATH = {
    "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
    "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
# The attention gradients: role -> (forward bmm, saved operand side).
_ATTN_BACKWARD = {"dA": ("O", "b"), "dV": ("O", "a"), "dQ": ("S", "b"), "dK": ("S", "a")}

_E = "model.embed_tokens.weight"
_GAMMA_FINAL = "model.norm.weight"


def matmul_count_llama(L: int, n_s: int, n_h: int) -> int:
    """Products per step for the Llama-family block, `M = L(21 + 6·n_s·n_h) + 3` (ref block §5)."""
    return L * (21 + 6 * n_s * n_h) + 3


def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


class LlamaComputation(DeclaredComputation):
    """``C`` for a ``LlamaForCausalLM`` of the given config, batch ``n_s × n`` and step size.

    ``source = (repo, revision)`` makes :meth:`build_model` an unmodified ``from_pretrained``
    load; without it the model is built from ``config`` with random weights (tests only).
    """

    def __init__(self, config: Any, *, n_s: int, n: int, eta: float,
                 source: tuple[str, str] | None = None) -> None:
        self.config = copy.deepcopy(config)
        self.source = source
        self._n_s, self.n, self._eta = int(n_s), int(n), float(eta)
        c = self.config
        self.L = int(c.num_hidden_layers)
        self.d = int(c.hidden_size)
        self.d_f = int(c.intermediate_size)
        self.n_h = int(c.num_attention_heads)
        self.n_kv = int(c.num_key_value_heads)
        self.d_h = int(getattr(c, "head_dim", None) or self.d // self.n_h)
        self.n_v = int(c.vocab_size)
        self._check_config()
        self.validate()

    @classmethod
    def from_pretrained(cls, repo: str, revision: str, *, n_s: int, n: int,
                        eta: float) -> LlamaComputation:
        config = load_model_config(repo, revision)
        return cls(config, n_s=n_s, n=n, eta=eta, source=(repo, revision))

    @classmethod
    def from_config(cls, cfg: RunConfig) -> LlamaComputation:
        """The run's instance: ``VERIF_MODEL`` at ``VERIF_MODEL_REVISION``, ``VERIF_BATCH ×
        VERIF_SEQ_LEN``, ``η = VERIF_ETA``."""
        return cls.from_pretrained(cfg.model, cfg.model_revision, n_s=cfg.batch, n=cfg.seq_len,
                                   eta=cfg.require_eta())

    def _check_config(self) -> None:
        c = self.config
        if c.model_type != "llama":
            raise ValueError(f"expected a llama config, got model_type={c.model_type!r}")
        bad = []
        if not c.tie_word_embeddings:
            bad.append("tie_word_embeddings must be True (ref block §1: W_E is tied)")
        if getattr(c, "attention_bias", False) or getattr(c, "mlp_bias", False):
            bad.append("linear biases must be off (ref block §1)")
        if c.hidden_act != "silu":
            bad.append(f"hidden_act={c.hidden_act!r}, ref block §3 has SiLU")
        if getattr(c, "pretraining_tp", 1) != 1:
            bad.append("pretraining_tp must be 1")
        if self.n_h % self.n_kv:
            bad.append(f"n_h={self.n_h} is not a multiple of n_kv={self.n_kv}")
        if bad:
            raise ValueError("unsupported config: " + "; ".join(bad))
        assert_no_dropout(c)

    # ---- declaration ----------------------------------------------------------------------

    @property
    def n_s(self) -> int:
        return self._n_s

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def N(self) -> int:
        return self.n_s * self.n

    def w(self, l: int, x: str) -> str:
        """Name of linear weight ``W_x`` of layer ``ℓ`` (1-based)."""
        return f"model.layers.{l - 1}.{_LINEAR_PATH[x]}.weight"

    def gamma_attn(self, l: int) -> str:
        return f"model.layers.{l - 1}.input_layernorm.weight"

    def gamma_mlp(self, l: int) -> str:
        return f"model.layers.{l - 1}.post_attention_layernorm.weight"

    @property
    def w_e(self) -> str:
        return _E

    @property
    def gamma_final(self) -> str:
        return _GAMMA_FINAL

    def linear_shape(self, x: str) -> tuple[int, int]:
        """``(o, i)`` of ``W_x``."""
        q_out, kv_out = self.n_h * self.d_h, self.n_kv * self.d_h
        return {"q": (q_out, self.d), "k": (kv_out, self.d), "v": (kv_out, self.d),
                "o": (self.d, q_out), "gate": (self.d_f, self.d), "up": (self.d_f, self.d),
                "down": (self.d, self.d_f)}[x]

    @cached_property
    def weight_names(self) -> tuple[str, ...]:
        names = [_E]
        for l in range(1, self.L + 1):
            names += [self.gamma_attn(l), *(self.w(l, x) for x in ("q", "k", "v", "o")),
                      self.gamma_mlp(l), *(self.w(l, x) for x in ("gate", "up", "down"))]
        return (*names, _GAMMA_FINAL)

    @cached_property
    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {_E: (self.n_v, self.d), _GAMMA_FINAL: (self.d,)}
        for l in range(1, self.L + 1):
            shapes[self.gamma_attn(l)] = shapes[self.gamma_mlp(l)] = (self.d,)
            for x in LINEARS:
                shapes[self.w(l, x)] = self.linear_shape(x)
        return {name: shapes[name] for name in self.weight_names}

    def _members(self):
        return ((s, h) for s in range(self.n_s) for h in range(self.n_h))

    @cached_property
    def _inventory(self) -> tuple[tuple[ProductSpec, ...], tuple[str, ...]]:
        """The specs in ref block §6 order, and each one's κ class."""
        N, n, d_h, d, n_v = self.N, self.n, self.d_h, self.d, self.n_v
        rows: list[tuple] = []  # (name, kind, a_shape, b_shape, weight, layer, member, class)

        def linear(l, x, kind):
            o, i = self.linear_shape(x)
            a, b = {ProductKind.FORWARD: ((N, i), (i, o)),
                    ProductKind.INPUT_GRAD: ((N, o), (o, i)),
                    ProductKind.WEIGHT_GRAD: ((o, N), (N, i))}[kind]
            role = {ProductKind.FORWARD: "Y", ProductKind.INPUT_GRAD: "dX",
                    ProductKind.WEIGHT_GRAD: "G"}[kind] + f"_{x}"
            rows.append((f"L{l}.{role}", kind, a, b, self.w(l, x), l, None, role))

        def attn(l, role, kind, a, b):
            for s, h in self._members():
                rows.append((f"L{l}.{role}[{s},{h}]", kind, a, b, None, l, (s, h), role))

        fwd, ig, wg, og = (ProductKind.FORWARD, ProductKind.INPUT_GRAD, ProductKind.WEIGHT_GRAD,
                           ProductKind.OPERAND_GRAD)
        for l in range(1, self.L + 1):
            for x in ("q", "k", "v"):
                linear(l, x, fwd)
            attn(l, "S", fwd, (n, d_h), (d_h, n))
            attn(l, "O", fwd, (n, n), (n, d_h))
            for x in ("o", "gate", "up", "down"):
                linear(l, x, fwd)
        rows.append(("Lambda", fwd, (N, d), (d, n_v), _E, None, None, "Lambda"))
        rows.append(("dF", ig, (N, n_v), (n_v, d), _E, None, None, "dF"))
        rows.append(("G_E_head", wg, (n_v, N), (N, d), _E, None, None, "G_E_head"))
        for l in range(self.L, 0, -1):
            for x in ("down", "gate", "up", "o"):
                linear(l, x, ig)
                linear(l, x, wg)
            for s, h in self._members():
                rows.append((f"L{l}.dA[{s},{h}]", og, (n, d_h), (d_h, n), None, l, (s, h), "dA"))
                rows.append((f"L{l}.dV[{s},{h}]", og, (n, n), (n, d_h), None, l, (s, h), "dV"))
            for s, h in self._members():
                rows.append((f"L{l}.dQ[{s},{h}]", og, (n, n), (n, d_h), None, l, (s, h), "dQ"))
                rows.append((f"L{l}.dK[{s},{h}]", og, (n, n), (n, d_h), None, l, (s, h), "dK"))
            for x in ("q", "k", "v"):
                linear(l, x, ig)
                linear(l, x, wg)
        specs = tuple(ProductSpec(m, name, kind, a, b, weight=w, layer=l, member=mem)
                      for m, (name, kind, a, b, w, l, mem, _) in enumerate(rows, start=1))
        return specs, tuple(r[-1] for r in rows)

    @property
    def products(self) -> tuple[ProductSpec, ...]:
        return self._inventory[0]

    @cached_property
    def _m_by_name(self) -> dict[str, int]:
        return {p.name: p.m for p in self.products}

    def m_of(self, name: str) -> int:
        return self._m_by_name[name]

    def product_class(self, spec: ProductSpec) -> str:
        """κ class (P3.c): the product's role, the same in every layer and member.

        Explicit names, one per row of ref block §5 split by weight: ``Y_q`` … ``G_down``, ``S``,
        ``O``, ``dA``, ``dV``, ``dQ``, ``dK``, ``Lambda``, ``dF`` and ``G_E_head``.
        """
        if self.product(spec.m) != spec:
            raise ValueError(f"{spec.name} is not product {spec.m} of this computation")
        return self._inventory[1][spec.m - 1]

    @cached_property
    def linear_weights(self) -> dict[str, int]:
        return {self.w(l, x): self.m_of(f"L{l}.G_{x}")
                for l in range(1, self.L + 1) for x in LINEARS}

    # glue_gradient_weights (the base class's derivation): W_E and every γ, for check 6b.

    # P6: fp32 at both scales; stated here so the instance's numerics are read in one place.
    @property
    def weight_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def product_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def operand_dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def accumulator_dtype(self) -> torch.dtype:
        return torch.float32

    # ---- shared ---------------------------------------------------------------------------

    def encode_record(self, record: Record) -> bytes:
        if not isinstance(record, Record):
            raise RecordError(f"a Llama record is a records.Record, got {type(record).__name__}")
        if len(record) > self.n:
            raise RecordError(f"record length {len(record)} > n = {self.n}")
        return encode_record(record)

    def build_model(self) -> nn.Module:
        """An unmodified ``LlamaForCausalLM``, eager attention, fp32, dropout asserted zero.

        Eval mode: dropout is zero anyway (S4d), and HF runs activation checkpointing only in
        train mode, which the capture forbids. Load ``W_t`` before use.
        """
        if self.source is not None:
            repo, revision = self.source
            model = load_pretrained(repo, revision, attn_impl="eager", dtype=torch.float32)
        else:
            from transformers import LlamaForCausalLM

            with torch.random.fork_rng(devices=[]):
                model = LlamaForCausalLM._from_config(
                    copy.deepcopy(self.config), attn_implementation="eager", dtype=torch.float32)
        model.eval()
        self._check_model(model)
        return model

    def _check_model(self, model: nn.Module) -> None:
        assert_no_dropout(model)
        if model.config._attn_implementation != "eager":
            raise ValueError(f"attention is {model.config._attn_implementation!r}, not eager")
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("activation checkpointing must be off (prover/capture.py)")
        if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
            raise ValueError("the output projection is not tied to W_E")
        shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
        if shapes != dict(self.weight_shapes):
            raise ValueError("model parameters differ from the declared weights")
        bad = [n for n, p in model.named_parameters() if p.dtype != self.weight_dtype]
        if bad:
            raise ValueError(f"parameters not {self.weight_dtype}: {bad[:3]}")

    def assemble(self, records: Sequence[Record]) -> Batch:
        """Glue: pad and stack the records (ref block §2). ``ρ`` comes from each ``ℓ``."""
        if len(records) != self.n_s:
            raise ValueError(f"batch has {len(records)} records, the computation declares {self.n_s}")
        for r in records:
            if not isinstance(r, Record):
                raise ValueError(f"a Llama record is a records.Record, got {type(r).__name__}")
        return assemble_batch(records, self.n)

    def logits(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """``Λ`` as ``[n_s, n, n_v]``: the forward pass, embedding to output projection."""
        return model(input_ids=batch.ids, attention_mask=batch.rho, use_cache=False).logits

    @staticmethod
    def loss_from_logits(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        """``ℒ = (1/Σμ)·Σ_i μ_i·(logsumexp(Λ_i) − Λ_{i,y_i})`` (ref block §3)."""
        mu = batch.mask.reshape(-1).to(logits.dtype)
        total = mu.sum()
        if not total > 0:
            raise ValueError("the batch has no loss-masked target")
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.targets.reshape(-1),
                             reduction="none")
        return (ce * mu).sum() / total

    # ---- verifier side --------------------------------------------------------------------

    def replay(self, leaves: LeafReader) -> LlamaReplay:
        return LlamaReplay(self, leaves)

    # ---- prover side ----------------------------------------------------------------------

    def loss(self, model: nn.Module, records: Sequence[Record]) -> torch.Tensor:
        batch = self.assemble(records)
        return self.loss_from_logits(self.logits(model, batch), batch)

    def label(self, capture: MatmulCapture, model: nn.Module) -> list[torch.Tensor]:
        """Map the captured records onto ``P_1..P_M`` by operand identity (module docstring)."""
        return _Labeler(self, capture, model).run()


class _Labeler:
    """One labeling pass. Every slot is filled exactly once, or :class:`LabelingError`."""

    def __init__(self, c: LlamaComputation, capture: MatmulCapture, model: nn.Module) -> None:
        self.c, self.cap, self.model = c, capture, model
        self.slots: list[torch.Tensor | None] = [None] * c.M
        # weight name -> (ℓ or None, x) for every weight that has linear products.
        self.role = {c.w(l, x): (l, x) for l in range(1, c.L + 1) for x in LINEARS}
        self.role[_E] = (None, "E")

    def run(self) -> list[torch.Tensor]:
        cap, c = self.cap, self.c
        for rec in cap.records + cap.glue_outer:
            if rec.a is None:
                raise LabelingError("capture operands were released before labeling")
        self._check_glue_outer()
        fwd = [r for r in cap.records if r.phase == "forward"]
        bwd = [r for r in cap.records if r.phase == "backward"]
        y_rec = self._forward_linears([r for r in fwd if r.batch is None])
        bmm_ops = self._forward_bmms([r for r in fwd if r.batch is not None], y_rec)
        self._backward_linears([r for r in bwd if r.batch is None], y_rec)
        self._backward_bmms([r for r in bwd if r.batch is not None], bmm_ops)
        missing = [c.products[i].name for i, t in enumerate(self.slots) if t is None]
        if missing:
            raise LabelingError(f"{len(missing)} slots never filled, first {missing[:5]}")
        return self.slots  # type: ignore[return-value]

    # ---- helpers -------------------------------------------------------------------------

    def _fill(self, name: str, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor,
              rec: MatmulRecord, leaf: torch.Tensor | None = None) -> None:
        """Put the product in slot ``name`` after checking the operand shapes against its spec.

        ``(a, b, out)`` are as captured. ``leaf`` is given when the record computed
        ``Pᵀ = Bᵀ·Aᵀ`` (the δK̃ case): it is the contiguous ``P``, and the captured shapes are
        checked against the spec transposed.
        """
        m = self.c.m_of(name)
        spec = self.c.product(m)
        if self.slots[m - 1] is not None:
            raise LabelingError(f"slot {name} filled twice (record #{rec.index})")
        got = (tuple(a.shape), tuple(b.shape), tuple(out.shape))
        want = (spec.a_shape, spec.b_shape, spec.p_shape)
        if leaf is not None:
            want = tuple(s[::-1] for s in (spec.b_shape, spec.a_shape, spec.p_shape))
        if got != want:
            raise LabelingError(f"{name} (record #{rec.index}): shapes {got} != declared {want}")
        p = out if leaf is None else leaf
        if tuple(p.shape) != spec.p_shape or p.dtype != self.c.product_dtype:
            raise LabelingError(f"{name}: leaf {p.dtype} {tuple(p.shape)} != declared "
                                f"{self.c.product_dtype} {spec.p_shape}")
        if not p.is_contiguous():
            raise LabelingError(f"{name} (record #{rec.index}): output is not contiguous")
        self.slots[m - 1] = p

    def _linear_name(self, weight: str, prefix: str) -> str:
        l, x = self.role[weight]
        if l is None:
            return {"Y": "Lambda", "dX": "dF", "G": "G_E_head"}[prefix]
        return f"L{l}.{prefix}_{x}"

    def _check_glue_outer(self) -> None:
        # P7: the one q=1 product is the RoPE angle table, computed once per forward pass from
        # public constants (inv_freq and the positions).
        outer = self.cap.glue_outer
        bad = [r for r in outer if r.phase != "forward" or r.a_info.param_name
               or r.b_info.param_name]
        if bad or len(outer) != 1:
            raise LabelingError(f"expected exactly one forward q=1 product (the RoPE table, P7), "
                                f"got {len(outer)} of which {len(bad)} unexpected")

    # ---- forward ---------------------------------------------------------------------------

    def _forward_linears(self, recs: list[MatmulRecord]) -> dict[str, MatmulRecord]:
        """``Y_x`` and ``Λ`` by weight storage; returns weight name -> its forward record."""
        out: dict[str, MatmulRecord] = {}
        for rec in recs:
            w = rec.b_info.param_name
            if w not in self.role:
                raise LabelingError(f"forward mm #{rec.index} has no weight operand")
            if w in out:
                raise LabelingError(f"weight {w} used in two forward products "
                                    f"(#{out[w].index}, #{rec.index})")
            out[w] = rec
            self._fill(self._linear_name(w, "Y"), rec.a, rec.b, rec.out, rec)
        return out

    def _forward_bmms(self, recs: list[MatmulRecord],
                      y_rec: dict[str, MatmulRecord]) -> dict[int, tuple[int, str, str]]:
        """``S`` and ``O`` members; returns operand storage -> ``(ℓ, "S"|"O", "a"|"b")``."""
        c = self.c
        taken: set[int] = set()
        ops: dict[int, tuple[int, str, str]] = {}
        linear_a = {_ptr(r.a) for r in y_rec.values()}
        for l in range(1, c.L + 1):
            try:
                lo = max(y_rec[c.w(l, x)].index for x in ("q", "k", "v"))
                hi = y_rec[c.w(l, "o")].index
            except KeyError as e:
                raise LabelingError(f"layer {l}: forward projection missing ({e})") from None
            inside = [r for r in recs if lo < r.index < hi]
            if len(inside) != 2:
                raise LabelingError(f"layer {l}: {len(inside)} forward bmms between Y_v and Y_o, "
                                    f"expected 2 (S, O)")
            s_rec, o_rec = sorted(inside, key=lambda r: r.index)  # O consumes softmax(S)
            self._check_row_stochastic(l, o_rec)
            self._check_kv_values(l, s_rec, o_rec, y_rec[c.w(l, "v")])
            for kind, rec in (("S", s_rec), ("O", o_rec)):
                taken.add(rec.index)
                self._fill_members(l, kind, rec)
                for side, t in (("a", rec.a), ("b", rec.b)):
                    p = _ptr(t)
                    if p in ops or p in linear_a:
                        raise LabelingError(f"layer {l} {kind}.{side} (record #{rec.index}) "
                                            f"shares storage with another forward operand")
                    ops[p] = (l, kind, side)
        stray = [r.index for r in recs if r.index not in taken]
        if stray:
            raise LabelingError(f"forward bmms outside every layer bracket: {stray}")
        return ops

    def _check_row_stochastic(self, l: int, rec: MatmulRecord) -> None:
        a = rec.a
        with torch.no_grad():
            ok = bool((a >= 0).all()) and float((a.sum(-1) - 1).abs().max()) < 1e-4
        if not ok:
            raise LabelingError(f"layer {l}: the second forward bmm (#{rec.index}) has an A that "
                                f"is not row-stochastic, so it is not O = A·V")

    def _check_kv_values(self, l: int, s_rec: MatmulRecord, o_rec: MatmulRecord,
                         v_rec: MatmulRecord) -> None:
        # repeat_kv copies each kv head to its g query heads, so these hold bit for bit.
        c = self.c
        g, shape = c.n_h // c.n_kv, (c.n_s, c.n_kv, c.n_h // c.n_kv)
        if s_rec.batch != c.n_s * c.n_h or o_rec.batch != c.n_s * c.n_h:
            return  # _fill_members reports the batch size
        with torch.no_grad():
            if tuple(v_rec.out.shape) != (c.N, c.n_kv * c.d_h) \
                    or tuple(o_rec.b.shape[1:]) != (c.n, c.d_h) \
                    or tuple(s_rec.b.shape[1:]) != (c.d_h, c.n):
                return  # _fill reports the shapes
            v = v_rec.out.view(c.n_s, c.n, c.n_kv, c.d_h).permute(0, 2, 1, 3)
            o_b = o_rec.b.reshape(*shape, c.n, c.d_h)
            if not torch.equal(o_b, v[:, :, None].expand_as(o_b)):
                raise LabelingError(f"layer {l}: O's B (record #{o_rec.index}) is not Y_v's "
                                    f"output, head by head")
            s_b = s_rec.b.reshape(*shape, c.d_h, c.n)
            if g > 1 and not torch.equal(s_b, s_b[:, :, :1].expand_as(s_b)):
                raise LabelingError(f"layer {l}: S's B (record #{s_rec.index}) differs across "
                                    f"query heads that share a kv head")

    def _fill_members(self, l: int, role: str, rec: MatmulRecord,
                      transposed: bool = False) -> None:
        c = self.c
        if rec.batch != c.n_s * c.n_h:
            raise LabelingError(f"layer {l} {role} (record #{rec.index}): batch {rec.batch} != "
                                f"n_s·n_h = {c.n_s * c.n_h}")
        # δK̃: the record holds δK̃ᵀ; one contiguous transpose of the whole batch gives the leaves.
        leaves = rec.out.transpose(1, 2).contiguous() if transposed else None
        for i, (a, b, out) in enumerate(rec.members()):
            s, h = divmod(i, c.n_h)
            self._fill(f"L{l}.{role}[{s},{h}]", a, b, out, rec,
                       leaf=None if leaves is None else leaves[i])

    # ---- backward --------------------------------------------------------------------------

    def _backward_linears(self, recs: list[MatmulRecord], y_rec: dict[str, MatmulRecord]) -> None:
        """``δX_x``/``δF`` by weight storage, then ``G_x``/``G_E^head`` by the saved inputs."""
        dy: dict[int, str] = {}  # storage of δY_x -> weight
        rest = []
        for rec in recs:
            w = rec.b_info.param_name
            if w is None:
                rest.append(rec)
                continue
            if w not in self.role:
                raise LabelingError(f"backward mm #{rec.index} has B = {w}, which has no products")
            p = _ptr(rec.a)
            if p in dy:
                raise LabelingError(f"δY of {w} (#{rec.index}) aliases δY of {dy[p]}")
            dy[p] = w
            self._fill(self._linear_name(w, "dX"), rec.a, rec.b, rec.out, rec)
        x_of: dict[int, set[str]] = {}  # storage of X_x -> the weights it feeds
        for w, rec in y_rec.items():
            x_of.setdefault(_ptr(rec.a), set()).add(w)
        for rec in rest:
            feeds = x_of.get(_ptr(rec.b), set())
            w = dy.get(_ptr(rec.a))
            if w is None or w not in feeds:
                raise LabelingError(f"backward mm #{rec.index} matches no weight gradient: A is "
                                    f"δY of {w}, B is the input of {sorted(feeds)}")
            if w != _E:
                grad = self.model.get_parameter(w).grad
                if grad is None or _ptr(grad) != _ptr(rec.out):
                    raise LabelingError(f"G of {w} (record #{rec.index}) is not {w}.grad")
            self._fill(self._linear_name(w, "G"), rec.a, rec.b, rec.out, rec)

    def _backward_bmms(self, recs: list[MatmulRecord],
                       ops: dict[int, tuple[int, str, str]]) -> None:
        """The four attention gradients, by which saved forward operand each one reads."""
        role_of = {v: k for k, v in _ATTN_BACKWARD.items()}  # (S|O, a|b) -> role
        glue: dict[tuple[int, str], int] = {}  # (ℓ, δO|δS) -> storage of the glue operand
        for rec in recs:
            hit_a, hit_b = ops.get(_ptr(rec.a)), ops.get(_ptr(rec.b))
            # grad_mat2 = Aᵀ·grad reads the saved A; grad_self = grad·Bᵀ reads the saved B.
            if (hit_a is None) == (hit_b is None):
                raise LabelingError(f"backward bmm #{rec.index}: expected exactly one saved "
                                    f"forward operand, got A→{hit_a}, B→{hit_b}")
            l, fwd, side = hit_a if hit_a is not None else hit_b
            if (side == "a") != (hit_a is not None):
                raise LabelingError(f"backward bmm #{rec.index} reads {fwd}.{side} in the "
                                    f"wrong operand position")
            role = role_of[(fwd, side)]
            other = rec.b if side == "a" else rec.a
            key = (l, "dO" if fwd == "O" else "dS")
            if glue.setdefault(key, _ptr(other)) != _ptr(other):
                raise LabelingError(f"layer {l} {role} (#{rec.index}): its {key[1]} operand "
                                    f"differs from the other {fwd} gradient's")
            self._fill_members(l, role, rec, transposed=(role == "dK"))


@dataclass
class _Unit:
    """One backward unit of the single pass: a decoder layer, or the head (``j = L+1``).

    ``x`` is the unit's input, a detached copy of what the model passed it (``X_j``) with
    ``requires_grad`` on, so the autograd graph is cut at the unit boundary. ``out`` is the
    unit's output with its graph: the layer's output, or the scalar ``ℒ`` for the head.
    ``fwd`` holds each forward product's received operands with their ``_version`` at the
    call, and ``saved`` the storages of ``S``'s and ``O``'s operands (storage -> (S|O, a|b)).
    """

    x: torch.Tensor
    out: torch.Tensor | None = None
    fwd: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]] = field(default_factory=dict)
    saved: dict[int, tuple[str, str]] = field(default_factory=dict)


class LlamaReplay(Replay):
    """Glue replay of one Llama step from committed leaves (checks 3, 5 and 6b).

    The replay does what training does: **one forward pass with grad on, then one backward
    pass**, both under :class:`~verification.computation.substitution.ProductSubstitution`, so
    every checked product is its committed leaf and only glue is computed.

    **Forward.** The first request runs ``logits`` on the committed batch, then
    ``loss_from_logits``, with grad on. A forward pre-hook on each decoder layer (and on the
    final norm) hands the module a detached copy of its input with ``requires_grad`` on: that
    is ``X_ℓ`` (``X_{L+1}`` for the head), and it cuts the autograd graph at the layer
    boundary, so each backward unit (a layer, or the head) keeps a graph of its own. During the
    pass the replay keeps, per unit, every operand its products receive, the unit's output with
    its graph, the storages of the ``S`` and ``O`` operands, and a hook on each linear
    product's output that records the storage of its output gradient ``δY_x``. Forward
    requests are then served from those kept operands in any order, with no rerun.

    The pass checks the call sequence by name, layer by layer (``Y_q, Y_k, Y_v, S, O, Y_o,
    Y_gate, Y_up, Y_down``, then ``Λ`` and nothing in the loss), and that each module receives
    exactly what the one before it returned: a layer's output is ``X_{ℓ+1}``, bit for bit.

    Operands are what the product op receives in the model's own code: ``(X·, W_xᵀ)`` for a
    linear, and for an attention product the bmm operands at batch index ``s·n_h + h``, i.e.
    ``(Q̃, K̃ᵀ)`` after RoPE and ``repeat_kv`` for ``S``, ``(softmax(S/√d_h + Ω), Ṽ)`` for ``O``.

    A model run that departs from ``C`` (an undeclared product, a different call order, a
    module that doesn't receive what the one before it returned) raises :class:`ReplayError`.
    The leaves were validated by check 2, so that is a verifier-side bug and propagates as a
    crash, not a rejection.

    **Backward (A9).** The backward products come in units: the head (``δF``, ``G_E^head``),
    then layers ``L … 1``. A unit's first request calls ``torch.autograd.grad`` on that unit's
    kept output: from ``ℒ`` for the head, with the held ``δX_{j+1}`` for layer ``j``, to
    ``X_j``, the unit's γ and its weights. Nothing is rerun. Each backward product it meets is
    replaced by its committed leaf, so RMSNorm, softmax, SiLU, RoPE, ``repeat_kv`` and the
    residual sums run in autograd on committed values. The products are identified at call
    time by operand identity, as in the labeling:

    - an mm whose ``B`` is a declared weight is ``δX_x`` (``δF`` for ``W_E``), and its ``A``
      must be that weight's output gradient ``δY_x``, whose storage the hook recorded;
    - another mm is ``G_x`` (``G_E^head``): its ``A`` is ``δY_xᵀ`` and its ``B`` the saved
      forward input of ``Y_x``. q, k, v (and gate, up) share that input; ``δY`` tells them
      apart;
    - a bmm is ``δA``, ``δV``, ``δQ̃`` or ``Q̃ᵀ·δS``, by which saved operand of ``S`` or ``O``
      it reads, and in which position. The last is supplied as the transposed ``δK̃`` leaves.

    Operands are what the op receives: ``(δY_x, W_x)`` and ``(δY_xᵀ, X_x)``, ``(δO, Ṽᵀ)``,
    ``(Aᵀ, δO)``, ``(δS, K̃)``, and for ``δK̃`` the received ``(Q̃ᵀ, δS)`` transposed to
    ``(δSᵀ, Q̃)`` (A7). Each operand's ``_version``, forward and backward, is recorded when
    the product reads it and checked when it is served.

    **Memory.** As in training: after the forward pass every unit's graph and forward operands
    are held. A unit's ``autograd.grad`` frees its graph's saved tensors, and the replay then
    drops the unit's forward operands, input and output, so memory falls unit by unit during
    the backward. The residual-stream frontier ``δX_j`` and the γ gradients persist; one
    unit's backward operands are held, and dropped after its last canonical product
    (``G_E_head``, or ``L{ℓ}.G_v``). The head is the largest unit: ``δΛ`` and the softmax
    intermediates are ``N × n_v`` each.

    **Order.** Canonical order ``1..M`` runs the forward and each unit's backward once. A
    forward request whose unit has already run its backward, or a backward request for a unit
    that already ran, rebuilds from scratch: a fresh pass, then the backward units from the head
    down to the one requested. So any order gives the same operands.

    :meth:`glue_gradients` is valid only right after ``operands(M)``, with ``δX_1`` held. It
    returns each γ's gradient from the units' backward, and ``W_E``'s full gradient: the
    committed ``G_E^head`` plus the embedding module's own backward of ``δX_1``.
    """

    def __init__(self, computation: LlamaComputation, leaves: LeafReader) -> None:
        c = self.c = computation
        self.view = TranscriptView(c, leaves)
        self.model = c.build_model()
        load_weights(c, self.model, {n: self.view.w_t(n) for n in c.weight_names})
        self._weight_of = param_storage_map(self.model)  # storage -> weight name
        self._role = {c.w(l, x): (l, x) for l in range(1, c.L + 1) for x in LINEARS}
        self._role[_E] = (None, "E")
        self.batch: Batch | None = None
        # The pass: whether it ran, its units by j (dropped as each one's backward runs) and
        # the next unit whose backward has not run (L+1 = the head; 0 once all have run).
        self._built = False
        self._units: dict[int, _Unit] = {}
        self._next_unit = c.L + 1
        # During the forward pass: every call's name in order, where the current unit's calls
        # start, the current unit's calls (name -> (A, B, output)), the layer whose S, O pair
        # is open (set by its Y_v) and the bmms taken from it.
        self._names: list[str] = []
        self._mark = 0
        self._calls: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._cur: int | None = None
        self._n_bmm = 0
        # During one unit's backward: the unit, its forward linears by name -> weight, its
        # backward product names, its forward operands, its S/O operand storages and the
        # output-gradient storages the hooks record (storage -> weight).
        self._unit: int | None = None
        self._ys: dict[str, str] = {}
        self._want: set[str] = set()
        self._fwd: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
        self._saved: dict[int, tuple[str, str]] = {}
        self._dy: dict[int, str] = {}
        # Held across requests: the unit whose backward operands are held and those operands
        # with their versions, the residual-stream frontier (j, δX_j), the γ gradients, and
        # whether the last request was operands(M).
        self._bunit: int | None = None
        self._bops: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
        self._dx: tuple[int, torch.Tensor] | None = None
        self._glue: dict[str, torch.Tensor] = {}
        self._served_last = False

    # ---- forward: the one pass ------------------------------------------------------------------

    def _supply(self, op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """A forward product's committed leaf, as a fresh tensor (substitution.py)."""
        c = self.c
        if op == "mm":
            w = self._weight_of.get(_ptr(b))
            if w not in self._role:
                raise ReplayError(f"replay: an mm whose B is not a declared weight ({w})")
            l, x = self._role[w]
            # S then O follow the layer's Y_v, as in the labeling (module docstring).
            self._cur, self._n_bmm = (l if x == "v" else None), 0
            name = "Lambda" if l is None else f"L{l}.Y_{x}"
            out = self.view.product(c.m_of(name)).clone()
        else:
            if self._cur is None or self._n_bmm > 1:
                raise ReplayError("replay: a bmm outside a layer's S, O pair")
            name = f"L{self._cur}.{('S', 'O')[self._n_bmm]}"
            self._n_bmm += 1
            out = torch.stack([self.view.product(c.m_of(f"{name}[{s},{h}]"))
                               for s, h in c._members()])
        self._names.append(name)
        if name not in self._calls:  # a repeat fails the unit's name check
            self._calls[name] = (a, b, out)
        return out

    @staticmethod
    def _layer_names(l: int) -> list[str]:
        return [f"L{l}.{r}" for r in ("Y_q", "Y_k", "Y_v", "S", "O", "Y_o", "Y_gate", "Y_up",
                                      "Y_down")]

    def _enter_unit(self, j: int, x: Any) -> torch.Tensor:
        """Start unit ``j`` on input ``x``; returns the graph-cutting copy the module gets."""
        c = self.c
        if not isinstance(x, torch.Tensor):
            raise ReplayError(f"replay: unit {j}'s input is not a tensor")
        if j > 1:  # a module receives exactly what the one before it returned
            prev = self._units[j - 1].out if j - 1 in self._units else None
            if prev is None or (x is not prev and not torch.allclose(
                    x, prev, rtol=0.0, atol=0.0, equal_nan=True)):
                what = "the final norm" if j == c.L + 1 else f"layer {j}"
                raise ReplayError(f"replay: layer {j - 1}'s output is not X_{j}, the input "
                                  f"{what} receives")
        if len(self._names) != self._mark or self._calls:
            raise ReplayError("replay: the forward pass does not have the declared products "
                              "(a product outside every layer)")
        xin = x.detach().requires_grad_()
        self._units[j] = _Unit(x=xin)
        return xin

    def _leave_unit(self, j: int, out: torch.Tensor) -> None:
        """Close unit ``j``: check its calls, keep its operands and output, hook its ``δY``."""
        c = self.c
        head = j == c.L + 1
        names = self._names[self._mark:]
        calls, self._calls, self._mark = self._calls, {}, len(self._names)
        if names != (["Lambda"] if head else self._layer_names(j)):
            where = "the head" if head else f"layer {j}"
            raise ReplayError(f"replay: {where} does not have the declared products")
        u = self._units[j]
        u.out = out
        u.fwd = {n: (a, b, a._version, b._version) for n, (a, b, _) in calls.items()}
        for kind in ([] if head else ["S", "O"]):
            fa, fb, _ = calls[f"L{j}.{kind}"]
            u.saved[_ptr(fa)], u.saved[_ptr(fb)] = (kind, "a"), (kind, "b")
        if len(u.saved) != (0 if head else 4):
            raise ReplayError(f"replay: layer {j}'s S and O operands share storage")
        for y, w in self._unit_names(j)[0].items():  # δY_x's storage identifies δX_x and G_x
            calls[y][2].register_hook(lambda g, w=w: self._dy.__setitem__(_ptr(g), w))

    def _build(self) -> None:
        """The one forward pass, embedding to ``ℒ``, with grad on under substitution."""
        c, model, inner = self.c, self.model, self.model.model
        self._built, self._units, self._next_unit = False, {}, c.L + 1
        self._bunit, self._bops, self._dx, self._glue = None, {}, None, {}
        self._names, self._mark, self._calls, self._cur, self._n_bmm = [], 0, {}, None, 0

        def enter_layer(j):
            def hook(module, args, kwargs):
                if len(args) != 1:
                    raise ReplayError(f"layer {j} called with {len(args)} positional args")
                return (self._enter_unit(j, args[0]),), kwargs
            return hook

        def leave_layer(j):
            def hook(module, args, output):
                self._leave_unit(j, output[0] if isinstance(output, tuple) else output)
            return hook

        def enter_head(module, args):
            return (self._enter_unit(c.L + 1, args[0]),)

        hooks = []
        for j, layer in enumerate(inner.layers, start=1):
            hooks.append(layer.register_forward_pre_hook(enter_layer(j), with_kwargs=True))
            hooks.append(layer.register_forward_hook(leave_layer(j)))
        hooks.append(inner.norm.register_forward_pre_hook(enter_head))
        batch = c.assemble(self.view.records())
        try:
            with torch.enable_grad(), ProductSubstitution(self._supply):
                loss = c.loss_from_logits(c.logits(model, batch), batch)
                if c.L + 1 not in self._units:
                    raise ReplayError("replay: the forward pass never reached the final norm")
                self._leave_unit(c.L + 1, loss)
            want = [n for l in range(1, c.L + 1) for n in self._layer_names(l)] + ["Lambda"]
            if self._names != want or set(self._units) != set(range(1, c.L + 2)):
                raise ReplayError("replay: the forward pass does not have the declared products")
        except BaseException:
            self._units = {}
            raise
        finally:
            for h in hooks:
                h.remove()
            self._names, self._calls = [], {}
        self.batch, self._built = batch, True

    def _forward_operands(self, spec: ProductSpec) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.c
        j = c.L + 1 if spec.layer is None else spec.layer
        if not self._built or j not in self._units:  # first request, or j's backward ran
            self._build()
        role = c.product_class(spec)
        key = role if spec.layer is None else f"L{j}.{role}"
        a, b, va, vb = self._units[j].fwd[key]
        if a._version != va or b._version != vb:
            raise ReplayError(f"replay: an operand of {key} changed after its product read it")
        if spec.member is not None:
            s, h = spec.member
            a, b = a[s * c.n_h + h], b[s * c.n_h + h]
        return a.detach(), b.detach()  # served without the graph they belong to

    # ---- backward (A9) -------------------------------------------------------------------------

    def _unit_names(self, j: int) -> tuple[dict[str, str], set[str]]:
        """Backward unit ``j`` (a layer, or ``L+1`` for the head): its forward linear products
        mapped to their weights, and the names of its backward products."""
        c = self.c
        if j == c.L + 1:
            return {"Lambda": _E}, {"dF", "G_E_head"}
        ys = {f"L{j}.Y_{x}": c.w(j, x) for x in LINEARS}
        want = {f"L{j}.{p}_{x}" for p in ("dX", "G") for x in LINEARS}
        return ys, want | {f"L{j}.{r}" for r in _ATTN_BACKWARD}

    def _linear_name(self, w: str, prefix: str) -> str:
        l, x = self._role[w]
        if l is None:
            return {"dX": "dF", "G": "G_E_head"}[prefix]
        return f"L{l}.{prefix}_{x}"

    def _supply_backward(self, op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """A backward product of the running unit, identified by operand identity as in the
        labeling (module docstring)."""
        c = self.c
        if op == "mm":
            w = self._weight_of.get(_ptr(b))
            if w is not None:  # δX_x = δY_x·W_x reads the weight itself
                if w not in self._role or self._dy.get(_ptr(a)) != w:
                    raise ReplayError(f"replay: a backward mm reads weight {w} but its A is "
                                      f"not that weight's output gradient")
                name = self._linear_name(w, "dX")
            else:  # G_x = δY_xᵀ·X_x reads the saved forward input
                w = self._dy.get(_ptr(a))
                y = next((n for n, v in self._ys.items() if v == w), None)
                if w is None or y is None or _ptr(b) != _ptr(self._fwd[y][0]):
                    raise ReplayError(f"replay: a backward mm matches no weight gradient "
                                      f"(A is the output gradient of {w})")
                name = self._linear_name(w, "G")
            out = self.view.product(c.m_of(name)).clone() if name in self._want else None
        else:
            hit_a, hit_b = self._saved.get(_ptr(a)), self._saved.get(_ptr(b))
            if (hit_a is None) == (hit_b is None):
                raise ReplayError(f"replay: a backward bmm reads {hit_a}, {hit_b}; expected "
                                  f"exactly one saved forward operand")
            fwd, side = hit_a if hit_a is not None else hit_b  # type: ignore[misc]
            # grad_mat2 = Aᵀ·grad reads the saved A; grad_self = grad·Bᵀ reads the saved B.
            if (side == "a") != (hit_a is not None):
                raise ReplayError(f"replay: a backward bmm reads {fwd}.{side} in the wrong "
                                  f"operand position")
            role = {v: k for k, v in _ATTN_BACKWARD.items()}[(fwd, side)]
            name = f"L{self._unit}.{role}"
            out = None
            if name in self._want:
                out = torch.stack([self.view.product(c.m_of(f"{name}[{s},{h}]"))
                                   for s, h in c._members()])
                if role == "dK":  # the op computes δK̃ᵀ = Q̃ᵀ·δS; the leaves hold δK̃
                    out = out.transpose(1, 2).contiguous()
        if out is None or name in self._bops:
            raise ReplayError(f"replay: backward product {name} is not declared in this unit "
                              f"or appears twice")
        self._bops[name] = (a, b, a._version, b._version)
        return out

    def _run_unit(self, j: int) -> None:
        """Backpropagate unit ``j`` of the pass: ``δX_{j+1}`` (the head starts from ``ℒ``)
        into ``δX_j`` and its γ, under substitution; then drop the unit's forward state."""
        c, model = self.c, self.model
        head = j == c.L + 1
        u = self._units.get(j)
        if u is None or u.out is None or self._next_unit != j:
            raise ReplayError(f"replay: backward unit {j} is not the next one of the pass")
        grad_out = None  # the head starts from ℒ; layer j from δX_{j+1}
        if not head:
            dx = self._dx
            if dx is None or dx[0] != j + 1:
                raise ReplayError(f"replay: layer {j}'s backward needs δX_{j + 1}")
            grad_out = dx[1]
        if head:
            gammas = [c.gamma_final]
            params = [model.get_parameter(c.gamma_final), model.get_parameter(_E)]
        else:
            gammas = [c.gamma_attn(j), c.gamma_mlp(j)]
            params = [model.get_parameter(n)
                      for n in (*gammas, *(c.w(j, x) for x in LINEARS))]
        self._ys, self._want = self._unit_names(j)
        self._unit, self._bunit, self._bops = j, None, {}
        self._fwd, self._saved, self._dy = u.fwd, u.saved, {}
        try:
            with ProductSubstitution(self._supply_backward):
                grads = torch.autograd.grad(
                    u.out, [u.x, *params], grad_outputs=grad_out, allow_unused=True)
        except BaseException:
            self._built, self._units = False, {}  # the graph may be half consumed
            raise
        finally:
            self._fwd, self._saved, self._dy = {}, {}, {}
            # The graph's saved tensors are freed; drop the unit's operands, input and output.
            self._units.pop(j, None)
            self._next_unit = j - 1
        if not self._bops:
            raise ReplayError(f"replay: no backward product of unit {j} reached the "
                              f"substitution; is the dispatch mode not active in backward?")
        if any(g is None for g in grads) or set(self._bops) != self._want:
            missing = sorted(self._want - set(self._bops))
            raise ReplayError(f"replay: backward unit {j} is missing {missing or 'a gradient'}")
        self._dx = (j, grads[0])
        for n, g in zip(gammas, grads[1:]):
            self._glue[n] = g
        self._bunit = j

    def _backward_operands(self, spec: ProductSpec) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.c
        j = c.L + 1 if spec.layer is None else spec.layer
        if self._bunit != j:
            # In order, unit j is the next one of the pass; otherwise rebuild from scratch.
            if not self._built or self._next_unit < j:
                self._build()
            while self._next_unit >= j:
                self._run_unit(self._next_unit)
        role = c.product_class(spec)
        key = role if spec.layer is None else f"L{j}.{role}"
        a, b, va, vb = self._bops[key]
        if a._version != va or b._version != vb:
            raise ReplayError(f"replay: an operand of {key} changed after its product read it")
        if spec.member is not None:
            s, h = spec.member
            i = s * c.n_h + h
            a, b = (b[i].mT, a[i].mT) if role == "dK" else (a[i], b[i])
        # The unit's last product in canonical order: drop the unit's operands.
        if spec.name == ("G_E_head" if spec.layer is None else f"L{j}.G_v"):
            self._bunit, self._bops = None, {}
        return a, b

    # ---- Replay ------------------------------------------------------------------------------

    def operands(self, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.c
        spec = c.product(m)
        if spec.kind is ProductKind.FORWARD:
            a, b = self._forward_operands(spec)
        else:
            a, b = self._backward_operands(spec)
        if a.dtype != c.operand_dtype or b.dtype != c.operand_dtype:
            raise ReplayError(f"{spec.name}: operands {a.dtype}, {b.dtype}, declared "
                              f"{c.operand_dtype}")
        self._served_last = m == c.M
        return a, b

    def glue_gradients(self) -> dict[str, torch.Tensor]:
        """Check 6b: each γ's gradient, and ``W_E``'s full gradient ``G_E^head + G_E^emb``.

        Valid right after ``operands(M)``, the last product of layer 1's backward. ``G_E^head``
        is the committed leaf. ``G_E^emb`` is the backward of the model's own embedding module
        at the committed ids, fed ``δX_1``, so it follows that module's ``padding_idx``. At
        test scale that has no numeric effect: SmolLM2's padding id 2 occurs only at padded
        positions, where ``δX_1`` is exactly 0, so a plain scatter-add gives the same bits.
        """
        c = self.c
        if not self._served_last or self._dx is None or self._dx[0] != 1:
            raise ReplayError("replay: glue_gradients() is valid only right after "
                              "operands(1..M) in order")
        assert self.batch is not None
        emb = self.model.model.embed_tokens

        def no_product(op, a, b):
            raise ReplayError("replay: a product in the embedding backward")

        with torch.enable_grad(), ProductSubstitution(no_product):
            (g_emb,) = torch.autograd.grad(emb(self.batch.ids), [emb.weight],
                                           grad_outputs=self._dx[1])
        grads = {**self._glue, _E: self.view.product(c.m_of("G_E_head")) + g_emb}
        out = {n: grads[n] for n in c.glue_gradient_weights}
        bad = [n for n, g in out.items() if g.dtype != c.weight_dtype]
        if bad:
            raise ReplayError(f"replay: glue gradients not {c.weight_dtype}: {bad[:3]}")
        return out
