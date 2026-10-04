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
same substitution; no backward is written by hand.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
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


class LlamaReplay(Replay):
    """Glue replay of one Llama step from committed leaves (check 3 for check 5).

    The first forward request runs the whole forward pass once (``logits`` on the committed
    batch) under substitution, with hooks that keep each layer's input ``X_ℓ`` (``X_{L+1}`` is
    the final norm's input) and the keyword arguments LlamaModel passes its layers: the causal
    mask over ``ρ``, the RoPE ``(cos, sin)``, the positions. That pass checks the call sequence
    by name and keeps no layer's operands, only ``Λ``'s. A request in layer ℓ then reruns
    ``layers[ℓ−1](X_ℓ, **kwargs)`` under substitution and keeps every operand that layer's
    products receive, which must reproduce ``X_{ℓ+1}``; they are dropped after ``Y_down``. Peak
    glue is one layer's operands plus the ``L+1`` residual states and ``Λ``'s operands. An
    out-of-order call reruns its layer, so any order gives the same operands.

    Operands are what the product op receives in the model's own code: ``(X·, W_xᵀ)`` for a
    linear, and for an attention product the bmm operands at batch index ``s·n_h + h``, i.e.
    ``(Q̃, K̃ᵀ)`` after RoPE and ``repeat_kv`` for ``S``, ``(softmax(S/√d_h + Ω), Ṽ)`` for ``O``.

    A model run that departs from ``C`` (an undeclared product, a different call order, a layer
    that doesn't reproduce its output) raises :class:`ReplayError`. The leaves were validated
    by check 2, so that is a verifier-side bug and propagates as a crash, not a rejection.

    ``X_1 … X_{L+1}``, the layer kwargs, the batch and ``Λ``'s operands stay for the whole step.
    The forward requests run under ``torch.no_grad()``; the backward reruns below run with grad
    on. The glue values are the same either way.

    **Backward (A9).** The backward products come in units: the head (``δF``, ``G_E^head``),
    then layers ``L … 1``. A unit's first request reruns it with grad on under substitution:
    the head as ``lm_head(norm(X_{L+1}))`` into ``loss_from_logits`` on the committed batch,
    which gives ``δΛ`` by autograd; a layer as ``layers[ℓ−1](X_ℓ, **kwargs)``, which must give
    the declared forward calls and reproduce ``X_{ℓ+1}``. ``torch.autograd.grad`` then
    backpropagates (the scalar ``ℒ``, or the held ``δX_{ℓ+1}``) to ``X_j``, the unit's γ and
    its weights. Each backward product it meets is replaced by its committed leaf, so RMSNorm,
    softmax, SiLU, RoPE, ``repeat_kv`` and the residual sums run in autograd on committed values.
    The products are identified at call time by operand identity, as in the labeling:

    - an mm whose ``B`` is a declared weight is ``δX_x`` (``δF`` for ``W_E``), and its ``A``
      must be that weight's output gradient ``δY_x``, whose storage a hook on the substituted
      forward output records;
    - another mm is ``G_x`` (``G_E^head``): its ``A`` is ``δY_xᵀ`` and its ``B`` the saved
      forward input of ``Y_x``. q, k, v (and gate, up) share that input; ``δY`` tells them
      apart;
    - a bmm is ``δA``, ``δV``, ``δQ̃`` or ``Q̃ᵀ·δS``, by which saved operand of ``S`` or ``O``
      it reads, and in which position. The last is supplied as the transposed ``δK̃`` leaves.

    Operands are what the op receives: ``(δY_x, W_x)`` and ``(δY_xᵀ, X_x)``, ``(δO, Ṽᵀ)``,
    ``(Aᵀ, δO)``, ``(δS, K̃)``, and for ``δK̃`` the received ``(Q̃ᵀ, δS)`` transposed to
    ``(δSᵀ, Q̃)`` (A7). Each operand's ``_version`` is checked when it is served.

    Memory: the residual-stream frontier ``δX_j`` and the γ gradients persist. One unit's
    operands are held, and are dropped after its last canonical product (``G_E_head``, or
    ``L{ℓ}.G_v``). So the backward peak is the forward's persistent state plus one unit's
    glue and its autograd graph. The head unit is the largest: ``δΛ`` and the softmax
    intermediates are ``N × n_v`` each. An out-of-order request recomputes the chain from the
    head, so any order gives the same operands.

    :meth:`glue_gradients` is valid only right after ``operands(M)``, with ``δX_1`` held. It
    returns each γ's gradient from the unit reruns, and ``W_E``'s full gradient: the committed
    ``G_E^head`` plus the embedding module's own backward of ``δX_1``.
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
        self.x: list[torch.Tensor] | None = None  # X_1..X_{L+1}
        self.layer_kwargs: list[dict[str, Any]] | None = None
        self.lambda_operands: tuple[torch.Tensor, torch.Tensor] | None = None
        self._layer: int | None = None  # the layer whose operands are held
        self._ops: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # Per substitution pass: the calls seen (operands only where `_keep` says), the layer
        # whose S, O pair is open (set by its Y_v) and the bmms taken from it.
        self._seen: list[tuple[str, torch.Tensor | None, torch.Tensor | None]] = []
        self._keep: Callable[[str], bool] = lambda name: True
        self._cur: int | None = None
        self._n_bmm = 0
        # Backward (A9). Per unit rerun: the phase, the unit, its forward calls (name ->
        # (A, B, output)), its weights by forward name, its backward product names, the saved
        # S/O operand storages and the output-gradient storages (storage -> weight).
        self._phase = "forward"
        self._unit: int | None = None
        self._fwd: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._ys: dict[str, str] = {}
        self._want: set[str] = set()
        self._saved: dict[int, tuple[str, str]] = {}
        self._dy: dict[int, str] = {}
        # Held across requests: the unit whose operands are held (L+1 = the head) and those
        # operands with their versions, the residual-stream frontier (j, δX_j), the γ
        # gradients, and whether the last request was operands(M).
        self._bunit: int | None = None
        self._bops: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
        self._dx: tuple[int, torch.Tensor] | None = None
        self._glue: dict[str, torch.Tensor] = {}
        self._served_last = False

    # ---- substitution ------------------------------------------------------------------------

    def _supply(self, op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The committed product for this call, as a fresh tensor (substitution.py)."""
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
        self._seen.append((name, a, b) if self._keep(name) else (name, None, None))
        return out

    def _substituted(self, run: Callable[[], None], keep: Callable[[str], bool]
                     ) -> list[tuple[str, torch.Tensor | None, torch.Tensor | None]]:
        """Run ``run`` under substitution; returns every call's name, with the operands of
        the calls ``keep`` accepts."""
        self._seen, self._keep, self._cur, self._n_bmm = [], keep, None, 0
        try:
            with torch.no_grad(), ProductSubstitution(self._supply):
                run()
            return self._seen
        finally:
            self._seen = []

    @staticmethod
    def _layer_names(l: int) -> list[str]:
        return [f"L{l}.{r}" for r in ("Y_q", "Y_k", "Y_v", "S", "O", "Y_o", "Y_gate", "Y_up",
                                      "Y_down")]

    def _forward(self) -> None:
        """The whole forward pass once; keeps ``X_ℓ``, the layer kwargs and ``Λ``'s operands."""
        c, inner = self.c, self.model.model
        x: list[Any] = [None] * (c.L + 1)
        kw: list[Any] = [None] * c.L

        def keep_layer(i):
            def hook(module, args, kwargs):
                if len(args) != 1:
                    raise ReplayError(f"layer {i + 1} called with {len(args)} positional args")
                x[i], kw[i] = args[0], dict(kwargs)
            return hook

        def keep_final(module, args):
            x[c.L] = args[0]

        hooks = [layer.register_forward_pre_hook(keep_layer(i), with_kwargs=True)
                 for i, layer in enumerate(inner.layers)]
        hooks.append(inner.norm.register_forward_pre_hook(keep_final))
        batch = c.assemble(self.view.records())
        try:
            seen = self._substituted(lambda: c.logits(self.model, batch),
                                     keep=lambda name: name == "Lambda")
        finally:
            for h in hooks:
                h.remove()
        want = [n for l in range(1, c.L + 1) for n in self._layer_names(l)] + ["Lambda"]
        if [n for n, _, _ in seen] != want or any(t is None for t in x):
            raise ReplayError("replay: the forward pass does not have the declared products")
        self.batch, self.x, self.layer_kwargs = batch, x, kw
        self.lambda_operands = seen[-1][1:]  # type: ignore[assignment]

    def _run_layer(self, l: int) -> None:
        """Rerun layer ℓ from ``X_ℓ`` and hold the operands of its nine products."""
        if self.x is None:
            self._forward()
        assert self.x is not None and self.layer_kwargs is not None
        self._layer, self._ops = None, {}
        out: Any = None

        def run():
            nonlocal out
            out = self.model.model.layers[l - 1](self.x[l - 1], **self.layer_kwargs[l - 1])

        seen = self._substituted(run, keep=lambda name: True)
        out = out[0] if isinstance(out, tuple) else out
        if [n for n, _, _ in seen] != self._layer_names(l):
            raise ReplayError(f"replay: layer {l} does not have the declared products")
        if not torch.allclose(out, self.x[l], rtol=0.0, atol=0.0, equal_nan=True):
            raise ReplayError(f"replay: rerunning layer {l} did not reproduce X_{l + 1}")
        self._layer, self._ops = l, {n: (a, b) for n, a, b in seen}  # type: ignore[misc]

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

    def _supply_unit(self, op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Supply for a backward unit's rerun: its forward products as :meth:`_supply` gives
        them, then its backward products, each identified by operand identity as in the
        labeling (module docstring)."""
        if self._phase == "forward":
            out = self._supply(op, a, b)
            self._fwd[self._seen[-1][0]] = (a, b, out)
            return out
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
        """Rerun backward unit ``j`` from ``X_j`` with grad on under substitution, and
        backpropagate ``δX_{j+1}`` (the head starts from ``ℒ``) into ``δX_j`` and its γ."""
        c, model = self.c, self.model
        head = j == c.L + 1
        assert self.x is not None and self.layer_kwargs is not None and self.batch is not None
        grad_out = None  # the head starts from ℒ; layer j from δX_{j+1}
        if not head:
            dx = self._dx
            if dx is None or dx[0] != j + 1:
                raise ReplayError(f"replay: layer {j}'s backward needs δX_{j + 1}")
            grad_out = dx[1]
        x = self.x[j - 1].detach().requires_grad_()
        if head:
            gammas = [c.gamma_final]
            params = [model.get_parameter(c.gamma_final), model.get_parameter(_E)]
        else:
            gammas = [c.gamma_attn(j), c.gamma_mlp(j)]
            params = [model.get_parameter(n)
                      for n in (*gammas, *(c.w(j, x) for x in LINEARS))]
        self._ys, self._want = self._unit_names(j)
        self._unit, self._bunit, self._bops = j, None, {}
        self._seen, self._keep, self._cur, self._n_bmm = [], lambda name: False, None, 0
        try:
            with torch.enable_grad(), ProductSubstitution(self._supply_unit):
                self._phase = "forward"
                if head:  # LlamaForCausalLM's tail: final norm, output projection, then ℒ
                    out = c.loss_from_logits(model.lm_head(model.model.norm(x)), self.batch)
                else:
                    out = model.model.layers[j - 1](x, **self.layer_kwargs[j - 1])
                    out = out[0] if isinstance(out, tuple) else out
                if [n for n, _, _ in self._seen] != (["Lambda"] if head
                                                       else self._layer_names(j)):
                    raise ReplayError(f"replay: backward unit {j}'s forward rerun does not "
                                      f"have the declared products")
                if not head and not torch.allclose(out, self.x[j], rtol=0.0, atol=0.0,
                                                   equal_nan=True):
                    raise ReplayError(f"replay: rerunning layer {j} did not reproduce X_{j + 1}")
                for kind in ([] if head else ["S", "O"]):
                    fa, fb, _ = self._fwd[f"L{j}.{kind}"]
                    self._saved[_ptr(fa)], self._saved[_ptr(fb)] = (kind, "a"), (kind, "b")
                if len(self._saved) != (0 if head else 4):
                    raise ReplayError(f"replay: layer {j}'s S and O operands share storage")
                for y, w in self._ys.items():  # δY_x's storage identifies δX_x and G_x
                    self._fwd[y][2].register_hook(
                        lambda g, w=w: self._dy.__setitem__(_ptr(g), w))
                self._phase = "backward"
                grads = torch.autograd.grad(
                    out, [x, *params], grad_outputs=grad_out, allow_unused=True)
        finally:
            self._phase, self._seen, self._fwd, self._saved, self._dy = "forward", [], {}, {}, {}
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
            if self.x is None:
                self._forward()
            # In order, δX_{j+1} is held from the unit just finished; otherwise the chain is
            # recomputed from the head.
            if j <= c.L and (self._dx is None or self._dx[0] != j + 1):
                for k in range(c.L + 1, j, -1):
                    self._run_unit(k)
            self._run_unit(j)
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
        if spec.kind is not ProductKind.FORWARD:
            a, b = self._backward_operands(spec)
        elif spec.layer is None:  # Λ
            if self.lambda_operands is None:
                self._forward()
            a, b = self.lambda_operands  # type: ignore[misc]
        else:
            l = spec.layer
            if self._layer != l:
                self._run_layer(l)
            role = c.product_class(spec)
            if spec.member is None:
                a, b = self._ops[f"L{l}.{role}"]
            else:
                s, h = spec.member
                a_all, b_all = self._ops[f"L{l}.{role}"]
                a, b = a_all[s * c.n_h + h], b_all[s * c.n_h + h]
            if role == "Y_down":  # the layer's last forward product
                self._layer, self._ops = None, {}
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
