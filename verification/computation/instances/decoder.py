"""The decoder instance of ``C`` (ref block §1–§6): the declaration, the prover-side labeling
and the verifier-side replay, for the Llama and Qwen2 families of S2.

The model is an unmodified Hugging Face ``LlamaForCausalLM`` or ``Qwen2ForCausalLM`` with
eager attention, fp32 weights and dropout zero (§8.A, S4d). The same class serves tiny randomly
initialised configs for the fast tests; the real instance is
:meth:`DecoderComputation.from_pretrained`. Two switches are read from the config, never from
the model name or the scale:

- **Output layer.** ``tie_word_embeddings`` true: the output projection is ``W_E``, as in the
  reference block (SmolLM2, Llama-3.2, Qwen2.5). False (Falcon3): it is its own linear weight
  ``W_out``, whose gradient ``G_head`` is a checked product (check 6a), and ``W_E`` keeps only
  the embedding's lookup gradient (check 6b). ``M`` is the same either way (S2).
- **q, k, v biases.** On for ``model_type == "qwen2"``, whose attention hardcodes them; Llama's
  ``attention_bias`` (which also biases ``o``) and ``mlp_bias`` must be off. A biased
  projection runs as one ``addmm``, and check 5 verifies it as one product of augmented
  operands (F14), so a bias adds no product. Each bias is a weight whose gradient is glue (check 6b).

Implementation pins (not fixed by the reference block):

- **Weight names** are the HF parameter names (``model.embed_tokens.weight``,
  ``model.layers.{ℓ−1}.self_attn.q_proj.weight``, ``model.norm.weight`` …), listed in the ref
  block §6 order, which differs from ``named_parameters()`` order. A tied output projection
  is ``W_E``; HF lists it once, under the embedding's name. A bias follows its weight
  (``W_q, b_q, W_k, b_k, W_v, b_v``, HF's order), and an untied ``lm_head.weight`` comes last,
  after ``γ_final``.
- **Product names** are ``L{ℓ}.Y_q`` … for the linear products, ``L{ℓ}.S[s,h]``,
  ``L{ℓ}.O[s,h]``, ``L{ℓ}.dA[s,h]``, ``L{ℓ}.dV[s,h]``, ``L{ℓ}.dQ[s,h]`` and ``L{ℓ}.dK[s,h]``
  for the attention members, and ``Lambda``, ``dF`` and ``G_E_head`` (``G_head`` when untied)
  outside the stack. ``member`` is the 0-based ``(s, h)``, the bmm batch index ``s·n_h + h``
  of HF's eager attention. Each product's κ class (:meth:`DecoderComputation.product_class`)
  is its role without the layer and member.
- **Batch assembly** is ``data.assemble_batch``: right padding with ``PAD_ID`` (SmolLM2's EOS,
  F8f), ``ρ`` from each record's ``ℓ``. ``ρ`` is the HF ``attention_mask``, so ``Ω`` is HF's
  causal mask over ``ρ``. Positions are ``0 … n−1`` in every sequence.
- **Loss** is ref block §3's ``ℒ = Σ_i μ_i·CE_i / Σ_i μ_i`` over the logits, with
  ``F.cross_entropy(reduction="none")`` per token. ``μ`` takes ``CE``'s dtype, fp32 under
  autocast too; in the bf16 logits' dtype ``Σ μ`` would round above 256. A batch with
  ``Σ μ = 0`` raises ``ValueError`` instead of producing a 0/0 NaN. ``data.build_record``
  never builds such a record (every record has at least the EOS target), so only a faulted
  batch can reach it.
- **Mixed precision** (F6b, F5a) is config, ``(MASTER_DTYPE, COMPUTE_DTYPE)``. The weights are
  the fp32 masters. At ``(fp32, bf16)`` forward and loss run under PyTorch autocast on
  ``VERIF_DEVICE``'s type (:meth:`DecoderComputation.autocast`): every product and every
  operand is bf16, accumulated in fp32, and the backward inherits the forward's dtypes. At
  ``(fp32, fp32)`` the call is a no-op and the step is unchanged bit for bit. CPU and CUDA
  autocast cast different op sets, so a bit-for-bit result on one device says nothing about
  the other.

Labeling (:meth:`DecoderComputation.label`) maps each captured matmul to its slot by operand
identity (storage), never by call order alone. Under autocast a weight operand (and a bias) is
autocast's copy of the weight, which the capture traces back to the weight
(``OperandInfo.param_copy``); nothing is matched by value:

- forward ``Y_x`` and ``Λ``: ``B`` is a view of the weight's storage; backward ``δX_x`` and
  ``δF``: ``B`` is the weight itself. A biased ``Y_x`` is an ``addmm`` whose bias is the
  declared ``b_x`` with ``α = β = 1``; any other product carrying a bias is refused;
- weight gradients ``G_x`` and the head's (``G_E^head`` or ``G_head``): ``B`` is the saved
  forward input (the storage of ``Y_x``'s ``A``, shared by q/k/v and by gate/up), and ``A`` is
  ``δYᵀ``, which shares storage with the ``A`` of the already-labeled ``δX_x``. For linear
  weights the output must also be the storage AccumulateGrad adopted as ``W_x.grad``, or under
  autocast the fp32 copy of the output that became ``W_x.grad``;
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

Replay (:class:`DecoderReplay`) is the verifier's side. It runs its own model, loaded from the
committed ``W_t``, under :class:`~verification.computation.substitution.ProductSubstitution`:
every checked product is replaced by its committed leaf, so all glue (embedding, RMSNorm, RoPE,
the causal mask over ``ρ``, softmax, SiLU, residual adds) runs through the model's own modules on
committed values (S4b, check 3), under the prover's autocast setting (F5a). A8 covers the
forward products. A9 covers the backward products and :meth:`DecoderReplay.glue_gradients`, by
autograd on the same modules under the same substitution; no backward is written by hand. As in
training, the replay runs one forward pass with grad on and one backward pass per unit (the
head, then layers ``L…1``), with no reruns; the class docstring has the details.

A biased ``Y_x``'s spec names its bias (``ProductSpec.bias``), and its committed leaf is
``A·B + 1·bᵀ``. The replay serves ``A`` and ``B`` without the bias; check 5 adds ``b`` from the
committed ``W_t`` and checks ``[A | 1]·[B ; bᵀ]`` (F14).
"""

from __future__ import annotations

import contextlib
import copy
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from setup.config import RunConfig, assert_no_dropout
from setup.data import PAD_ID, Batch, assemble_batch
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

__all__ = ["LINEARS", "DecoderComputation", "DecoderReplay", "matmul_count_decoder"]

# Linear weights of one layer, as ref block §2 names them, with their HF module paths.
LINEARS = ("q", "k", "v", "o", "gate", "up", "down")
_LINEAR_PATH = {
    "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
    "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}
# The linears that carry a bias when the switch is on (Qwen2's attention, F14).
_BIASED = ("q", "k", "v")
# The attention gradients: role -> (forward bmm, saved operand side).
_ATTN_BACKWARD = {"dA": ("O", "b"), "dV": ("O", "a"), "dQ": ("S", "b"), "dK": ("S", "a")}

_E = "model.embed_tokens.weight"
_W_OUT = "lm_head.weight"  # the output projection when untied
_FAMILIES = ("llama", "qwen2")
# A module's hook tables; any entry left on the verifier's reused model is a leak.
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
_GAMMA_FINAL = "model.norm.weight"


def matmul_count_decoder(L: int, n_s: int, n_h: int) -> int:
    """Products per step for the decoder block, `M = L(21 + 6·n_s·n_h) + 3` (ref block §5).

    The same with an untied output layer or q, k, v biases (S2, F14)."""
    return L * (21 + 6 * n_s * n_h) + 3


def _ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def _param(info: Any) -> str | None:
    """The parameter a captured operand is, or is autocast's copy of."""
    return info.param_name or info.param_copy


class DecoderComputation(DeclaredComputation):
    """``C`` for a Llama or Qwen2 causal LM of the given config, batch ``n_s × n`` and step size.

    ``source = (repo, revision)`` makes :meth:`build_model` an unmodified ``from_pretrained``
    load; without it the model is built from ``config`` with random weights (tests only).
    ``tied`` and ``qkv_bias`` are the config's switches (module docstring).
    ``(master_dtype, compute_dtype)`` is ``(MASTER_DTYPE, COMPUTE_DTYPE)``, and ``device`` the
    device whose autocast runs when the two differ (F6b).
    """

    def __init__(self, config: Any, *, n_s: int, n: int, eta: float,
                 source: tuple[str, str] | None = None,
                 master_dtype: torch.dtype = torch.float32,
                 compute_dtype: torch.dtype = torch.float32,
                 device: str | torch.device = "cpu") -> None:
        self.config = copy.deepcopy(config)
        self.source = source
        self._n_s, self.n, self._eta = int(n_s), int(n), float(eta)
        # F6b: fp32 masters, bf16 compute through autocast. fp16 compute needs loss scaling,
        # which no decision covers.
        if master_dtype != torch.float32:
            raise ValueError(f"MASTER_DTYPE must be float32 (F6b), got {master_dtype}")
        if compute_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"COMPUTE_DTYPE must be float32 or bfloat16, got {compute_dtype}")
        self.master_dtype, self.compute_dtype = master_dtype, compute_dtype
        self.device_type = torch.device(device).type
        c = self.config
        self.L = int(c.num_hidden_layers)
        self.d = int(c.hidden_size)
        self.d_f = int(c.intermediate_size)
        self.n_h = int(c.num_attention_heads)
        self.n_kv = int(c.num_key_value_heads)
        self.d_h = int(getattr(c, "head_dim", None) or self.d // self.n_h)
        self.n_v = int(c.vocab_size)
        self.tied = bool(c.tie_word_embeddings)
        self.qkv_bias = c.model_type == "qwen2"
        self._check_config()
        self.validate()
        # The verifier's model, built once and reset for each replay (_replay_model); the
        # prover never sees it.
        self._vmodel: nn.Module | None = None
        self._vbuffers: dict[str, torch.Tensor] = {}
        self._vowner: weakref.ref[DecoderReplay] | None = None

    @classmethod
    def from_pretrained(cls, repo: str, revision: str, *, n_s: int, n: int, eta: float,
                        master_dtype: torch.dtype = torch.float32,
                        compute_dtype: torch.dtype = torch.float32,
                        device: str | torch.device = "cpu") -> DecoderComputation:
        config = load_model_config(repo, revision)
        return cls(config, n_s=n_s, n=n, eta=eta, source=(repo, revision),
                   master_dtype=master_dtype, compute_dtype=compute_dtype, device=device)

    @classmethod
    def from_config(cls, cfg: RunConfig) -> DecoderComputation:
        """The run's instance: ``VERIF_MODEL`` at ``VERIF_MODEL_REVISION``, ``VERIF_BATCH ×
        VERIF_SEQ_LEN``, ``η = VERIF_ETA``, ``(VERIF_MASTER_DTYPE, VERIF_COMPUTE_DTYPE)`` on
        ``VERIF_DEVICE``."""
        return cls.from_pretrained(cfg.model, cfg.model_revision, n_s=cfg.batch, n=cfg.seq_len,
                                   eta=cfg.require_eta(), master_dtype=cfg.master_dtype,
                                   compute_dtype=cfg.compute_dtype, device=cfg.device)

    def _check_config(self) -> None:
        c = self.config
        if c.model_type not in _FAMILIES:
            raise ValueError(f"expected a config of model_type {_FAMILIES}, got "
                             f"{c.model_type!r}")
        bad = []
        if getattr(c, "attention_bias", False):
            bad.append("attention_bias must be off: it also biases o (F14 covers q, k, v)")
        if getattr(c, "mlp_bias", False):
            bad.append("mlp_bias must be off")
        if any(t != "full_attention" for t in getattr(c, "layer_types", None) or ()):
            bad.append(f"every layer must use full attention, got {c.layer_types}")
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

    def bias(self, l: int, x: str) -> str | None:
        """Name of ``W_x``'s bias ``b_x`` in layer ``ℓ``, or ``None`` when it has none."""
        if self.qkv_bias and x in _BIASED:
            return f"model.layers.{l - 1}.{_LINEAR_PATH[x]}.bias"
        return None

    def gamma_attn(self, l: int) -> str:
        return f"model.layers.{l - 1}.input_layernorm.weight"

    def gamma_mlp(self, l: int) -> str:
        return f"model.layers.{l - 1}.post_attention_layernorm.weight"

    @property
    def w_e(self) -> str:
        return _E

    @property
    def w_out(self) -> str:
        """The output projection: ``W_E`` when tied, else ``lm_head.weight``."""
        return _E if self.tied else _W_OUT

    @property
    def g_head(self) -> str:
        """The head's weight-gradient product: ``G_E_head``, a summand of ``W_E``'s glue
        gradient, when tied; ``G_head``, ``W_out``'s linear gradient, when untied."""
        return "G_E_head" if self.tied else "G_head"

    @property
    def gamma_final(self) -> str:
        return _GAMMA_FINAL

    def linear_shape(self, x: str) -> tuple[int, int]:
        """``(o, i)`` of ``W_x``."""
        q_out, kv_out = self.n_h * self.d_h, self.n_kv * self.d_h
        return {"q": (q_out, self.d), "k": (kv_out, self.d), "v": (kv_out, self.d),
                "o": (self.d, q_out), "gate": (self.d_f, self.d), "up": (self.d_f, self.d),
                "down": (self.d, self.d_f)}[x]

    def _linear_params(self, l: int, x: str) -> list[str]:
        b = self.bias(l, x)
        return [self.w(l, x)] if b is None else [self.w(l, x), b]

    @cached_property
    def weight_names(self) -> tuple[str, ...]:
        names = [_E]
        for l in range(1, self.L + 1):
            names += [self.gamma_attn(l),
                      *(n for x in ("q", "k", "v", "o") for n in self._linear_params(l, x)),
                      self.gamma_mlp(l), *(self.w(l, x) for x in ("gate", "up", "down"))]
        names.append(_GAMMA_FINAL)
        if not self.tied:
            names.append(_W_OUT)
        return tuple(names)

    @cached_property
    def weight_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {_E: (self.n_v, self.d), _GAMMA_FINAL: (self.d,),
                                              _W_OUT: (self.n_v, self.d)}
        for l in range(1, self.L + 1):
            shapes[self.gamma_attn(l)] = shapes[self.gamma_mlp(l)] = (self.d,)
            for x in LINEARS:
                shapes[self.w(l, x)] = self.linear_shape(x)
                b = self.bias(l, x)
                if b is not None:
                    shapes[b] = self.linear_shape(x)[:1]
        return {name: shapes[name] for name in self.weight_names}

    @cached_property
    def biases(self) -> dict[str, str]:
        """Each biased linear weight -> its bias, in ``weight_names`` order."""
        return {self.w(l, x): b for l in range(1, self.L + 1) for x in LINEARS
                if (b := self.bias(l, x)) is not None}

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
        w_out, g_head = self.w_out, self.g_head
        rows.append(("Lambda", fwd, (N, d), (d, n_v), w_out, None, None, "Lambda"))
        rows.append(("dF", ig, (N, n_v), (n_v, d), w_out, None, None, "dF"))
        rows.append((g_head, wg, (n_v, N), (N, d), w_out, None, None, g_head))
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
        biases = self.biases  # F14: Y_x = X·W_xᵀ + 1·b_xᵀ is one product
        specs = tuple(ProductSpec(m, name, kind, a, b, weight=w, layer=l, member=mem,
                                  bias=biases.get(w) if kind is fwd else None)
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
        ``O``, ``dA``, ``dV``, ``dQ``, ``dK``, ``Lambda``, ``dF`` and ``G_E_head`` (``G_head``
        when untied).
        """
        if self.product(spec.m) != spec:
            raise ValueError(f"{spec.name} is not product {spec.m} of this computation")
        return self._inventory[1][spec.m - 1]

    @cached_property
    def linear_weights(self) -> dict[str, int]:
        out = {self.w(l, x): self.m_of(f"L{l}.G_{x}")
               for l in range(1, self.L + 1) for x in LINEARS}
        if not self.tied:
            out[_W_OUT] = self.m_of("G_head")
        return out

    # glue_gradient_weights (the base class's derivation): W_E, every γ and every bias, for
    # check 6b.

    # P6, F6b: the weights are the fp32 masters; products and operands are in the compute
    # format, accumulated in fp32 (F5a turns cuBLAS's reduced-precision reductions off).
    @property
    def weight_dtype(self) -> torch.dtype:
        return self.master_dtype

    @property
    def product_dtype(self) -> torch.dtype:
        return self.compute_dtype

    @property
    def operand_dtype(self) -> torch.dtype:
        return self.compute_dtype

    @property
    def accumulator_dtype(self) -> torch.dtype:
        return torch.float32

    def autocast(self) -> contextlib.AbstractContextManager[Any]:
        """Autocast to the compute format on the run's device type; a no-op at
        ``(fp32, fp32)`` (F6b)."""
        if self.compute_dtype == self.master_dtype:
            return contextlib.nullcontext()
        return torch.autocast(self.device_type, dtype=self.compute_dtype)

    # ---- shared ---------------------------------------------------------------------------

    def encode_record(self, record: Record) -> bytes:
        if not isinstance(record, Record):
            raise RecordError(f"a decoder record is a records.Record, got "
                              f"{type(record).__name__}")
        if len(record) > self.n:
            raise RecordError(f"record length {len(record)} > n = {self.n}")
        return encode_record(record)

    def build_model(self) -> nn.Module:
        """An unmodified HF causal LM for the config (``LlamaForCausalLM``,
        ``Qwen2ForCausalLM``), eager attention, weights in the master format, dropout
        asserted zero.

        Eval mode: dropout is zero anyway (S4d), and HF runs activation checkpointing only in
        train mode, which the capture forbids. Load ``W_t`` before use.
        """
        if self.source is not None:
            repo, revision = self.source
            model = load_pretrained(repo, revision, attn_impl="eager", dtype=self.weight_dtype)
        else:
            from transformers import AutoModelForCausalLM

            with torch.random.fork_rng(devices=[]):
                model = AutoModelForCausalLM.from_config(
                    copy.deepcopy(self.config), attn_implementation="eager",
                    dtype=self.weight_dtype)
        model.eval()
        self._check_model(model)
        return model

    def _check_model(self, model: nn.Module) -> None:
        assert_no_dropout(model)
        if model.config._attn_implementation != "eager":
            raise ValueError(f"attention is {model.config._attn_implementation!r}, not eager")
        if getattr(model, "is_gradient_checkpointing", False):
            raise ValueError("activation checkpointing must be off (prover/capture.py)")
        tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
        if tied != self.tied:
            raise ValueError(f"the output projection is {'' if tied else 'not '}tied to W_E, "
                             f"the config says tie_word_embeddings={self.tied}")
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
                raise ValueError(f"a decoder record is a records.Record, got {type(r).__name__}")
        # F8f: the pad is the model's EOS. PAD_ID is SmolLM2's; the first real-model run (FS4)
        # wires each model's EOS from data.run_pin.
        return assemble_batch(records, self.n, pad_id=PAD_ID)

    def logits(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """``Λ`` as ``[n_s, n, n_v]``: the forward pass, embedding to output projection."""
        return model(input_ids=batch.ids, attention_mask=batch.rho, use_cache=False).logits

    @staticmethod
    def loss_from_logits(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        """``ℒ = (1/Σμ)·Σ_i μ_i·(logsumexp(Λ_i) − Λ_{i,y_i})`` (ref block §3)."""
        mask = batch.mask.reshape(-1)
        if not bool(mask.any()):
            raise ValueError("the batch has no loss-masked target")
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch.targets.reshape(-1),
                             reduction="none")
        mu = mask.to(ce.dtype)  # CE's dtype, not the logits' (module docstring)
        return (ce * mu).sum() / mu.sum()

    def _forward_loss(self, model: nn.Module, batch: Batch) -> torch.Tensor:
        """Forward and loss under :meth:`autocast`, for the prover's ``loss`` and the replay's
        pass alike (F5a)."""
        with self.autocast():
            return self.loss_from_logits(self.logits(model, batch), batch)

    # ---- verifier side --------------------------------------------------------------------

    def replay(self, leaves: LeafReader) -> DecoderReplay:
        return DecoderReplay(self, leaves)

    def _replay_model(self, owner: DecoderReplay) -> nn.Module:
        """The verifier's model for ``owner``, built on first use and reset on every reuse.

        Building a model per step reloads it from disk (about 0.12 s for SmolLM2). Reuse keeps
        the steps independent:

        - every parameter gets fresh storage, so no tensor served by an earlier replay (a
          linear's ``B`` is a view of its weight) aliases the next step's weights; the caller
          then loads ``W_t`` into all of them (``load_weights`` requires full coverage);
        - every buffer (the RoPE ``inv_freq``) is restored from its value at build time;
        - a hook left on any module, a changed buffer set or a failed model check raises;
        - the previous owner loses the model, so a stale replay can't run on another step's
          weights.

        Only :class:`DecoderReplay` calls this; the prover builds its own model.
        """
        model = self._vmodel
        if model is None:
            model = self._vmodel = self.build_model()
            self._vbuffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        else:
            prev = self._vowner() if self._vowner is not None else None
            if prev is not None:
                prev.model = None
            hooked = [n or "<root>" for n, mod in model.named_modules()
                      if any(getattr(mod, h) for h in _HOOK_DICTS)]
            if hooked:
                raise ReplayError(f"replay: hooks left on the verifier's model: {hooked[:3]}")
            buffers = dict(model.named_buffers())
            if buffers.keys() != self._vbuffers.keys():
                raise ReplayError("replay: the verifier's model buffers changed")
            with torch.no_grad():
                for n, b in buffers.items():
                    b.copy_(self._vbuffers[n])
                for p in model.parameters():
                    p.data = torch.empty_like(p)
                    p.grad = None
                    p.requires_grad_(True)
            model.eval()
            self._check_model(model)
        self._vowner = weakref.ref(owner)
        return model

    # ---- prover side ----------------------------------------------------------------------

    def loss(self, model: nn.Module, records: Sequence[Record]) -> torch.Tensor:
        return self._forward_loss(model, self.assemble(records))

    def label(self, capture: MatmulCapture, model: nn.Module) -> list[torch.Tensor]:
        """Map the captured records onto ``P_1..P_M`` by operand identity (module docstring)."""
        return _Labeler(self, capture, model).run()


class _Labeler:
    """One labeling pass. Every slot is filled exactly once, or :class:`LabelingError`."""

    def __init__(self, c: DecoderComputation, capture: MatmulCapture, model: nn.Module) -> None:
        self.c, self.cap, self.model = c, capture, model
        self.slots: list[torch.Tensor | None] = [None] * c.M
        # weight name -> (ℓ or None, x) for every weight that has linear products.
        self.role = {c.w(l, x): (l, x) for l in range(1, c.L + 1) for x in LINEARS}
        self.role[c.w_out] = (None, "head")

    def run(self) -> list[torch.Tensor]:
        cap, c = self.cap, self.c
        for rec in cap.records + cap.glue_outer:
            if rec.a is None:
                raise LabelingError("capture operands were released before labeling")
        # Only a forward linear may carry a bias, and only its declared one (_check_bias).
        stray = [r.index for r in cap.records + cap.glue_outer
                 if r.bias is not None and (r.phase != "forward" or r.batch is not None)]
        if stray:
            raise LabelingError(f"records {stray[:5]} carry a bias; only a forward linear may")
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
            return {"Y": "Lambda", "dX": "dF", "G": self.c.g_head}[prefix]
        return f"L{l}.{prefix}_{x}"

    def _check_bias(self, weight: str, rec: MatmulRecord) -> None:
        """``Y_x`` carries exactly its declared bias, added once (F14); an unbiased one none."""
        want = self.c.biases.get(weight)
        if want is None:
            if rec.bias is not None:
                raise LabelingError(f"forward mm #{rec.index} of {weight} carries a bias, none "
                                    f"is declared")
            return
        if rec.bias is None or self.cap.param_of(rec.bias) != want \
                or rec.alpha != 1 or rec.beta != 1:
            raise LabelingError(f"forward mm #{rec.index} of {weight} does not add its declared "
                                f"bias {want} once (α = β = 1)")

    def _check_glue_outer(self) -> None:
        # P7: the one q=1 product is the RoPE angle table, computed once per forward pass from
        # public constants (inv_freq and the positions).
        outer = self.cap.glue_outer
        bad = [r for r in outer if r.phase != "forward" or _param(r.a_info)
               or _param(r.b_info)]
        if bad or len(outer) != 1:
            raise LabelingError(f"expected exactly one forward q=1 product (the RoPE table, P7), "
                                f"got {len(outer)} of which {len(bad)} unexpected")

    # ---- forward ---------------------------------------------------------------------------

    def _forward_linears(self, recs: list[MatmulRecord]) -> dict[str, MatmulRecord]:
        """``Y_x`` and ``Λ`` by weight storage; returns weight name -> its forward record."""
        out: dict[str, MatmulRecord] = {}
        for rec in recs:
            w = _param(rec.b_info)
            if w not in self.role:
                raise LabelingError(f"forward mm #{rec.index} has no weight operand")
            if w in out:
                raise LabelingError(f"weight {w} used in two forward products "
                                    f"(#{out[w].index}, #{rec.index})")
            out[w] = rec
            self._check_bias(w, rec)
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
        # Under autocast A is the fp32 softmax rounded to bf16, which moves a row sum by at
        # most bf16's unit roundoff, half its eps.
        tol = max(1e-4, torch.finfo(a.dtype).eps)
        with torch.no_grad():
            ok = bool((a >= 0).all()) and float((a.float().sum(-1) - 1).abs().max()) < tol
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
        """``δX_x``/``δF`` by weight storage, then ``G_x`` and the head's by the saved inputs."""
        dy: dict[int, str] = {}  # storage of δY_x -> weight
        rest = []
        for rec in recs:
            w = _param(rec.b_info)
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
            if w in self.c.linear_weights:  # not the tied W_E, whose grad is a sum
                grad = self.model.get_parameter(w).grad
                if grad is None or (_ptr(grad) != _ptr(rec.out)
                                    and not self.cap.is_copy_of(grad, rec.index)):
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


class DecoderReplay(Replay):
    """Glue replay of one decoder step from committed leaves (checks 3, 5 and 6b).

    The replay does what training does: **one forward pass with grad on, then one backward
    pass**, both under :class:`~verification.computation.substitution.ProductSubstitution`, so
    every checked product is its committed leaf and only glue is computed.

    **Forward.** The first request runs ``logits`` on the committed batch, then
    ``loss_from_logits``, with grad on and under the computation's autocast (F5a). A forward
    pre-hook on each decoder layer (and on the final norm) hands the module a detached copy of
    its input with ``requires_grad`` on: that is ``X_ℓ`` (``X_{L+1}`` for the head), and it
    cuts the autograd graph at the layer boundary, so each backward unit (a layer, or the head)
    keeps a graph of its own. During the pass the replay keeps, per unit, every operand its
    products receive, the unit's output with its graph, the storages of the ``S`` and ``O``
    operands, and a hook on each linear product's output that records the storage of its
    output gradient ``δY_x``. Forward requests are then served from those kept operands in any
    order, with no rerun.

    The pass checks the call sequence by name, layer by layer (``Y_q, Y_k, Y_v, S, O, Y_o,
    Y_gate, Y_up, Y_down``, then ``Λ`` and nothing in the loss), and that each module receives
    exactly what the one before it returned: a layer's output is ``X_{ℓ+1}``, bit for bit.

    Under autocast a linear's ``B`` and bias are autocast's bf16 copies of the weight and
    bias, which the substitution reports as they are made; the replay holds each copy until
    its unit's backward has run, so a weight is still told by storage.

    Operands are what the product op receives in the model's own code: ``(X·, W_xᵀ)`` for a
    linear, and for an attention product the bmm operands at batch index ``s·n_h + h``, i.e.
    ``(Q̃, K̃ᵀ)`` after RoPE and ``repeat_kv`` for ``S``, ``(softmax(S/√d_h + Ω), Ṽ)`` for ``O``.

    A model run that departs from ``C`` (an undeclared product, a different call order, a
    module that doesn't receive what the one before it returned) raises :class:`ReplayError`.
    The leaves were validated by check 2, so that is a verifier-side bug and propagates as a
    crash, not a rejection.

    **Backward (A9).** The backward products come in units: the head (``δF`` and
    ``G_E^head`` or ``G_head``), then layers ``L … 1``. A unit's first request calls
    ``torch.autograd.grad`` on that unit's kept output: from ``ℒ`` for the head, with the held
    ``δX_{j+1}`` for layer ``j``, to ``X_j``, the unit's γ, biases and weights. Nothing is
    rerun. Each backward product it meets is
    replaced by its committed leaf, so RMSNorm, softmax, SiLU, RoPE, ``repeat_kv`` and the
    residual sums run in autograd on committed values. The products are identified at call
    time by operand identity, as in the labeling:

    - an mm whose ``B`` is a declared weight is ``δX_x`` (``δF`` for the output projection),
      and its ``A`` must be that weight's output gradient ``δY_x``, whose storage the hook
      recorded;
    - another mm is ``G_x`` (or the head's): its ``A`` is ``δY_xᵀ`` and its ``B`` the saved
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
    the backward. The residual-stream frontier ``δX_j`` and the γ and bias gradients persist;
    one unit's backward operands are held, and dropped after its last canonical product (the
    head's weight gradient, or ``L{ℓ}.G_v``). The head is the largest unit: ``δΛ`` and the
    softmax intermediates are ``N × n_v`` each.

    **Order.** Canonical order ``1..M`` runs the forward and each unit's backward once. A
    forward request whose unit has already run its backward, or a backward request for a unit
    that already ran, rebuilds from scratch: a fresh pass, then the backward units from the head
    down to the one requested. So any order gives the same operands.

    :meth:`glue_gradients` is valid only right after ``operands(M)``, with ``δX_1`` held. It
    returns each γ's and each bias's gradient from the units' backward, and ``W_E``'s full
    gradient: the embedding module's own backward of ``δX_1``, plus the committed ``G_E^head``
    when tied.
    """

    def __init__(self, computation: DecoderComputation, leaves: LeafReader) -> None:
        c = self.c = computation
        self.view = TranscriptView(c, leaves)
        self.model = c._replay_model(self)
        load_weights(c, self.model, {n: self.view.w_t(n) for n in c.weight_names})
        self._weight_of = param_storage_map(self.model)  # storage -> weight name
        # Autocast's copies of the weights in this pass: storage -> (weight, copy).
        self._copies: dict[int, tuple[str, torch.Tensor]] = {}
        self._role = {c.w(l, x): (l, x) for l in range(1, c.L + 1) for x in LINEARS}
        self._role[c.w_out] = (None, "head")
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
        # with their versions, the residual-stream frontier (j, δX_j), the γ and bias
        # gradients, and whether the last request was operands(M).
        self._bunit: int | None = None
        self._bops: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
        self._dx: tuple[int, torch.Tensor] | None = None
        self._glue: dict[str, torch.Tensor] = {}
        self._served_last = False

    # ---- forward: the one pass ------------------------------------------------------------------

    def _weight(self, t: torch.Tensor) -> str | None:
        """The weight ``t`` is a view of, or of autocast's copy of (F5a)."""
        p = _ptr(t)
        w = self._weight_of.get(p)
        return w if w is not None else self._copies.get(p, (None,))[0]

    def _copied(self, src: torch.Tensor, out: torch.Tensor) -> None:
        w = self._weight(src)
        if w is not None:
            self._copies[_ptr(out)] = (w, out)

    def _supply(self, op: str, a: torch.Tensor, b: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor:
        """A forward product's committed leaf, as a fresh tensor (substitution.py). A biased
        ``Y_x`` arrives as ``addmm`` and its leaf is the op's output, bias included (F14)."""
        c = self.c
        if op in ("mm", "addmm"):
            w = self._weight(b)
            if w not in self._role:
                raise ReplayError(f"replay: an mm whose B is not a declared weight ({w})")
            want = c.biases.get(w)
            got = None if bias is None else (self._weight(bias) or "an undeclared tensor")
            if got != want:
                raise ReplayError(f"replay: {w}'s product adds bias {got}, declared {want}")
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
        self._copies = {}

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
            with torch.enable_grad(), ProductSubstitution(self._supply, copied=self._copied):
                loss = c._forward_loss(model, batch)
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
            return {"Lambda": c.w_out}, {"dF", c.g_head}
        ys = {f"L{j}.Y_{x}": c.w(j, x) for x in LINEARS}
        want = {f"L{j}.{p}_{x}" for p in ("dX", "G") for x in LINEARS}
        return ys, want | {f"L{j}.{r}" for r in _ATTN_BACKWARD}

    def _linear_name(self, w: str, prefix: str) -> str:
        l, x = self._role[w]
        if l is None:
            return {"dX": "dF", "G": self.c.g_head}[prefix]
        return f"L{l}.{prefix}_{x}"

    def _supply_backward(self, op: str, a: torch.Tensor, b: torch.Tensor,
                         bias: torch.Tensor | None) -> torch.Tensor:
        """A backward product of the running unit, identified by operand identity as in the
        labeling (module docstring)."""
        c = self.c
        if bias is not None:
            raise ReplayError(f"replay: a backward {op} adds a bias")
        if op == "mm":
            w = self._weight(b)
            if w is not None:  # δX_x = δY_x·W_x reads the weight itself (or its copy)
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
        into ``δX_j``, its γ and its biases, under substitution; then drop the unit's forward
        state."""
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
        # The unit's glue-gradient weights first, then its linear weights.
        if head:
            glue = [c.gamma_final]
            names = [*glue, c.w_out]
        else:
            glue = [c.gamma_attn(j), c.gamma_mlp(j),
                    *(b for x in LINEARS if (b := c.bias(j, x)) is not None)]
            names = [*glue, *(c.w(j, x) for x in LINEARS)]
        params = [model.get_parameter(n) for n in names]
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
            # The graph's saved tensors are freed; drop the unit's operands, input and output,
            # and autocast's copies of its weights.
            self._units.pop(j, None)
            self._next_unit = j - 1
            self._copies = {k: v for k, v in self._copies.items() if v[0] not in names}
        if not self._bops:
            raise ReplayError(f"replay: no backward product of unit {j} reached the "
                              f"substitution; is the dispatch mode not active in backward?")
        if any(g is None for g in grads) or set(self._bops) != self._want:
            missing = sorted(self._want - set(self._bops))
            raise ReplayError(f"replay: backward unit {j} is missing {missing or 'a gradient'}")
        self._dx = (j, grads[0])
        for n, g in zip(glue, grads[1:]):
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
        if spec.name == (c.g_head if spec.layer is None else f"L{j}.G_v"):
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
        """Check 6b: each γ's and each bias's gradient, and ``W_E``'s full gradient,
        ``G_E^head + G_E^emb`` when tied and ``G_E^emb`` alone when untied.

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

        def no_product(op, a, b, bias):
            raise ReplayError("replay: a product in the embedding backward")

        # Under the prover's autocast (F5a), so the lookup runs as training ran it.
        with torch.enable_grad(), c.autocast(), ProductSubstitution(no_product):
            (g_emb,) = torch.autograd.grad(emb(self.batch.ids), [emb.weight],
                                           grad_outputs=self._dx[1])
        # F6b: the committed G_E^head widened exactly to W_E's format, plus the lookup's.
        g_e = (self.view.product(c.m_of("G_E_head")).to(c.weight_dtype) + g_emb if c.tied
               else g_emb)
        grads = {**self._glue, _E: g_e}
        out = {n: grads[n] for n in c.glue_gradient_weights}
        bad = [n for n, g in out.items() if g.dtype != c.weight_dtype]
        if bad:
            raise ReplayError(f"replay: glue gradients not {c.weight_dtype}: {bad[:3]}")
        return out
