# One Leaf per Attention Product — Draft Plan

> **🔓 EDIT STATUS: UNLOCKED — free for any session to edit.**
> Shared, multi-session file. Before editing, set this line to `🔒 LOCKED — <session/date>`
> and re-read the file. The instant you finish, set it back to `🔓 UNLOCKED`. Never end a
> turn with the file left LOCKED.

**Status: drafted 2026-10-26; approved by the user 2026-10-28** ("set a plan … and then you
will begin orchestrating its building by an agentic team of smaller AIs"), with the agent
breakdown of §5 and the interface pins of §2.6 added that day and open point 7.1 taken on its
recommendation (`0x03`, `ndim = 3`). The user took up speed item O18 on 2026-10-26, the
same day as a protocol change (`PERFORMANCE_TASKS.md`): commit all `(s, h)` members of one
attention product as one leaf of the step's Merkle tree, while every member stays its own
matmul `P_m` with its own label `m` and its own `k` challenges. The two same-bits
alternatives, batching the per-leaf work (O15) and the cross-process store (O16), are archived
in `archive/LEAF_SPEED_ALTERNATIVES.md`. **Built and merged 2026-10-28 (c3f8d37); measured on
four H100s (FS49, §4.3) and closed by the user at Milestone FM12 on 2026-10-29.**

What changes: the leaf sequence of spec §4.1 (a batched product's members share one leaf),
P9b's "one leaf per mathematical object" for those products (`DECISIONS_SETUP.md`), the
reference block's §6 wording, the leaf counts and every root, and the code that builds, hashes,
stores and reads leaves. What stays: the matmuls, their canonical order and labels (P8a), the
challenges' derivation, check 5's formulas and bands, `k`, the band files, the records' and
weights' leaves, and the Merkle rule (P9a).

## 1. Why, and what it costs

### 1.1 Measured (FS42, four H100s, 2026-10-25)

- Llama-3.2-1B's 393,996 leaves include 393,216 attention members: 128 sequences × 32 heads × 6
  attention products × 16 layers. Qwen2.5-0.5B (259,284 leaves) and Falcon3-1B (111,452) follow
  the same count.
- Step time follows the leaf count, not bytes or FLOPs: per-leaf host work is 85–90% of the
  step (`STATUS.md`, FS42's bottleneck list). On Llama, pipelined: labeling 3.1 s and the
  commit 2.3 s on GPU 0 (about 7 µs a leaf each), check 2 6.5 s (its read 4.4 s, the
  cross-process store rebuilding each leaf on first read), the store write 1.0 s; the step
  interval is 12.5 s against 0.92 s of plain training.

### 1.2 The detection floor stays (spec §8.2, `VERIFICATION_PARAMETER_SIZING.md` §7)

1. The floor is `f_achieved = c·τ·e·2^(N/k)`, with `N = λ + log₂T + log₂M + log₂G`; 0.95 at
   full scale, `k = 21`.
2. No term counts leaves. `M` counts matmuls (393,555 for Llama in the sizing), and each member
   keeps its own label and challenges, so `M` is unchanged.
3. `G = 2⁵²` is chosen. A grinding attempt re-hashes an altered leaf and its path, derives the
   target's fresh challenges and evaluates its residual (at least about 5·10⁴ operations for a
   rank-one deviation, sizing §12.5).
4. The path shortens from 19 to 10 node hashes. The cheapest leaf to alter is a small weight
   tensor within check 6's band (a norm's 2,048 entries) in both designs, so an attempt gets
   cheaper by under 1%: far under a tenth of a bit, against a chosen `log₂G`.
5. So `N`, `k = 21`, `f_achieved` and the bands stay; nothing is recalibrated. The test-scale
   band file stays valid too: its context keys (model, dtypes, `h_D`, `n_s`, `n`, `η`, `M`,
   `T`, `k`, devices) hold no root, and honest residual statistics don't depend on which
   challenges a root draws.

### 1.3 The speed floors

- The profile's hashing floor is bytes × `hash_kernel.per_byte_s` + leaves ×
  `hash_kernel.per_leaf_s` (`evaluation/results/profile/analyze.py`), with FS42's probe at
  about 1.2·10⁻¹¹ s a byte and 2.2 µs a leaf per GPU. Check 2's floor on Llama, 0.51 s per
  verifier GPU, is about 0.21 s of bytes and 0.29 s of leaves; with 876 leaves it falls to
  about 0.21 s (estimate). The commit's floor carries the same leaf term, and the outer tree
  (0.09–0.11 s measured) falls to about zero.
- Labeling's floor stays about zero; check 5's and training's don't change, since measuring
  still runs per member.
- The step's floor stays about 1 s (training). What goes is the measured per-leaf work above
  the floors: labeling, the commit and check 2 take 8.6 s of Llama's sequential step.
- Expected, Llama (estimate, 1.6): the pipelined interval from 12.5 s to about 6.3 s, around
  PoTS's training plus verify (FS42's comparison put our 12.5 s at 1.9 times it); sequential
  from 15.7 s to about 7.0 s. Check 5 then sets the pace, and its host side (O2) is next.

### 1.4 Security

- **Soundness: no change** (1.2). Check 5 still tests each member with its own challenges and
  names the failing member.
- **Binding: no change, on one condition.** The grouped leaf's encoding must stay injective
  with no free bytes (spec §4.1): member order and shapes follow from `C`, never from the
  prover. Padding between members, or a header field the prover could vary, would act as a
  nonce for grinding without touching any value. The encoding of 2.2 has none, and a test
  pins that the leaf's bytes are exactly its header plus the members' values.
- **Challenges: no change.** They still come from the root, which still covers every byte;
  labels stay member positions (P8a), injective as before. Only P9b's remark that the leaf and
  the label name the same object goes.
- **Openings: coarser, the one deficit.** Today one member can be shown against the root with
  its bytes and a 19-hash path. After the change the outer tree's smallest unit for these
  products is a whole attention product, 67–200 MB on Llama. BLAKE3 (F23) is itself a tree
  over 1 KiB chunks, so a member could still be opened with its bytes plus about 1 KB of proof
  (a verified slice), but no code does that, and nothing needs it today: the verifier reads
  every product, no code opens a product leaf with its path (check 4's record paths into
  `h_D` are the only path checks), and the sampling draft samples whole steps. A future
  member-sampling verifier or a third-party dispute would need that slice code.
- **Memory per read: no change.** The largest attention product (about 200 MB on Llama) is
  below the output layer's 525 MB leaf, so the largest leaf doesn't grow.
- **Implementation risk: new.** Prover and verifier need the same map from member `m` to its
  leaf and position. A disagreement could make check 5 measure bytes other than the committed
  ones; §4's tests corrupt single members and require rejection at that member.

### 1.5 Rejected alternatives

- *Same bits, per-leaf batching and the store fix (O15, O16, archived).* They keep the spec and
  the roots but keep about 394k leaves: the hash kernel's 2.2 µs a leaf (about 0.29 s per
  verifier GPU in check 2), the outer tree, and whatever host work per leaf batching leaves.
  Estimated 0.5 s a step slower pipelined and 0.8 s sequential on Llama than this change (1.6),
  with more new code: every layer must handle the leaves in bulk while still producing one
  digest, one store row and one disk file per member.
- *One leaf per sequence's members of a product* (12,288 product leaves on Llama). It keeps
  per-sequence openings, which nothing uses, and leaves about 15 times the per-leaf work of
  2.4's count.
- *One leaf per layer's six attention products.* 16 leaves instead of 96 on Llama, a negligible
  saving, with three shapes in one leaf and a header that must describe them.
- *Sharing challenges across a product's members.* It would cut challenge drawing, but it
  changes P8a's independence argument and the sizing's union bound; out of scope here.
- *Committing `δK̃` in the bmm's own layout (`δK̃ᵀ`).* It would save one transpose copy per
  layer, but the leaf would no longer be the product as `P = A·B` defines it, and check 5's
  operands and widths would change; 2.2 keeps the transpose, batched.

### 1.6 Shared and different parts, per option (Llama-3.2-1B, four H100s)

Measured is FS42's pipelined step (2026-10-25); every other column is an estimate. The
same-bits columns assume a well-built bulk path leaving about 0.5–1 µs of host work a leaf; at
2–3 µs a leaf they lose about 1 s more per side. The hash kernel's 2.2 µs a leaf is the
probe's measure and stays in them, unless kernel work cuts it (not measured).

| part (seconds per step) | measured | batching (O15) | store (O16) | O15 + O16 | one leaf per product |
|---|---|---|---|---|---|
| labeling (prover) | 3.1 | 0.3 | 3.1 | 0.3 | 0.2 |
| the commit (prover) | 2.3 | 0.3 | 2.3 | 0.3 | 0.15 |
| the store write (prover, pipelined only) | 1.0 | 1.0 | 0.1 | 0.1 | 0.05 |
| the hand-off's byte count (prover) | 0.6 | 0.05 | 0.6 | 0.05 | 0 |
| check 2's read (verifier) | 4.4 | 3.2 | 1.4 | 0.2 | 0.05 |
| check 2's hashing and the rest (verifier) | 2.1 | 0.6 | 2.1 | 0.6 | 0.3 |
| the root comparison (verifier) | 0.1 | 0.1 | 0.1 | 0.1 | 0 |
| shared: check 5 3.4, checks 4, 6a, 6b, 7 0.3 (verifier) | 3.7 | 3.7 | 3.7 | 3.7 | 3.7 |
| shared: training under capture 0.8, cache release and serialization 0.13 (prover) | 0.93 | 0.93 | 0.93 | 0.93 | 0.93 |
| shared: between steps (O17) and the release wait | 2.2 | 2.2 | 2.2 | 2.2 | 2.2 |
| **pipelined interval** | **12.5** | **9.8** | **9.5** | **6.8** | **6.3** |
| **sequential interval** | **15.7** | **7.8** | **15.7** | **7.8** | **7.0** |

The pipelined interval is the verifier's sections plus the shared 2.2 s, since the prover
(1.3–2.6 s with batching or one leaf per product, 7.0–7.9 s otherwise) runs beside it
and finishes first. The sequential interval is both
sides' sections plus training and 1.7 s between steps; O16 does nothing there, since one
process uses no cross-process store. The shared rows, about 6 s pipelined, are what remains
under every option; check 5's host side (O2) and the between-step costs (O17) are the next
levers.

## 2. The protocol change

### 2.1 The leaf sequence

The canonical product order (ref block §6, `decoder._inventory`) and every `m` stay. The leaf
sequence follows it: walking `m = 1 … M`, a product that is not a member is a leaf at its own
position, and the members of one batched product (one attention role of one layer: `S`, `O`,
`δA`, `δV`, `δQ̃` or `δK̃`) form one leaf at the position of the role's first member; later
members add no leaf. Per layer that gives, forward: `Y_q, Y_k, Y_v, [S], [O], Y_o, …`, and
backward: `…, G_o, [δA], [δV], [δQ̃], [δK̃], δX_q, …`. In product order the backward pairs
alternate per member (`δA_{s,h}, δV_{s,h}`, stride 2 in `m`); each role's members still sit in
their own leaf, consecutively, at member index `i = s·n_h + h`.

### 2.2 The leaf encoding

- A grouped leaf is a tensor leaf under the product tag `0x03`, with `ndim = 3`:
  `0x03 ‖ dtype ‖ 3 ‖ n_s·n_h ‖ a ‖ b ‖ raw bytes`, the members in C order, member `i` at
  `[i]`. It is the role's bmm output itself (`[n_s·n_h, a, b]`, contiguous), so the GPU hashes
  it in place with no copy, and the header differs from every 2-D product leaf's by `ndim`.
  Its slot, shape and dtype come from `C`, which every hash call already checks per leaf.
- `δK̃` stays the product as `P = A·B` defines it. The captured bmm returns `Q̃ᵀ·δS = δK̃ᵀ` per
  member; labeling already makes one contiguous batched transpose of the whole output
  (`_fill_members`: `rec.out.transpose(1, 2).contiguous()`, on the output's device), and that
  tensor becomes the `δK̃` leaf itself. Every member's values are those of today's leaf.
- No new tag (7.1, decided 2026-10-28 on its recommendation). A new tag (`0x05`, "product
  run") would read more clearly in a hex dump but adds nothing to injectivity, since the slot
  fixes the object and `ndim` already separates the forms; it would also touch
  `test_tags.py`'s prefix-free set.
- A grouped leaf is 3-D whatever its member count: a declared batched product with one member
  (`n_s = n_h = 1`, tests only) is `[1, a, b]`, never a 2-D leaf. The form follows from `C`'s
  declaration, never from a count.

### 2.3 Labels, challenges and the member map

- `r(m, j) = Expand(PRF(h, ⟨m, j⟩); c_m)` and `⟨m, j⟩ = 0x10 ‖ m(4) ‖ j(1)` are unchanged.
- `C` gains one map, shared by prover and verifier: `leaf_of(m) -> (leaf index, member index
  or None)`, its inverse per leaf, and the product leaves' list. The decoder builds it from
  `member_runs` (`(ℓ, role) -> (m₀, stride)`): member `i` of a run is product `m₀ + stride·i`
  and sits at `[i]` of the run's leaf. The MLP has no members, so its map is the identity.
- `n_leaves = n_s + 2·n_w + K`, with `K` the number of product leaves:
  `K = M − Σ_runs (n_members − 1)`. For the decoder, `K = L·(21 + 6) + 3B = 27L + 3B`.
  Invariant 7 holds as before: the verifier takes `n_leaves` from `C`, never from the
  transcript.

### 2.4 Counts

| model | `M` | leaves before | product leaves `K` | leaves after | tree depth |
|---|---|---|---|---|---|
| SmolLM2-135M, test scale (`n_s = 4`, `B = 1`) | 7,113 | 7,661 | 813 | 1,361 | 13 → 11 |
| Llama-3.2-1B (`n_s = 128`, `B = 8`) | 393,576 | 393,996 | 456 | 876 | 19 → 10 |
| Qwen2.5-0.5B (24 layers, 14 heads) | 258,576 | 259,284 | 672 | 1,380 | 18 → 11 |
| Falcon3-1B (18 layers, 8 heads) | 110,994 | 111,452 | 510 | 968 | 17 → 10 |

The full-scale rows are computed from `M = L·(21 + 6·n_s·n_h) + 3B` and FS42's leaf counts,
which they reproduce; FS45's tests compute every row from `C`.

### 2.5 Text of the amendments (applied once the plan is approved, task FS44)

- **Spec §4.1**, after "in a fixed canonical order.": "The members of one batched product
  (Section 2) share a leaf: it holds their values in member order under the canonical
  serialization, at the position of the first member. Every other product is a leaf of its
  own. A member remains its own matmul, with its own position `m` and its own challenges
  (Section 5)."
- **Reference block §6.** The sentence "The order is a fixed labeling of the leaves" becomes
  "The order is a fixed labeling of the products, and the leaves follow it: each attention
  product's `n_s·n_h` members form one leaf, an `n_s·n_h × n × n` or `n_s·n_h × n × d_h`
  array in lexicographic order of `(s, h)`, at the position of its first member." Items 3–4
  keep their order. The label sentence gains: "A member's label is its own position, so the
  members of one leaf keep distinct labels." §5 gains the leaf count,
  `n_s + 2n_w + L·27 + 3` (1,361 at test scale).
- **`DECISIONS_SETUP.md` P9b.** A sub-bullet "**Amended (user, 2026-10-26; O18): an attention
  product's members share one leaf**", with 1.1–1.5's reasoning and alternatives; the
  "Rejected: one leaf per batched operation" bullet is marked "reversed 2026-10-26, below", its
  premise "nothing is saved" refuted by FS42. P8a gains one line: the label is a member's
  position, and since O18 a leaf may hold several labels.
- **`verification/CLAUDE.md` and `verification/README.md`** (the coders, with the code):
  the constants line (7,661 → 1,361 leaves), the leaf-order block and the "own leaf" sentence,
  invariant 7's formula, the `product_index` entry, the capture's arrival-hashing counts
  (`used = M − dK`), the Merkle and store figures (7,661 leaves, 7,663 files), the replay's
  "member leaves stacked at `s·n_h + h`", check 5's member serving.
- **`STATUS.md`, `PERFORMANCE_TASKS.md` O18 and `IMPLEMENTATION_PLAN.md`**: the tasks of §5
  and their outcome.

### 2.6 Interface pins (2026-10-28)

The names and contracts every task codes against. A coder who finds that one of them can't
hold stops and reports; none is changed in a brief.

**The declared computation (`interface.py`).**
- `ProductLeaf`, a frozen dataclass: `g` (0-based position among the product leaves), `m0`,
  `stride`, `n` (members; 1 for a lone product), `grouped` (bool), `shape` (`p_shape` for a
  lone product, `(n, *p_shape)` for a group), `name` (the spec's name for a lone product; the
  first member's name without its `[s,h]` suffix for a group, as `L3.S`), and the property
  `ms`, `range(m0, m0 + stride·n, stride)`.
- `DeclaredComputation.product_groups -> tuple[tuple[int, int, int], ...]`: `(m0, stride, n)`
  of each batched product, declared by the instance; the default is `()`.
- Derived and cached: `product_leaves` (in leaf order: walking `m = 1 … M`, a product in no
  group is a leaf, a group is one leaf at its `m0`, its later members add none);
  `n_product_leaves` (`K`); `product_leaf(g)` (`IndexError` outside `0 … K−1`); `leaf_of(m) ->
  (g, i)`, with `i` the member index or `None` for a lone product (`IndexError` outside
  `1 … M`).
- `n_leaves = n_s + 2·n_w + n_product_leaves`; `product_leaf_index(g) = n_s + n_w + g`;
  `w_next_index(name) = n_s + n_w + n_product_leaves + position`.
- `product_index(m)` is removed, so every caller is revisited: it names a leaf
  (`product_leaf_index(leaf_of(m)[0])`) or reads a product through `TranscriptView.product(m)`.
- `validate()` adds, per group: `n ≥ 1`, `stride ≥ 1`, every member in `1 … M`, no product in
  two groups, members sharing `p_shape`, `kind` and `layer`, each with a `member`; and every
  product with a `member` is in a group.
- `label(capture, model)` returns the product leaves in leaf order, `n_product_leaves` of them:
  a lone product's leaf is its `P`; a group's is `[n, *p_shape]`, contiguous, member `i`
  (product `m0 + stride·i`) at `[i]`. It raises `LabelingError` unless every leaf is filled
  exactly once with its declared shape and dtype.

**The instances.** The decoder's `product_groups` is one group per layer and attention role
(`S`, `O`, `dA`, `dV`, `dQ`, `dK`), from `member_runs`, sorted by `m0`; construction raises if
a member product is in no run (every decoder run is evenly spaced). The labeler keeps one slot
per product leaf: `_fill_run` stores the whole bmm output (for `δK̃`, the batched transpose it
already makes) as the group's leaf, and the per-member fallback keeps every check and error of
`_fill` and then stacks the members into the leaf. The MLP declares no groups, so its
`label` is unchanged (`K = M`).

**Placement.** `product_leaf_device(g)` is `product_device(m0)` (a group's members share a
layer, so a unit and a device); `leaf_device(i)` follows the new layout; `loads()` is unchanged
(the same bytes). `product_device(m)` stays: the capture's hints use it.

**The commitment.** `_tensor_slot(c, i)` of a product leaf gives `TAG_PRODUCT`, the leaf's
`shape`, `product_dtype`, and the slot name `P_{m0} (name)` for a lone product,
`P_{m0}..P_{m_last} (name)` for a group. `leaf_totals` follows it. The encoding doesn't change.

**The reads (`TranscriptView`).**
- `product_leaf(g)` reads leaf `product_leaf_index(g)` and checks a tensor's shape against
  the declared one, raising `LeafShapeError` as `leaf_parts` words it.
- `product(m)` is the leaf of a lone product, or `leaf[i]`, a view, for a member.
- `product_run(m0, stride, n, own=False)`: when the `n` products are members of one group at
  member indices `i0 + step·t`, the slice `leaf[i0 : i0 + step·(n − 1) + 1 : step]`: a view
  when `step = 1` and the slice starts 64-byte aligned (`own` returns a clone), else one aligned
  contiguous copy, so a BLAS kernel sees what `torch.stack` gives it (the old `stack_block`'s
  rule; FS47x, 2026-10-28); otherwise `product_block`.
- `product_block(ms, own=False)`: equal bit for bit to `torch.stack([product(m) for m in ms])`;
  a progression inside one group is the slice. Any `m` outside `1 … M` raises `IndexError`
  before any read, as today.
- `stack_block` (reader), `CommittedLeaves.tensor_block`, `tensor_run` and `_prove_run` go,
  with their tests: no caller is left (checked 2026-10-28), since a member is a slice of one
  leaf, whose `_version` `CommittedLeaves.leaf` checks on every read.

**The prover (`prove_step`).** `StepOutput.products` holds the `K` product leaves in leaf order,
each moved to `product_leaf_device(g)`, with one version each.
`cap.count_hint_misses(leaves, placement.product_leaf_device)` iterates the product leaves with
`g` from 0. `arrival_digests` is keyed by `product_leaf_index(g)`. `missing` and
`missing_first` count leaves without a digest by their first product's kind, and name them by
the leaf's name. `perturb` stays keyed by `m` (`ProverFault.perturb`): a lone product as today;
a member is written as `fn(member clone)` into a clone of its group's leaf (one clone per group
per step, shared by several perturbed members of it), its shape and dtype checked as today, and
the leaf's arrival digest is dropped.

**The capture.** `_hash_arrival` hashes a whole output as one table entry under
`tensor_leaf_header(TAG_PRODUCT, out)`, 3-D for a bmm; `_arrived_from` keeps one digest per
record; `arrival_digest(p)` returns it only when `p` is exactly that whole output (storage,
offset, shape, dtype, contiguity, version), so a member view gets `None`;
`summary()["arrival_hash"]["leaves"]` counts outputs hashed.

**The runs.** On CUDA the commit hashes the records, both weight sets and the `L` `δK̃` leaves
itself; every other product leaf comes from its arrival digest. So `used` and `commit known`
are `K − L`, and `hashed` is `K` (the raw `δK̃ᵀ` outputs among them, unused). `bring_up`'s
device-leaves line becomes `arrival digests U, hashed H, commit known N, product leaves K, dK
leaves D`, and `fs11_session`'s gate requires `U = N = K − D` and `H = K`.

## 3. Code changes by module

From two read-only surveys of the code (2026-10-26). Line numbers are as of 387b991.

### 3.1 The declared computation (`verification/computation/`)

- `interface.py`: `product_index(m) = n_s + n_w + m − 1` (:395–401) goes and `n_leaves = n_s +
  2n_w + M` becomes the map of 2.3; `w_next_index` shifts by `K` instead of `M`. New:
  `ProductLeaf`, `product_groups`, `product_leaves`, `product_leaf(g)`, `leaf_of(m)`,
  `n_product_leaves`, `product_leaf_index(g)` (2.6). `label` returns `K` leaf tensors (2.2),
  not `M`.
- `instances/decoder.py`: the groups from `member_runs` (:557–575); `_fill_run` and
  `_fill_members` (:1137–1162) fill a run's leaf as the bmm output itself, and `δK̃` as one
  batched transpose (:1275); the replay's `supply` serves an `S`/`O` bmm the committed group
  leaf directly (today it stacks the member leaves at `s·n_h + h`), and `δK̃` as its transpose.
  `operand_group`/`operand_blocks` are unchanged (they serve operands, not leaves).
- `instances/mlp.py`: identity map.
- `leaf_sizes.py` and `placement.py` (`leaf_device(i)`, `loads()`): per product leaf.

### 3.2 The commitment (`verification/commitment/`)

- `leaves.py` `_tensor_slot` (:171–185), `leaf_parts`/`leaf_hash` (:187–208), `leaf_hashes`
  (:699), `leaf_totals` (:878–887): a product slot is a product leaf, its shape the group's.
- `encoding.py`: no change (3-D tensor leaves already encode); a test for the 3-D header.
- `kernel_check.py`: new shapes join the per-shape cross-check key; no code change expected.
- `gpu_blake3.py`: leaves up to about 200 MB, under the 622 MB already hashed; no change.

### 3.3 The prover (`verification/prover/`)

- `step.py`: `StepOutput.products` per product leaf (:242 builds one entry per `m` today),
  each moved to `placement.product_device`; `leaves()` in the new order; `arrival_digests`
  keyed by product leaf; `perturb` on a member writes the member into a copy of its leaf
  (the prover's fault hook, invariant 5), whose digest is then dropped.
- `capture.py`: `_hash_arrival` (:684–701) hashes a bmm output as one table entry with the
  3-D header, in place of one entry per member; `arrival_digest` (:703–737) matches a whole
  output; the `δK̃` output's arrival digest stays unused (its leaf is the transpose). The
  summary's counts (`leaves`, `used`) change.

### 3.4 The transcript and the verifier

- `transcript/reader.py`: `product(m)` returns `leaf(g)[i]` for a member, `leaf(g)` otherwise;
  `product_run(m₀, stride, n)` returns the slice `leaf(g)[i₀ : i₀ + n]` (members of one role
  are consecutive in their leaf even where their `m`s have stride 2); `product_block(ms)` the
  matching index. `stack_block` is no longer needed for members.
- `transcript/store.py`, `shared.py`: no logic change; one file and one table row per product
  leaf. The cross-process store's per-leaf rebuild (`set_` on first read) now runs about 900
  times a step instead of about 394k.
- `verifier/context.py` `CommittedLeaves` (:481–606): `tensor_block`, `tensor_run` and
  `_prove_run` proved member leaves to be equal-shape views at consecutive offsets of one
  storage; a member is now a slice of one validated leaf, so the proof reduces to the leaf's
  own checks (shape, dtype, `_version`), and the three methods are removed (2.6).
- The decoder replay reads members through `TranscriptView.product_run` and `product_block`
  (`_member_block`) and products through `product(m)`; with 2.6's reads it needs no change.
  Check 5's serving (`_served_group`, `_served_blocks`, `_served_view`) reads the same three
  and needs none either.
- `verifier/checks.py`: check 2 (:208–257) iterates `range(c.n_leaves)` and needs nothing
  else; check 5's serving (`_served_view`, `_served_blocks`, :1124–1192) goes through the
  reader's views, keyed by `m`; checks 4, 6, 7, 0 and 8 read records and weights, whose
  indices before the products don't move (`w_next_index` moves with `C`).
- The challenge code (`challenges.py`, `challenge_source.py`, `freivalds.py`,
  `measure_kernel.py`) is keyed by `m` and doesn't change.

### 3.5 Runs and tools

- `runs/loop.py`, `runs/pipeline.py`: the handshake compares `n_leaves`, `M` and the products'
  digest, which follow `C`.
- `runs/llama_step.py` `work_rows` (:202–251): the commit's leaf rule (`leaves − known`) and
  `hash_leaves` per row; `runs/fs11_session.py`'s device-leaves gate (`known = M − dK`,
  `hashed = M`) becomes `known = K − L` (all but the `δK̃` leaves) and hashed on arrival `K`
  outputs (the raw `δK̃ᵀ` outputs among them, unused).
- `runs/flipped_matmul_sweep.py` (:369–797): planting a member writes it into a copy of its
  leaf and re-roots with `perturb_leaf`, which now re-hashes the whole leaf (test scale: about
  2.4 MB, about 0.3 ms a trial; about 7 s more per member target over its 23,200 trials).
- `runs/card_identity.py`: digests per leaf; the first differing leaf named by its role and
  layer (`L3.S`).
- `runs/store_crosscheck.py`, `run_verified.py` (kept-step file counts), `kernel_speed.py` and
  `benchmarks/hashing` (through `leaf_sizes`): counts only.
- `evaluation/results/profile/analyze.py` (the profiling session's tool): no change.
  `us_per_leaf` keeps its definition and its hash floor counts bytes and leaves as recorded;
  FS49 compares sections in seconds, not per leaf, and its report says that the leaf count
  changed.

### 3.6 Tests that pin the old layout

`test_leaf_sizes.py`, `test_decoder.py` (`(7113, 272, 7661)` and the product counts),
`test_store.py`, `test_shared.py`, `test_shared_tester.py`, `test_reader.py`,
`test_placement.py`, `test_interface.py`, `test_mlp.py`, `test_leaves.py`,
`test_leaves_known.py`, `test_leaves_known_tester.py`, `test_leaves_native.py`,
`test_hostcopy.py`, `test_hostcopy_stress.py`, `test_fs34_tester.py`, `test_fs34_routing.py`,
`test_step.py`, `test_llama_step.py`, `test_fs31_tester.py`, `test_metrics.py`,
`test_fs11_session.py`, `test_analyze.py`, `test_trace.py`, and the layout-dependent CUDA and
sharded tests (`test_decoder_split_replay.py`, `test_check_devices.py`, `test_sharded_run.py`,
`test_full_scale_models.py`). The step roots cited in prose (`daa53d7c…`, step 4 of the honest
run) change; the final weights' root (`bb9aec2a…`), `h_D` and the band-file hashes don't.

## 4. Tests and acceptance

### 4.1 New tests (the security conditions of 1.4)

1. **No free bytes.** For every product leaf of a SmolLM2 step and of the tiny decoders, the
   encoded length is exactly the header plus `n_members·a·b·itemsize`, and the header is the
   one `C` gives.
2. **The map.** `leaf_of(m)` and its inverse round-trip for every `m`; every `m` is in exactly
   one leaf; leaves are in the order of 2.1; `n_leaves` from `C` matches a real `prove_step`.
3. **A corrupted member byte** in a committed group leaf, after the commit, rejects at check 2
   (root mismatch), on the in-memory, disk and shared stores.
4. **A wrong member committed consistently** (the prover's `perturb` on one member, re-rooted)
   rejects at check 5 naming that member's `m`, for a member in the middle of a backward pair
   (`δV`, stride 2) and for a `δK̃` member.
5. **Invariant 7.** `verify_path` with `n_leaves` from `C` on the new tree; a transcript with
   one leaf more or less fails.

### 4.2 Test-scale acceptance (the Mac, CPU)

- The fast suite, and the slow suite (label fill, the replay bit for bit, the full-scale
  models' replay test where the weights are cached).
- `run_verified` (10 steps, the existing band file): accepted, final weights equal to the
  plain baseline's bit for bit, step 4 kept (now 1,363 files).
- The cheats (A13) exact under the same band file: A1 at (1, 4), A2 at (1, 5) on `P_1`, A3 at
  (1, 6a), the hidden step at (2, 7), the `δF` flip at (4, 5) on `P_2372`.
- `store_crosscheck`: memory and disk give identical decisions.
- One short sweep on a member target (`L15.S[0,0]`, `--x 2 --trials 1`), whose self-check
  requires check 5's numbers bit for bit.
- `k_tunability` on the new stored step: the configured `k` reproduces its archive.

### 4.3 GPU acceptance

- **One A40** (about $0.6 an hour, under an hour, about $1, estimate): the CUDA test files
  (capture's arrival hashing of whole outputs, device leaves, the kernel's leaf table, the
  shared store over CUDA IPC) and a one-step `bring_up --no-tamper` on a tiny decoder with
  `VERIF_VERIFIER_DEVICES=cuda:0`. Under the $20 cap, so no confirmation is needed; the price,
  estimate and cap are stated when it starts. *Done 2026-10-28* (code at 5d1dfab; about 18 min,
  about $0.2): 2,788 CUDA tests and the step table's 164 passed, none failed; the bring-up ran
  through a scratch script (`bring_up(..., tamper=False)`, two steps) on four arms (tied fp32 on
  the host and on `cuda:0`, untied bf16 Qwen2 on `cuda:0`, untied bf16 Llama on the host), each
  step accepted, the prover's root equal to the CPU path's, 60 leaves hashed on arrival and 58
  used (`K − D`).
- **Four H100s, the measurement (FS49).** The session FS42 ran (`four-h100-profile`): the
  multi-GPU tests, the probe, and per model the plain run, the card and both loops, all three
  models accepted on every step with equal roots across loops. Added: Llama-3.2-1B's two
  loops on the code before this change (387b991), on the same pod, so the comparison is A/B
  on one host (same-type pods differ in host-side costs, and this change is host-side). The
  profile tools then give the sections against FS42's and against the A/B arms. At FS42's
  $15.96 an hour: the parts about 76 minutes (the A/B arms add 10), with setup and fetch about
  91 minutes, about $24 (estimate); the job deadline (11,700 s) caps it near $52 and the boot
  guard (14,100 s) near $63, over the $20 cap, so this pod waits for the user's go, with the
  upload approved before the pod is created. *User, 2026-10-28:* go given for this run, then,
  on the host-side session's proposal, "wait and combine": FS49 runs on one pod together with
  that session's added arms (`HOST_SIDE_PLAN_DRAFT.md`, FS54), after the host-side changes and
  FS54 are merged on top of this change, so every comparison is on one host. The other
  session's figures for the combined run: parts about 96 minutes, about $30; job deadline
  14,100 s (about $63), boot guard 16,500 s (about $73). The exact price is quoted to the user
  again before the pod is rented. *Done 2026-10-29* (code at b18a3f0, the host-side changes
  merged on this change; the pod lived about 75 minutes, about $20 by time × price, and is
  deleted). Steady steps 2–3, measured; profiles in
  `trainer_output/verification/fs49_four_h100_profile_2026-10-29/profiles/`.
  - *Acceptance.* Every arm accepted on every step. The arms on the code before the change
    have other roots, as they must; this change alone, this change with the host-side fixes,
    and the arm with the cache release forced on give identical roots, step records and
    residual archives. The largest GPU peak is 55 GiB, on the prover's GPU; the verifier GPUs
    stay at or under 42 GiB.
  - *Step interval, Llama-3.2-1B, same pod.* Before the change 14 s sequential and 11 s
    pipelined; this change alone 4.2 s and 3.8 s; with the host-side fixes 2.8 s and 2.5 s.
    Qwen2.5-0.5B and Falcon3-1B ran with both changes only: 2.5 s and 2.0 s, and 2.6 s and
    2.2 s, against FS42's 12 s and 9.5 s, and 6.5 s and 5.2 s (another pod and an older
    commit, so a rough "before").
  - *What the change removed (Llama-3.2-1B).* Labeling 2.8 s → 0.03 s; the commit's hashing
    2.3 s → about 0, its digests now coming from the capture's arrival hashing; check 2 3.5 s
    sequential and 6.6 s pipelined → 0.07 s; the store write and the hand-off's totals 1.0 s
    and 0.6 s → 0. Every row 1.3 targeted came in at or under its estimate. The step beat the
    6.3 s estimate because two rows 1.3 held fixed were smaller on this pod: check 5 took
    2.5 s on the code before the change, not FS42's 3.4 s, and the time between steps fell
    from about 1.4 s to 1.0 s.
  - *The training yardstick.* Forward, backward and update take 0.27 s, 0.41 s and 0.44 s
    (Qwen2.5-0.5B, Llama-3.2-1B, Falcon3-1B). The plain run's step also copies every fp32
    weight from host memory to GPU 0 (0.19 s, 0.47 s, 0.61 s), which the verified run doesn't,
    so its step time overstates training. Against training itself, the pipelined step interval
    is 7.4×, 6.1× and 5.0×, the verifier's time 6.1×, 5.0× and 4.4×, and the accumulated GPU
    time per pipelined step 5.7 s, 6.9 s and 6.5 s.
  - *Remaining bottlenecks (Llama-3.2-1B, pipelined, 2.5 s; the verifier sets the pace).*
    Check 5 1.6 s against a floor of 0.04–0.1 s: in the traced step one verifier GPU at a time
    is busy for 1.4 s of its 1.7 s, each GPU's own work being about 0.5 s, about 0.43 s of it
    the measuring kernel at about 10% occupancy (O2). The time between steps 0.43 s (O17).
    Check 6a 0.28 s, host-bound: about 26,000 kernel launches and 1,900 reads to the host a
    step (O1). Training under capture 0.48–0.67 s against 0.27–0.44 s plain, off the critical
    path when pipelined. Step 1 builds the verifier's model inside check 5 (O14), 6.6 s once
    per run.

## 5. Tasks

**Branches (2026-10-28).** The feature is built on `o18-leaves`, made from `verification-impl`
at 387b991. Each coder and tester works in a worktree made by hand from the tip of `o18-leaves`
at its launch (`.claude/worktrees/<task>`, branch `<task>`), and I merge into `o18-leaves`.
`verification-impl` takes the feature in one merge, after FS48 and the validator, so it stays
green while the switch is half built: between FS45a and FS45b the leaf count and the prover
disagree and every end-to-end test fails, as planned.

| task | role | what | depends on | done when |
|---|---|---|---|---|
| FS44 | orchestrator | The amendments of 2.5 to the spec, the reference block, `DECISIONS_SETUP.md` P9b and P8a; `STATUS.md`, `PERFORMANCE_TASKS.md` O18, `IMPLEMENTATION_PLAN.md` | the user's approval | the texts are in, the files unlocked |
| FS45a | coder | 2.6's declared computation (`interface.py`, the decoder's and the MLP's groups), placement, the commitment's slots and `leaf_totals`, `leaf_sizes`, the reads (`TranscriptView`, `CommittedLeaves`), with their own tests and the layout text in `verification/CLAUDE.md` and `README.md` | the user's approval; FS44 runs beside it | the brief's unit tests pass; `leaf_sizes` gives 1,361 leaves for SmolLM2 and 876 for Llama-3.2-1B at `n_s = 128`, `B = 8` |
| FS45t | tester | 4.1's five tests, written from 2.6 and 4.1 alone, in new files | the user's approval; FS44 runs beside it | written; tests 1, 2 and 5 pass on FS45a, 3 and 4 on FS45b's merge |
| FS46 | coder | The capture's arrival hashing of whole outputs (2.6, the capture), its CPU tests with the existing fakes and the CUDA tests' expectations (run on the A40 in FS48) | the user's approval; FS44 runs beside it | the capture's tests pass on the Mac |
| FS45b | coder | Labeling into product leaves, `prove_step` (2.6, the prover), `count_hint_misses` per leaf, and the end-to-end tests of `computation/`, `prover/` and `verifier/` that the switch breaks | FS45a merged | the brief's test files pass; an honest SmolLM2 step (slow test) commits 1,361 leaves and is accepted |
| FS47 | coder | The runs and tools of 3.5 (sweep, work rows, the device-leaves line and gate, card names, kept-step counts), their tests and their `verification/CLAUDE.md` entries | FS45b merged | the brief's test files pass |
| FS47x | coders, one per share | The fast suite's remaining failures after FS45b and FS47, split by test directory, each a test that pins the old layout (3.6) | FS45b, FS47 merged | the fast suite passes on `o18-leaves` |
| — | reviewer | One per coder task, on its branch: the diff against its brief and this plan | that task | findings fixed, at most three rounds |
| FS48 | orchestrator | Test-scale acceptance (4.2) and the A40 run (4.3) | all of the above merged | every item of 4.2 and 4.3 holds |
| — | validator | A fresh Opus agent at xhigh effort: this plan with the decisions it cites against the merged code, read-only | FS48 | no mismatch, at most three validations |
| FS49 | orchestrator | The four-H100 measurement (4.3), with the user's go on the pod | the validator | all three models accepted in both loops, roots equal across loops; the sections and the step interval against FS42 and the A/B arms |

Waves: FS45a, FS45t and FS46 start together (FS46 touches only the capture, which doesn't import
`C`); FS45b starts when FS45a is merged, and FS45a's reviewer runs beside it. FS47 starts
beside FS45a (user, 2026-10-28: "you may parallelize in building"), coded against 2.6's pins,
and runs its tests once FS45b is merged into its branch; the FS45b and FS46 reviewers run next; then the FS47x shares, split by what the fast suite still
fails. Coders, testers and reviewers run on Sonnet at medium effort, at most ten at once, each
with a brief in the session scratchpad that names its files, steps, tests and done criteria
(`IMPLEMENTATION_PLAN.md`, "Working rules for the agent team"). Agents read files under
`verification/` with `grep` and `sed`, never the Read tool, and edit them with the exact-replace
helper, since opening a file there loads `verification/CLAUDE.md`. In `verification/CLAUDE.md`
and `README.md`, text that states the current layout changes; dated measurement paragraphs keep
their figures. I check each branch against its brief and this plan before merging. **Milestone
FM12** follows FS49: the step interval before and after, per model and loop, and the remaining
bottleneck list, for the user's approval.

## 6. Coordination with other sessions

- **The profiling session (FS42's).** Its tools (`evaluation/results/profile/`, the
  `four-h100-profile` plan in `fs11_session.py`, `llama_step.work_rows`) are touched in 3.5;
  FS47 changes only the counts and the work-row rule, and FS49 reuses its session plan with one
  added A/B part. Its next work, between-step and hand-off costs (O17), touches `runs/loop.py`,
  `pipeline.py` and `metrics.py`; FS47 touches `loop.py` and `pipeline.py` only through `C`'s
  counts, so either can merge first and the other rebases.
- **The host-side session (`HOST_SIDE_PLAN_DRAFT.md`).** Merge order (2026-10-28): this change
  into `verification-impl`; the host-side changes merged onto it; its FS54 arms rebased onto
  that; then FS49's pod on that tip, prepared with `--base-commit 387b991` and `--o18-commit`
  this change's merge commit, so the session runs the code before this change, this change
  alone, and both, on one pod.
- **Check 5's host side (O2)** stays in this session and is planned from FS49's trace, since
  this change moves check 5's serving onto slices of one leaf.

## 7. Open points

1. **The tag** — decided 2026-10-28 on the recommendation: `0x03` with `ndim = 3` (2.2). A
   new tag `0x05` would be equally injective and only read more clearly in a hex dump.
2. **The planted-error sweep's cost.** About 7 s more per member target at test scale (3.5);
   at full scale the re-hash per trial is a whole product (67–200 MB, about 1–3 ms on the GPU),
   to be priced when the full-scale sweep (FS13) is planned. It joins O7's item.
3. **A verified slice opening** (1.4) is not built; it would be a separate item if a
   member-sampling verifier or third-party disputes are ever taken up.
