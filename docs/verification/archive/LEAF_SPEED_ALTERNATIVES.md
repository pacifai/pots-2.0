# Same-Bits Alternatives to One Leaf per Attention Product

> **Frozen archive. Don't edit this file.** Speed items O15 and O16 of `../PERFORMANCE_TASKS.md`,
> archived on 2026-10-26 when the user took up O18 instead: commit each attention product's
> members as one leaf, a protocol change planned in `../ATTENTION_LEAF_PLAN_DRAFT.md`. Both
> items keep every root and every bit, and both remove per-leaf work that O18 removes by
> cutting the leaf count. They are kept as the alternatives to fall back on if O18 is dropped
> or reversed, and for any per-product work O18 leaves. The text below is the items as they
> stood when archived; the figures are FS42's (four H100s, 2026-10-25).

- **O15 — Per-leaf work in the commitment, on both sides.** *From FS42's profile (four H100s,
  2026-10-25); added with the user's yes, 2026-10-26; waits on O18.* On Llama-3.2-1B 393,216
  of the 393,576 products are attention members (128 sequences × 32 heads × 6 products × 16
  layers), each its own leaf (P9b). Pipelined, labeling takes 3.1 s and the commit 2.3 s on
  GPU 0, about 7 µs a leaf each, and check 2's hash producer 1.5 s on the verifier GPUs,
  measured; their floors are about 0 (launches), 0.1 s and 0.5 s (hashing). Qwen2.5-0.5B (259k
  leaves) and Falcon3-1B (111k) scale with the leaf count. Option, same roots: handle each
  attention product's members as one run, with one label table, one batched hash launch over
  its members and no Python per member, on the prover and in check 2. Meets O16 at the store's
  read call. Most of it goes away if the user takes O18.
- **O16 — The cross-process store doubles check 2's read and the store write when pipelined.**
  *From FS42's profile; added with the user's yes, 2026-10-26; waits on O18.* With the prover
  and the verifier in separate processes, check 2's leaf read takes 4.4 s on Llama-3.2-1B
  against 1.4 s sequential (the store rebuilds each leaf on its first read, about 10 µs a leaf,
  first seen in FS35's session), and the store write 1.0 s against 0.1 s, measured. Option, same
  bits: share each product's buffer between the processes once, and read and write per product
  rather than per leaf.
