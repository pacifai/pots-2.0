/* BLAKE3 for the GPU hash kernel (decision F23), as code that both NVRTC and a host C compiler
 * build. gpu_blake3.cu wraps the phase functions below in CUDA kernels, and the host test
 * harness (tests/verification/commitment/gpu_blake3_host.c) runs the same functions in
 * loops, so the index math, flags, padding and tree schedule are tested without a GPU.
 *
 * Leaf i is BLAKE3(0x00 || header || payload), unkeyed, 32 bytes. Its message has
 * L = 1 + hdr_len + payload_len bytes and C = ceil(L / 1024) chunks (L >= 1, so C >= 1).
 *
 * The tree is reduced in stages:
 *   stage 0   a tile is a group of G = 2^POTS_G_LOG chunks of one leaf, aligned in the leaf.
 *             One thread hashes one chunk, then the group's chaining values reduce in shared
 *             memory. A leaf with one group takes ROOT here and writes its digest.
 *   stage s   a tile is R = 2^POTS_R_LOG aligned nodes of one leaf's level s-1, reduced in
 *             shared memory to one node of level s. A leaf with one tile takes ROOT here.
 * Level 0 has n0 = ceil(C / G) nodes, level s has ceil(n_{s-1} / R). An aligned group of a
 * leaf's bottom-up pairing, with the odd last node promoted, is a subtree of BLAKE3's
 * left-balanced tree, so a partial last tile reduces to the same node as the global tree.
 * A stage's tiles of all leaves are numbered flat through a prefix sum of per-leaf tile
 * counts, so one launch covers every leaf.
 */
#ifndef POTS_GPU_BLAKE3_CORE_H
#define POTS_GPU_BLAKE3_CORE_H

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define POTS_ON_GPU 1
typedef unsigned char pots_u8;
typedef unsigned int pots_u32;
typedef unsigned long long pots_u64;
#else
#include <stdint.h>
#include <string.h>
typedef uint8_t pots_u8;
typedef uint32_t pots_u32;
typedef uint64_t pots_u64;
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#endif

#define POTS_CHUNK_LEN 1024u
#ifndef POTS_G_LOG
#define POTS_G_LOG 3
#endif
#ifndef POTS_R_LOG
#define POTS_R_LOG 6
#endif
#define POTS_G (1u << POTS_G_LOG)
#define POTS_R (1u << POTS_R_LOG)
#define POTS_CHUNK_THREADS 128u
#define POTS_TILES_PER_BLOCK (POTS_CHUNK_THREADS / POTS_G)
#define POTS_REDUCE_THREADS POTS_R
#define POTS_NONE 0xFFFFFFFFu
#define POTS_WORD_STRIDE 17u /* words per slot in shared memory: 16 plus one against bank conflicts */
#define POTS_NODE_STRIDE 9u  /* likewise for a chaining value's 8 words */

#define POTS_CHUNK_START 1u
#define POTS_CHUNK_END 2u
#define POTS_PARENT 4u
#define POTS_ROOT 8u

/* One leaf of a table. hdr_ptr and payload_ptr are addresses in the kernel's address space. */
typedef struct {
  pots_u64 payload_ptr;
  pots_u64 payload_len;
  pots_u64 hdr_ptr;
  pots_u32 hdr_len;
  pots_u32 kind; /* 0 none, 1 float32, 2 bfloat16, 3 float16 */
} pots_leaf_t;

/* One stage's arguments. All pointers are in the kernel's address space. */
typedef struct {
  pots_u32 n;           /* leaves */
  pots_u32 stage;       /* 0 for the chunk stage */
  pots_u64 total_tiles; /* this stage's tiles over all leaves */
  const pots_leaf_t *leaf;
  const pots_u64 *tile_prefix; /* n + 1 entries: tiles of the leaves before i */
  const pots_u64 *in_off;      /* stage >= 1: node index of leaf i's first level-(stage-1) node */
  const pots_u64 *out_off;     /* node index of leaf i's first level-stage node */
  pots_u32 *nodes;             /* 8 words per node */
  pots_u32 *digests;           /* 8 words per leaf */
  pots_u32 *first_bad;         /* the smallest leaf with a non-finite payload, else n */
} pots_params_t;

/* ---- compression ---- */

#define POTS_IV0 0x6A09E667u
#define POTS_IV1 0xBB67AE85u
#define POTS_IV2 0x3C6EF372u
#define POTS_IV3 0xA54FF53Au
#define POTS_IV4 0x510E527Fu
#define POTS_IV5 0x9B05688Cu
#define POTS_IV6 0x1F83D9ABu
#define POTS_IV7 0x5BE0CD19u

#define POTS_ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define POTS_MIX(v, a, b, c, d, x, y)                         \
  do {                                                        \
    v[a] = v[a] + v[b] + (x);                                 \
    v[d] = POTS_ROTR(v[d] ^ v[a], 16);                        \
    v[c] = v[c] + v[d];                                       \
    v[b] = POTS_ROTR(v[b] ^ v[c], 12);                        \
    v[a] = v[a] + v[b] + (y);                                 \
    v[d] = POTS_ROTR(v[d] ^ v[a], 8);                         \
    v[c] = v[c] + v[d];                                       \
    v[b] = POTS_ROTR(v[b] ^ v[c], 7);                         \
  } while (0)
#define POTS_ROUND(v, m)                \
  do {                                  \
    POTS_MIX(v, 0, 4, 8, 12, m[0], m[1]);   \
    POTS_MIX(v, 1, 5, 9, 13, m[2], m[3]);   \
    POTS_MIX(v, 2, 6, 10, 14, m[4], m[5]);  \
    POTS_MIX(v, 3, 7, 11, 15, m[6], m[7]);  \
    POTS_MIX(v, 0, 5, 10, 15, m[8], m[9]);  \
    POTS_MIX(v, 1, 6, 11, 12, m[10], m[11]); \
    POTS_MIX(v, 2, 7, 8, 13, m[12], m[13]); \
    POTS_MIX(v, 3, 4, 9, 14, m[14], m[15]); \
  } while (0)
/* The message permutation between rounds: word i of the next round is word perm[i] of this. */
#define POTS_PERMUTE(m, t)                                                                  \
  do {                                                                                      \
    t[0] = m[2]; t[1] = m[6]; t[2] = m[3]; t[3] = m[10]; t[4] = m[7]; t[5] = m[0];          \
    t[6] = m[4]; t[7] = m[13]; t[8] = m[1]; t[9] = m[11]; t[10] = m[12]; t[11] = m[5];      \
    t[12] = m[9]; t[13] = m[14]; t[14] = m[15]; t[15] = m[8];                               \
    for (int pi_ = 0; pi_ < 16; pi_++) m[pi_] = t[pi_];                                     \
  } while (0)

/* BLAKE3's compression with the 64-bit counter's high word 0. out gets the first 8 words of
 * the output (the chaining value, and the digest's 32 bytes when flags has ROOT). out may
 * alias cv. */
static __host__ __device__ __forceinline__ void pots_compress(
    const pots_u32 *cv, const pots_u32 *block, pots_u32 counter, pots_u32 block_len,
    pots_u32 flags, pots_u32 *out) {
  pots_u32 m[16], v[16], t[16];
  for (int i = 0; i < 16; i++) m[i] = block[i];
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = POTS_IV0; v[9] = POTS_IV1; v[10] = POTS_IV2; v[11] = POTS_IV3;
  v[12] = counter; v[13] = 0; v[14] = block_len; v[15] = flags;
  POTS_ROUND(v, m); POTS_PERMUTE(m, t);
  POTS_ROUND(v, m); POTS_PERMUTE(m, t);
  POTS_ROUND(v, m); POTS_PERMUTE(m, t);
  POTS_ROUND(v, m); POTS_PERMUTE(m, t);
  POTS_ROUND(v, m); POTS_PERMUTE(m, t);
  POTS_ROUND(v, m); POTS_PERMUTE(m, t);
  POTS_ROUND(v, m);
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

static __host__ __device__ __forceinline__ void pots_iv(pots_u32 *cv) {
  cv[0] = POTS_IV0; cv[1] = POTS_IV1; cv[2] = POTS_IV2; cv[3] = POTS_IV3;
  cv[4] = POTS_IV4; cv[5] = POTS_IV5; cv[6] = POTS_IV6; cv[7] = POTS_IV7;
}

/* The parent of two chaining values, flags PARENT (| ROOT). out may alias left or right. */
static __host__ __device__ __forceinline__ void pots_parent(
    const pots_u32 *left, const pots_u32 *right, pots_u32 flags, pots_u32 *out) {
  pots_u32 cv[8], m[16];
  pots_iv(cv);
  for (int i = 0; i < 8; i++) {
    m[i] = left[i];
    m[i + 8] = right[i];
  }
  pots_compress(cv, m, 0, 64, flags, out);
}

/* ---- the leaf's message ---- */

static __host__ __device__ __forceinline__ pots_u64 pots_msg_len(const pots_leaf_t *lf) {
  return 1ull + lf->hdr_len + lf->payload_len;
}

static __host__ __device__ __forceinline__ pots_u64 pots_chunks(const pots_leaf_t *lf) {
  return (pots_msg_len(lf) + (POTS_CHUNK_LEN - 1)) / POTS_CHUNK_LEN;
}

/* Nodes of level s: ceil(C / G) at level 0, then ceil(n / R) per level. */
static __host__ __device__ __forceinline__ pots_u64 pots_level_count(const pots_leaf_t *lf,
                                                                       pots_u32 s) {
  pots_u64 n = (pots_chunks(lf) + (POTS_G - 1)) >> POTS_G_LOG;
  for (pots_u32 i = 0; i < s; i++) n = (n + (POTS_R - 1)) >> POTS_R_LOG;
  return n;
}

/* Tiles of stage s: all level-0 groups at stage 0, then one per R nodes of a level that still
 * has more than one node. */
static __host__ __device__ __forceinline__ pots_u64 pots_stage_tiles(const pots_leaf_t *lf,
                                                                       pots_u32 s) {
  if (s == 0) return pots_level_count(lf, 0);
  return pots_level_count(lf, s - 1) > 1 ? pots_level_count(lf, s) : 0;
}

static __host__ __device__ __forceinline__ pots_u32 pots_msg_byte(const pots_leaf_t *lf,
                                                                    pots_u64 o) {
  pots_u64 pre = 1ull + lf->hdr_len;
  if (o == 0) return 0;
  if (o < pre) return ((const pots_u8 *)lf->hdr_ptr)[o - 1];
  o -= pre;
  if (o < lf->payload_len) return ((const pots_u8 *)lf->payload_ptr)[o];
  return 0;
}

static __host__ __device__ __forceinline__ pots_u32 pots_load32(const pots_u8 *p) {
#ifdef POTS_ON_GPU
  return *(const pots_u32 *)p;
#else
  pots_u32 w;
  memcpy(&w, p, 4);
  return w;
#endif
}

/* The little-endian word at message offset `off` (a multiple of 4), zero past the end, read
 * byte by byte: the path of the prefix, the tail and an unaligned payload. The aligned payload
 * words take one load in pots_chunk_load. */
static __host__ __device__ __forceinline__ pots_u32 pots_msg_word_bytes(const pots_leaf_t *lf,
                                                                          pots_u64 off) {
  return pots_msg_byte(lf, off) | (pots_msg_byte(lf, off + 1) << 8) |
         (pots_msg_byte(lf, off + 2) << 16) | (pots_msg_byte(lf, off + 3) << 24);
}

/* 1 when the word holds a NaN or Inf of `kind`: all exponent bits set (f32 0x7F800000, bf16
 * 0x7F80, f16 0x7C00). The zero padding of a payload's last word is finite, so a trailing
 * partial value never counts, as in native.c. */
static __host__ __device__ __forceinline__ int pots_word_bad(pots_u32 kind, pots_u32 w) {
  if (kind == 1) return (w & 0x7F800000u) == 0x7F800000u;
  if (kind == 2) return (w & 0x7F80u) == 0x7F80u || (w & 0x7F800000u) == 0x7F800000u;
  if (kind == 3) return (w & 0x7C00u) == 0x7C00u || (w & 0x7C000000u) == 0x7C000000u;
  return 0;
}

static __host__ __device__ __forceinline__ void pots_note_bad(pots_u32 *first_bad,
                                                                pots_u32 leaf) {
#ifdef POTS_ON_GPU
  /* Most bad words come after a smaller leaf is already noted: skip the atomic then. */
  if (leaf < *(volatile pots_u32 *)first_bad) atomicMin(first_bad, leaf);
#else
  if (leaf < *first_bad) *first_bad = leaf;
#endif
}

/* The leaf that owns flat tile `tile`: the last i with prefix[i] <= tile, for tile below
 * prefix[n]. A leaf with no tiles repeats its neighbour's prefix and is skipped. */
static __host__ __device__ __forceinline__ pots_u32 pots_find_leaf(const pots_u64 *prefix,
                                                                     pots_u32 n, pots_u64 tile) {
  pots_u32 lo = 0, hi = n;
  while (lo < hi) {
    pots_u32 mid = lo + ((hi - lo) >> 1);
    if (prefix[mid] > tile) hi = mid;
    else lo = mid + 1;
  }
  return lo - 1;
}

/* One level of a tile's reduction, for the thread at slot `tid` holding lane `lane` of a tile
 * of `gn` live nodes. Node `lane` takes its right neighbour 2^l lanes up when it exists, and
 * stays as it is otherwise (the odd node's promotion). ROOT goes on the compression that
 * leaves one node, in a leaf's last tile. */
static __host__ __device__ __forceinline__ void pots_tile_level(pots_u32 *nodes, pots_u32 tid,
                                                                  pots_u32 lane, pots_u32 gn,
                                                                  pots_u32 l, int final_tile) {
  if ((lane & ((2u << l) - 1u)) == 0 && lane + (1u << l) < gn) {
    pots_u32 flags = POTS_PARENT | ((final_tile && gn <= (2u << l)) ? POTS_ROOT : 0u);
    pots_parent(nodes + POTS_NODE_STRIDE * tid, nodes + POTS_NODE_STRIDE * (tid + (1u << l)),
                flags, nodes + POTS_NODE_STRIDE * tid);
  }
}

static __host__ __device__ __forceinline__ void pots_store_digest(const pots_params_t *P,
                                                                    pots_u32 leaf,
                                                                    const pots_u32 *cv) {
  for (int i = 0; i < 8; i++) P->digests[8ull * leaf + i] = cv[i];
}

/* ---- stage 0: chunks and their groups ----
 * A block of POTS_CHUNK_THREADS threads is POTS_CHUNK_THREADS slots, one chunk each, in
 * POTS_TILES_PER_BLOCK tiles of G consecutive slots. The kernel runs, with a barrier after
 * every call: setup; for b in 0..15 {load(b); compute(b)}; store; level(l) for
 * l in 0..G_LOG-1; finish (no barrier needed after). */

typedef struct {
  pots_u32 leaf[POTS_CHUNK_THREADS]; /* POTS_NONE for a slot with no chunk */
  pots_u32 chunk[POTS_CHUNK_THREADS];
  pots_u32 gn[POTS_CHUNK_THREADS]; /* live chunks of the slot's tile */
  /* Per slot, once, for the load: the chunk's message bytes (0 for a slot with no chunk), the
   * prefix bytes it holds (the prefix for chunk 0, else 0), the leaf's kind in bits 0-1 and
   * bit 2 when payload words can be read in one load, and the address that chunk-relative
   * offset 0 would have in the payload. */
  pots_u32 clen[POTS_CHUNK_THREADS];
  pots_u32 prel[POTS_CHUNK_THREADS];
  pots_u32 mode[POTS_CHUNK_THREADS];
  pots_u64 wp[POTS_CHUNK_THREADS];
  pots_u32 words[POTS_CHUNK_THREADS * POTS_WORD_STRIDE];
  pots_u32 node[POTS_CHUNK_THREADS * POTS_NODE_STRIDE];
} pots_chunk_smem_t;

static __host__ __device__ __forceinline__ void pots_chunk_setup(
    pots_chunk_smem_t *sm, const pots_params_t *P, pots_u64 blk, pots_u32 tid) {
  pots_u64 tile = blk * POTS_TILES_PER_BLOCK + (tid >> POTS_G_LOG);
  pots_u32 lane = tid & (POTS_G - 1);
  sm->leaf[tid] = POTS_NONE;
  sm->chunk[tid] = 0;
  sm->gn[tid] = 0;
  sm->clen[tid] = 0;
  if (tile < P->total_tiles) {
    pots_u32 leaf = pots_find_leaf(P->tile_prefix, P->n, tile);
    pots_u64 grp = tile - P->tile_prefix[leaf];
    pots_u64 c = pots_chunks(&P->leaf[leaf]);
    pots_u64 left = c - grp * POTS_G;
    sm->gn[tid] = left < POTS_G ? (pots_u32)left : POTS_G;
    if (lane < left) {
      sm->leaf[tid] = leaf;
      sm->chunk[tid] = (pots_u32)(grp * POTS_G + lane);
      const pots_leaf_t *lf = &P->leaf[leaf];
      pots_u64 pre = 1ull + lf->hdr_len;
      pots_u64 start = (grp * POTS_G + lane) * POTS_CHUNK_LEN;
      pots_u64 rest = pre + lf->payload_len - start;
      sm->clen[tid] = rest < POTS_CHUNK_LEN ? (pots_u32)rest : POTS_CHUNK_LEN;
      sm->prel[tid] = start == 0 ? (pots_u32)pre : 0u;
      sm->mode[tid] = lf->kind | (((pre & 3) == 0 && (lf->payload_ptr & 3) == 0) ? 4u : 0u);
      sm->wp[tid] = lf->payload_ptr + start - pre;
    }
  }
}

/* Block b of every slot's chunk into sm->words: the block's threads read the 16 words of 8
 * consecutive slots at a time, so 16 neighbouring threads read 64 contiguous bytes. The
 * finiteness test runs on the words as they are read. A word at chunk offset q is zero from
 * the chunk's end on; a payload word that lies whole inside the chunk, of an aligned slot, is
 * one load; any other word (prefix, tail, unaligned payload) goes byte by byte. */
static __host__ __device__ __forceinline__ void pots_chunk_load(
    pots_chunk_smem_t *sm, const pots_params_t *P, pots_u32 tid, pots_u32 b) {
  for (pots_u32 i = 0; i < 16; i++) {
    pots_u32 idx = tid + i * POTS_CHUNK_THREADS;
    pots_u32 s = idx >> 4, w = idx & 15;
    pots_u32 q = 64u * b + 4u * w;
    pots_u32 clen = sm->clen[s];
    pots_u32 word = 0;
    if (q < clen) {
      pots_u32 mode = sm->mode[s];
      int payload = q >= sm->prel[s];
      if (payload && (mode & 4u) && q + 4u <= clen)
        word = pots_load32((const pots_u8 *)(sm->wp[s] + q));
      else
        word = pots_msg_word_bytes(&P->leaf[sm->leaf[s]],
                                   (pots_u64)sm->chunk[s] * POTS_CHUNK_LEN + q);
      if (payload && (mode & 3u) && pots_word_bad(mode & 3u, word))
        pots_note_bad(P->first_bad, sm->leaf[s]);
    }
    sm->words[s * POTS_WORD_STRIDE + w] = word;
  }
}

/* Slot tid's chunk takes its block b, if it has one. cv is the slot's chaining value, kept
 * by the caller between calls. */
static __host__ __device__ __forceinline__ void pots_chunk_compute(
    pots_chunk_smem_t *sm, const pots_params_t *P, pots_u32 tid, pots_u32 b, pots_u32 *cv) {
  pots_u32 lid = sm->leaf[tid];
  if (lid == POTS_NONE) return;
  const pots_leaf_t *lf = &P->leaf[lid];
  pots_u64 c = sm->chunk[tid];
  pots_u32 clen = sm->clen[tid];
  pots_u32 nb = (clen + 63) >> 6;
  if (b >= nb) return;
  if (b == 0) pots_iv(cv);
  int last = b == nb - 1;
  pots_u32 flags = (b == 0 ? POTS_CHUNK_START : 0u) | (last ? POTS_CHUNK_END : 0u);
  if (last && pots_chunks(lf) == 1) flags |= POTS_ROOT;
  pots_compress(cv, sm->words + tid * POTS_WORD_STRIDE, (pots_u32)c, last ? clen - 64 * b : 64,
                flags, cv);
}

static __host__ __device__ __forceinline__ void pots_chunk_store(pots_chunk_smem_t *sm,
                                                                   pots_u32 tid,
                                                                   const pots_u32 *cv) {
  if (sm->leaf[tid] == POTS_NONE) return;
  for (int i = 0; i < 8; i++) sm->node[tid * POTS_NODE_STRIDE + i] = cv[i];
}

static __host__ __device__ __forceinline__ void pots_chunk_level(
    pots_chunk_smem_t *sm, const pots_params_t *P, pots_u32 tid, pots_u32 l) {
  pots_u32 lid = sm->leaf[tid];
  if (lid == POTS_NONE) return;
  pots_u32 final_tile = pots_level_count(&P->leaf[lid], 0) == 1;
  pots_tile_level(sm->node, tid, tid & (POTS_G - 1), sm->gn[tid], l, final_tile);
}

/* Lane 0 of each tile writes the leaf's digest when the tile is its only one, else its node. */
static __host__ __device__ __forceinline__ void pots_chunk_finish(
    const pots_chunk_smem_t *sm, const pots_params_t *P, pots_u32 tid) {
  pots_u32 lid = sm->leaf[tid];
  if (lid == POTS_NONE || (tid & (POTS_G - 1)) != 0) return;
  const pots_u32 *cv = sm->node + tid * POTS_NODE_STRIDE;
  if (pots_level_count(&P->leaf[lid], 0) == 1) {
    pots_store_digest(P, lid, cv);
  } else {
    pots_u64 at = P->out_off[lid] + (sm->chunk[tid] >> POTS_G_LOG);
    for (int i = 0; i < 8; i++) P->nodes[8ull * at + i] = cv[i];
  }
}

/* ---- stages 1 and up: R nodes of one leaf per block of POTS_REDUCE_THREADS threads ----
 * Per block: setup; level(l) for l in 0..R_LOG-1; finish; barriers between calls. */

typedef struct {
  pots_u32 node[POTS_REDUCE_THREADS * POTS_NODE_STRIDE];
} pots_reduce_smem_t;

typedef struct {
  pots_u32 leaf; /* POTS_NONE when the block has no tile */
  pots_u32 gn;   /* live nodes of the tile */
  pots_u32 tile; /* the tile's index within its leaf */
  pots_u32 final_tile;
} pots_reduce_thread_t;

static __host__ __device__ __forceinline__ void pots_reduce_setup(
    pots_reduce_smem_t *sm, const pots_params_t *P, pots_u64 blk, pots_u32 tid,
    pots_reduce_thread_t *st) {
  st->leaf = POTS_NONE;
  st->gn = 0;
  st->tile = 0;
  st->final_tile = 0;
  if (blk >= P->total_tiles) return;
  pots_u32 leaf = pots_find_leaf(P->tile_prefix, P->n, blk);
  const pots_leaf_t *lf = &P->leaf[leaf];
  pots_u64 t = blk - P->tile_prefix[leaf];
  pots_u64 cnt = pots_level_count(lf, P->stage - 1);
  pots_u64 left = cnt - t * POTS_R;
  st->leaf = leaf;
  st->tile = (pots_u32)t;
  st->gn = left < POTS_R ? (pots_u32)left : POTS_R;
  st->final_tile = pots_level_count(lf, P->stage) == 1;
  if (tid < st->gn) {
    const pots_u32 *src = P->nodes + 8ull * (P->in_off[leaf] + t * POTS_R + tid);
    for (int i = 0; i < 8; i++) sm->node[tid * POTS_NODE_STRIDE + i] = src[i];
  }
}

static __host__ __device__ __forceinline__ void pots_reduce_level(
    pots_reduce_smem_t *sm, const pots_reduce_thread_t *st, pots_u32 tid, pots_u32 l) {
  if (st->leaf == POTS_NONE) return;
  pots_tile_level(sm->node, tid, tid, st->gn, l, st->final_tile);
}

static __host__ __device__ __forceinline__ void pots_reduce_finish(
    const pots_reduce_smem_t *sm, const pots_params_t *P, pots_u32 tid,
    const pots_reduce_thread_t *st) {
  if (st->leaf == POTS_NONE || tid != 0) return;
  if (st->final_tile) {
    pots_store_digest(P, st->leaf, sm->node);
  } else {
    pots_u64 at = P->out_off[st->leaf] + st->tile;
    for (int i = 0; i < 8; i++) P->nodes[8ull * at + i] = sm->node[i];
  }
}

#endif
