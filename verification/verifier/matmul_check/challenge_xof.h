/* The keyed XOF behind check 5's challenge entries (FS41, P8b, F23), as code that both NVRTC and
 * a host C compiler build. challenge_xof.cu wraps pots_xof_unit in a kernel; the host test
 * (tests/verification/verifier/matmul_check/challenge_xof_host.c) loops over it.
 *
 * A unit is one 64-byte output block b of the challenge vector j (1-based) of one item. The
 * label 0x10 || m (4, big-endian) || j (1) fits one block, so a unit is one compression, with
 * the key h as chaining value and all 16 output words kept. Word w is entry 16b + w, mapped as
 * P8b says, and written at row 16b + w, column j - 1 of the item's [width, k] matrix. An
 * item's units run with j fastest.
 */
#ifndef POTS_CHALLENGE_XOF_H
#define POTS_CHALLENGE_XOF_H

#include "gpu_blake3_core.h"

typedef struct {  /* 24 bytes */
  pots_u32 m;       /* the product's label m, 1 <= m < 2^32 */
  pots_u32 width;   /* entries per vector, >= 1 */
  pots_u64 unit0;   /* the item's first unit: sum over earlier items of k*ceil(width/16) */
  pots_u64 out_off; /* the item's first entry in out, in float32 entries */
} pots_xitem_t;

typedef struct {  /* 64 bytes */
  pots_u32 key[8]; /* h as 8 little-endian words */
  pots_u32 k;      /* 1 <= k <= 255 */
  pots_u32 n;      /* items */
  pots_u64 units;  /* sum of k*ceil(width/16) over all items */
  pots_u64 items;  /* address of pots_xitem_t[n] */
  pots_u64 out;    /* address of the float32 output */
} pots_xparams_t;

/* BLAKE3's compression with all 16 output words (the XOF's block), counter high word 0. */
static __host__ __device__ __forceinline__ void pots_compress_xof(
    const pots_u32 *cv, const pots_u32 *block, pots_u32 counter, pots_u32 block_len,
    pots_u32 flags, pots_u32 *out) {
  pots_u32 m[16], v[16], t[16];
#ifdef POTS_ON_GPU
#pragma unroll
#endif
  for (int i = 0; i < 16; i++) m[i] = block[i];
#ifdef POTS_ON_GPU
#pragma unroll
#endif
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
#ifdef POTS_ON_GPU
#pragma unroll
#endif
  for (int i = 0; i < 8; i++) {
    out[i] = v[i] ^ v[i + 8];
    out[i + 8] = v[i + 8] ^ cv[i];
  }
}

/* The label block: bytes 0x10, m big-endian, j, then zeros. */
static __host__ __device__ __forceinline__ void pots_xof_label(pots_u32 m, pots_u32 j,
                                                               pots_u32 *block) {
  block[0] = 0x10u | (((m >> 24) & 0xffu) << 8) | (((m >> 16) & 0xffu) << 16) |
             (((m >> 8) & 0xffu) << 24);
  block[1] = (m & 0xffu) | (j << 8);
  for (int i = 2; i < 16; i++) block[i] = 0;
}

/* The item holding unit u: the last i with unit0[i] <= u. */
static __host__ __device__ __forceinline__ pots_u32 pots_xof_find(const pots_xparams_t *P,
                                                                  pots_u64 u) {
  const pots_xitem_t *items = (const pots_xitem_t *)P->items;
  pots_u32 lo = 0, hi = P->n;  /* items[lo].unit0 <= u, and u < items[hi].unit0 (or hi == n) */
  while (hi - lo > 1) {
    pots_u32 mid = lo + (hi - lo) / 2;
    if (items[mid].unit0 <= u) lo = mid; else hi = mid;
  }
  return lo;
}

/* P8b: a = w >> 8, r = (2a + 1 - 2^24) / 2^24, exact in float32. */
static __host__ __device__ __forceinline__ float pots_xof_entry(pots_u32 w) {
  int numerator = 2 * (int)(w >> 8) + 1 - (1 << 24);
  return (float)numerator * (1.0f / 16777216.0f);
}

static __host__ __device__ __forceinline__ void pots_xof_unit(const pots_xparams_t *P,
                                                              pots_u64 u) {
  const pots_xitem_t *it = (const pots_xitem_t *)P->items + pots_xof_find(P, u);
  pots_u64 local = u - it->unit0;
  pots_u32 k = P->k;
  pots_u32 b, j;
  if (local < 0x100000000ull) {
    pots_u32 l = (pots_u32)local;
    b = l / k;
    j = l % k + 1;
  } else {
    b = (pots_u32)(local / k);
    j = (pots_u32)(local % k) + 1;
  }
  pots_u32 block[16], words[16];
  pots_xof_label(it->m, j, block);
  pots_compress_xof(P->key, block, b, 6, 27, words);
  float *out = (float *)P->out + it->out_off + (pots_u64)j - 1;
#ifdef POTS_ON_GPU
#pragma unroll
#endif
  for (pots_u32 w = 0; w < 16; w++) {
    pots_u64 row = 16ull * b + w;
    if (row < it->width) out[row * k] = pots_xof_entry(words[w]);
  }
}

#endif
