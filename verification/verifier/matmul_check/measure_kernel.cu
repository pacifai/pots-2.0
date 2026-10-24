// The fused measuring kernel (FS37, decision O2; one sweep per row, FS37b). Compiled by NVRTC at
// run time (measure_kernel.py) with measure_core.h and measure_phases.h passed in memory, with
// --fmad=false. Every decision about what is computed lives in those headers; this file maps
// the work onto blocks, warps, lanes, shared-memory tiles and shuffles. One launch per level of
// the read plan (pots_measure_level).
//
// A block holds ROWS_PER_BLOCK *sets*: a set is one (row, entry) pair, one accumulator set in
// one warp's lanes (each warp owns ROWS_PER_BLOCK / POTS_MK_WARPS consecutive sets). A read
// group with e entries is run E_MAX entries at a time (e_b = min(e, E_MAX) of them per batch);
// the block takes RB = ROWS_PER_BLOCK / min(e, E_MAX) rows of the shared view, and set s is
// entry s / RB, row s % RB. The block walks the view's columns in tiles of TILE_COLS: the view's
// RB rows are loaded once, coalesced, into shared memory (widened to fp32), with every entry's
// vector tile; lane p then takes the tile's columns p, p+32, ... in increasing order for each
// of its sets, so every accumulator sees exactly the operations of its step's partial in
// measure_core.h (the per-element updates are measure_phases.h's, shared with the host
// emulation). A column past the view's end is skipped, never zero-filled. After the last tile
// the warp's halving tree reduces each accumulator and lane 0 stores through pots_phase_store.
// A phase-1 entry then sweeps its own P rows (all the batch's P views in one tile). The
// geometry (the macros below) is passed as -D options and never changes a bit.
#ifndef POTS_MK_WARPS
#define POTS_MK_WARPS 8u /* warps per block */
#endif
#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 32u /* rows of a view per block */
#endif
#ifndef TILE_COLS
#define TILE_COLS 64u /* columns per tile, a multiple of 32 */
#endif
#ifndef K_CHUNK
#define K_CHUNK 16u /* challenge columns per sweep */
#endif
#ifndef E_MAX
#define E_MAX 4u /* entries of a read group swept together */
#endif
#include "measure_core.h"
#include "measure_phases.h"

#define POTS_FULL 0xffffffffu
#define POTS_NT (POTS_MK_WARPS * 32u)
#define POTS_RPW (ROWS_PER_BLOCK / POTS_MK_WARPS) /* rows per warp */
#define POTS_XPAD (TILE_COLS + 1u)                /* the x tile's row pitch: no bank conflicts */
#define POTS_YP (TILE_COLS + 1u)                  /* the vector tile's column pitch */

static_assert(TILE_COLS % 32u == 0u && TILE_COLS >= 32u, "TILE_COLS must be a multiple of 32");
static_assert(ROWS_PER_BLOCK % POTS_MK_WARPS == 0u && ROWS_PER_BLOCK >= POTS_MK_WARPS,
              "ROWS_PER_BLOCK must be a multiple of the warps");
static_assert(K_CHUNK >= 1u, "K_CHUNK must be at least 1");
static_assert(E_MAX >= 1u && E_MAX <= ROWS_PER_BLOCK, "E_MAX must be in 1 .. ROWS_PER_BLOCK");

enum { POTS_SW_SHARED = 0, POTS_SW_P = 1, POTS_SW_BIAS = 2 };

// One level's launch, passed by value.
typedef struct {
  pots_u64 prods;    // pots_mprod_t[n_prod]
  pots_u64 works;    // pots_mwork_t[n_prod + 1] (the scratch addresses)
  pots_u64 rows;     // float64 [n_prod, ROW_FIXED + k]
  pots_u64 reads;    // pots_mread_t[n_groups + 1], the level's groups and its sentinel
  pots_u64 entries;  // pots_mentry_t
  pots_u64 starts;   // uint64 [n_groups + 1]: the first row block of each group, then the total
  pots_u64 n_groups;
  pots_u64 n_blocks; // row blocks of this level
  pots_u32 k;
  pots_u32 pad;
} pots_mlargs_t;

// ---- the halving trees ----
// Lane i takes s[i] + s[i + off] for off = 16, 8, 4, 2, 1; lane 0 ends with the result (the
// other lanes' values are never used).
static __device__ __forceinline__ float pots_tree_f(float v) {
  for (pots_u32 off = 16; off >= 1; off >>= 1) v = v + __shfl_down_sync(POTS_FULL, v, off);
  return v;
}
static __device__ __forceinline__ double pots_tree_d(double v) {
  for (pots_u32 off = 16; off >= 1; off >>= 1) v = v + __shfl_down_sync(POTS_FULL, v, off);
  return v;
}

// ---- loads, specialised by dtype and layout; two passes (gather, then scatter) ----
//
// Every tile load is two passes over the thread's elements: the global loads go to a small
// register array (its size a compile-time constant, the loops unrolled), then the shared stores.
// No shared store sits between two global loads, so the loads of a pass are all in flight
// together (the compiler can't move a global load past a store through a generic pointer).

#define POTS_NPX ((ROWS_PER_BLOCK * TILE_COLS + POTS_NT - 1u) / POTS_NT) /* x elements per thread */
#define POTS_NPV ((K_CHUNK * TILE_COLS + POTS_NT - 1u) / POTS_NT)        /* vector elements */
#define POTS_NPB ((TILE_COLS + POTS_NT - 1u) / POTS_NT)                  /* b2, b1 elements */

// DT 0: fp32, 1: bf16 (the 16 bits shifted left by 16, as pots_mload). LAY 0: row-major
// (col_stride == 1), 1: transposed (row_stride == 1), 2: any strides.
template <int DT>
static __device__ __forceinline__ float pots_ld(pots_u64 addr, pots_i64 at) {
  if (DT == 1) return pots_bits_float((pots_u32)((const pots_u16 *)addr)[at] << 16);
  return ((const float *)addr)[at];
}

// Element e of an x tile of nrows rows (nrows <= rbp) and ncols columns: its row r and tile
// column cc; false when e is no element. The block's threads take consecutive elements along
// the view's contiguous dimension (for a transposed view the rows, in runs of rbp).
template <int LAY>
static __device__ __forceinline__ bool pots_xe(pots_u32 e, pots_u32 nrows, pots_u32 rbp,
                                               pots_u32 ncols, pots_u32 &r, pots_u32 &cc) {
  if (LAY == 1) {
    if (e >= rbp * ncols) return false;
    r = e % rbp;
    cc = e / rbp;
    return r < nrows;
  }
  if (e >= nrows * TILE_COLS) return false;
  r = e / TILE_COLS;
  cc = e % TILE_COLS;
  return cc < ncols;
}

// Pass 1: rows r0 .. r0+nrows-1 and columns c0 .. c0+ncols-1 of view m into the registers v.
template <int DT, int LAY>
static __device__ __forceinline__ void pots_gather_x(const pots_mview_t *__restrict__ m,
                                                     pots_i64 r0, pots_u32 nrows, pots_u32 rbp,
                                                     pots_i64 c0, pots_u32 ncols,
                                                     float (&v)[POTS_NPX]) {
  const pots_i64 rs = m->row_stride, cs = m->col_stride;
  const pots_u64 addr = m->addr;
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPX; i++) {
    pots_u32 r = 0, cc = 0;
    v[i] = 0.0f;
    if (pots_xe<LAY>(threadIdx.x + i * POTS_NT, nrows, rbp, ncols, r, cc)) {
      const pots_i64 at = LAY == 1 ? (r0 + r) + (c0 + cc) * cs
                                   : (r0 + r) * rs + (LAY == 0 ? (c0 + cc) : (c0 + cc) * cs);
      v[i] = pots_ld<DT>(addr, at);
    }
  }
}

// Pass 2: the registers into sx[r * POTS_XPAD + cc].
template <int LAY>
static __device__ __forceinline__ void pots_scatter_x(pots_u32 nrows, pots_u32 rbp,
                                                      pots_u32 ncols, const float (&v)[POTS_NPX],
                                                      float *__restrict__ sx) {
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPX; i++) {
    pots_u32 r = 0, cc = 0;
    if (pots_xe<LAY>(threadIdx.x + i * POTS_NT, nrows, rbp, ncols, r, cc))
      sx[r * POTS_XPAD + cc] = v[i];
  }
}

static __device__ __forceinline__ int pots_mode(const pots_mview_t *m) {
  const int lay = m->col_stride == 1 ? 0 : (m->row_stride == 1 ? 1 : 2);
  return (m->dtype == POTS_DT_BF16 ? 3 : 0) + lay;
}

// The switch on (dtype, layout), made once per tile outside every compute loop.
static __device__ __forceinline__ void pots_gather_x_mode(int mode, const pots_mview_t *m,
                                                          pots_i64 r0, pots_u32 nrows,
                                                          pots_u32 rbp, pots_i64 c0,
                                                          pots_u32 ncols, float (&v)[POTS_NPX]) {
  switch (mode) {
    case 0: pots_gather_x<0, 0>(m, r0, nrows, rbp, c0, ncols, v); break;
    case 1: pots_gather_x<0, 1>(m, r0, nrows, rbp, c0, ncols, v); break;
    case 2: pots_gather_x<0, 2>(m, r0, nrows, rbp, c0, ncols, v); break;
    case 3: pots_gather_x<1, 0>(m, r0, nrows, rbp, c0, ncols, v); break;
    case 4: pots_gather_x<1, 1>(m, r0, nrows, rbp, c0, ncols, v); break;
    default: pots_gather_x<1, 2>(m, r0, nrows, rbp, c0, ncols, v); break;
  }
}
static __device__ __forceinline__ void pots_scatter_x_mode(int mode, pots_u32 nrows,
                                                           pots_u32 rbp, pots_u32 ncols,
                                                           const float (&v)[POTS_NPX],
                                                           float *__restrict__ sx) {
  switch (mode % 3) {
    case 0: pots_scatter_x<0>(nrows, rbp, ncols, v, sx); break;
    case 1: pots_scatter_x<1>(nrows, rbp, ncols, v, sx); break;
    default: pots_scatter_x<2>(nrows, rbp, ncols, v, sx); break;
  }
}

// The vector tile of one entry: R's columns j0 .. j0+nj-1 at tile columns 0 .. ncols-1 (element
// e: tile column e / nj, column j0 + e % nj: R's entries are consecutive in memory), or an
// A-kind entry's stored t rows (element e: row e / TILE_COLS, tile column e % TILE_COLS) and,
// on `first`, its b2 and b1. Stored at sy[j * POTS_YP + cc] (the +1 pitch: consecutive threads
// of R's order store consecutive j without bank conflicts).
static __device__ __forceinline__ void pots_gather_r(const float *__restrict__ r, pots_u32 k,
                                                     pots_u32 j0, pots_u32 nj, pots_i64 c0,
                                                     pots_u32 ncols, float (&v)[POTS_NPV]) {
  const pots_u32 tot = ncols * nj;
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPV; i++) {
    const pots_u32 e = threadIdx.x + i * POTS_NT;
    v[i] = 0.0f;
    if (e < tot) {
      const pots_u32 cc = e / nj, j = e - cc * nj;
      v[i] = r[(pots_u64)(c0 + cc) * k + j0 + j];
    }
  }
}
static __device__ __forceinline__ void pots_scatter_r(pots_u32 nj, pots_u32 ncols,
                                                      const float (&v)[POTS_NPV],
                                                      float *__restrict__ sy) {
  const pots_u32 tot = ncols * nj;
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPV; i++) {
    const pots_u32 e = threadIdx.x + i * POTS_NT;
    if (e < tot) {
      const pots_u32 cc = e / nj, j = e - cc * nj;
      sy[j * POTS_YP + cc] = v[i];
    }
  }
}
static __device__ __forceinline__ void pots_gather_t(const float *__restrict__ tt, pots_i64 q,
                                                     pots_u32 j0, pots_u32 nj, pots_i64 c0,
                                                     pots_u32 ncols, float (&v)[POTS_NPV]) {
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPV; i++) {
    const pots_u32 e = threadIdx.x + i * POTS_NT;
    const pots_u32 j = e / TILE_COLS, cc = e % TILE_COLS;
    v[i] = 0.0f;
    if (j < nj && cc < ncols) v[i] = tt[(pots_u64)(j0 + j) * (pots_u64)q + (pots_u64)(c0 + cc)];
  }
}
static __device__ __forceinline__ void pots_scatter_t(pots_u32 nj, pots_u32 ncols,
                                                      const float (&v)[POTS_NPV],
                                                      float *__restrict__ sy) {
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPV; i++) {
    const pots_u32 e = threadIdx.x + i * POTS_NT;
    const pots_u32 j = e / TILE_COLS, cc = e % TILE_COLS;
    if (j < nj && cc < ncols) sy[j * POTS_YP + cc] = v[i];
  }
}
static __device__ __forceinline__ void pots_gather_b(const float *__restrict__ b2,
                                                     const float *__restrict__ b1, int guard,
                                                     pots_i64 c0, pots_u32 ncols,
                                                     float (&v2)[POTS_NPB], float (&v1)[POTS_NPB]) {
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPB; i++) {
    const pots_u32 cc = threadIdx.x + i * POTS_NT;
    v2[i] = 0.0f;
    v1[i] = 0.0f;
    if (cc < ncols) {
      v2[i] = b2[c0 + cc];
      if (guard) v1[i] = b1[c0 + cc];
    }
  }
}
static __device__ __forceinline__ void pots_scatter_b(int guard, pots_u32 ncols,
                                                      const float (&v2)[POTS_NPB],
                                                      const float (&v1)[POTS_NPB],
                                                      float *__restrict__ sb2,
                                                      float *__restrict__ sb1) {
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPB; i++) {
    const pots_u32 cc = threadIdx.x + i * POTS_NT;
    if (cc < ncols) {
      sb2[cc] = v2[i];
      if (guard) sb1[cc] = v1[i];
    }
  }
}

// ---- sets: (row, entry) pairs of a batch ----

typedef struct {
  float acc[POTS_RPW][K_CHUNK]; // lane p's accumulator of challenge column j of its set rr
  float e1[POTS_RPW];           // b2 | u
  float e2[POTS_RPW];           // b1 | nu | p1
  double ed[POTS_RPW];          // the float64 sum of squares: the bias's, or pp
} pots_lane_t;

// One batch: up to E_MAX entries of a read group over one block of rows, and the warp's sets.
// Everything in it is the same for all threads of a warp (the arrays are indexed by the unrolled
// set number rr, a compile-time constant in every loop).
typedef struct {
  const pots_mprod_t *prods;
  const pots_mwork_t *works;
  const pots_mentry_t *ent; // the batch's entries
  double *rows;
  pots_u32 k;
  pots_u32 eb;     // entries in the batch
  pots_u32 rb;     // rows of the view per block (RB); the sets of entry b are b*rb .. b*rb+rb-1
  pots_i64 r0;     // the block's first row
  pots_u32 nrows;  // the block's rows of the group (<= rb)
  pots_u32 nld;    // rows of the shared view to load (the group's rows, or the B rows)
  pots_i64 pmax;   // the widest P view of the batch's phase-1 entries (0: none)
  pots_u32 e_of[POTS_RPW];   // the set's entry
  pots_u32 row_of[POTS_RPW]; // the set's row within the block
  pots_i64 pcols[POTS_RPW];  // columns of the set's own P view (phase-1 sets)
  pots_u32 live;   // bit rr: the set exists and has a row
  pots_u32 plive;  // bit rr: a live set of a phase-1 entry
  pots_u32 akind;  // bit rr: an A-row set (phase 1); clear: a B-row set (phase 0)
  pots_u32 guardm; // bit rr: the set's product has the guard
} pots_batch_t;

static __device__ __forceinline__ const pots_mprod_t *pots_bprod(const pots_batch_t &B,
                                                                 pots_u32 b) {
  return B.prods + B.ent[b].prod;
}

// The batch of entries ent[0 .. eb-1] over rows r0 .. r0+nrows-1 of the group (rb: the block's
// row capacity RB). Bias: the single bias row of ent[0]'s product instead (one set).
static __device__ __forceinline__ pots_batch_t pots_make_batch(
    const pots_mprod_t *prods, const pots_mwork_t *works, const pots_mentry_t *ent,
    double *rows, pots_u32 k, pots_u32 eb, pots_u32 rb, pots_i64 r0, pots_u32 nrows, int bias) {
  const pots_u32 warp = threadIdx.x >> 5;
  pots_batch_t B;
  B.prods = prods;
  B.works = works;
  B.ent = ent;
  B.rows = rows;
  B.k = k;
  B.eb = eb;
  B.rb = rb;
  B.r0 = r0;
  B.nrows = nrows;
  B.live = B.plive = B.akind = B.guardm = 0u;
  B.pmax = 0;
  {
    const pots_mprod_t *t0 = prods + ent[0].prod;
    const pots_i64 qb0 = t0->b.rows;
    B.nld = ent[0].phase != 0 ? nrows
                              : (r0 >= qb0 ? 0u : (qb0 - r0 < (pots_i64)nrows ? (pots_u32)(qb0 - r0) : nrows));
  }
  if (bias) B.nld = 1u;
  for (pots_u32 b = 0; b < eb; b++) {
    const pots_mprod_t *t = prods + ent[b].prod;
    if (ent[b].phase == 1 && t->p.cols > B.pmax) B.pmax = t->p.cols;
  }
  _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
    const pots_u32 s = warp * POTS_RPW + rr;
    const pots_u32 b = s / rb, row = s % rb;
    B.e_of[rr] = b;
    B.row_of[rr] = row;
    B.pcols[rr] = 0;
    if (bias) {
      if (s == 0u) {
        const pots_mprod_t *t = prods + ent[0].prod;
        B.live |= 1u << rr;
        if ((t->flags & POTS_M_GUARD) != 0) B.guardm |= 1u << rr;
      }
    } else if (b < eb) {
      const pots_mentry_t en = ent[b];
      const pots_mprod_t *t = prods + en.prod;
      pots_u32 nb = nrows;
      if (en.phase == 0) {
        const pots_i64 qb = t->b.rows;
        nb = r0 >= qb ? 0u : (qb - r0 < (pots_i64)nrows ? (pots_u32)(qb - r0) : nrows);
      }
      if (row < nb) {
        B.live |= 1u << rr;
        if ((t->flags & POTS_M_GUARD) != 0) B.guardm |= 1u << rr;
        if (en.phase == 1) {
          B.akind |= 1u << rr;
          B.plive |= 1u << rr;
          B.pcols[rr] = t->p.cols;
        }
      }
    }
  }
  return B;
}

// ---- one sweep over a batch ----

// The registers a tile's global loads land in: the x tile, one vector tile per entry, and each
// entry's b2 and b1. Every index into them is a compile-time constant (the loops over entries
// are unrolled to E_MAX with a runtime `b < eb` guard), so they stay in registers.
typedef struct {
  float x[POTS_NPX];
  float v[E_MAX][POTS_NPV];
  float b2[E_MAX][POTS_NPB];
  float b1[E_MAX][POTS_NPB];
} pots_tile_regs_t;

// Pass 1 of a tile of the shared view `vw` (SW SHARED: every entry's vector tile; BIAS: the one
// bias row and ent[0]'s R).
template <int SW>
static __device__ __forceinline__ void pots_gather_shared(const pots_batch_t &B,
                                                          const pots_mview_t *vw, int mode,
                                                          pots_i64 c0, pots_u32 ncols,
                                                          pots_u32 j0, pots_u32 nj, int first,
                                                          pots_tile_regs_t &R) {
  if (SW == POTS_SW_BIAS) {
    pots_gather_x_mode(mode, vw, 0, 1u, 1u, c0, ncols, R.x);
    pots_gather_r((const float *)pots_bprod(B, 0)->r_addr, B.k, j0, nj, c0, ncols, R.v[0]);
    return;
  }
  pots_gather_x_mode(mode, vw, B.r0, B.nld, B.rb, c0, ncols, R.x);
  _Pragma("unroll") for (pots_u32 b = 0; b < E_MAX; b++) {
    if (b < B.eb) {
      const pots_mprod_t *t = pots_bprod(B, b);
      if (B.ent[b].phase == 0) {
        pots_gather_r((const float *)t->r_addr, B.k, j0, nj, c0, ncols, R.v[b]);
      } else {
        const pots_mscr_t sc = pots_mscr_make(t, B.works[B.ent[b].prod].scratch);
        pots_gather_t(sc.tt, t->a.cols, j0, nj, c0, ncols, R.v[b]);
        if (first)
          pots_gather_b(sc.b2, sc.b1, (t->flags & POTS_M_GUARD) != 0, c0, ncols, R.b2[b],
                        R.b1[b]);
      }
    }
  }
}

// Pass 2 of that tile.
template <int SW>
static __device__ __forceinline__ void pots_scatter_shared(
    const pots_batch_t &B, int mode, pots_u32 ncols, pots_u32 nj, int first,
    const pots_tile_regs_t &R, float *__restrict__ sx, float *__restrict__ sy,
    float *__restrict__ sb2, float *__restrict__ sb1) {
  if (SW == POTS_SW_BIAS) {
    pots_scatter_x_mode(mode, 1u, 1u, ncols, R.x, sx);
    pots_scatter_r(nj, ncols, R.v[0], sy);
    return;
  }
  pots_scatter_x_mode(mode, B.nld, B.rb, ncols, R.x, sx);
  _Pragma("unroll") for (pots_u32 b = 0; b < E_MAX; b++) {
    if (b < B.eb) {
      float *yb = sy + b * K_CHUNK * POTS_YP;
      if (B.ent[b].phase == 0) {
        pots_scatter_r(nj, ncols, R.v[b], yb);
      } else {
        pots_scatter_t(nj, ncols, R.v[b], yb);
        if (first)
          pots_scatter_b((pots_bprod(B, b)->flags & POTS_M_GUARD) != 0, ncols, R.b2[b], R.b1[b],
                         sb2 + b * TILE_COLS, sb1 + b * TILE_COLS);
      }
    }
  }
}

// One pass over the batch's view (SW SHARED: the view vw, all entries; P: each phase-1
// entry's own P view; BIAS: the one bias row) for the challenge columns j0 .. j0+nj-1 (first:
// also the extras), into the lanes' accumulators. Every thread of the block calls it with the
// same arguments (it has barriers).
//
// The compute walks the lane's columns c (increasing) and, for each, the warp's sets: when a
// set's entry differs from the previous set's, that entry's nj vector values at c (and b2[c],
// b1[c] of an A-kind entry) go from shared memory into registers once, and every set of the
// entry uses them. Each accumulator still sees its columns in increasing order with the same
// operation, so the bits are those of the column-inner order.
template <int SW>
static __device__ __forceinline__ void pots_sweep(const pots_batch_t &B, const pots_mview_t *vw,
                                                  pots_u32 j0, pots_u32 nj, int first,
                                                  pots_lane_t &L, float *__restrict__ sx,
                                                  float *__restrict__ sy,
                                                  float *__restrict__ sb2,
                                                  float *__restrict__ sb1) {
  const pots_u32 lane = threadIdx.x & 31u;
  const pots_u32 live = SW == POTS_SW_P ? B.plive : B.live;
  _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
    _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++) L.acc[rr][j] = 0.0f;
    L.e1[rr] = 0.0f;
    L.e2[rr] = 0.0f;
    L.ed[rr] = 0.0;
  }
  const pots_i64 cols = SW == POTS_SW_P ? B.pmax : vw->cols;
  const int mode = SW == POTS_SW_P ? 0 : pots_mode(vw);
  pots_tile_regs_t R;
  for (pots_i64 c0 = 0; c0 < cols; c0 += TILE_COLS) {
    const pots_i64 left = cols - c0;
    const pots_u32 ncols = left < (pots_i64)TILE_COLS ? (pots_u32)left : TILE_COLS;
    __syncthreads(); // the previous tile has been consumed
    if (SW == POTS_SW_P) {
      for (pots_u32 b = 0; b < B.eb; b++) {
        if (B.ent[b].phase != 1) continue;
        const pots_mprod_t *t = pots_bprod(B, b);
        const pots_i64 lf = t->p.cols - c0;
        if (lf <= 0) continue;
        const pots_u32 ncb = lf < (pots_i64)TILE_COLS ? (pots_u32)lf : TILE_COLS;
        const int pmode = pots_mode(&t->p);
        pots_gather_x_mode(pmode, &t->p, B.r0, B.nrows, B.rb, c0, ncb, R.x);
        pots_gather_r((const float *)t->r_addr, B.k, j0, nj, c0, ncb, R.v[0]);
        pots_scatter_x_mode(pmode, B.nrows, B.rb, ncb, R.x, sx + b * B.rb * POTS_XPAD);
        pots_scatter_r(nj, ncb, R.v[0], sy + b * K_CHUNK * POTS_YP);
      }
    } else {
      pots_gather_shared<SW>(B, vw, mode, c0, ncols, j0, nj, first, R);
      pots_scatter_shared<SW>(B, mode, ncols, nj, first, R, sx, sy, sb2, sb1);
    }
    __syncthreads();
    for (pots_u32 c = lane; c < ncols; c += 32u) {
      pots_u32 cur = 0xffffffffu; // the entry whose values are in yv
      float yv[K_CHUNK];
      float b2c = 0.0f, b1c = 0.0f;
      _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
        if ((live >> rr) & 1u) {
          const pots_u32 b = B.e_of[rr];
          const int guard = (B.guardm >> rr) & 1u;
          const int akind = (B.akind >> rr) & 1u;
          bool in = true;
          if (SW == POTS_SW_P) in = (pots_i64)c0 + (pots_i64)c < B.pcols[rr];
          if (in) {
            if (b != cur) {
              cur = b;
              const float *yb = sy + b * K_CHUNK * POTS_YP + c;
              _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++) yv[j] = j < nj ? yb[j * POTS_YP] : 0.0f;
              if (SW == POTS_SW_SHARED && first && akind) {
                b2c = sb2[b * TILE_COLS + c];
                b1c = guard ? sb1[b * TILE_COLS + c] : 0.0f;
              }
            }
            const float x = sx[((SW == POTS_SW_P ? b * B.rb : 0u) + B.row_of[rr]) * POTS_XPAD + c];
            if (SW == POTS_SW_P) {
              pots_upd_p(x, yv, 1, nj, L.acc[rr], first, guard, &L.ed[rr], &L.e2[rr]);
            } else if (SW == POTS_SW_BIAS) {
              pots_upd_bias(x, yv, 1, nj, L.acc[rr], first, guard, &L.ed[rr], &L.e2[rr]);
            } else if (akind) {
              pots_upd_a(x, yv, 1, nj, L.acc[rr], first, guard, b2c, b1c, &L.e1[rr], &L.e2[rr]);
            } else {
              pots_upd_b(x, yv, 1, nj, L.acc[rr], first, guard, &L.e1[rr], &L.e2[rr]);
            }
          }
        }
      }
    }
  }
}

// ---- the stores of a sweep ----

// The tree of one accumulator and lane 0's store.
static __device__ __forceinline__ void pots_store_f(const pots_mprod_t *t, const pots_mscr_t &sc,
                                                    double *rowp, pots_u32 phase, pots_i64 item,
                                                    pots_u32 step, float acc) {
  const float v = pots_tree_f(acc);
  if ((threadIdx.x & 31u) == 0u) pots_phase_store(t, &sc, rowp, phase, item, step, (double)v);
}
static __device__ __forceinline__ void pots_store_d(const pots_mprod_t *t, const pots_mscr_t &sc,
                                                    double *rowp, pots_u32 phase, pots_i64 item,
                                                    pots_u32 step, double acc) {
  const double v = pots_tree_d(acc);
  if ((threadIdx.x & 31u) == 0u) pots_phase_store(t, &sc, rowp, phase, item, step, v);
}

// Every live set's results of the sweep SW: the dots (steps step0 + stride * j), then the extras
// on `first`, in the order of today's step numbering. The bias stores its sum of squares before
// its dots.
template <int SW>
static __device__ __forceinline__ void pots_store_sets(const pots_batch_t &B, pots_u32 j0,
                                                       pots_u32 nj, int first,
                                                       const pots_lane_t &L) {
  const pots_u32 warp = threadIdx.x >> 5;
  const pots_u32 k = B.k;
  (void)warp;
  _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
    const pots_u32 live = SW == POTS_SW_P ? B.plive : B.live;
    if (!((live >> rr) & 1u)) continue;
    const pots_u32 b = B.e_of[rr];
    const pots_mentry_t en = B.ent[b];
    const pots_mprod_t *t = B.prods + en.prod;
    const pots_mscr_t sc = pots_mscr_make(t, B.works[en.prod].scratch);
    double *rowp = B.rows + (pots_u64)en.prod * (POTS_ROW_FIXED + k);
    const int guard = (B.guardm >> rr) & 1u;
    const pots_i64 item = SW == POTS_SW_BIAS ? t->b.rows : B.r0 + (pots_i64)B.row_of[rr];
    const pots_u32 phase = en.phase;
    if (SW == POTS_SW_BIAS) {
      if (first) pots_store_d(t, sc, rowp, 0, item, 0, L.ed[rr]);
      _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++)
        if (j < nj) pots_store_f(t, sc, rowp, 0, item, 1 + j0 + j, L.acc[rr][j]);
      if (first && guard) pots_store_f(t, sc, rowp, 0, item, k + 1, L.e2[rr]);
    } else if (SW == POTS_SW_P) {
      _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++)
        if (j < nj) pots_store_f(t, sc, rowp, 1, item, 2 * (j0 + j) + 1, L.acc[rr][j]);
      if (first) {
        pots_store_d(t, sc, rowp, 1, item, 2 * k + 1, L.ed[rr]);
        if (guard) pots_store_f(t, sc, rowp, 1, item, 2 * k + 3, L.e2[rr]);
      }
    } else if (phase == 0) {
      _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++)
        if (j < nj) pots_store_f(t, sc, rowp, 0, item, j0 + j, L.acc[rr][j]);
      if (first) {
        pots_store_f(t, sc, rowp, 0, item, k, L.e1[rr]);
        if (guard) pots_store_f(t, sc, rowp, 0, item, k + 1, L.e2[rr]);
      }
    } else {
      _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++)
        if (j < nj) pots_store_f(t, sc, rowp, 1, item, 2 * (j0 + j), L.acc[rr][j]);
      if (first) {
        pots_store_f(t, sc, rowp, 1, item, 2 * k, L.e1[rr]);
        if (guard) pots_store_f(t, sc, rowp, 1, item, 2 * k + 2, L.e2[rr]);
      }
    }
  }
}

// ---- a batch: every chunk, the shared sweep, then the entries' own P sweeps ----

static __device__ __forceinline__ void pots_run_batch(const pots_batch_t &B, int bias,
                                                      pots_lane_t &L, float *sx, float *sy,
                                                      float *sb2, float *sb1) {
  const pots_u32 k = B.k;
  const pots_mprod_t *t0 = pots_bprod(B, 0);
  const pots_mview_t *vw = bias ? &t0->bias : (B.ent[0].phase == 0 ? &t0->b : &t0->a);
  for (pots_u32 j0 = 0; j0 < k; j0 += K_CHUNK) {
    const pots_u32 nj = k - j0 < K_CHUNK ? k - j0 : K_CHUNK;
    const int first = j0 == 0;
    if (bias) {
      pots_sweep<POTS_SW_BIAS>(B, vw, j0, nj, first, L, sx, sy, sb2, sb1);
      pots_store_sets<POTS_SW_BIAS>(B, j0, nj, first, L);
      continue;
    }
    pots_sweep<POTS_SW_SHARED>(B, vw, j0, nj, first, L, sx, sy, sb2, sb1);
    pots_store_sets<POTS_SW_SHARED>(B, j0, nj, first, L);
    if (B.pmax > 0) {
      pots_sweep<POTS_SW_P>(B, vw, j0, nj, first, L, sx, sy, sb2, sb1);
      pots_store_sets<POTS_SW_P>(B, j0, nj, first, L);
    }
  }
}

// ---- phase 2: one warp per item, as FS37 ----

static __device__ __forceinline__ void pots_measure_item2(const pots_mprod_t *t,
                                                          const pots_mscr_t &sc, double *rowp,
                                                          pots_u64 item, pots_u32 lane) {
  const double part = pots_phase_partial(t, &sc, 2, (pots_i64)item, 0, lane);
  const double v = pots_tree_d(part);
  if (lane == 0) pots_phase_store(t, &sc, rowp, 2, (pots_i64)item, 0, v);
  __syncwarp();
}

// One level (FS38b): a block is a run of RB rows of one read group (a phase-2 group is one
// block, its warps taking the 5 + k items); the group's entries are run E_MAX at a time, each
// batch reading the shared view's tiles once for all its entries.
extern "C" __global__ void __launch_bounds__(POTS_MK_WARPS * 32, 2) pots_measure_level(
    pots_mlargs_t A) {
  __shared__ float s_x[ROWS_PER_BLOCK * POTS_XPAD];
  __shared__ float s_y[E_MAX * K_CHUNK * POTS_YP];
  __shared__ float s_b2[E_MAX * TILE_COLS];
  __shared__ float s_b1[E_MAX * TILE_COLS];
  pots_lane_t L;
  const pots_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  const pots_mprod_t *prods = (const pots_mprod_t *)A.prods;
  const pots_mwork_t *works = (const pots_mwork_t *)A.works;
  const pots_mread_t *reads = (const pots_mread_t *)A.reads;
  const pots_mentry_t *entries = (const pots_mentry_t *)A.entries;
  const pots_u64 *starts = (const pots_u64 *)A.starts;
  for (pots_u64 b = blockIdx.x; b < A.n_blocks; b += gridDim.x) {
    pots_u64 lo = 0, hi = A.n_groups; // invariant: starts[lo] <= b < starts[hi]
    while (hi - lo > 1) {
      const pots_u64 mid = lo + (hi - lo) / 2;
      if (starts[mid] <= b) lo = mid;
      else hi = mid;
    }
    const pots_mread_t rd = reads[lo];
    const pots_u64 rbk = b - starts[lo];
    double *rows = (double *)A.rows;
    if (entries[rd.entry0].phase == 2) {
      const pots_mentry_t en = entries[rd.entry0];
      const pots_mprod_t *t = prods + en.prod;
      const pots_mscr_t sc = pots_mscr_make(t, works[en.prod].scratch);
      double *rowp = rows + (pots_u64)en.prod * (POTS_ROW_FIXED + A.k);
      for (pots_u64 it = warp; it < rd.count; it += POTS_MK_WARPS)
        pots_measure_item2(t, sc, rowp, it, lane);
    } else {
      const pots_u32 ne = rd.n_entries;
      const pots_u32 rb = ROWS_PER_BLOCK / (ne < E_MAX ? ne : E_MAX); // RB
      const pots_u64 r0 = rbk * rb;
      const pots_u64 rem = rd.count - r0;
      const pots_u32 nrows = rem < (pots_u64)rb ? (pots_u32)rem : rb;
      for (pots_u32 e0 = 0; e0 < ne; e0 += E_MAX) {
        const pots_u32 eb = ne - e0 < E_MAX ? ne - e0 : E_MAX;
        const pots_mentry_t *ent = entries + rd.entry0 + e0;
        const pots_batch_t B = pots_make_batch(prods, works, ent, rows, A.k, eb, rb,
                                               (pots_i64)r0, nrows, 0);
        if (B.nld > 0) pots_run_batch(B, 0, L, s_x, s_y, s_b2, s_b1);
        const pots_mprod_t *t0 = prods + ent[0].prod;
        if (ne == 1u && ent[0].phase == 0 && (t0->flags & POTS_M_BIASED) != 0 &&
            (pots_i64)r0 + (pots_i64)nrows > t0->b.rows) { // the bias item
          const pots_batch_t Bb = pots_make_batch(prods, works, ent, rows, A.k, 1u, rb,
                                                  (pots_i64)r0, nrows, 1);
          pots_run_batch(Bb, 1, L, s_x, s_y, s_b2, s_b1);
        }
      }
    }
    __syncthreads(); // the shared tiles are reused by the block's next row block
  }
}
