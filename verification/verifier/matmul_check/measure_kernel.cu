// The fused measuring kernel (FS37, decision O2; one sweep per row, FS37b). Compiled by NVRTC at
// run time (measure_kernel.py) with measure_core.h and measure_phases.h passed in memory, with
// --fmad=false. Every decision about what is computed lives in those headers; this file maps
// the work onto blocks, warps, lanes, shared-memory tiles and shuffles. One launch per level of
// the read plan (pots_measure_level).
//
// A block takes ROWS_PER_BLOCK consecutive rows of one read group's view, each warp
// ROWS_PER_BLOCK / POTS_MK_WARPS of them, and walks the columns in tiles of TILE_COLS. A tile is
// loaded coalesced into shared memory (widened to fp32) with the vector entries it needs; lane p
// then takes the tile's columns p, p+32, ... in increasing order for each of its rows, so every
// accumulator sees exactly the operations of its step's partial in measure_core.h (the per-element
// updates are measure_phases.h's, shared with the host emulation). A column past the view's end
// is skipped, never zero-filled. After the last tile the warp's halving tree reduces each
// accumulator and lane 0 stores through pots_phase_store. The geometry (the four macros below) is
// passed as -D options and never changes a bit.
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
#include "measure_core.h"
#include "measure_phases.h"

#define POTS_FULL 0xffffffffu
#define POTS_NT (POTS_MK_WARPS * 32u)
#define POTS_RPW (ROWS_PER_BLOCK / POTS_MK_WARPS) /* rows per warp */
#define POTS_XPAD (TILE_COLS + 1u)                /* the x tile's row pitch: no bank conflicts */

static_assert(TILE_COLS % 32u == 0u && TILE_COLS >= 32u, "TILE_COLS must be a multiple of 32");
static_assert(ROWS_PER_BLOCK % POTS_MK_WARPS == 0u && ROWS_PER_BLOCK >= POTS_MK_WARPS,
              "ROWS_PER_BLOCK must be a multiple of the warps");
static_assert(K_CHUNK >= 1u, "K_CHUNK must be at least 1");

enum { POTS_SW_B = 0, POTS_SW_BIAS = 1, POTS_SW_A = 2, POTS_SW_P = 3 };

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

// ---- loads, specialised by dtype and layout ----

// DT 0: fp32, 1: bf16 (the 16 bits shifted left by 16, as pots_mload). LAY 0: row-major
// (col_stride == 1), 1: transposed (row_stride == 1), 2: any strides.
template <int DT>
static __device__ __forceinline__ float pots_ld(pots_u64 addr, pots_i64 at) {
  if (DT == 1) return pots_bits_float((pots_u32)((const pots_u16 *)addr)[at] << 16);
  return ((const float *)addr)[at];
}

// Rows r0 .. r0+nrows-1 and columns c0 .. c0+ncols-1 of view m into sx[r * POTS_XPAD + cc],
// the block's threads taking consecutive elements along the view's contiguous dimension.
template <int DT, int LAY>
static __device__ __forceinline__ void pots_load_x(const pots_mview_t *m, pots_i64 r0,
                                                   pots_u32 nrows, pots_i64 c0, pots_u32 ncols,
                                                   float *sx) {
  const pots_i64 rs = m->row_stride, cs = m->col_stride;
  const pots_u64 addr = m->addr;
  if (LAY == 1) {
    const pots_u32 tot = ROWS_PER_BLOCK * ncols;
    for (pots_u32 e = threadIdx.x; e < tot; e += POTS_NT) {
      const pots_u32 r = e % ROWS_PER_BLOCK, cc = e / ROWS_PER_BLOCK;
      if (r < nrows) sx[r * POTS_XPAD + cc] = pots_ld<DT>(addr, (r0 + r) + (c0 + cc) * cs);
    }
  } else {
    const pots_u32 tot = nrows * TILE_COLS;
    for (pots_u32 e = threadIdx.x; e < tot; e += POTS_NT) {
      const pots_u32 r = e / TILE_COLS, cc = e % TILE_COLS;
      if (cc < ncols) {
        const pots_i64 at = (r0 + r) * rs + (LAY == 0 ? (c0 + cc) : (c0 + cc) * cs);
        sx[r * POTS_XPAD + cc] = pots_ld<DT>(addr, at);
      }
    }
  }
}

static __device__ __forceinline__ int pots_mode(const pots_mview_t *m) {
  const int lay = m->col_stride == 1 ? 0 : (m->row_stride == 1 ? 1 : 2);
  return (m->dtype == POTS_DT_BF16 ? 3 : 0) + lay;
}

// The switch on (dtype, layout), made once per tile outside every compute loop.
static __device__ __forceinline__ void pots_load_x_mode(int mode, const pots_mview_t *m,
                                                        pots_i64 r0, pots_u32 nrows,
                                                        pots_i64 c0, pots_u32 ncols, float *sx) {
  switch (mode) {
    case 0: pots_load_x<0, 0>(m, r0, nrows, c0, ncols, sx); break;
    case 1: pots_load_x<0, 1>(m, r0, nrows, c0, ncols, sx); break;
    case 2: pots_load_x<0, 2>(m, r0, nrows, c0, ncols, sx); break;
    case 3: pots_load_x<1, 0>(m, r0, nrows, c0, ncols, sx); break;
    case 4: pots_load_x<1, 1>(m, r0, nrows, c0, ncols, sx); break;
    default: pots_load_x<1, 2>(m, r0, nrows, c0, ncols, sx); break;
  }
}

// ---- one sweep ----

typedef struct {
  float acc[POTS_RPW][K_CHUNK]; // lane p's accumulator of challenge column j of its row rr
  float e1[POTS_RPW];           // b2 | u
  float e2[POTS_RPW];           // b1 | nu | p1
  double ed[POTS_RPW];          // the float64 sum of squares: the bias's, or pp
} pots_lane_t;

// One pass over rows r0 .. r0+nrows-1 of view m (<= ROWS_PER_BLOCK rows) for the challenge
// columns j0 .. j0+nj-1 (`first`: also the extras), into the lanes' accumulators. Every thread
// of the block calls it with the same arguments (it has barriers).
template <int KIND>
static __device__ __forceinline__ void pots_sweep(const pots_mprod_t *t, const pots_mscr_t &sc,
                                                  const pots_mview_t *m, pots_i64 r0,
                                                  pots_u32 nrows, pots_u32 j0, pots_u32 nj,
                                                  int first, pots_lane_t &L, float *sx,
                                                  float *sy, float *sb2, float *sb1) {
  const pots_u32 k = t->k, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  const pots_i64 q = t->a.cols;
  const int guard = (t->flags & POTS_M_GUARD) != 0;
  const float *r = (const float *)t->r_addr;
  const int mode = pots_mode(m);
  _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
    _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++) L.acc[rr][j] = 0.0f;
    L.e1[rr] = 0.0f;
    L.e2[rr] = 0.0f;
    L.ed[rr] = 0.0;
  }
  for (pots_i64 c0 = 0; c0 < m->cols; c0 += TILE_COLS) {
    const pots_i64 left = m->cols - c0;
    const pots_u32 ncols = left < (pots_i64)TILE_COLS ? (pots_u32)left : TILE_COLS;
    __syncthreads(); // the previous tile has been consumed
    pots_load_x_mode(mode, m, r0, nrows, c0, ncols, sx);
    if (KIND == POTS_SW_A) {
      for (pots_u32 e = threadIdx.x; e < nj * TILE_COLS; e += POTS_NT) {
        const pots_u32 j = e / TILE_COLS, cc = e % TILE_COLS;
        if (cc < ncols) sy[e] = sc.tt[(pots_u64)(j0 + j) * (pots_u64)q + (pots_u64)(c0 + cc)];
      }
      if (first) {
        for (pots_u32 cc = threadIdx.x; cc < ncols; cc += POTS_NT) {
          sb2[cc] = sc.b2[c0 + cc];
          if (guard) sb1[cc] = sc.b1[c0 + cc];
        }
      }
    } else {
      const pots_u32 tot = ncols * nj;
      for (pots_u32 e = threadIdx.x; e < tot; e += POTS_NT) {
        const pots_u32 cc = e / nj, j = e - cc * nj;
        sy[j * TILE_COLS + cc] = r[(pots_u64)(c0 + cc) * k + j0 + j];
      }
    }
    __syncthreads();
    _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
      const pots_u32 row = warp * POTS_RPW + rr;
      if (row < nrows) {
        const float *xr = sx + row * POTS_XPAD;
        for (pots_u32 c = lane; c < ncols; c += 32u) {
          const float x = xr[c];
          if (KIND == POTS_SW_B)
            pots_upd_b(x, sy + c, TILE_COLS, nj, L.acc[rr], first, guard, &L.e1[rr], &L.e2[rr]);
          else if (KIND == POTS_SW_BIAS)
            pots_upd_bias(x, sy + c, TILE_COLS, nj, L.acc[rr], first, guard, &L.ed[rr],
                          &L.e2[rr]);
          else if (KIND == POTS_SW_A)
            pots_upd_a(x, sy + c, TILE_COLS, nj, L.acc[rr], first, guard, first ? sb2[c] : 0.0f,
                       (first && guard) ? sb1[c] : 0.0f, &L.e1[rr], &L.e2[rr]);
          else
            pots_upd_p(x, sy + c, TILE_COLS, nj, L.acc[rr], first, guard, &L.ed[rr], &L.e2[rr]);
        }
      }
    }
  }
}

// Reduce and store the dot accumulators of a sweep: for each of the warp's rows within nrows,
// each chunk column j, the tree and lane 0's store of step step0 + stride * j for item
// r0 + row. A warp's rows past nrows do nothing.
static __device__ __forceinline__ void pots_store_dots(const pots_mprod_t *t,
                                                       const pots_mscr_t &sc, double *rowp,
                                                       pots_u32 phase, pots_i64 item0,
                                                       pots_u32 nrows, pots_u32 nj,
                                                       pots_u32 step0, pots_u32 stride,
                                                       const pots_lane_t &L) {
  const pots_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
    const pots_u32 row = warp * POTS_RPW + rr;
    if (row < nrows) {
      _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++) {
        if (j < nj) {
          const float v = pots_tree_f(L.acc[rr][j]);
          if (lane == 0)
            pots_phase_store(t, &sc, rowp, phase, item0 + row, step0 + stride * j, (double)v);
        }
      }
    }
  }
}

// The extras: kind 0 reduces e1 (float32), 1 e2 (float32), 2 ed (float64), each stored at
// `step`.
static __device__ __forceinline__ void pots_store_extra(const pots_mprod_t *t,
                                                        const pots_mscr_t &sc, double *rowp,
                                                        pots_u32 phase, pots_i64 item0,
                                                        pots_u32 nrows, pots_u32 step, int which,
                                                        const pots_lane_t &L) {
  const pots_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
    const pots_u32 row = warp * POTS_RPW + rr;
    if (row < nrows) {
      double v;
      if (which == 2) v = pots_tree_d(L.ed[rr]);
      else v = (double)pots_tree_f(which == 0 ? L.e1[rr] : L.e2[rr]);
      if (lane == 0) pots_phase_store(t, &sc, rowp, phase, item0 + row, step, v);
    }
  }
}

// ---- one entry (a node of the group) over the block's rows ----

static __device__ __forceinline__ void pots_run_entry(const pots_mprod_t *t,
                                                      const pots_mscr_t &sc, double *rowp,
                                                      pots_u32 phase, pots_i64 r0,
                                                      pots_u32 nrows, pots_lane_t &L, float *sx,
                                                      float *sy, float *sb2, float *sb1) {
  const pots_u32 k = t->k;
  const int guard = (t->flags & POTS_M_GUARD) != 0, biased = (t->flags & POTS_M_BIASED) != 0;
  if (phase == 0) {
    const pots_i64 qb = t->b.rows; // B's rows; item qb of a biased product is the bias
    const pots_u32 nb = r0 >= qb ? 0u : (qb - r0 < (pots_i64)nrows ? (pots_u32)(qb - r0) : nrows);
    if (nb > 0) {
      for (pots_u32 j0 = 0; j0 < k; j0 += K_CHUNK) {
        const pots_u32 nj = k - j0 < K_CHUNK ? k - j0 : K_CHUNK;
        const int first = j0 == 0;
        pots_sweep<POTS_SW_B>(t, sc, &t->b, r0, nb, j0, nj, first, L, sx, sy, sb2, sb1);
        pots_store_dots(t, sc, rowp, 0, r0, nb, nj, j0, 1, L);
        if (first) {
          pots_store_extra(t, sc, rowp, 0, r0, nb, k, 0, L);
          if (guard) pots_store_extra(t, sc, rowp, 0, r0, nb, k + 1, 1, L);
        }
      }
    }
    if (biased && r0 + (pots_i64)nrows > qb) { // the bias item: one row of the bias view
      for (pots_u32 j0 = 0; j0 < k; j0 += K_CHUNK) {
        const pots_u32 nj = k - j0 < K_CHUNK ? k - j0 : K_CHUNK;
        const int first = j0 == 0;
        pots_sweep<POTS_SW_BIAS>(t, sc, &t->bias, 0, 1u, j0, nj, first, L, sx, sy, sb2, sb1);
        if (first) pots_store_extra(t, sc, rowp, 0, qb, 1u, 0, 2, L);
        pots_store_dots(t, sc, rowp, 0, qb, 1u, nj, 1 + j0, 1, L);
        if (first && guard) pots_store_extra(t, sc, rowp, 0, qb, 1u, k + 1, 1, L);
      }
    }
    return;
  }
  // phase 1: a row of A, then the same row of P, per chunk. The stores of s_j (step 2j) come
  // before those of pi_j (step 2j + 1, which reads d and beta) for every j.
  for (pots_u32 j0 = 0; j0 < k; j0 += K_CHUNK) {
    const pots_u32 nj = k - j0 < K_CHUNK ? k - j0 : K_CHUNK;
    const int first = j0 == 0;
    pots_sweep<POTS_SW_A>(t, sc, &t->a, r0, nrows, j0, nj, first, L, sx, sy, sb2, sb1);
    pots_store_dots(t, sc, rowp, 1, r0, nrows, nj, 2 * j0, 2, L);
    if (first) {
      pots_store_extra(t, sc, rowp, 1, r0, nrows, 2 * k, 0, L);
      if (guard) pots_store_extra(t, sc, rowp, 1, r0, nrows, 2 * k + 2, 1, L);
    }
    __syncwarp();
    pots_sweep<POTS_SW_P>(t, sc, &t->p, r0, nrows, j0, nj, first, L, sx, sy, sb2, sb1);
    pots_store_dots(t, sc, rowp, 1, r0, nrows, nj, 2 * j0 + 1, 2, L);
    if (first) {
      pots_store_extra(t, sc, rowp, 1, r0, nrows, 2 * k + 1, 2, L);
      if (guard) pots_store_extra(t, sc, rowp, 1, r0, nrows, 2 * k + 3, 1, L);
    }
    __syncwarp();
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

// One level (FS38b): a block is a run of ROWS_PER_BLOCK rows of one read group (a phase-2 group
// is one block, its warps taking the 5 + k items); every entry (product, phase) of the group
// runs its own rows in turn, so the shared view's tiles are read again by each entry (from L2).
extern "C" __global__ void __launch_bounds__(POTS_MK_WARPS * 32) pots_measure_level(
    pots_mlargs_t A) {
  __shared__ float s_x[ROWS_PER_BLOCK * POTS_XPAD];
  __shared__ float s_y[K_CHUNK * TILE_COLS];
  __shared__ float s_b2[TILE_COLS];
  __shared__ float s_b1[TILE_COLS];
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
    const pots_u64 rb = b - starts[lo];
    if (entries[rd.entry0].phase == 2) {
      const pots_mentry_t en = entries[rd.entry0];
      const pots_mprod_t *t = prods + en.prod;
      const pots_mscr_t sc = pots_mscr_make(t, works[en.prod].scratch);
      double *rowp = (double *)A.rows + (pots_u64)en.prod * (POTS_ROW_FIXED + A.k);
      for (pots_u64 it = warp; it < rd.count; it += POTS_MK_WARPS)
        pots_measure_item2(t, sc, rowp, it, lane);
    } else {
      const pots_u64 r0 = rb * ROWS_PER_BLOCK;
      const pots_u64 rem = rd.count - r0;
      const pots_u32 nrows = rem < (pots_u64)ROWS_PER_BLOCK ? (pots_u32)rem : ROWS_PER_BLOCK;
      for (pots_u32 e = 0; e < rd.n_entries; e++) {
        const pots_mentry_t en = entries[rd.entry0 + e];
        const pots_mprod_t *t = prods + en.prod;
        const pots_mscr_t sc = pots_mscr_make(t, works[en.prod].scratch);
        double *rowp = (double *)A.rows + (pots_u64)en.prod * (POTS_ROW_FIXED + A.k);
        pots_run_entry(t, sc, rowp, en.phase, (pots_i64)r0, nrows, L, s_x, s_y, s_b2, s_b1);
      }
    }
    __syncthreads(); // the shared tiles are reused by the block's next row block
  }
}
