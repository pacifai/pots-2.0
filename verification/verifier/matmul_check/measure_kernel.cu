// The fused measuring kernel (FS37, decision O2; one sweep per row, FS37b round 3). Compiled by
// NVRTC at run time (measure_kernel.py) with measure_core.h and measure_phases.h passed in memory,
// with --fmad=false. Every decision about what is computed lives in those headers; this file maps
// the work onto blocks, warps, lanes, shared-memory tiles and shuffles. One launch per level of
// the read plan (pots_measure_level).
//
// A block takes ROWS_PER_BLOCK consecutive rows of one read group's view (the table the host
// builds gives it its group and its row block: nothing is searched); warp w owns rows w*RPW ..
// w*RPW + RPW - 1 of the block (RPW = ROWS_PER_BLOCK / 8). The group's entries run one after
// another over the same rows. For each entry the block copies the entry's product descriptor
// into shared memory, then makes one sweep per row kind: a B row (phase 0), or an A row and then
// a P row (phase 1), plus the bias item of a biased product. A sweep walks the view's columns in
// tiles of TILE_COLS: the rows' tile goes coalesced into shared memory (widened to fp32) with the
// tile's vector entries, and lane p takes the tile's columns p, p+32, ... in increasing order.
// For each column the warp's rows go inside, with the vector values of the column in registers
// once. Each accumulator still sees exactly the operations of its step's partial in
// measure_core.h (the per-element updates are measure_phases.h's, shared with the host
// emulation). A column past the view's end is skipped, never zero-filled.
//
// Stores: after a sweep chunk every lane holds NV fp32 accumulators (the warp's rows x the
// chunk's dots, then each row's e1, e2); a transposed butterfly over the 32 lanes reduces them
// all at once with the halving tree's pairings (the lower lane's value on the left), leaving
// accumulator m reduced in lane m. The values go through shared memory and consecutive threads
// write them on consecutive rows. An A row's dots s_j stay in shared memory until the same
// chunk's P sweep, which forms d_j from them (as pots_phase_store's step 2j+1 does), so s_j never
// goes to global scratch. The float64 sums keep the lane-0 tree. The bias item and phase 2 keep
// the lane-0 path.
//
// Every tile load is two passes: the global loads go to a register array of compile-time size
// (unrolled), then the shared stores; each sweep prefetches the next tile's global loads before
// computing the current one. K_CHUNK is min(k, cap) per build, and __launch_bounds__(256, 2) asks
// for at most 128 registers. The geometry (the macros below) is passed as -D options and never
// changes a bit.
#ifndef POTS_MK_WARPS
#define POTS_MK_WARPS 8u /* warps per block */
#endif
#ifndef ROWS_PER_BLOCK
#define ROWS_PER_BLOCK 16u /* rows of a view per block */
#endif
#ifndef TILE_COLS
#define TILE_COLS 256u /* columns per tile, a multiple of 32 */
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
#define POTS_YP (TILE_COLS + 1u)                  /* the vector tile's column pitch */
#define POTS_SLOTS (K_CHUNK + 2u)                 /* staged values per row: the dots, e1, e2 */
#define POTS_NP ((POTS_RPW * POTS_SLOTS + 31u) / 32u) /* butterfly passes of 32 values */

static_assert(TILE_COLS % 32u == 0u && TILE_COLS >= 32u, "TILE_COLS must be a multiple of 32");
static_assert(ROWS_PER_BLOCK % POTS_MK_WARPS == 0u && ROWS_PER_BLOCK >= POTS_MK_WARPS,
              "ROWS_PER_BLOCK must be a multiple of the warps");
static_assert(K_CHUNK >= 1u, "K_CHUNK must be at least 1");

enum { POTS_SW_B = 0, POTS_SW_A = 1, POTS_SW_P = 2, POTS_SW_BIAS = 3 };

// One level's launch, passed by value.
typedef struct {
  pots_u64 prods;    // pots_mprod_t[n_prod]
  pots_u64 works;    // pots_mwork_t[n_prod + 1] (the scratch addresses)
  pots_u64 rows;     // float64 [n_prod, ROW_FIXED + k]
  pots_u64 reads;    // pots_mread_t[n_groups + 1], the level's groups and its sentinel
  pots_u64 entries;  // pots_mentry_t
  pots_u64 blocks;   // pots_mblk_t[n_blocks]: each block's group and row block
  pots_u64 n_groups;
  pots_u64 n_blocks; // row blocks of this level
  pots_u32 k;
  pots_u32 pad;
} pots_mlargs_t;

// One block of a level: its group (an index into the level's reads) and its row block within the
// group (a phase-2 group has one block, 0).
typedef struct {
  pots_u32 group;
  pots_u32 rbk;
} pots_mblk_t;

// The block's copy of the current entry's product, in shared memory.
typedef struct {
  pots_mprod_t t;
  pots_u64 scratch; // the product's scratch base
  pots_u32 prod;    // the entry's product
  pots_u32 phase;   // and phase
} pots_sdesc_t;

// ---- the halving trees (the float64 sums, the bias item) ----
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

// ---- the transposed butterfly ----
// 32 values per lane, x[m] being the lane's partial of accumulator m. One stage (OFF = 16, 8, 4,
// 2, 1) leaves a lane 2*OFF values and keeps OFF of them: the lower OFF if its lane bit OFF is
// clear, else the upper OFF; it sends the others to the lane OFF away and adds what it gets to
// what it keeps with the lower lane's value on the left, which is the tree's s[i] + s[i + OFF].
// After the five stages lane m holds accumulator m reduced exactly as pots_tree_f reduces it.
// Every index is a compile-time constant after unrolling.
template <int OFF>
static __device__ __forceinline__ void pots_bfly_stage(const float (&in)[2 * OFF],
                                                       float (&out)[OFF], pots_u32 lane) {
  const bool up = (lane & (pots_u32)OFF) != 0u;
  _Pragma("unroll") for (int i = 0; i < OFF; i++) {
    const float lo = in[i], hi = in[i + OFF];
    const float send = up ? lo : hi;
    const float keep = up ? hi : lo;
    const float recv = __shfl_xor_sync(POTS_FULL, send, OFF);
    out[i] = up ? recv + keep : keep + recv;
  }
}
static __device__ __forceinline__ float pots_butterfly(const float (&x)[32], pots_u32 lane) {
  float a[16], b[8], c[4], d[2], e[1];
  pots_bfly_stage<16>(x, a, lane);
  pots_bfly_stage<8>(a, b, lane);
  pots_bfly_stage<4>(b, c, lane);
  pots_bfly_stage<2>(c, d, lane);
  pots_bfly_stage<1>(d, e, lane);
  return e[0];
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

// The vector tile of a sweep, K_CHUNK rows of pitch POTS_YP: row j, tile column cc at
// sy[j * POTS_YP + cc]. Rows j >= nj are zero (a chunk shorter than K_CHUNK), so the compute loop
// reads all K_CHUNK rows unpredicated. R's columns j0 .. j0+nj-1 at tile columns 0 .. ncols-1
// (element e: tile column e / K_CHUNK, row e % K_CHUNK: R's entries are consecutive in memory),
// or an A sweep's stored t rows (element e: row e / TILE_COLS, tile column e % TILE_COLS).
static __device__ __forceinline__ void pots_gather_r(const float *__restrict__ r, pots_u32 k,
                                                     pots_u32 j0, pots_u32 nj, pots_i64 c0,
                                                     pots_u32 ncols, float (&v)[POTS_NPV]) {
  const pots_u32 tot = ncols * K_CHUNK;
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPV; i++) {
    const pots_u32 e = threadIdx.x + i * POTS_NT;
    const pots_u32 cc = e / K_CHUNK, j = e - cc * K_CHUNK;
    v[i] = 0.0f;
    if (e < tot && j < nj) v[i] = r[(pots_u64)(c0 + cc) * k + j0 + j];
  }
}
static __device__ __forceinline__ void pots_scatter_r(pots_u32 ncols, const float (&v)[POTS_NPV],
                                                      float *__restrict__ sy) {
  const pots_u32 tot = ncols * K_CHUNK;
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPV; i++) {
    const pots_u32 e = threadIdx.x + i * POTS_NT;
    const pots_u32 cc = e / K_CHUNK, j = e - cc * K_CHUNK;
    if (e < tot) sy[j * POTS_YP + cc] = v[i];
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
static __device__ __forceinline__ void pots_scatter_t(pots_u32 ncols, const float (&v)[POTS_NPV],
                                                      float *__restrict__ sy) {
  _Pragma("unroll") for (pots_u32 i = 0; i < POTS_NPV; i++) {
    const pots_u32 e = threadIdx.x + i * POTS_NT;
    const pots_u32 j = e / TILE_COLS, cc = e % TILE_COLS;
    if (j < K_CHUNK && cc < ncols) sy[j * POTS_YP + cc] = v[i];
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

// ---- one sweep ----

// A lane's accumulators for the warp's RPW rows: the chunk's dots, e1 (b2 | u), e2 (b1 | nu |
// p1 | the bias's beta1) and the float64 sum of squares (pp, or the bias's). Every index is a
// compile-time constant in every loop.
typedef struct {
  float acc[POTS_RPW][K_CHUNK];
  float e1[POTS_RPW];
  float e2[POTS_RPW];
  double ed[POTS_RPW];
} pots_lane_t;

// The registers a tile's global loads land in: the x tile, the vector tile and (A) b2, b1.
typedef struct {
  float x[POTS_NPX];
  float v[POTS_NPV];
  float b2[POTS_NPB];
  float b1[POTS_NPB];
} pots_tile_regs_t;

static __device__ __forceinline__ const pots_mview_t *pots_sw_view(const pots_mprod_t *t, int sw) {
  return sw == POTS_SW_B ? &t->b : (sw == POTS_SW_A ? &t->a : (sw == POTS_SW_P ? &t->p : &t->bias));
}

// Pass 1 of a tile of the sweep SW's view: rows r0 .. r0+nrows-1 (the bias: its one row), columns
// c0 .. c0+ncols-1, with the tile's vector entries (A: the stored t, and on `ex` b2 and b1).
template <int SW>
static __device__ __forceinline__ void pots_gather_tile(const pots_sdesc_t &D, int mode,
                                                        pots_i64 r0, pots_u32 nrows,
                                                        pots_i64 c0, pots_u32 ncols, pots_u32 j0,
                                                        pots_u32 nj, int ex,
                                                        pots_tile_regs_t &R) {
  const pots_mprod_t *t = &D.t;
  pots_gather_x_mode(mode, pots_sw_view(t, SW), r0, nrows, SW == POTS_SW_BIAS ? 1u : ROWS_PER_BLOCK,
                     c0, ncols, R.x);
  if (SW == POTS_SW_A) {
    const pots_mscr_t sc = pots_mscr_make(t, D.scratch);
    pots_gather_t(sc.tt, t->a.cols, j0, nj, c0, ncols, R.v);
    if (ex >= 1) pots_gather_b(sc.b2, sc.b1, ex == 2, c0, ncols, R.b2, R.b1);
  } else {
    pots_gather_r((const float *)t->r_addr, t->k, j0, nj, c0, ncols, R.v);
  }
}

// Pass 2 of that tile.
template <int SW>
static __device__ __forceinline__ void pots_scatter_tile(int mode, pots_u32 nrows,
                                                         pots_u32 ncols, int ex,
                                                         const pots_tile_regs_t &R,
                                                         float *__restrict__ sx,
                                                         float *__restrict__ sy,
                                                         float *__restrict__ sb2,
                                                         float *__restrict__ sb1) {
  pots_scatter_x_mode(mode, nrows, SW == POTS_SW_BIAS ? 1u : ROWS_PER_BLOCK, ncols, R.x, sx);
  if (SW == POTS_SW_A) {
    pots_scatter_t(ncols, R.v, sy);
    if (ex >= 1) pots_scatter_b(ex == 2, ncols, R.b2, R.b1, sb2, sb1);
  } else {
    pots_scatter_r(ncols, R.v, sy);
  }
}

// The compute of one tile. EX: 0 the dots only, 1 also the extras, 2 the extras with the guard
// (block-uniform, so each is its own branch-free loop). The warp's RPW rows go inside the lane's
// columns, with the column's vector values in registers once; rows past the block's `nlive` do
// nothing.
template <int SW, int EX>
static __device__ __forceinline__ void pots_compute(pots_u32 ncols, pots_u32 nlive,
                                                    pots_lane_t &L, const float *__restrict__ sx,
                                                    const float *__restrict__ sy,
                                                    const float *__restrict__ sb2,
                                                    const float *__restrict__ sb1) {
  const pots_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  for (pots_u32 c = lane; c < ncols; c += 32u) {
    float yv[K_CHUNK];
    _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++) yv[j] = sy[j * POTS_YP + c];
    float b2c = 0.0f, b1c = 0.0f;
    if (SW == POTS_SW_A && EX >= 1) {
      b2c = sb2[c];
      if (EX == 2) b1c = sb1[c];
    }
    _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
      const pots_u32 row = warp * POTS_RPW + rr;
      if (row < nlive) {
        const float x = sx[row * POTS_XPAD + c];
        if (SW == POTS_SW_B)
          pots_upd_b(x, yv, 1, K_CHUNK, L.acc[rr], EX >= 1, EX == 2, &L.e1[rr], &L.e2[rr]);
        else if (SW == POTS_SW_A)
          pots_upd_a(x, yv, 1, K_CHUNK, L.acc[rr], EX >= 1, EX == 2, b2c, b1c, &L.e1[rr],
                     &L.e2[rr]);
        else if (SW == POTS_SW_P)
          pots_upd_p(x, yv, 1, K_CHUNK, L.acc[rr], EX >= 1, EX == 2, &L.ed[rr], &L.e2[rr]);
        else
          pots_upd_bias(x, yv, 1, K_CHUNK, L.acc[rr], EX >= 1, EX == 2, &L.ed[rr], &L.e2[rr]);
      }
    }
  }
}

// One pass over the sweep SW's view for the challenge columns j0 .. j0+nj-1 (ex: also the extras;
// 2: with the guard), into the lanes' accumulators. `nlive` rows from row r0 (the bias: its one
// row). Every thread of the block calls it with the same arguments (it has barriers).
template <int SW>
static __device__ __forceinline__ void pots_sweep(const pots_sdesc_t &D, pots_i64 r0,
                                                  pots_u32 nlive, pots_u32 j0, pots_u32 nj,
                                                  int ex, pots_lane_t &L, float *__restrict__ sx,
                                                  float *__restrict__ sy,
                                                  float *__restrict__ sb2,
                                                  float *__restrict__ sb1) {
  _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
    _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++) L.acc[rr][j] = 0.0f;
    L.e1[rr] = 0.0f;
    L.e2[rr] = 0.0f;
    L.ed[rr] = 0.0;
  }
  const pots_mview_t *vw = pots_sw_view(&D.t, SW);
  const pots_i64 cols = vw->cols;
  const int mode = pots_mode(vw);
  pots_tile_regs_t R;
  // Prefetch: tile 0's global loads are issued here, and each later tile's right after the
  // barrier that follows the scatter of the one before, so they are in flight while the lanes
  // compute. Only the order of the loads changes, never a value.
  if (cols > 0) {
    const pots_i64 l0 = cols < (pots_i64)TILE_COLS ? cols : (pots_i64)TILE_COLS;
    pots_gather_tile<SW>(D, mode, r0, nlive, 0, (pots_u32)l0, j0, nj, ex, R);
  }
  for (pots_i64 c0 = 0; c0 < cols; c0 += TILE_COLS) {
    const pots_i64 left = cols - c0;
    const pots_u32 ncols = left < (pots_i64)TILE_COLS ? (pots_u32)left : TILE_COLS;
    __syncthreads(); // the previous tile has been consumed
    pots_scatter_tile<SW>(mode, nlive, ncols, ex, R, sx, sy, sb2, sb1);
    __syncthreads();
    if (c0 + (pots_i64)TILE_COLS < cols) {
      const pots_i64 ln = cols - (c0 + (pots_i64)TILE_COLS);
      pots_gather_tile<SW>(D, mode, r0, nlive, c0 + (pots_i64)TILE_COLS,
                           ln < (pots_i64)TILE_COLS ? (pots_u32)ln : TILE_COLS, j0, nj, ex, R);
    }
    switch (ex) {
      case 0: pots_compute<SW, 0>(ncols, nlive, L, sx, sy, sb2, sb1); break;
      case 1: pots_compute<SW, 1>(ncols, nlive, L, sx, sy, sb2, sb1); break;
      default: pots_compute<SW, 2>(ncols, nlive, L, sx, sy, sb2, sb1); break;
    }
  }
}

// ---- the stores of a sweep ----

// Every accumulator of a sweep through the butterfly, into the staging array
// stg[row * POTS_SLOTS + slot] (slot j < K_CHUNK: dot j, K_CHUNK: e1, K_CHUNK + 1: e2), and the
// float64 sums (the rows' ed) through the lane-0 tree into stdbl[row] when `dbl`. Values of
// unused slots are staged too and never read.
static __device__ __forceinline__ void pots_stage(const pots_lane_t &L, int dbl,
                                                  float *__restrict__ stg,
                                                  double *__restrict__ stdbl) {
  const pots_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  _Pragma("unroll") for (pots_u32 p = 0; p < POTS_NP; p++) {
    float x[32];
    _Pragma("unroll") for (pots_u32 i = 0; i < 32; i++) {
      const pots_u32 m = p * 32u + i, rr = m / POTS_SLOTS, sl = m % POTS_SLOTS;
      float v = 0.0f;
      if (rr < POTS_RPW) v = sl < K_CHUNK ? L.acc[rr][sl] : (sl == K_CHUNK ? L.e1[rr] : L.e2[rr]);
      x[i] = v;
    }
    const float out = pots_butterfly(x, lane);
    const pots_u32 m = p * 32u + lane;
    if (m < POTS_RPW * POTS_SLOTS) stg[warp * POTS_RPW * POTS_SLOTS + m] = out;
  }
  if (dbl) {
    _Pragma("unroll") for (pots_u32 rr = 0; rr < POTS_RPW; rr++) {
      const double v = pots_tree_d(L.ed[rr]);
      if (lane == 0u) stdbl[warp * POTS_RPW + rr] = v;
    }
  }
}

// The writes of the staged values of a B sweep (phase 0), by consecutive threads on consecutive
// rows: the dots (steps j0 + j), then on `ex` b2 (step k) and with the guard b1 (step k + 1).
static __device__ __forceinline__ void pots_flush_b(const pots_sdesc_t &D, pots_i64 r0,
                                                    pots_u32 nlive, pots_u32 j0, pots_u32 nj,
                                                    int ex, const float *__restrict__ stg) {
  const pots_mprod_t *t = &D.t;
  const pots_mscr_t sc = pots_mscr_make(t, D.scratch);
  for (pots_u32 idx = threadIdx.x; idx < ROWS_PER_BLOCK * POTS_SLOTS; idx += POTS_NT) {
    const pots_u32 row = idx % ROWS_PER_BLOCK, sl = idx / ROWS_PER_BLOCK;
    if (row >= nlive) continue;
    const double v = (double)stg[row * POTS_SLOTS + sl];
    const pots_i64 item = r0 + (pots_i64)row;
    if (sl < nj) pots_phase_store(t, &sc, 0, 0, item, j0 + sl, v);
    else if (ex >= 1 && sl == K_CHUNK) pots_phase_store(t, &sc, 0, 0, item, t->k, v);
    else if (ex == 2 && sl == K_CHUNK + 1u) pots_phase_store(t, &sc, 0, 0, item, t->k + 1u, v);
  }
}

// The writes of an A sweep's extras (first chunk only; its dots stay staged for the P sweep):
// u (step 2k) and with the guard nu (step 2k + 2).
static __device__ __forceinline__ void pots_flush_a(const pots_sdesc_t &D, pots_i64 r0,
                                                    pots_u32 nlive, int ex,
                                                    const float *__restrict__ stg) {
  const pots_mprod_t *t = &D.t;
  const pots_mscr_t sc = pots_mscr_make(t, D.scratch);
  for (pots_u32 idx = threadIdx.x; idx < ROWS_PER_BLOCK * 2u; idx += POTS_NT) {
    const pots_u32 row = idx % ROWS_PER_BLOCK, sl = K_CHUNK + idx / ROWS_PER_BLOCK;
    if (row >= nlive) continue;
    const double v = (double)stg[row * POTS_SLOTS + sl];
    const pots_i64 item = r0 + (pots_i64)row;
    if (sl == K_CHUNK) pots_phase_store(t, &sc, 0, 1, item, 2u * t->k, v);
    else if (ex == 2) pots_phase_store(t, &sc, 0, 1, item, 2u * t->k + 2u, v);
  }
}

// The writes of a P sweep: d_j = (s_j + beta_j) - pi_j (biased) or s_j - pi_j, from the A sweep's
// staged s_j (stA) and this sweep's pi_j (stg), with the fp32 operations of pots_phase_store's
// step 2j + 1; on `ex` pp (step 2k + 1, from stdbl) and with the guard p1 (step 2k + 3).
static __device__ __forceinline__ void pots_flush_p(const pots_sdesc_t &D, pots_i64 r0,
                                                    pots_u32 nlive, pots_u32 j0, pots_u32 nj,
                                                    int ex, const float *__restrict__ stA,
                                                    const float *__restrict__ stg,
                                                    const double *__restrict__ stdbl) {
  const pots_mprod_t *t = &D.t;
  const pots_mscr_t sc = pots_mscr_make(t, D.scratch);
  const int biased = (t->flags & POTS_M_BIASED) != 0;
  const pots_i64 n = t->a.rows;
  for (pots_u32 idx = threadIdx.x; idx < ROWS_PER_BLOCK * POTS_SLOTS; idx += POTS_NT) {
    const pots_u32 row = idx % ROWS_PER_BLOCK, sl = idx / ROWS_PER_BLOCK;
    if (row >= nlive) continue;
    const pots_i64 item = r0 + (pots_i64)row;
    if (sl < nj) {
      const pots_u32 j = j0 + sl;
      const float s = stA[row * POTS_SLOTS + sl], pi = stg[row * POTS_SLOTS + sl];
      float dd;
      if (biased) {
        const float sb = s + sc.beta[j];
        dd = sb - pi;
      } else {
        dd = s - pi;
      }
      sc.d[(pots_u64)j * (pots_u64)n + (pots_u64)item] = dd;
    } else if (ex >= 1 && sl == K_CHUNK) {
      pots_phase_store(t, &sc, 0, 1, item, 2u * t->k + 1u, stdbl[row]);
    } else if (ex == 2 && sl == K_CHUNK + 1u) {
      pots_phase_store(t, &sc, 0, 1, item, 2u * t->k + 3u, (double)stg[row * POTS_SLOTS + sl]);
    }
  }
}

// The bias item's stores, lane 0 of the one warp that holds it, as before: the sum of squares
// (step 0), the dots (steps 1 + j0 + j) and with the guard beta1 (step k + 1).
static __device__ __forceinline__ void pots_store_bias(const pots_sdesc_t &D, pots_u32 j0,
                                                       pots_u32 nj, int ex,
                                                       const pots_lane_t &L) {
  const pots_u32 warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  if (warp != 0u) return;
  const pots_mprod_t *t = &D.t;
  const pots_mscr_t sc = pots_mscr_make(t, D.scratch);
  const pots_i64 item = t->b.rows;
  if (ex >= 1) {
    const double v = pots_tree_d(L.ed[0]);
    if (lane == 0u) pots_phase_store(t, &sc, 0, 0, item, 0, v);
  }
  _Pragma("unroll") for (pots_u32 j = 0; j < K_CHUNK; j++) {
    if (j < nj) {
      const float v = pots_tree_f(L.acc[0][j]);
      if (lane == 0u) pots_phase_store(t, &sc, 0, 0, item, 1u + j0 + j, (double)v);
    }
  }
  if (ex == 2) {
    const float v = pots_tree_f(L.e2[0]);
    if (lane == 0u) pots_phase_store(t, &sc, 0, 0, item, t->k + 1u, (double)v);
  }
}

// ---- an entry: every chunk of its sweeps ----

// The entry's work over the block's rows r0 .. r0+nrows-1 of its group (D holds its product).
// Phase 0: a B sweep (the B rows among them), then the bias item when this block holds it; phase
// 1: per chunk, the A sweep, then the P sweep, which forms d_j from the staged s_j.
static __device__ __forceinline__ void pots_run_entry(const pots_sdesc_t &D, pots_u32 ne,
                                                      pots_i64 r0, pots_u32 nrows,
                                                      pots_lane_t &L, float *sx, float *sy,
                                                      float *sb2, float *sb1, float *stA,
                                                      float *stP, double *stdbl) {
  const pots_mprod_t *t = &D.t;
  const pots_u32 k = t->k;
  const int guard = (t->flags & POTS_M_GUARD) != 0;
  if (D.phase == 0) {
    const pots_i64 qb = t->b.rows;
    const pots_u32 nb = r0 >= qb ? 0u : (qb - r0 < (pots_i64)nrows ? (pots_u32)(qb - r0) : nrows);
    if (nb > 0) {
      for (pots_u32 j0 = 0; j0 < k; j0 += K_CHUNK) {
        const pots_u32 nj = k - j0 < K_CHUNK ? k - j0 : K_CHUNK;
        const int ex = j0 == 0 ? (guard ? 2 : 1) : 0;
        pots_sweep<POTS_SW_B>(D, r0, nb, j0, nj, ex, L, sx, sy, sb2, sb1);
        pots_stage(L, 0, stA, stdbl);
        __syncthreads();
        pots_flush_b(D, r0, nb, j0, nj, ex, stA);
      }
    }
    if (ne == 1u && (t->flags & POTS_M_BIASED) != 0 && r0 + (pots_i64)nrows > qb) {
      for (pots_u32 j0 = 0; j0 < k; j0 += K_CHUNK) {
        const pots_u32 nj = k - j0 < K_CHUNK ? k - j0 : K_CHUNK;
        const int ex = j0 == 0 ? (guard ? 2 : 1) : 0;
        pots_sweep<POTS_SW_BIAS>(D, 0, 1u, j0, nj, ex, L, sx, sy, sb2, sb1);
        pots_store_bias(D, j0, nj, ex, L);
      }
    }
    return;
  }
  for (pots_u32 j0 = 0; j0 < k; j0 += K_CHUNK) {
    const pots_u32 nj = k - j0 < K_CHUNK ? k - j0 : K_CHUNK;
    const int ex = j0 == 0 ? (guard ? 2 : 1) : 0;
    pots_sweep<POTS_SW_A>(D, r0, nrows, j0, nj, ex, L, sx, sy, sb2, sb1);
    pots_stage(L, 0, stA, stdbl);
    if (ex >= 1) {
      __syncthreads();
      pots_flush_a(D, r0, nrows, ex, stA);
    }
    pots_sweep<POTS_SW_P>(D, r0, nrows, j0, nj, ex, L, sx, sy, sb2, sb1);
    pots_stage(L, ex >= 1, stP, stdbl);
    __syncthreads();
    pots_flush_p(D, r0, nrows, j0, nj, ex, stA, stP, stdbl);
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

// The product descriptor of an entry into shared memory, by consecutive threads (4-byte words),
// then a barrier. Every later read of the product goes to the copy.
static __device__ __forceinline__ void pots_load_desc(pots_sdesc_t &D, const pots_mprod_t *src,
                                                      pots_u64 scratch, pots_mentry_t en) {
  const pots_u32 *s = (const pots_u32 *)src;
  pots_u32 *d = (pots_u32 *)&D.t;
  if (threadIdx.x < sizeof(pots_mprod_t) / 4u) d[threadIdx.x] = s[threadIdx.x];
  if (threadIdx.x == sizeof(pots_mprod_t) / 4u) {
    D.scratch = scratch;
    D.prod = en.prod;
    D.phase = en.phase;
  }
  __syncthreads();
}

// One level (FS38b): a block is a run of ROWS_PER_BLOCK rows of one read group, given by the
// block table (a phase-2 group is one block, its warps taking the 5 + k items); the group's
// entries run one after another over the block's rows.
extern "C" __global__ void __launch_bounds__(POTS_MK_WARPS * 32, 2) pots_measure_level(
    pots_mlargs_t A) {
  __shared__ pots_sdesc_t s_d;
  __shared__ double s_std[ROWS_PER_BLOCK];
  __shared__ float s_x[ROWS_PER_BLOCK * POTS_XPAD];
  __shared__ float s_y[K_CHUNK * POTS_YP];
  __shared__ float s_b2[TILE_COLS];
  __shared__ float s_b1[TILE_COLS];
  __shared__ float s_sa[ROWS_PER_BLOCK * POTS_SLOTS];
  __shared__ float s_sp[ROWS_PER_BLOCK * POTS_SLOTS];
  pots_lane_t L;
  const pots_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
  const pots_mprod_t *prods = (const pots_mprod_t *)A.prods;
  const pots_mwork_t *works = (const pots_mwork_t *)A.works;
  const pots_mread_t *reads = (const pots_mread_t *)A.reads;
  const pots_mentry_t *entries = (const pots_mentry_t *)A.entries;
  const pots_mblk_t *blocks = (const pots_mblk_t *)A.blocks;
  for (pots_u64 b = blockIdx.x; b < A.n_blocks; b += gridDim.x) {
    const pots_mblk_t bk = blocks[b];
    const pots_mread_t rd = reads[bk.group];
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
      const pots_u64 r0 = (pots_u64)bk.rbk * ROWS_PER_BLOCK;
      const pots_u64 rem = rd.count - r0;
      const pots_u32 nrows = rem < (pots_u64)ROWS_PER_BLOCK ? (pots_u32)rem : ROWS_PER_BLOCK;
      for (pots_u32 e = 0; e < ne; e++) {
        const pots_mentry_t en = entries[rd.entry0 + e];
        pots_load_desc(s_d, prods + en.prod, works[en.prod].scratch, en);
        pots_run_entry(s_d, ne, (pots_i64)r0, nrows, L, s_x, s_y, s_b2, s_b1, s_sa, s_sp, s_std);
        __syncthreads(); // the entry's shared state is reused by the next one
      }
    }
    __syncthreads(); // the shared tiles are reused by the block's next row block
  }
}
