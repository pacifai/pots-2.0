/* The fused measuring kernel's work split (FS37), as plain C that NVRTC and a host C compiler
 * both build, on top of measure_core.h's arithmetic. The kernel (measure_kernel.cu) gives each
 * warp one *item* and runs the item's *steps*: step by step, lane p computes partial p of the
 * step's dot or sum (the core's partial functions), a halving tree over the 32 lanes reduces it
 * (__shfl_down_sync on the GPU, a 32-slot array on the host: measure_core_host.c), and lane 0
 * stores the result. Nothing else is shared between lanes, so there are no atomics, and every
 * number is the value of MEASURE_KERNEL_PLAN_DRAFT.md §2.1a on logical indices.
 *
 * (FS37b: the kernel now runs one sweep per row over tiles, with these same updates; the items
 * and steps below still define which partial each accumulator is.)
 *
 * Three phases, each a launch, since a phase reads what the one before it wrote:
 *   0. B's rows, one item per row l of B (and one for the bias, if any): t_j[l] = B[l,.]·R[.,j],
 *      b2[l] = B[l,.]²·1, and with the guard b1[l] = |B[l,.]|·1; the bias item makes ‖b‖²,
 *      β_j = b·R[.,j] and with the guard β₁ = |b|·1.
 *   1. A's and P's rows, one item per row i: s_j[i] = A[i,.]·t_j, π_j[i] = P[i,.]·R[.,j],
 *      d_j[i] = (s_j[i] + β_j) − π_j[i], u_i = A[i,.]²·b2, Σ_c P[i,c]² (float64), and with the
 *      guard ν_i = |A[i,.]|·b1 (+ β₁ when biased) and p1_i = |P[i,.]|·1.
 *   2. The row, one item per column of it: each float64 sum over the n rows, then its sqrt.
 * Per-product scratch (below) carries phase to phase; it needs no initialization, because a
 * phase writes every cell a later phase reads (without the guard the cells of ν, p1 and b1
 * are never read).
 */
#ifndef POTS_MEASURE_PHASES_H
#define POTS_MEASURE_PHASES_H

#include "measure_core.h"

#define POTS_PHASES 3u
#define POTS_ROW_FIXED 5u /* ν, ‖|P|·1‖, ‖P‖_F, U², ‖b‖², then k residuals */

/* One product's place in the launch: its scratch, and the id of its first item per phase.
 * The array has one entry per product and a last one whose `start` holds the totals. */
typedef struct {
  pots_u64 scratch; /* absolute address, 8-aligned */
  pots_u64 start[POTS_PHASES];
} pots_mwork_t;

typedef struct {
  double *pp;      /* [n]  Σ_c P[i,c]² in float64 */
  double *bias_sq; /* [1]  ‖b‖² */
  float *tt;       /* [k, q]  t_j[l] at j*q + l */
  float *b1;       /* [q] */
  float *b2;       /* [q] */
  float *nu;       /* [n] */
  float *u;        /* [n] */
  float *p1;       /* [n] */
  float *d;        /* [k, n]  d_j[i] at j*n + i, s_j[i] first */
  float *beta;     /* [k] */
  float *beta1;    /* [1] */
} pots_mscr_t;

/* Bytes of scratch of a product: the float64 cells first, then the float32 ones. */
static __host__ __device__ __forceinline__ pots_u64 pots_mscratch_bytes(pots_i64 n, pots_i64 q,
                                                                        pots_u32 k) {
  pots_u64 f = (pots_u64)k * (pots_u64)q + 2u * (pots_u64)q + 3u * (pots_u64)n +
               (pots_u64)k * (pots_u64)n + (pots_u64)k + 1u;
  return (8u * ((pots_u64)n + 1u) + 4u * f + 7u) & ~(pots_u64)7u;
}

static __host__ __device__ __forceinline__ pots_mscr_t pots_mscr_make(const pots_mprod_t *t,
                                                                      pots_u64 base) {
  const pots_i64 n = t->a.rows, q = t->a.cols;
  const pots_u64 k = t->k;
  pots_mscr_t s;
  s.pp = (double *)base;
  s.bias_sq = s.pp + n;
  s.tt = (float *)(s.bias_sq + 1);
  s.b1 = s.tt + k * (pots_u64)q;
  s.b2 = s.b1 + q;
  s.nu = s.b2 + q;
  s.u = s.nu + n;
  s.p1 = s.u + n;
  s.d = s.p1 + n;
  s.beta = s.d + k * (pots_u64)n;
  s.beta1 = s.beta + k;
  return s;
}

static __host__ __device__ __forceinline__ pots_u64 pots_phase_items(const pots_mprod_t *t,
                                                                     pots_u32 phase) {
  if (phase == 0) return (pots_u64)t->b.rows + ((t->flags & POTS_M_BIASED) != 0);
  if (phase == 1) return (pots_u64)t->a.rows;
  return POTS_ROW_FIXED + t->k;
}

/* The product and the item within it of global item `item` of `phase`: the last product whose
 * first item is at or before it (products with no items in a phase never match). */
static __host__ __device__ __forceinline__ pots_u64 pots_mfind(const pots_mwork_t *w,
                                                               pots_u64 n_prod, pots_u32 phase,
                                                               pots_u64 item, pots_u64 *local) {
  pots_u64 lo = 0, hi = n_prod; /* invariant: w[lo].start <= item < w[hi].start */
  while (hi - lo > 1) {
    pots_u64 mid = lo + (hi - lo) / 2;
    if (w[mid].start[phase] <= item) lo = mid;
    else hi = mid;
  }
  *local = item - w[lo].start[phase];
  return lo;
}

static __host__ __device__ __forceinline__ pots_u32 pots_phase_steps(const pots_mprod_t *t,
                                                                     pots_u32 phase,
                                                                     pots_i64 item) {
  const pots_u32 g = (t->flags & POTS_M_GUARD) != 0;
  if (phase == 0) return t->k + 1 + g; /* a B row, or the bias item (same count) */
  if (phase == 1) return 2 * t->k + 2 + 2 * g;
  (void)item;
  return 1;
}

/* 1 when the step's reduction is a float64 sum (float64 tree), 0 for a float32 dot. */
static __host__ __device__ __forceinline__ pots_u32 pots_phase_double(const pots_mprod_t *t,
                                                                      pots_u32 phase,
                                                                      pots_i64 item,
                                                                      pots_u32 step) {
  if (phase == 0) return item == t->b.rows && step == 0;
  if (phase == 1) return step == 2 * t->k + 1;
  return 1;
}

/* Lane `lane`'s partial of step `step` of item `item`, as float64 (a float32 partial is
 * widened exactly). */
static __host__ __device__ __forceinline__ double pots_phase_partial(
    const pots_mprod_t *t, const pots_mscr_t *sc, pots_u32 phase, pots_i64 item, pots_u32 step,
    pots_u32 lane) {
  const float *r = (const float *)t->r_addr;
  const pots_u32 k = t->k;
  const pots_i64 n = t->a.rows, q = t->a.cols;
  const int guard = (t->flags & POTS_M_GUARD) != 0, biased = (t->flags & POTS_M_BIASED) != 0;
  if (phase == 0) {
    if (item < q) {
      if (step < k) return pots_dot32_partial(&t->b, item, lane, POTS_F_ID, r + step, k);
      if (step == k) return pots_dot32_partial(&t->b, item, lane, POTS_F_SQ, 0, 0);
      return pots_dot32_partial(&t->b, item, lane, POTS_F_ABS, 0, 0);
    }
    if (step == 0) return pots_sum64_sq_partial(&t->bias, 0, lane);
    if (step <= k) return pots_dot32_partial(&t->bias, 0, lane, POTS_F_ID, r + (step - 1), k);
    return pots_dot32_partial(&t->bias, 0, lane, POTS_F_ABS, 0, 0);
  }
  if (phase == 1) {
    if (step < 2 * k) {
      pots_u32 j = step >> 1;
      if ((step & 1) == 0) return pots_dot32_partial(&t->a, item, lane, POTS_F_ID, sc->tt + (pots_u64)j * q, 1);
      return pots_dot32_partial(&t->p, item, lane, POTS_F_ID, r + j, k);
    }
    if (step == 2 * k) return pots_dot32_partial(&t->a, item, lane, POTS_F_SQ, sc->b2, 1);
    if (step == 2 * k + 1) return pots_sum64_sq_partial(&t->p, item, lane);
    if (step == 2 * k + 2) return pots_dot32_partial(&t->a, item, lane, POTS_F_ABS, sc->b1, 1);
    return pots_dot32_partial(&t->p, item, lane, POTS_F_ABS, 0, 0);
  }
  if (item == 0) return guard ? pots_sum64_f_partial(sc->nu, n, lane, 1) : 0.0;
  if (item == 1) return guard ? pots_sum64_f_partial(sc->p1, n, lane, 1) : 0.0;
  if (item == 2) return pots_sum64_d_partial(sc->pp, n, lane);
  if (item == 3) return pots_sum64_f_partial(sc->u, n, lane, 0);
  if (item == 4) return (biased && lane == 0) ? *sc->bias_sq : 0.0;
  return pots_sum64_f_partial(sc->d + (pots_u64)(item - POTS_ROW_FIXED) * (pots_u64)n, n, lane, 1);
}

static __host__ __device__ __forceinline__ double pots_nan64(void) {
  union {
    pots_u64 u;
    double d;
  } x;
  x.u = 0x7ff8000000000000ull;
  return x.d;
}

/* Lane 0's store of step `step`'s reduced value `v` (a float32 result is exact in float64);
 * `row` is the product's output row. */
static __host__ __device__ __forceinline__ void pots_phase_store(
    const pots_mprod_t *t, const pots_mscr_t *sc, double *row, pots_u32 phase, pots_i64 item,
    pots_u32 step, double v) {
  const pots_u32 k = t->k;
  const pots_i64 n = t->a.rows, q = t->a.cols;
  const int guard = (t->flags & POTS_M_GUARD) != 0, biased = (t->flags & POTS_M_BIASED) != 0;
  if (phase == 0) {
    if (item < q) {
      if (step < k) sc->tt[(pots_u64)step * q + item] = (float)v;
      else if (step == k) sc->b2[item] = (float)v;
      else sc->b1[item] = (float)v;
    } else if (step == 0) {
      *sc->bias_sq = v;
    } else if (step <= k) {
      sc->beta[step - 1] = (float)v;
    } else {
      *sc->beta1 = (float)v;
    }
    return;
  }
  if (phase == 1) {
    if (step < 2 * k) {
      pots_u32 j = step >> 1;
      float *dst = sc->d + (pots_u64)j * n + item;
      if ((step & 1) == 0) {
        *dst = (float)v; /* s_j[i] */
      } else {
        float s = *dst, pi = (float)v, dd;
        if (biased) {
          float sb = s + sc->beta[j];
          dd = sb - pi;
        } else {
          dd = s - pi;
        }
        *dst = dd;
      }
    } else if (step == 2 * k) {
      sc->u[item] = (float)v;
    } else if (step == 2 * k + 1) {
      sc->pp[item] = v;
    } else if (step == 2 * k + 2) {
      float nu = (float)v;
      if (biased) nu = nu + *sc->beta1;
      sc->nu[item] = nu;
    } else {
      sc->p1[item] = (float)v;
    }
    return;
  }
  if (item == 0 || item == 1) row[item] = guard ? pots_sqrt64(v) : pots_nan64();
  else if (item == 2) row[2] = pots_sqrt64(v);
  else if (item == 3 || item == 4) row[item] = v;
  else row[item] = pots_sqrt64(v);
}

/* Shared reads (FS38b). A level's launch runs *read groups*: the nodes (product, phase) that
 * read one stored view (the same address, shape, strides and dtype) form a group, and one item
 * of the group is one row of that view, run for every entry of the group in turn so the row is
 * read once while it is hot. `reads` holds a level's groups in order and then a sentinel whose
 * `first` is the level's item total; `entries` (a group's `entry0 .. entry0 + n_entries - 1`)
 * name each node by product and phase. An unshared node is a group of one entry. */
typedef struct {
  pots_u64 first;     /* the level's running item count at the group's first row */
  pots_u64 count;     /* rows of the group */
  pots_u32 entry0;    /* index of the group's first entry in `entries` */
  pots_u32 n_entries; /* nodes sharing the group's view */
} pots_mread_t;

typedef struct {
  pots_u32 prod;
  pots_u32 phase;
} pots_mentry_t;

/* The group holding item `item` of one level's groups `r[0..n_groups-1]` (r[n_groups] is the
 * level's sentinel: first = the level's item total, count = entry0 = n_entries = 0), and the row
 * within it; binary search on `first`, as pots_mfind does. */
static __host__ __device__ __forceinline__ pots_u64 pots_rfind(const pots_mread_t *r,
                                                               pots_u64 n_groups, pots_u64 item,
                                                               pots_u64 *local) {
  pots_u64 lo = 0, hi = n_groups; /* invariant: r[lo].first <= item < r[hi].first */
  while (hi - lo > 1) {
    pots_u64 mid = lo + (hi - lo) / 2;
    if (r[mid].first <= item) lo = mid;
    else hi = mid;
  }
  *local = item - r[lo].first;
  return lo;
}

/* ---- sweeps (FS37b) ----
 *
 * The level kernel makes one pass over a row and keeps one accumulator per step in each lane;
 * a lane visits its columns c = p, p+32, ... in increasing order, so every accumulator sees
 * exactly the operations of its step's pots_dot32_partial or pots_sum64_sq_partial, and the
 * bits are those of the step-by-step run. The four functions below are the per-element updates
 * of the four sweep kinds, shared by the kernel and the host emulation (measure_core_host.c).
 * `x` is the element already widened to fp32. `y` holds the vector values at the element's
 * column for the chunk's challenge columns j = 0 .. nj-1, entry j at y[j*jstride] (R: R + c*k +
 * j0, stride 1; the stored t: tt + j0*q + c, stride q; shared memory: [j][c], stride TILE).
 * `first` is 1 for a row's first chunk, which also carries the extras. A masked column (past
 * the view's end) is never passed in: nothing is added for it, not even a zero. */

#if defined(POTS_ON_GPU) && defined(K_CHUNK)
#define POTS_FOR_J(nj) _Pragma("unroll") for (pots_u32 j = 0; j < (pots_u32)K_CHUNK; j++) if (j < (nj))
#else
#define POTS_FOR_J(nj) for (pots_u32 j = 0; j < (nj); j++)
#endif

/* acc[j] = fmaf(x, y_j, acc[j]) for the chunk's nj dots. */
static __host__ __device__ __forceinline__ void pots_upd_dots(float x, const float *y,
                                                              pots_i64 jstride, pots_u32 nj,
                                                              float *acc) {
  POTS_FOR_J(nj) acc[j] = fmaf(x, y[(pots_i64)j * jstride], acc[j]);
}

/* A row of B (phase 0): the dots with R, then (first chunk) b2 = Σ x²·1 (the square rounded to
 * fp32 first) and, with the guard, b1 = Σ |x|·1. */
static __host__ __device__ __forceinline__ void pots_upd_b(float x, const float *y,
                                                           pots_i64 jstride, pots_u32 nj,
                                                           float *acc, int first, int guard,
                                                           float *b2, float *b1) {
  pots_upd_dots(x, y, jstride, nj, acc);
  if (first) {
    const float x2 = x * x;
    *b2 = fmaf(x2, 1.0f, *b2);
    if (guard) *b1 = fmaf(fabsf(x), 1.0f, *b1);
  }
}

/* The bias (phase 0, 1 row): the dots with R, then (first chunk) the float64 sum of squares
 * and, with the guard, β₁ = Σ |x|·1. */
static __host__ __device__ __forceinline__ void pots_upd_bias(float x, const float *y,
                                                              pots_i64 jstride, pots_u32 nj,
                                                              float *acc, int first, int guard,
                                                              double *sq, float *b1) {
  pots_upd_dots(x, y, jstride, nj, acc);
  if (first) {
    const double xd = (double)x;
    *sq = fma(xd, xd, *sq); /* xd * xd is exact in float64, so the fma gives the same value */
    if (guard) *b1 = fmaf(fabsf(x), 1.0f, *b1);
  }
}

/* A row of A (phase 1): the dots with the stored t, then (first chunk) u = Σ x²·b2[c] and, with
 * the guard, ν = Σ |x|·b1[c]; `b2c` and `b1c` are the vectors' values at the column. */
static __host__ __device__ __forceinline__ void pots_upd_a(float x, const float *y,
                                                           pots_i64 jstride, pots_u32 nj,
                                                           float *acc, int first, int guard,
                                                           float b2c, float b1c, float *u,
                                                           float *nu) {
  pots_upd_dots(x, y, jstride, nj, acc);
  if (first) {
    const float x2 = x * x;
    *u = fmaf(x2, b2c, *u);
    if (guard) *nu = fmaf(fabsf(x), b1c, *nu);
  }
}

/* A row of P (phase 1): the dots with R, then (first chunk) the float64 sum of squares and,
 * with the guard, p1 = Σ |x|·1. */
static __host__ __device__ __forceinline__ void pots_upd_p(float x, const float *y,
                                                           pots_i64 jstride, pots_u32 nj,
                                                           float *acc, int first, int guard,
                                                           double *pp, float *p1) {
  pots_upd_dots(x, y, jstride, nj, acc);
  if (first) {
    const double xd = (double)x;
    *pp = fma(xd, xd, *pp); /* xd * xd is exact in float64, so the fma gives the same value */
    if (guard) *p1 = fmaf(fabsf(x), 1.0f, *p1);
  }
}

#endif /* POTS_MEASURE_PHASES_H */
