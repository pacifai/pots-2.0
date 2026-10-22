/* Host build of verification/verifier/matmul_check/measure_core.h (decision O2): computes every
 * product's row by the kernel's work decomposition, one partial after another and then the
 * tree, so the pinned order of MEASURE_KERNEL_PLAN_DRAFT.md §2.1a is tested on a machine with
 * no GPU. Build with -ffp-contract=off. */
#include <stdlib.h>
#include "measure_core.h"
#include "measure_phases.h"

size_t pots_host_sizeof_mview(void) { return sizeof(pots_mview_t); }
size_t pots_host_sizeof_mprod(void) { return sizeof(pots_mprod_t); }
size_t pots_host_offsets_mview(size_t *out) {
  pots_mview_t *p = NULL;
  size_t n = 0;
  out[n++] = (size_t)&p->addr;
  out[n++] = (size_t)&p->rows;
  out[n++] = (size_t)&p->cols;
  out[n++] = (size_t)&p->row_stride;
  out[n++] = (size_t)&p->col_stride;
  out[n++] = (size_t)&p->dtype;
  out[n++] = (size_t)&p->pad;
  return n;
}
size_t pots_host_offsets_mprod(size_t *out) {
  pots_mprod_t *p = NULL;
  size_t n = 0;
  out[n++] = (size_t)&p->a;
  out[n++] = (size_t)&p->b;
  out[n++] = (size_t)&p->p;
  out[n++] = (size_t)&p->bias;
  out[n++] = (size_t)&p->r_addr;
  out[n++] = (size_t)&p->k;
  out[n++] = (size_t)&p->flags;
  return n;
}
int pots_host_check(const pots_mprod_t *t) { return pots_mprod_check(t); }

/* dot32 of row `row` of m under f with vector vec: the 32 partials, then the tree. */
static float dot32(const pots_mview_t *m, pots_i64 row, pots_u32 f, const float *vec,
                   pots_i64 vec_stride) {
  float s[POTS_LANES];
  for (pots_u32 p = 0; p < POTS_LANES; p++) s[p] = pots_dot32_partial(m, row, p, f, vec, vec_stride);
  return pots_tree32_f(s);
}

static double sum64_sq_view(const pots_mview_t *m, pots_i64 row) {
  double s[POTS_LANES];
  for (pots_u32 p = 0; p < POTS_LANES; p++) s[p] = pots_sum64_sq_partial(m, row, p);
  return pots_tree32_d(s);
}

static double sum64_f(const float *x, pots_i64 len, pots_u32 square) {
  double s[POTS_LANES];
  for (pots_u32 p = 0; p < POTS_LANES; p++) s[p] = pots_sum64_f_partial(x, len, p, square);
  return pots_tree32_d(s);
}

static double sum64_d(const double *x, pots_i64 len) {
  double s[POTS_LANES];
  for (pots_u32 p = 0; p < POTS_LANES; p++) s[p] = pots_sum64_d_partial(x, len, p);
  return pots_tree32_d(s);
}

static int measure_one(const pots_mprod_t *t, double *row) {
  const pots_i64 n = t->a.rows, q = t->a.cols;
  const pots_u32 k = t->k;
  const int guard = (t->flags & POTS_M_GUARD) != 0, biased = (t->flags & POTS_M_BIASED) != 0;
  const float *r = (const float *)t->r_addr;
  float *tt = malloc(sizeof(float) * (size_t)k * (size_t)q); /* t_j[l] at j*q + l */
  float *b1 = malloc(sizeof(float) * (size_t)q);
  float *b2 = malloc(sizeof(float) * (size_t)q);
  float *nu_i = malloc(sizeof(float) * (size_t)n);
  float *u_i = malloc(sizeof(float) * (size_t)n);
  float *p1_i = malloc(sizeof(float) * (size_t)n);
  double *pp_i = malloc(sizeof(double) * (size_t)n);
  float *d = malloc(sizeof(float) * (size_t)k * (size_t)n); /* d_j[i] at j*n + i */
  float *beta = malloc(sizeof(float) * (size_t)k);
  float beta1 = 0.0f;
  double bias_sq = 0.0;
  pots_i64 i, l;
  pots_u32 j;

  if (!tt || !b1 || !b2 || !nu_i || !u_i || !p1_i || !pp_i || !d || !beta) {
    free(tt); free(b1); free(b2); free(nu_i); free(u_i); free(p1_i); free(pp_i); free(d); free(beta);
    return POTS_ERR_NOMEM;
  }

  /* 1. B's rows against R's columns, |B|·1 and B²·1. */
  for (l = 0; l < q; l++) {
    for (j = 0; j < k; j++) tt[(size_t)j * (size_t)q + (size_t)l] = dot32(&t->b, l, POTS_F_ID, r + j, k);
    if (guard) b1[l] = dot32(&t->b, l, POTS_F_ABS, NULL, 0);
    b2[l] = dot32(&t->b, l, POTS_F_SQ, NULL, 0);
  }
  if (biased) {
    if (guard) beta1 = dot32(&t->bias, 0, POTS_F_ABS, NULL, 0);
    bias_sq = sum64_sq_view(&t->bias, 0);
    for (j = 0; j < k; j++) beta[j] = dot32(&t->bias, 0, POTS_F_ID, r + j, k);
  }
  /* 2 and 3. A's rows and P's rows. */
  for (i = 0; i < n; i++) {
    for (j = 0; j < k; j++) {
      float s = dot32(&t->a, i, POTS_F_ID, tt + (size_t)j * (size_t)q, 1);
      float pi = dot32(&t->p, i, POTS_F_ID, r + j, k);
      if (biased) {
        d[(size_t)j * (size_t)n + (size_t)i] = (s + beta[j]) - pi;
      } else {
        d[(size_t)j * (size_t)n + (size_t)i] = s - pi;
      }
    }
    if (guard) {
      float nu = dot32(&t->a, i, POTS_F_ABS, b1, 1);
      if (biased) nu = nu + beta1;
      nu_i[i] = nu;
      p1_i[i] = dot32(&t->p, i, POTS_F_ABS, NULL, 0);
    }
    u_i[i] = dot32(&t->a, i, POTS_F_SQ, b2, 1);
    pp_i[i] = sum64_sq_view(&t->p, i);
  }
  /* 5. The row. */
  row[0] = guard ? pots_sqrt64(sum64_f(nu_i, n, 1)) : NAN;
  row[1] = guard ? pots_sqrt64(sum64_f(p1_i, n, 1)) : NAN;
  row[2] = pots_sqrt64(sum64_d(pp_i, n));
  row[3] = sum64_f(u_i, n, 0);
  row[4] = bias_sq;
  for (j = 0; j < k; j++) row[5 + j] = pots_sqrt64(sum64_f(d + (size_t)j * (size_t)n, n, 1));
  free(tt); free(b1); free(b2); free(nu_i); free(u_i); free(p1_i); free(pp_i); free(d); free(beta);
  return POTS_ERR_OK;
}

/* Every product's row (width 5 + k each, float64), or the first refusal (no row written). */
int pots_host_measure(const pots_mprod_t *tab, uint64_t n, double *rows) {
  uint64_t i;
  for (i = 0; i < n; i++) {
    int e = pots_mprod_check(&tab[i]);
    if (e != POTS_ERR_OK) return e;
  }
  for (i = 0; i < n; i++) {
    int e = measure_one(&tab[i], rows);
    if (e != POTS_ERR_OK) return e;
    rows += 5 + tab[i].k;
  }
  return POTS_ERR_OK;
}

/* ---- the fused kernel's decomposition (measure_kernel.cu), run on the host ---- */

size_t pots_host_sizeof_mwork(void) { return sizeof(pots_mwork_t); }
uint64_t pots_host_scratch_bytes(int64_t n, int64_t q, uint32_t k) {
  return pots_mscratch_bytes(n, q, k);
}

/* The GPU's tree: every lane adds the value `off` lanes above it (its own where that is past
 * the warp), for off = 16, 8, 4, 2, 1, all lanes at once; lane 0's value is the result. */
static void shfl_tree_f(float *v) {
  for (pots_u32 off = 16; off >= 1; off >>= 1) {
    float w[POTS_LANES];
    for (pots_u32 i = 0; i < POTS_LANES; i++) w[i] = v[i] + (i + off < POTS_LANES ? v[i + off] : v[i]);
    for (pots_u32 i = 0; i < POTS_LANES; i++) v[i] = w[i];
  }
}
static void shfl_tree_d(double *v) {
  for (pots_u32 off = 16; off >= 1; off >>= 1) {
    double w[POTS_LANES];
    for (pots_u32 i = 0; i < POTS_LANES; i++) w[i] = v[i] + (i + off < POTS_LANES ? v[i + off] : v[i]);
    for (pots_u32 i = 0; i < POTS_LANES; i++) v[i] = w[i];
  }
}

/* One item of one product's phase as the kernel runs it (measure_kernel.cu's per-item helper):
 * for each step the 32 lane partials, the shuffle tree, and lane 0's store. */
static void host_measure_item(const pots_mprod_t *t, const pots_mscr_t *sc, double *row,
                              pots_u32 phase, uint64_t item) {
  pots_u32 steps = pots_phase_steps(t, phase, (pots_i64)item), s, lane;
  for (s = 0; s < steps; s++) {
    double part[POTS_LANES];
    for (lane = 0; lane < POTS_LANES; lane++)
      part[lane] = pots_phase_partial(t, sc, phase, (pots_i64)item, s, lane);
    if (pots_phase_double(t, phase, (pots_i64)item, s)) {
      shfl_tree_d(part);
    } else {
      float f[POTS_LANES];
      for (lane = 0; lane < POTS_LANES; lane++) f[lane] = (float)part[lane];
      shfl_tree_f(f);
      part[0] = (double)f[0];
    }
    pots_phase_store(t, sc, row, phase, (pots_i64)item, s, part[0]);
  }
}

/* The three phases over every product, as the kernel runs them: for each item (a warp) each
 * step takes its 32 lane partials, the shuffle tree, and lane 0's store. `works` has n + 1
 * entries (the last holds the totals) and every scratch address is set, as the Python side
 * builds them. `reverse` runs each phase's items last to first:
 * the items of a phase are independent, so the order gives the same bits. Rows are width 5 + k. */
int pots_host_measure_phased(const pots_mprod_t *tab, uint64_t n, const pots_mwork_t *works,
                             double *rows, int reverse) {
  uint64_t i, g;
  pots_u32 phase;
  for (i = 0; i < n; i++) {
    int e = pots_mprod_check(&tab[i]);
    if (e != POTS_ERR_OK) return e;
  }
  for (phase = 0; phase < POTS_PHASES; phase++) {
    uint64_t total = works[n].start[phase];
    for (g = 0; g < total; g++) {
      uint64_t gi = reverse ? total - 1 - g : g;
      uint64_t item, pi = pots_mfind(works, n, phase, gi, &item);
      const pots_mprod_t *t = &tab[pi];
      pots_mscr_t sc = pots_mscr_make(t, works[pi].scratch);
      host_measure_item(t, &sc, rows + pi * (POTS_ROW_FIXED + t->k), phase, item);
    }
  }
  return POTS_ERR_OK;
}

/* The levels' launches (FS38b), as pots_measure_level runs them: level l's groups are
 * reads[level_off[l] .. level_off[l+1]-2] and its sentinel reads[level_off[l+1]-1]. Levels in
 * order; within a level every item (last to first when `reverse`), and for each item every entry
 * of its group in order. */
int pots_host_measure_levels(const pots_mprod_t *tab, uint64_t n, const pots_mwork_t *works,
                             const pots_mread_t *reads, const uint64_t *level_off,
                             uint32_t n_levels, const pots_mentry_t *entries, double *rows,
                             int reverse) {
  uint64_t i, g;
  uint32_t l, e;
  for (i = 0; i < n; i++) {
    int err = pots_mprod_check(&tab[i]);
    if (err != POTS_ERR_OK) return err;
  }
  for (l = 0; l < n_levels; l++) {
    const pots_mread_t *r = reads + level_off[l];
    uint64_t n_groups = level_off[l + 1] - level_off[l] - 1, total = r[n_groups].first;
    for (g = 0; g < total; g++) {
      uint64_t gi = reverse ? total - 1 - g : g;
      uint64_t item, ri = pots_rfind(r, n_groups, gi, &item);
      for (e = 0; e < r[ri].n_entries; e++) {
        const pots_mentry_t *en = &entries[r[ri].entry0 + e];
        const pots_mprod_t *t = &tab[en->prod];
        pots_mscr_t sc = pots_mscr_make(t, works[en->prod].scratch);
        host_measure_item(t, &sc, rows + (uint64_t)en->prod * (POTS_ROW_FIXED + t->k), en->phase,
                          item);
      }
    }
  }
  return POTS_ERR_OK;
}
