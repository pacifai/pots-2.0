/* Check 5's measuring arithmetic for the fused GPU kernel (decision O2), as code that both
 * NVRTC and a host C compiler build. The kernel will wrap the partial functions below in CUDA
 * thread code, and the host test driver (tests/verification/verifier/matmul_check/
 * measure_core_host.c) runs the same functions in loops, so the order of every addition is
 * tested without a GPU. Nothing calls this yet besides that test.
 *
 * It computes one product's row, the columns of freivalds.ROW, for P = A·B (+ 1·bᵀ):
 *   [ν, ‖|P|·1‖, ‖P‖_F, U² terms, ‖b‖², residual_1 .. residual_k]
 * in float64, with the pinned order of MEASURE_KERNEL_PLAN_DRAFT.md §2.1a:
 *   - dot32(x, y, L): 32 partial sums, partial p takes c = p, p+32, ... < L in increasing order
 *     with s[p] = fmaf(x_c, y_c, s[p]) from +0.0f, then the tree s[i] += s[i+off] for
 *     off = 16, 8, 4, 2, 1, and the result is s[0]. A row term applies f (identity, |m| or m*m
 *     rounded to fp32) to the matrix entry first; a vector of ones is 1.0f per entry.
 *   - sum64(x, L): the same partials and tree in float64, s[p] = s[p] + x_c, a square formed in
 *     float64 from the widened fp32 value (exact).
 * A dot is only ever formed as its 32 partials and then the tree, so the host build exercises
 * the decomposition the GPU uses (one partial per thread of a warp). Build with no fast math and
 * no contraction (-ffp-contract=off, NVRTC --fmad=false): every fused multiply-add is an
 * explicit fmaf. NaN and Inf propagate as IEEE does.
 */
#ifndef POTS_MEASURE_CORE_H
#define POTS_MEASURE_CORE_H

#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
#define POTS_ON_GPU 1
typedef unsigned short pots_u16;
typedef unsigned int pots_u32;
typedef unsigned long long pots_u64;
typedef long long pots_i64;
#else
/* The host build also passes -ffp-contract=off (gcc ignores the pragma, clang honours it); NVRTC passes --fmad=false (FS37). */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif
#include <math.h>
#include <stdint.h>
typedef uint16_t pots_u16;
typedef uint32_t pots_u32;
typedef uint64_t pots_u64;
typedef int64_t pots_i64;
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

#define POTS_LANES 32u /* partial sums of every dot and sum */

#define POTS_DT_FP32 1u /* the leaf dtype codes of commitment/encoding.py */
#define POTS_DT_BF16 2u

#define POTS_F_ID 0u /* what a row term applies to a matrix entry before the multiply */
#define POTS_F_ABS 1u
#define POTS_F_SQ 2u

#define POTS_M_GUARD 1u  /* test 1 on: ν and ‖|P|·1‖ are measured, else NaN */
#define POTS_M_BIASED 2u /* the product has a bias vector */

#define POTS_ERR_OK 0
#define POTS_ERR_DTYPE 1 /* a view's dtype is neither fp32 nor bf16 */
#define POTS_ERR_K 2     /* k == 0, or no challenge matrix */
#define POTS_ERR_SHAPE 3 /* A, B, P, bias shapes disagree, or a dimension is below 1 */
#define POTS_ERR_NOMEM 4 /* the host build could not allocate its scratch */

/* A strided matrix: element (i, c) is at addr + (i*row_stride + c*col_stride) elements. */
typedef struct {
  pots_u64 addr;
  pots_i64 rows;
  pots_i64 cols;
  pots_i64 row_stride;
  pots_i64 col_stride;
  pots_u32 dtype;
  pots_u32 pad;
} pots_mview_t;

/* One product. r_addr is fp32 [w, k] contiguous, entry (c, j) at c*k + j. Unbiased: bias is
 * zeroed (a vector is a view with one row). */
typedef struct {
  pots_mview_t a;
  pots_mview_t b;
  pots_mview_t p;
  pots_mview_t bias;
  pots_u64 r_addr;
  pots_u32 k;
  pots_u32 flags;
} pots_mprod_t;

/* ---- loads ---- */

static __host__ __device__ __forceinline__ float pots_bits_float(pots_u32 u) {
  union {
    pots_u32 u;
    float f;
  } x;
  x.u = u;
  return x.f;
}

/* Element (i, c) of a view, widened exactly to fp32 (bf16: the 16 bits shifted left by 16). */
static __host__ __device__ __forceinline__ float pots_mload(const pots_mview_t *v, pots_i64 i,
                                                            pots_i64 c) {
  pots_i64 at = i * v->row_stride + c * v->col_stride;
  if (v->dtype == POTS_DT_BF16) {
    const pots_u16 *p = (const pots_u16 *)v->addr;
    return pots_bits_float((pots_u32)p[at] << 16);
  }
  return ((const float *)v->addr)[at];
}

static __host__ __device__ __forceinline__ double pots_sqrt64(double x) { return sqrt(x); }

/* ---- the partials and the trees ---- */

/* Partial `part` of dot32(f(M[row, .]), y, M.cols): y[c] is vec[c*vec_stride], or 1.0f when
 * vec == 0. */
static __host__ __device__ __forceinline__ float pots_dot32_partial(
    const pots_mview_t *m, pots_i64 row, pots_u32 part, pots_u32 f, const float *vec,
    pots_i64 vec_stride) {
  float s = 0.0f;
  for (pots_i64 c = part; c < m->cols; c += POTS_LANES) {
    float x = pots_mload(m, row, c);
    if (f == POTS_F_ABS) x = fabsf(x);
    else if (f == POTS_F_SQ) x = x * x;
    float y = vec ? vec[c * vec_stride] : 1.0f;
    s = fmaf(x, y, s);
  }
  return s;
}

/* Partial `part` of sum64 over a view row's squares, each formed in float64. */
static __host__ __device__ __forceinline__ double pots_sum64_sq_partial(const pots_mview_t *m,
                                                                        pots_i64 row,
                                                                        pots_u32 part) {
  double s = 0.0;
  for (pots_i64 c = part; c < m->cols; c += POTS_LANES) {
    double x = (double)pots_mload(m, row, c);
    s = s + x * x;
  }
  return s;
}

/* Partial `part` of sum64 over x[0..len): the widened fp32 values, or their float64 squares. */
static __host__ __device__ __forceinline__ double pots_sum64_f_partial(const float *x,
                                                                       pots_i64 len,
                                                                       pots_u32 part,
                                                                       pots_u32 square) {
  double s = 0.0;
  for (pots_i64 c = part; c < len; c += POTS_LANES) {
    double v = (double)x[c];
    s = s + (square ? v * v : v);
  }
  return s;
}

/* Partial `part` of sum64 over float64 values x[0..len). */
static __host__ __device__ __forceinline__ double pots_sum64_d_partial(const double *x,
                                                                       pots_i64 len,
                                                                       pots_u32 part) {
  double s = 0.0;
  for (pots_i64 c = part; c < len; c += POTS_LANES) s = s + x[c];
  return s;
}

static __host__ __device__ __forceinline__ float pots_tree32_f(float *s) {
  for (pots_u32 off = 16; off >= 1; off >>= 1)
    for (pots_u32 i = 0; i < off; i++) s[i] = s[i] + s[i + off];
  return s[0];
}

static __host__ __device__ __forceinline__ double pots_tree32_d(double *s) {
  for (pots_u32 off = 16; off >= 1; off >>= 1)
    for (pots_u32 i = 0; i < off; i++) s[i] = s[i] + s[i + off];
  return s[0];
}

/* ---- refusals ---- */

static __host__ __device__ __forceinline__ int pots_mview_ok(const pots_mview_t *v) {
  return v->dtype == POTS_DT_FP32 || v->dtype == POTS_DT_BF16;
}

/* POTS_ERR_OK, or the first refusal the C side can see in the product's descriptors. */
static __host__ __device__ __forceinline__ int pots_mprod_check(const pots_mprod_t *t) {
  int biased = (t->flags & POTS_M_BIASED) != 0;
  if (!pots_mview_ok(&t->a) || !pots_mview_ok(&t->b) || !pots_mview_ok(&t->p)) return POTS_ERR_DTYPE;
  if (biased && !pots_mview_ok(&t->bias)) return POTS_ERR_DTYPE;
  if (t->k == 0 || t->r_addr == 0) return POTS_ERR_K;
  if (t->a.rows < 1 || t->a.cols < 1 || t->b.rows < 1 || t->b.cols < 1) return POTS_ERR_SHAPE;
  if (t->b.rows != t->a.cols || t->p.rows != t->a.rows || t->p.cols != t->b.cols)
    return POTS_ERR_SHAPE;
  if (biased && (t->bias.rows != 1 || t->bias.cols != t->b.cols)) return POTS_ERR_SHAPE;
  return POTS_ERR_OK;
}

#endif /* POTS_MEASURE_CORE_H */
