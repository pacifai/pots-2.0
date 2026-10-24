// The fused measuring kernel (FS37, decision O2). Compiled by NVRTC at run time
// (measure_kernel.py) with measure_core.h and measure_phases.h passed in memory, with
// --fmad=false. Every decision about what is computed lives in those headers; this file maps
// their items and steps onto warps, lanes and shuffles. One launch per level of the read plan (pots_measure_level).
#include "measure_core.h"
#include "measure_phases.h"

#define POTS_MK_WARPS 8u /* warps per block */
#define POTS_FULL 0xffffffffu

// One level's launch (FS38b), passed by value.
typedef struct {
  pots_u64 prods;   // pots_mprod_t[n_prod]
  pots_u64 works;   // pots_mwork_t[n_prod + 1] (the scratch addresses)
  pots_u64 rows;    // float64 [n_prod, ROW_FIXED + k]
  pots_u64 reads;   // pots_mread_t[n_groups + 1], the level's groups and its sentinel
  pots_u64 entries; // pots_mentry_t
  pots_u64 n_groups;
  pots_u64 total;   // items of this level
  pots_u32 k;
  pots_u32 pad;
} pots_mlargs_t;

// One warp's work on one item of product `t`'s phase `phase`: every step (the lanes' partials,
// the halving tree, lane 0's store).
static __device__ __forceinline__ void pots_measure_item(const pots_mprod_t *t, const pots_mscr_t &sc,
                                                         double *row, pots_u32 phase,
                                                         pots_u64 item, pots_u32 k,
                                                         pots_u32 lane) {
  (void)k;
  const pots_u32 steps = pots_phase_steps(t, phase, (pots_i64)item);
  for (pots_u32 s = 0; s < steps; s++) {
    const double part = pots_phase_partial(t, &sc, phase, (pots_i64)item, s, lane);
    double v;
    // The halving tree: lane i takes s[i] + s[i + off] for off = 16, 8, 4, 2, 1; lane 0 ends
    // with the result (the other lanes' values are never used).
    if (pots_phase_double(t, phase, (pots_i64)item, s)) {
      v = part;
      for (pots_u32 off = 16; off >= 1; off >>= 1) v = v + __shfl_down_sync(POTS_FULL, v, off);
    } else {
      float f = (float)part;
      for (pots_u32 off = 16; off >= 1; off >>= 1) f = f + __shfl_down_sync(POTS_FULL, f, off);
      v = (double)f;
    }
    if (lane == 0) pots_phase_store(t, &sc, row, phase, (pots_i64)item, s, v);
    __syncwarp();
  }
}

// One level (FS38b): an item is row `r` of a read group; every entry (product, phase) of the
// group runs its own item `r` in turn.
extern "C" __global__ void __launch_bounds__(POTS_MK_WARPS * 32)
pots_measure_level(pots_mlargs_t A) {
  const pots_u32 lane = threadIdx.x & 31u;
  const pots_u64 stride = (pots_u64)gridDim.x * POTS_MK_WARPS;
  const pots_mprod_t *prods = (const pots_mprod_t *)A.prods;
  const pots_mwork_t *works = (const pots_mwork_t *)A.works;
  const pots_mread_t *reads = (const pots_mread_t *)A.reads;
  const pots_mentry_t *entries = (const pots_mentry_t *)A.entries;
  for (pots_u64 g = (pots_u64)blockIdx.x * POTS_MK_WARPS + (threadIdx.x >> 5); g < A.total;
       g += stride) {
    pots_u64 item;
    const pots_u64 gi = pots_rfind(reads, A.n_groups, g, &item);
    const pots_mread_t rd = reads[gi];
    for (pots_u32 e = 0; e < rd.n_entries; e++) {
      const pots_mentry_t en = entries[rd.entry0 + e];
      const pots_mprod_t *t = prods + en.prod;
      const pots_mscr_t sc = pots_mscr_make(t, works[en.prod].scratch);
      double *row = (double *)A.rows + (pots_u64)en.prod * (POTS_ROW_FIXED + A.k);
      pots_measure_item(t, sc, row, en.phase, item, A.k, lane);
    }
  }
}
