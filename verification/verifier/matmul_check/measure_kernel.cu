// The fused measuring kernel (FS37, decision O2). Compiled by NVRTC at run time
// (measure_kernel.py) with measure_core.h and measure_phases.h passed in memory, with
// --fmad=false. Every decision about what is computed lives in those headers; this file maps
// their items and steps onto warps, lanes and shuffles. One launch per phase; the phases and
// their order are in measure_phases.h.
#include "measure_core.h"
#include "measure_phases.h"

#define POTS_MK_WARPS 8u /* warps per block */
#define POTS_FULL 0xffffffffu

// One launch's argument, passed by value.
typedef struct {
  pots_u64 prods;   // pots_mprod_t[n_prod]
  pots_u64 works;   // pots_mwork_t[n_prod + 1]
  pots_u64 rows;    // float64 [n_prod, ROW_FIXED + k]
  pots_u64 n_prod;
  pots_u64 total;   // items of this phase
  pots_u32 phase;
  pots_u32 k;
} pots_mkargs_t;

extern "C" __global__ void __launch_bounds__(POTS_MK_WARPS * 32)
pots_measure_phase(pots_mkargs_t A) {
  const pots_u32 lane = threadIdx.x & 31u;
  const pots_u64 stride = (pots_u64)gridDim.x * POTS_MK_WARPS;
  const pots_mprod_t *prods = (const pots_mprod_t *)A.prods;
  const pots_mwork_t *works = (const pots_mwork_t *)A.works;
  for (pots_u64 g = (pots_u64)blockIdx.x * POTS_MK_WARPS + (threadIdx.x >> 5); g < A.total;
       g += stride) {
    pots_u64 item;
    const pots_u64 pi = pots_mfind(works, A.n_prod, A.phase, g, &item);
    const pots_mprod_t *t = prods + pi;
    const pots_mscr_t sc = pots_mscr_make(t, works[pi].scratch);
    double *row = (double *)A.rows + pi * (POTS_ROW_FIXED + A.k);
    const pots_u32 steps = pots_phase_steps(t, A.phase, (pots_i64)item);
    for (pots_u32 s = 0; s < steps; s++) {
      const double part = pots_phase_partial(t, &sc, A.phase, (pots_i64)item, s, lane);
      double v;
      // The halving tree: lane i takes s[i] + s[i + off] for off = 16, 8, 4, 2, 1; lane 0 ends
      // with the result (the other lanes' values are never used).
      if (pots_phase_double(t, A.phase, (pots_i64)item, s)) {
        v = part;
        for (pots_u32 off = 16; off >= 1; off >>= 1) v = v + __shfl_down_sync(POTS_FULL, v, off);
      } else {
        float f = (float)part;
        for (pots_u32 off = 16; off >= 1; off >>= 1) f = f + __shfl_down_sync(POTS_FULL, f, off);
        v = (double)f;
      }
      if (lane == 0) pots_phase_store(t, &sc, row, A.phase, (pots_i64)item, s, v);
      __syncwarp();
    }
  }
}
