// FS41: draws check 5's challenge entries on the GPU, one pots_xof_unit per thread step.
#include "challenge_xof.h"

extern "C" __global__ void __launch_bounds__(256) pots_challenge_xof(pots_xparams_t P) {
  pots_u64 step = (pots_u64)gridDim.x * blockDim.x;
  for (pots_u64 u = (pots_u64)blockIdx.x * blockDim.x + threadIdx.x; u < P.units; u += step)
    pots_xof_unit(&P, u);
}
