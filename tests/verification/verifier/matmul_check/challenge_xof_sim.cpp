// The challenge kernel (challenge_xof.cu) run on the host: every (block, thread) of a grid in
// turn, with the CUDA keywords stubbed, so the Mac checks the kernel's own grid-stride loop.
// Built by test_challenge_xof_host.py.
#define __global__
#define __launch_bounds__(...)

struct sim_dim3 {
  unsigned x;
};
static sim_dim3 threadIdx, blockIdx, blockDim, gridDim;

#include "challenge_xof.cu"

extern "C" void pots_sim_xof(const pots_xparams_t *P, unsigned grid, unsigned block) {
  gridDim.x = grid;
  blockDim.x = block;
  for (unsigned b = 0; b < grid; b++)
    for (unsigned t = 0; t < block; t++) {
      blockIdx.x = b;
      threadIdx.x = t;
      pots_challenge_xof(*P);
    }
}
