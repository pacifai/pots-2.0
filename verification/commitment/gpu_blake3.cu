// The BLAKE3 hash kernels (decision F23). Compiled by NVRTC at run time (gpu_blake3.py), with
// gpu_blake3_core.h passed in memory. Every decision about what is hashed lives in the core;
// this file only maps the core's phases onto threads, shared memory and barriers.
#include "gpu_blake3_core.h"

// One block hashes POTS_TILES_PER_BLOCK groups of G chunks of 1 KiB: one thread a chunk. It
// walks the 16 blocks of its chunk, the block's threads loading 64 contiguous bytes at a time
// for each chunk into shared memory (and testing finiteness as they load), and the group's
// chaining values then reduce in shared memory.
extern "C" __global__ void __launch_bounds__(POTS_CHUNK_THREADS)
pots_chunk_stage(pots_params_t P) {
  __shared__ pots_chunk_smem_t sm;
  const pots_u32 tid = threadIdx.x;
  pots_u32 cv[8];
  pots_chunk_setup(&sm, &P, blockIdx.x, tid);
  __syncthreads();
  for (pots_u32 b = 0; b < 16; b++) {
    pots_chunk_load(&sm, &P, tid, b);
    __syncthreads();
    pots_chunk_compute(&sm, &P, tid, b, cv);
    __syncthreads();
  }
  pots_chunk_store(&sm, tid, cv);
  __syncthreads();
  for (pots_u32 l = 0; l < POTS_G_LOG; l++) {
    pots_chunk_level(&sm, &P, tid, l);
    __syncthreads();
  }
  pots_chunk_finish(&sm, &P, tid);
}

// One block reduces one tile of R nodes of one leaf's level stage-1 to one node.
extern "C" __global__ void __launch_bounds__(POTS_REDUCE_THREADS)
pots_reduce_stage(pots_params_t P) {
  __shared__ pots_reduce_smem_t sm;
  const pots_u32 tid = threadIdx.x;
  pots_reduce_thread_t st;
  pots_reduce_setup(&sm, &P, blockIdx.x, tid, &st);
  __syncthreads();
  for (pots_u32 l = 0; l < POTS_R_LOG; l++) {
    pots_reduce_level(&sm, &st, tid, l);
    __syncthreads();
  }
  pots_reduce_finish(&sm, &P, tid, &st);
}
