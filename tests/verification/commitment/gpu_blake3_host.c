/* Host build of verification/commitment/gpu_blake3_core.h (decision F23): runs the kernel's
 * work decomposition, one thread after another, so its tile and tree logic is tested on a
 * machine with no GPU. The Python test builds a plan with the same code the GPU path uses and
 * calls pots_host_stage once per stage. */
#include <stdlib.h>
#include "gpu_blake3_core.h"

size_t pots_host_sizeof_params(void) { return sizeof(pots_params_t); }
size_t pots_host_sizeof_leaf(void) { return sizeof(pots_leaf_t); }
size_t pots_host_offsets(size_t *out) {
  pots_params_t *p = NULL;
  size_t n = 0;
  out[n++] = (size_t)&p->n;
  out[n++] = (size_t)&p->stage;
  out[n++] = (size_t)&p->total_tiles;
  out[n++] = (size_t)&p->leaf;
  out[n++] = (size_t)&p->tile_prefix;
  out[n++] = (size_t)&p->in_off;
  out[n++] = (size_t)&p->out_off;
  out[n++] = (size_t)&p->nodes;
  out[n++] = (size_t)&p->digests;
  out[n++] = (size_t)&p->first_bad;
  return n;
}
unsigned host_g(void) { return POTS_G; }
unsigned host_r(void) { return POTS_R; }

static void chunk_stage(const pots_params_t *P) {
  pots_chunk_smem_t *sm = malloc(sizeof *sm);
  pots_u32(*cv)[8] = malloc(sizeof(pots_u32[POTS_CHUNK_THREADS][8]));
  pots_u64 blocks = (P->total_tiles + POTS_TILES_PER_BLOCK - 1) / POTS_TILES_PER_BLOCK;
  for (pots_u64 blk = 0; blk < blocks; blk++) {
    for (pots_u32 t = 0; t < POTS_CHUNK_THREADS; t++) pots_chunk_setup(sm, P, blk, t);
    for (pots_u32 b = 0; b < 16; b++) {
      for (pots_u32 t = 0; t < POTS_CHUNK_THREADS; t++) pots_chunk_load(sm, P, t, b);
      for (pots_u32 t = 0; t < POTS_CHUNK_THREADS; t++) pots_chunk_compute(sm, P, t, b, cv[t]);
    }
    for (pots_u32 t = 0; t < POTS_CHUNK_THREADS; t++) pots_chunk_store(sm, t, cv[t]);
    for (pots_u32 l = 0; l < POTS_G_LOG; l++)
      for (pots_u32 t = 0; t < POTS_CHUNK_THREADS; t++) pots_chunk_level(sm, P, t, l);
    for (pots_u32 t = 0; t < POTS_CHUNK_THREADS; t++) pots_chunk_finish(sm, P, t);
  }
  free(cv);
  free(sm);
}

static void reduce_stage(const pots_params_t *P) {
  pots_reduce_smem_t *sm = malloc(sizeof *sm);
  pots_reduce_thread_t st[POTS_REDUCE_THREADS];
  for (pots_u64 blk = 0; blk < P->total_tiles; blk++) {
    for (pots_u32 t = 0; t < POTS_REDUCE_THREADS; t++) pots_reduce_setup(sm, P, blk, t, &st[t]);
    for (pots_u32 l = 0; l < POTS_R_LOG; l++)
      for (pots_u32 t = 0; t < POTS_REDUCE_THREADS; t++) pots_reduce_level(sm, &st[t], t, l);
    for (pots_u32 t = 0; t < POTS_REDUCE_THREADS; t++) pots_reduce_finish(sm, P, t, &st[t]);
  }
  free(sm);
}

void pots_host_stage(const pots_params_t *P) {
  if (P->stage == 0) chunk_stage(P);
  else reduce_stage(P);
}
