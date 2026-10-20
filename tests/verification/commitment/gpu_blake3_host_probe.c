/* The host harness of gpu_blake3_host.c with every 4-byte memcpy of the core (the one in
 * pots_load32, the stand-in for the kernel's aligned 32-bit load) counted, and counted again
 * when its source is not 4-byte aligned, where the GPU's load would fault. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#undef memcpy
static unsigned long probe_wide, probe_misaligned;
static void *probe_memcpy(void *d, const void *s, size_t n) {
  if (n == 4) {
    probe_wide++;
    if ((uintptr_t)s & 3) probe_misaligned++;
  }
  return (memcpy)(d, s, n);
}
#define memcpy(d, s, n) probe_memcpy(d, s, n)
#include "gpu_blake3_host.c"
#undef memcpy

void probe_reset(void) { probe_wide = probe_misaligned = 0; }
void probe_counts(unsigned long *out) {
  out[0] = probe_wide;
  out[1] = probe_misaligned;
}
unsigned probe_load32(const unsigned char *p) { return pots_load32(p); }
