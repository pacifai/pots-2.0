/* Host build of the challenge XOF core (challenge_xof.h, FS41), for
 * test_challenge_xof_host.py. */
#include <stddef.h>

#include "challenge_xof.h"

size_t pots_xof_sizeof_params(void) { return sizeof(pots_xparams_t); }
size_t pots_xof_sizeof_item(void) { return sizeof(pots_xitem_t); }

size_t pots_xof_offsets(size_t *out) {
  size_t n = 0;
  out[n++] = offsetof(pots_xparams_t, key);
  out[n++] = offsetof(pots_xparams_t, k);
  out[n++] = offsetof(pots_xparams_t, n);
  out[n++] = offsetof(pots_xparams_t, units);
  out[n++] = offsetof(pots_xparams_t, items);
  out[n++] = offsetof(pots_xparams_t, out);
  out[n++] = offsetof(pots_xitem_t, m);
  out[n++] = offsetof(pots_xitem_t, width);
  out[n++] = offsetof(pots_xitem_t, unit0);
  out[n++] = offsetof(pots_xitem_t, out_off);
  return n;
}

pots_u32 pots_xof_find_host(const pots_xparams_t *P, pots_u64 u) { return pots_xof_find(P, u); }

void pots_host_xof(const pots_xparams_t *P) {
  for (pots_u64 u = 0; u < P->units; u++) pots_xof_unit(P, u);
}
