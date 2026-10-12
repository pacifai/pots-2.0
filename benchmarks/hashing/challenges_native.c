/* Batched challenge generation (P8b) over BLAKE3's C library, for E5(a').
 *
 * Product m's vector j is the keyed XOF of the label 0x10 || m (4 bytes BE) || j (1 byte),
 * read as little-endian uint32 words w, each giving the float32 entry
 * (2(w >> 8) + 1 - 2^24) / 2^24. Item i's matrix is [widths[i], k], row-major, at
 * out + offsets[i], with vector j in column j - 1. Both steps are exact in float32, so the
 * output matches the numpy path bit for bit.
 *
 * Work is handed out in units of ROWS rows of one item, all k vectors at once. A thread then
 * owns a contiguous slab of the output, so threads don't write the same cache lines (each
 * vector alone is a strided column, and splitting by vector would interleave the threads).
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "blake3.h"
#include "blake3_impl.h"

#define ROWS 256 /* rows per unit: 1 KB of XOF output per vector, 16 BLAKE3 blocks */
#define UNITS_PER_GRAB 8
/* Below this much work per thread, use fewer threads: 16384 entries are 1024 output blocks,
 * about 50 us on one Apple M5 core, against tens of us to start and join a thread. */
#define ENTRIES_PER_THREAD 16384

typedef struct {
  const uint8_t *key;
  uint32_t k;
  size_t n;
  const uint32_t *ms;
  const uint32_t *widths;
  const uint64_t *offsets;
  float *out;
  const uint64_t *first; /* first[i]: item i's first unit; first[n]: the number of units */
  atomic_size_t next;
} job_t;

/* Rows [e0, e1) of item m's matrix, every vector. */
static void fill_rows(const uint8_t *key, uint32_t m, uint32_t k, uint32_t e0, uint32_t e1,
                      float *mat) {
  uint8_t buf[4 * ROWS];
  size_t rows = e1 - e0;
  for (uint32_t j = 1; j <= k; j++) {
    uint8_t label[6] = {0x10, (uint8_t)(m >> 24), (uint8_t)(m >> 16), (uint8_t)(m >> 8),
                        (uint8_t)m, (uint8_t)j};
    blake3_hasher h;
    blake3_hasher_init_keyed(&h, key);
    blake3_hasher_update(&h, label, sizeof label);
    blake3_hasher_finalize_seek(&h, 4 * (uint64_t)e0, buf, 4 * rows);
    float *col = mat + (size_t)e0 * k + (j - 1);
    for (size_t e = 0; e < rows; e++) {
      const uint8_t *b = buf + 4 * e;
      uint32_t w = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 |
                   (uint32_t)b[3] << 24;
      int32_t numerator = 2 * (int32_t)(w >> 8) + 1 - (1 << 24);
      col[e * k] = (float)numerator / 16777216.0f;
    }
  }
}

static void *worker(void *arg) {
  job_t *job = arg;
  size_t total = job->first[job->n];
  for (;;) {
    size_t lo = atomic_fetch_add(&job->next, UNITS_PER_GRAB);
    if (lo >= total) break;
    size_t hi = lo + UNITS_PER_GRAB < total ? lo + UNITS_PER_GRAB : total;
    size_t a = 0, b = job->n; /* the item holding unit lo: first[a] <= lo < first[a + 1] */
    while (b - a > 1) {
      size_t mid = a + (b - a) / 2;
      if (job->first[mid] <= lo) a = mid;
      else b = mid;
    }
    for (size_t u = lo; u < hi; u++) {
      while (job->first[a + 1] <= u) a++;
      uint32_t e0 = (uint32_t)(u - job->first[a]) * ROWS;
      uint32_t e1 = e0 + ROWS < job->widths[a] ? e0 + ROWS : job->widths[a];
      fill_rows(job->key, job->ms[a], job->k, e0, e1, job->out + job->offsets[a]);
    }
  }
  return NULL;
}

/* Fill every item's matrix on up to n_threads threads (the caller's included). Returns 0;
 * -1 when out of memory (nothing written); -2 when a thread couldn't be started (the work
 * then finishes on the threads that did start, so the output is still complete). */
int pots_challenges(const uint8_t key[32], uint32_t k, size_t n, const uint32_t *ms,
                    const uint32_t *widths, const uint64_t *offsets, float *out,
                    uint32_t n_threads) {
  if (n == 0) return 0;
  uint64_t *first = malloc((n + 1) * sizeof *first);
  if (!first) return -1;
  uint64_t entries = 0;
  first[0] = 0;
  for (size_t i = 0; i < n; i++) {
    first[i + 1] = first[i] + (widths[i] + ROWS - 1) / ROWS;
    entries += (uint64_t)widths[i] * k;
  }
  job_t job = {key, k, n, ms, widths, offsets, out, first, 0};
  uint64_t useful = entries / ENTRIES_PER_THREAD;
  if (useful > first[n]) useful = first[n];
  uint32_t t = n_threads < 1 ? 1 : n_threads;
  if (useful < t) t = useful < 1 ? 1 : (uint32_t)useful;
  if (t > 256) t = 256;
  pthread_t tids[256];
  uint32_t started = 0;
  int rc = 0;
  for (; started + 1 < t; started++)
    if (pthread_create(&tids[started], NULL, worker, &job) != 0) {
      rc = -2;
      break;
    }
  worker(&job);
  for (uint32_t i = 0; i < started; i++) pthread_join(tids[i], NULL);
  free(first);
  return rc;
}

/* Lanes of the widest SIMD path BLAKE3's dispatch picked on this CPU: 16 AVX-512, 8 AVX2,
 * 4 SSE4.1, SSE2 or NEON, 1 portable. */
size_t pots_simd_degree(void) { return blake3_simd_degree(); }

const char *pots_blake3_version(void) { return blake3_version(); }
