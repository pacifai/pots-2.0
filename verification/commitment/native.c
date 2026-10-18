/* The protocol's native BLAKE3 batch, over BLAKE3's C library (decision F15b).
 *
 *   pots_challenges   check 5's challenge matrices, bit for bit the numpy path's.
 *   pots_hash_leaves  leaf digests BLAKE3(0x00 || header || payload) over many leaves, with
 *                     the finiteness test fused into the same pass over the payload.
 *   pots_all_finite   the finiteness test alone, over one buffer.
 *   pots_merkle_tree  every subtree hash of the RFC 6962 tree, in pre-order.
 *
 * Challenge entries: product m's vector j is the keyed XOF of the label 0x10 || m (4 bytes
 * BE) || j (1 byte), read as little-endian uint32 words w, each giving the float32 entry
 * (2(w >> 8) + 1 - 2^24) / 2^24. Item i's matrix is [widths[i], k], row-major, at
 * out + offsets[i], with vector j in column j - 1. Both steps are exact in float32.
 *
 * Challenge work is handed out in units of ROWS rows of one item, all k vectors at once. A
 * thread then owns a contiguous slab of the output, so threads don't write the same cache
 * lines (each vector alone is a strided column, and splitting by vector would interleave the
 * threads).
 */

#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "blake3.h"
#include "blake3_impl.h"

#define MAX_THREADS 256

/* Run fn(arg) on the calling thread and on up to t - 1 new ones, and return once all are
 * done. Returns 0, or -2 when a thread couldn't be started; the callers' work queues let the
 * threads that did start finish the work, so the output is still complete. */
static int run_threads(void *(*fn)(void *), void *arg, uint32_t t) {
  pthread_t tids[MAX_THREADS];
  uint32_t started = 0;
  int rc = 0;
  if (t > MAX_THREADS) t = MAX_THREADS;
  for (; started + 1 < t; started++)
    if (pthread_create(&tids[started], NULL, fn, arg) != 0) {
      rc = -2;
      break;
    }
  fn(arg);
  for (uint32_t i = 0; i < started; i++) pthread_join(tids[i], NULL);
  return rc;
}

/* ---- challenges ---- */

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

static void *challenge_worker(void *arg) {
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
  int rc = run_threads(challenge_worker, &job, t);
  free(first);
  return rc;
}

/* ---- finiteness ---- */

#define KIND_F32 1
#define KIND_BF16 2
#define KIND_F16 3

static inline size_t kind_size(uint8_t kind) { return kind == KIND_F32 ? 4 : 2; }

/* 1 when every value in the `nbytes` bytes at `p` is finite, 0 when one has all its exponent
 * bits set (NaN or Inf). A trailing partial value is ignored. Reads go through memcpy, so `p`
 * needs no alignment. */
static int block_finite(const uint8_t *p, size_t nbytes, uint8_t kind) {
  uint32_t bad = 0;
  if (kind == KIND_F32) {
    for (size_t i = 0; i + 4 <= nbytes; i += 4) {
      uint32_t w;
      memcpy(&w, p + i, 4);
      bad |= (w & 0x7f800000u) == 0x7f800000u;
    }
  } else if (kind == KIND_BF16 || kind == KIND_F16) {
    uint16_t mask = kind == KIND_BF16 ? 0x7f80 : 0x7c00;
    for (size_t i = 0; i + 2 <= nbytes; i += 2) {
      uint16_t h;
      memcpy(&h, p + i, 2);
      bad |= (uint16_t)(h & mask) == mask;
    }
  }
  return !bad;
}

/* ---- leaf hashing ---- */

#define BLOCK_BYTES (64 << 10) /* payload bytes checked, then hashed, at a time */
#define GRAB_BYTES (1 << 20)   /* about this much payload per grab of leaves */
#define BYTES_PER_THREAD (1 << 20)

typedef struct {
  size_t n;
  const uint8_t *hdr_buf;
  const uint64_t *hdr_off;
  const uint32_t *hdr_len;
  const uint32_t *hdr_idx;
  const uint64_t *payload_ptr;
  const uint64_t *payload_len;
  const uint8_t *kind;
  uint8_t *out;
  size_t per_grab;
  atomic_size_t next;
  atomic_size_t bad; /* the smallest index known to hold a non-finite value, else n */
} leaf_job_t;

static void note_bad(leaf_job_t *job, size_t i) {
  size_t cur = atomic_load(&job->bad);
  while (i < cur && !atomic_compare_exchange_weak(&job->bad, &cur, i)) {
  }
}

/* Leaf i's digest into out + 32 i; 0 when its payload holds a non-finite value. */
static int hash_leaf(const leaf_job_t *job, size_t i) {
  static const uint8_t prefix = 0x00;
  uint32_t u = job->hdr_idx[i];
  const uint8_t *p = (const uint8_t *)(uintptr_t)job->payload_ptr[i];
  uint64_t len = job->payload_len[i];
  uint8_t kind = job->kind[i];
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, &prefix, 1);
  if (job->hdr_len[u]) blake3_hasher_update(&h, job->hdr_buf + job->hdr_off[u], job->hdr_len[u]);
  for (uint64_t pos = 0; pos < len;) {
    size_t take = len - pos < BLOCK_BYTES ? (size_t)(len - pos) : BLOCK_BYTES;
    if (kind && !block_finite(p + pos, take, kind)) return 0;
    blake3_hasher_update(&h, p + pos, take);
    pos += take;
  }
  blake3_hasher_finalize(&h, job->out + 32 * i, 32);
  return 1;
}

static void *leaf_worker(void *arg) {
  leaf_job_t *job = arg;
  for (;;) {
    size_t lo = atomic_fetch_add(&job->next, job->per_grab);
    if (lo >= job->n) break;
    size_t hi = lo + job->per_grab < job->n ? lo + job->per_grab : job->n;
    for (size_t i = lo; i < hi; i++) {
      if (i > atomic_load(&job->bad)) break;
      if (!hash_leaf(job, i)) {
        note_bad(job, i);
        break;
      }
    }
  }
  return NULL;
}

/* Leaf i's digest is BLAKE3(0x00 || header || payload), written to out + 32 i, where the
 * header is hdr_buf[hdr_off[u] : + hdr_len[u]] for u = hdr_idx[i] and the payload is
 * payload_len[i] bytes at the address payload_ptr[i]. kind[i] says which values to test:
 * 0 none, 1 float32, 2 bfloat16, 3 float16.
 *
 * Returns the smallest i whose payload holds a non-finite value, or n when none does. The
 * digests before it are valid; those at or after it are unspecified. */
size_t pots_hash_leaves(size_t n, const uint8_t *hdr_buf, const uint64_t *hdr_off,
                        const uint32_t *hdr_len, const uint32_t *hdr_idx,
                        const uint64_t *payload_ptr, const uint64_t *payload_len,
                        const uint8_t *kind, uint8_t *out, uint32_t n_threads) {
  if (n == 0) return 0;
  uint64_t total = 0;
  for (size_t i = 0; i < n; i++) total += payload_len[i] + hdr_len[hdr_idx[i]];
  leaf_job_t job = {n, hdr_buf, hdr_off, hdr_len, hdr_idx, payload_ptr, payload_len,
                    kind, out, 1, 0, n};
  uint64_t avg = total / n;
  job.per_grab = avg >= GRAB_BYTES ? 1 : (size_t)(GRAB_BYTES / (avg ? avg : 1));
  uint64_t useful = total / BYTES_PER_THREAD;
  uint32_t t = n_threads < 1 ? 1 : n_threads;
  if (useful < t) t = useful < 1 ? 1 : (uint32_t)useful;
  run_threads(leaf_worker, &job, t);
  return atomic_load(&job.bad);
}

/* ---- finiteness of one buffer ---- */

#define FINITE_BYTES_PER_THREAD (4 << 20)

typedef struct {
  const uint8_t *p;
  uint64_t nbytes;
  uint8_t kind;
  uint32_t t;
  atomic_uint next;
  atomic_int ok;
} finite_job_t;

static void *finite_worker(void *arg) {
  finite_job_t *job = arg;
  size_t sz = kind_size(job->kind);
  uint64_t elems = job->nbytes / sz;
  for (;;) {
    uint32_t r = atomic_fetch_add(&job->next, 1);
    if (r >= job->t) break;
    if (!atomic_load(&job->ok)) break;
    uint64_t lo = elems * r / job->t * sz, hi = elems * (r + 1) / job->t * sz;
    for (uint64_t pos = lo; pos < hi;) {
      size_t take = hi - pos < BLOCK_BYTES ? (size_t)(hi - pos) : BLOCK_BYTES;
      if (!block_finite(job->p + pos, take, job->kind)) {
        atomic_store(&job->ok, 0);
        return NULL;
      }
      pos += take;
      if (!atomic_load(&job->ok)) return NULL;
    }
  }
  return NULL;
}

/* 1 when every value in the buffer is finite (kind as above; kind 0 always), else 0. The
 * buffer is split into contiguous ranges, aligned to the value size, over up to n_threads
 * threads. */
int pots_all_finite(const uint8_t *p, uint64_t nbytes, uint8_t kind, uint32_t n_threads) {
  if (nbytes == 0 || kind < KIND_F32 || kind > KIND_F16) return 1;
  finite_job_t job = {p, nbytes, kind, 1, 0, 1};
  uint64_t useful = nbytes / FINITE_BYTES_PER_THREAD;
  uint32_t t = n_threads < 1 ? 1 : n_threads;
  if (useful < t) t = useful < 1 ? 1 : (uint32_t)useful;
  job.t = t;
  run_threads(finite_worker, &job, t);
  return atomic_load(&job.ok);
}

/* ---- Merkle tree ---- */

/* The subtree hashes of leaves [lo, hi), pre-order from node p: the root of the range at p,
 * its left child (the first k leaves, k the largest power of two below the count) at p + 1
 * and its right child at p + 2k, since the left subtree holds 2k - 1 nodes. */
static void build_tree(const uint8_t *leaves, size_t lo, size_t hi, uint8_t *nodes, size_t p) {
  if (hi - lo == 1) {
    memcpy(nodes + 32 * p, leaves + 32 * lo, 32);
    return;
  }
  size_t k = 1;
  while (k * 2 < hi - lo) k *= 2;
  build_tree(leaves, lo, lo + k, nodes, p + 1);
  build_tree(leaves, lo + k, hi, nodes, p + 2 * k);
  uint8_t buf[65];
  buf[0] = 0x01;
  memcpy(buf + 1, nodes + 32 * (p + 1), 32);
  memcpy(buf + 33, nodes + 32 * (p + 2 * k), 32);
  blake3_hasher h;
  blake3_hasher_init(&h);
  blake3_hasher_update(&h, buf, sizeof buf);
  blake3_hasher_finalize(&h, nodes + 32 * p, 32);
}

/* The 2n - 1 subtree hashes of the RFC 6962 tree over n >= 1 leaf digests, pre-order. */
void pots_merkle_tree(size_t n, const uint8_t *leaves, uint8_t *nodes) {
  build_tree(leaves, 0, n, nodes, 0);
}

/* Lanes of the widest SIMD path BLAKE3's dispatch picked on this CPU: 16 AVX-512, 8 AVX2,
 * 4 SSE4.1, SSE2 or NEON, 1 portable. */
size_t pots_simd_degree(void) { return blake3_simd_degree(); }

const char *pots_blake3_version(void) { return blake3_version(); }
