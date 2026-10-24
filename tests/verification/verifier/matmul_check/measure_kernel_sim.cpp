// The level kernel (measure_kernel.cu) run on the host, one block at a time with one OS thread
// per CUDA thread, so the Mac checks the CUDA source's own control flow (tile loads, barriers,
// shuffles, the block's row mapping, the grid stride) against the host build. Nothing here is
// the kernel: the CUDA keywords are stubbed, __syncthreads and __syncwarp are barriers,
// __shfl_down_sync reads the warp's slots between two warp barriers, and `__shared__` arrays are
// statics (blocks run one after the other). A divergent barrier or shuffle hangs, which the test
// turns into a failure by its timeout. Built by test_measure_kernel_sim_host.py with
// -ffp-contract=off and the geometry as -D options.
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __shared__ static
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(x)

struct sim_dim3 {
  unsigned x;
};
static thread_local sim_dim3 threadIdx, blockIdx;
static sim_dim3 gridDim;

class SimBarrier {
 public:
  explicit SimBarrier(unsigned n) : n_(n) {}
  void wait() {
    std::unique_lock<std::mutex> lk(m_);
    unsigned gen = gen_;
    if (++count_ == n_) {
      count_ = 0;
      gen_++;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen != gen_; });
    }
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  unsigned n_, count_ = 0, gen_ = 0;
};

static SimBarrier *g_block;
static std::vector<SimBarrier *> g_warp;
static double g_slot[64][32];

static void __syncthreads() { g_block->wait(); }
static void __syncwarp() { g_warp[threadIdx.x >> 5]->wait(); }

template <typename T>
static T __shfl_down_sync(unsigned, T v, unsigned off) {
  const unsigned w = threadIdx.x >> 5, lane = threadIdx.x & 31u;
  g_slot[w][lane] = (double)v;
  g_warp[w]->wait();
  T out = lane + off < 32u ? (T)g_slot[w][lane + off] : v;
  g_warp[w]->wait();
  return out;
}

#include "measure_kernel.cu"

extern "C" void sim_launch(const pots_mlargs_t *args, unsigned grid) {
  const unsigned nt = POTS_MK_WARPS * 32u;
  SimBarrier block(nt);
  std::vector<SimBarrier *> warps;
  for (unsigned w = 0; w < POTS_MK_WARPS; w++) warps.push_back(new SimBarrier(32));
  g_block = &block;
  g_warp = warps;
  gridDim.x = grid;
  for (unsigned b = 0; b < grid; b++) {
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < nt; t++)
      ts.emplace_back([=] {
        threadIdx.x = t;
        blockIdx.x = b;
        pots_measure_level(*args);
      });
    for (auto &th : ts) th.join();
  }
  for (auto *w : warps) delete w;
}
