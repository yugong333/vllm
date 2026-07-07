// ============================================================================
// Standalone microbenchmark for the software grid / expert / col-stripe
// barriers defined in `src/moe_grid_barrier.h`.
//
// Measures per-call latency of:
//   * grid_barrier<GRID_SIZE>          — 128 arrivals
//   * expert_barrier                   — 8 arrivals, one id (matches the
//                                        production site #2 for one expert
//                                        group)
//   * colstripe_barrier                — 16 arrivals × 8 concurrent ids
//                                        (matches the production site #3)
//
// Grid_Barrier kernels use `__launch_bounds__(BLOCK_SIZE, 1)` to match the
// monokernel's one-block-per-SM occupancy — without it two measurement
// blocks can pack onto one SM and report artificially low latency.
//
// Timing: cudaEvent elapsed time per kernel; sweeps N_BARRIERS in
// {1, 10, 100, 1000} with warmup at the largest N, reporting
// elapsed * 1000 / N as per-call µs.  Launch overhead shows up equally for
// every primitive, so the relative ranking is unaffected; at N = 1000 the
// absolute numbers are barrier-dominated anyway.
//
// Build
// -----
//   nvcc -std=c++17 -arch=sm_90 -O3
//        -I vllm/csrc/moe/moe_monokernel/src
//        vllm/csrc/moe/moe_monokernel/bench_grid_barrier.cu
//        -o /tmp/bench_grid_barrier
//
// Run (on H200)
// -------------
//   /tmp/bench_grid_barrier
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// `moe_grid_barrier.h` is guarded with
// `#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION #error` so it can't be
// included from outside the kernel TU by accident. For the microbench
// we intentionally flip the guard — we are the implementation here, in
// the sense that we consume the device primitive directly.
#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#include "src/moe_grid_barrier.h"
#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION

// ── Configuration ────────────────────────────────────────────────────────

// Must match the production BS8 TMA+WGMMA variant so the result is
// comparable to the end-to-end overhead budget.  128 blocks × 384 threads
// = one block per SM on H200 (132 SMs).
constexpr uint32_t GRID_SIZE = 128u;
constexpr uint32_t BLOCK_SIZE = 384u;

// Arrival-set sizes for the sub-grid barriers.
constexpr uint32_t UP_GRID = 8u;       // Expert_Barrier arrival count
constexpr uint32_t DOWN_GRID = 8u;     // number of concurrent col-stripe
                                       // barriers
constexpr uint32_t DOWN_GROUPS = 16u;  // ColStripe_Barrier arrival count

// Counter-region sizing: one Counter_Pair (2 × uint32_t) per barrier id.
constexpr uint32_t GRID_COUNTERS = 2u;            // one pair
constexpr uint32_t EXPERT_COUNTERS = 16u * 2u;    // 16 pairs (only id=0
                                                  // is hit; provisioned to
                                                  // match the scratchpad)
constexpr uint32_t COLSTRIPE_COUNTERS = 8u * 2u;  // 8 Counter_Pairs
                                                  // — one per col
                                                  // stripe.

// H200 shader clock (2.11 GHz).  Unused by the event-based timing; kept
// for cross-checking against clock64() reads inside device code.
[[maybe_unused]] constexpr double H200_SHADER_GHZ = 2.11;

constexpr int WARMUP_LAUNCHES = 10;

// ── CUDA error helper ────────────────────────────────────────────────────

#define CUDA_CHECK(expr)                                               \
  do {                                                                 \
    cudaError_t _err = (expr);                                         \
    if (_err != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, \
                   __LINE__, cudaGetErrorName(_err),                   \
                   cudaGetErrorString(_err));                          \
      std::exit(2);                                                    \
    }                                                                  \
  } while (0)

// ── Device-side microbench kernels ──────────────────────────────────────

/**
 * @brief Grid_Barrier microbench — every block arrives every iteration.
 */
template <uint32_t GRID_SIZE_STATIC>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void grid_barrier_microbench(
    uint32_t* __restrict__ counters, uint32_t N_BARRIERS) {
  uint32_t phase = 0u;
  for (uint32_t i = 0; i < N_BARRIERS; ++i) {
    moe_monokernel::grid_barrier<GRID_SIZE_STATIC>(counters, phase);
  }
}

/**
 * @brief Expert_Barrier microbench — only the first `UP_GRID = 8` blocks
 *        participate, all calling with the same barrier id (= 0).
 *
 * Blocks outside the arrival set must NOT call `expert_barrier` for this
 * id (caller contract in `moe_grid_barrier.h`), so we gate the call on
 * `blockIdx.x < UP_GRID`. The remaining 120 blocks spin waiting for the
 * kernel to end but do not touch the counter.
 *
 * Seed block is blockIdx.x == 0 (lowest blockIdx.x in the arrival set),
 * matching the production recipe `up_group * UP_GRID` with up_group = 0.
 */
__global__ __launch_bounds__(BLOCK_SIZE, 1) void expert_barrier_microbench(
    uint32_t* __restrict__ expert_counters, uint32_t N_BARRIERS) {
  uint32_t phase = 0u;
  if (blockIdx.x < UP_GRID) {
    for (uint32_t i = 0; i < N_BARRIERS; ++i) {
      moe_monokernel::expert_barrier(expert_counters,
                                     /*expert_id=*/0u,
                                     /*arrival_count=*/UP_GRID,
                                     /*seed_thread_blockidx=*/0u, phase);
    }
  }
}

/**
 * @brief ColStripe_Barrier microbench — all 128 blocks call the barrier
 *        every iteration, using `col_stripe = blockIdx.x % DOWN_GRID`
 *        as the id. `DOWN_GRID = 8` concurrent barriers run, each with
 *        `DOWN_GROUPS = 16` arrivals.
 *
 * Seed block for col stripe c is the block whose `blockIdx.x == c`
 * (matches the production recipe where the Phase-5 writer for col
 * stripe c has blockIdx.x == c).
 */
__global__ __launch_bounds__(BLOCK_SIZE, 1) void colstripe_barrier_microbench(
    uint32_t* __restrict__ colstripe_counters, uint32_t N_BARRIERS) {
  const uint32_t col_stripe = blockIdx.x % DOWN_GRID;
  uint32_t phase = 0u;
  for (uint32_t i = 0; i < N_BARRIERS; ++i) {
    moe_monokernel::colstripe_barrier(colstripe_counters,
                                      /*col_stripe=*/col_stripe,
                                      /*arrival_count=*/DOWN_GROUPS,
                                      /*seed_thread_blockidx=*/col_stripe,
                                      phase);
  }
}

// ── Host-side timing helpers ────────────────────────────────────────────

/// Time a single kernel launch (on the default stream) using
/// `cudaEventRecord` / `cudaEventElapsedTime`. Returns elapsed
/// milliseconds.
template <typename Launch>
static float time_kernel_ms(Launch&& launch) {
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

struct BenchResult {
  uint32_t n_barriers;
  double per_call_us;
};

/// Sweep `N_BARRIERS ∈ {1, 10, 100, 1000}` for a given launch closure.
/// Performs `WARMUP_LAUNCHES` warmup iterations at the largest N before
/// timing.
template <typename Launch>
static void sweep_bench(const char* label, Launch&& launch,
                        BenchResult results[4]) {
  const uint32_t Ns[4] = {1u, 10u, 100u, 1000u};

  // Warm up on the largest N to prime the instruction cache and let the
  // ping-pong slot reach steady-state before the timed runs.
  for (int w = 0; w < WARMUP_LAUNCHES; ++w) {
    launch(Ns[3]);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::printf("%s:\n", label);
  for (int i = 0; i < 4; ++i) {
    const uint32_t N = Ns[i];
    const float ms = time_kernel_ms([&]() { launch(N); });
    const double us_per_call =
        (static_cast<double>(ms) * 1000.0) / static_cast<double>(N);
    results[i] = {N, us_per_call};
    std::printf("  N=%-5u latency = %.3f μs\n", N, us_per_call);
  }
}

// ── main ────────────────────────────────────────────────────────────────

int main(int /*argc*/, char** /*argv*/) {
  // Sanity-check the device is SM90+ (H100 / H200). Matches the
  // production monokernel's target. The microbench itself doesn't use
  // SM90-specific features beyond what `grid_barrier.h` uses (device-
  // scope atomics + threadfences, available from SM60), but the latency
  // numbers are only meaningful against the SM90 co-residency
  // invariant, so we require it.
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  std::printf("Device %d: %s (compute %d.%d, %d SMs)\n", device, props.name,
              props.major, props.minor, props.multiProcessorCount);
  if (props.major < 9) {
    std::fprintf(stderr,
                 "bench_grid_barrier: device has compute capability %d.%d; "
                 "this benchmark expects SM90 (H100) or newer.\n",
                 props.major, props.minor);
    return 77;  // conventional "skip" exit code.
  }
  if (static_cast<uint32_t>(props.multiProcessorCount) < GRID_SIZE) {
    std::fprintf(stderr,
                 "bench_grid_barrier: device has %d SMs but GRID_SIZE = %u. "
                 "The software-barrier co-residency invariant requires "
                 "GRID_SIZE <= SM count, otherwise the kernel can deadlock.\n",
                 props.multiProcessorCount, GRID_SIZE);
    return 2;
  }

  // Allocate + zero-init the three counter regions.
  //   grid:       1  Counter_Pair  = 2 × uint32_t       (8 B)
  //   expert:     16 Counter_Pairs = 32 × uint32_t      (128 B)
  //   colstripe:  8  Counter_Pairs = 16 × uint32_t      (64 B)
  // Mirrors the MoEGemmSpec<Dims>::partial_barrier scratchpad layout.
  // Zero-init is required on first use (moe_grid_barrier.h contract).
  uint32_t* d_grid_counters = nullptr;
  uint32_t* d_expert_counters = nullptr;
  uint32_t* d_colstripe_counters = nullptr;
  CUDA_CHECK(cudaMalloc(&d_grid_counters, GRID_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_expert_counters, EXPERT_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMalloc(&d_colstripe_counters, COLSTRIPE_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_grid_counters, 0, GRID_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMemset(d_expert_counters, 0, EXPERT_COUNTERS * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_colstripe_counters, 0,
                        COLSTRIPE_COUNTERS * sizeof(uint32_t)));

  std::printf("GRID_SIZE=%u, BLOCK_SIZE=%u\n", GRID_SIZE, BLOCK_SIZE);
  std::printf("UP_GRID=%u, DOWN_GRID=%u, DOWN_GROUPS=%u\n\n", UP_GRID,
              DOWN_GRID, DOWN_GROUPS);

  BenchResult grid_results[4]{};
  BenchResult expert_results[4]{};
  BenchResult colstripe_results[4]{};

  sweep_bench(
      "Grid_Barrier (128 arrivals)",
      [&](uint32_t N) {
        grid_barrier_microbench<GRID_SIZE>
            <<<GRID_SIZE, BLOCK_SIZE>>>(d_grid_counters, N);
        CUDA_CHECK(cudaGetLastError());
      },
      grid_results);

  sweep_bench(
      "Expert_Barrier (8 arrivals, 1 concurrent)",
      [&](uint32_t N) {
        expert_barrier_microbench<<<GRID_SIZE, BLOCK_SIZE>>>(d_expert_counters,
                                                             N);
        CUDA_CHECK(cudaGetLastError());
      },
      expert_results);

  sweep_bench(
      "ColStripe_Barrier (16 arrivals, 8 concurrent)",
      [&](uint32_t N) {
        colstripe_barrier_microbench<<<GRID_SIZE, BLOCK_SIZE>>>(
            d_colstripe_counters, N);
        CUDA_CHECK(cudaGetLastError());
      },
      colstripe_results);

  // Summary — compare each sub-grid barrier to the Grid_Barrier at
  // N = 1000, where per-launch overhead is amortized.
  const double grid_us = grid_results[3].per_call_us;
  const double expert_us = expert_results[3].per_call_us;
  const double colstripe_us = colstripe_results[3].per_call_us;

  std::printf("\nSummary (N=1000):\n");
  std::printf(
      "  Grid_Barrier       : %.3f μs   (budget: ≤ 10 μs — %s)\n",
      grid_us, grid_us <= 10.0 ? "PASS" : "FAIL");
  std::printf(
      "  Expert_Barrier     : %.3f μs   (%.1f%% of Grid, target ≤ 50%% — %s)\n",
      expert_us, grid_us > 0.0 ? (100.0 * expert_us / grid_us) : 0.0,
      (grid_us > 0.0 && expert_us <= 0.5 * grid_us) ? "PASS" : "FAIL");
  std::printf(
      "  ColStripe_Barrier  : %.3f μs   (%.1f%% of Grid, target ≤ 50%% — %s)\n",
      colstripe_us, grid_us > 0.0 ? (100.0 * colstripe_us / grid_us) : 0.0,
      (grid_us > 0.0 && colstripe_us <= 0.5 * grid_us) ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(d_grid_counters));
  CUDA_CHECK(cudaFree(d_expert_counters));
  CUDA_CHECK(cudaFree(d_colstripe_counters));
  return 0;
}
