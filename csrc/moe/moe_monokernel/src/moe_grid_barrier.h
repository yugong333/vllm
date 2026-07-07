#pragma once
#ifndef MOE_GRID_BARRIER_H
  #define MOE_GRID_BARRIER_H

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cstdint>

// Software grid / sub-grid barriers for the MoE monokernel.
//
// Replace `cooperative_groups::this_grid().sync()` so the kernel launches
// via plain `cudaLaunchKernel` and can be captured into a CUDA Graph.
//
// Deadlock safety: a spin barrier is only safe when the grid runs in one
// temporal wave.  The monokernel enforces one-block-per-SM co-residency
// (GRID_SIZE <= SM count, `__launch_bounds__(BLOCK_SIZE, 1)`, opt-in
// dynamic SHM > half an SM's budget); the host launcher checks the runtime
// half.
//
// Protocol (shared by all variants; per (region, id) counter pair):
//   1. `__syncthreads()`; thread 0 `__threadfence()` releases the block's
//      pre-barrier global writes.
//   2. The seed block `atomicExch`es SEED = 0x80000000 - (arrival_count-1)
//      and folds back any early arrivals that landed before the exchange:
//      the entry value's high bit is the PREVIOUS call's exit marker (not
//      a count), so only `prior & 0x7FFFFFFF` is folded.  This preserves
//      the invariant SEED + (arrival_count - 1) = 0x80000000 regardless of
//      the exchange/add interleaving.  Non-seed blocks `atomicAdd(+1)`.
//   3. Every thread spins on `atomicAdd(c, 0)` (an uncached device-scope
//      read; compiles to ld.acquire.gpu on SM90) until bit 31 sets, then
//      `__threadfence()` + `__syncthreads()` (acquire + re-gather).
//   4. `++phase` flips the ping-pong slot so a block racing one call ahead
//      cannot observe the previous call's exit state on the same slot.
//
// Counters must be zero on first use (host zeroes the scratchpad once);
// the seed exchange makes them self-maintaining afterwards.

namespace moe_monokernel {

/**
 * @brief Grid-wide software barrier.
 *
 * Blocks until all GRID_SIZE_STATIC blocks arrive; on exit every prior
 * global write of any block is visible to every subsequent read.  Block 0
 * seeds.  `counters` points at a two-slot ping-pong pair (e.g.
 * `&spec->grid_barrier.slot[0]`); `phase` is a register-resident counter
 * initialized to 0 at kernel entry.
 */
template <uint32_t GRID_SIZE_STATIC>
__device__ __forceinline__ void grid_barrier(uint32_t* __restrict__ counters,
                                             uint32_t& phase) {
  // A single-block grid has no cross-block ordering to enforce.
  if constexpr (GRID_SIZE_STATIC == 1) {
    ++phase;
    return;
  }

  constexpr uint32_t SEED = 0x80000000u - (GRID_SIZE_STATIC - 1u);
  const uint32_t slot = phase & 1u;
  uint32_t* c = counters + slot;

  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();

    if (blockIdx.x == 0) {
      const uint32_t prior = atomicExch(c, SEED);
      const uint32_t to_fold = prior & 0x7FFFFFFFu;
      if (to_fold != 0u) {
        atomicAdd(c, to_fold);
      }
    } else {
      atomicAdd(c, 1u);
    }
  }

  while ((atomicAdd(c, 0u) & 0x80000000u) == 0u) {
  }

  __threadfence();
  __syncthreads();
  ++phase;
}

/**
 * @brief Sub-grid software barrier over a caller-specified arrival set.
 *
 * Same protocol as grid_barrier; the slot address is
 * `counter_region + id * 2 + (phase & 1)` (one Counter_Pair per id, so
 * disjoint ids don't interfere) and the seed is thread 0 of the block with
 * `blockIdx.x == seed_thread_blockidx` (by convention the lowest blockIdx
 * in the arrival set).
 *
 * Callers not in the arrival set for `id` must not call this for that id.
 * `arrival_count` is a runtime arg (compile-time constant at every call
 * site, so the compiler folds SEED and the degenerate gate) because the
 * primitive is shared by call sites with different counts.
 */
__device__ __forceinline__ void partial_barrier(
    uint32_t* __restrict__ counter_region, uint32_t id, uint32_t arrival_count,
    uint32_t seed_thread_blockidx, uint32_t& phase) {
  if (arrival_count == 1u) {
    ++phase;
    return;
  }

  const uint32_t SEED = 0x80000000u - (arrival_count - 1u);
  const uint32_t slot = phase & 1u;
  uint32_t* c = counter_region + id * 2u + slot;

  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();

    if (blockIdx.x == seed_thread_blockidx) {
      const uint32_t prior = atomicExch(c, SEED);
      const uint32_t to_fold = prior & 0x7FFFFFFFu;
      if (to_fold != 0u) {
        atomicAdd(c, to_fold);
      }
    } else {
      atomicAdd(c, 1u);
    }
  }

  while ((atomicAdd(c, 0u) & 0x80000000u) == 0u) {
  }

  __threadfence();
  __syncthreads();
  ++phase;
}

/**
 * @brief Site #2 alias (Phase 3 → 4, coupled carve): id = up_group,
 * arrival_count = UP_GRID, seed = up_group * UP_GRID (the lowest blockIdx
 * whose up_group == g).
 */
__device__ __forceinline__ void expert_barrier(
    uint32_t* __restrict__ expert_counters, uint32_t expert_id,
    uint32_t arrival_count, uint32_t seed_thread_blockidx, uint32_t& phase) {
  partial_barrier(expert_counters, expert_id, arrival_count,
                  seed_thread_blockidx, phase);
}

/**
 * @brief Site #3 alias (Phase 4 → 5): id = blockIdx.x % DOWN_GRID,
 * arrival_count = DOWN_GROUPS, seed = the col stripe id (that block is
 * both in the arrival set at group 0 and the Phase-5 writer).
 */
__device__ __forceinline__ void colstripe_barrier(
    uint32_t* __restrict__ colstripe_counters, uint32_t col_stripe,
    uint32_t arrival_count, uint32_t seed_thread_blockidx, uint32_t& phase) {
  partial_barrier(colstripe_counters, col_stripe, arrival_count,
                  seed_thread_blockidx, phase);
}

// ── Asymmetric producer→consumer barrier (DECOUPLED up/down carve) ────────
//
// When UP_GROUPS != DOWN_GROUPS the producer set of an expert's temp_fp8
// (the UP_GRID up-blocks of one up_group) is not the consumer set (the
// down-blocks that later read it), so the symmetric expert_barrier —
// where every participant both arrives AND waits on its own up_group
// slot — is invalid.  Split the two halves, keyed by up_group:
//
//   * expert_produce_arrive — release + seed/arrive, NO spin.  Called by
//     every up-block after writing temp_fp8; arrival_count counts only
//     producers (UP_GRID).
//   * expert_consume_wait   — spin + acquire, NO arrive.  Called by every
//     down-block once per distinct up_group in its down-set before Phase 4
//     reads that group's temp_fp8.  Wait-only observers never touch the
//     counter, so the reset discipline is unchanged.
//
// The ping-pong slot is pinned to 0: each slot completes exactly once per
// launch (every up_group arrives unconditionally), and cross-launch safety
// comes from the seeder's atomicExch reset, not slot alternation.
//
// Deadlock-freedom: every block completes its non-blocking arrive before
// any blocking wait, so all arrivals land regardless of wait interleaving.
__device__ __forceinline__ void expert_produce_arrive(
    uint32_t* __restrict__ expert_counters, uint32_t up_group,
    uint32_t arrival_count, uint32_t seed_thread_blockidx) {
  // Single-producer group: nothing seeds the slot, so a matching wait
  // would spin forever; the release fence alone publishes temp_fp8.
  if (arrival_count == 1u) {
    __syncthreads();
    if (threadIdx.x == 0) __threadfence();
    return;
  }
  const uint32_t SEED = 0x80000000u - (arrival_count - 1u);
  uint32_t* c = expert_counters + up_group * 2u;  // slot 0
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    if (blockIdx.x == seed_thread_blockidx) {
      const uint32_t prior = atomicExch(c, SEED);
      const uint32_t to_fold = prior & 0x7FFFFFFFu;
      if (to_fold != 0u) {
        atomicAdd(c, to_fold);
      }
    } else {
      atomicAdd(c, 1u);
    }
  }
}

__device__ __forceinline__ void expert_consume_wait(
    uint32_t* __restrict__ expert_counters, uint32_t up_group,
    uint32_t arrival_count) {
  // Matches the single-producer short-circuit above: nothing to spin on.
  if (arrival_count == 1u) {
    __threadfence();
    __syncthreads();
    return;
  }
  uint32_t* c = expert_counters + up_group * 2u;  // slot 0
  while ((atomicAdd(c, 0u) & 0x80000000u) == 0u) {
  }
  __threadfence();
  __syncthreads();
}

}  // namespace moe_monokernel

#endif  // MOE_GRID_BARRIER_H
