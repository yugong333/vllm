#pragma once
#ifndef MOE_DEBUG_H
  #define MOE_DEBUG_H

  #ifdef DEBUG_MOE_PRINT

    #include <cstdio>

    // Print from block 0, threadIdx.x == 0 only
    #define DBG_B0T0(fmt, ...)                   \
      do {                                       \
        if (blockIdx.x == 0 && threadIdx.x == 0) \
          printf(fmt "\n", ##__VA_ARGS__);       \
      } while (0)

    // Print from block 0, any thread (use sparingly)
    #define DBG_B0(fmt, ...)                                          \
      do {                                                            \
        if (blockIdx.x == 0)                                          \
          printf("[blk0 t%u] " fmt "\n", threadIdx.x, ##__VA_ARGS__); \
      } while (0)

    // Print from block 0, specific threadIdx.x
    #define DBG_B0_THREAD(tid, fmt, ...)             \
      do {                                           \
        if (blockIdx.x == 0 && threadIdx.x == (tid)) \
          printf(fmt "\n", ##__VA_ARGS__);           \
      } while (0)

    // ── Step-by-step intermediate dumping ──────────────────────────────
    //
    // These helpers print a labeled slice of a value array from block 0,
    // thread 0 so the host can compare each phase (routing → quant → up →
    // down → final) against a Python reference.  All are no-ops unless
    // DEBUG_MOE_PRINT is defined, so they cost nothing in production
    // builds.  Use a SMALL count (≤ 16) — printf from device is slow and
    // serializes the launch.
    //
    // `MONO_DBG_PHASE("label")` — print a phase banner once (block0/t0).
    #define MONO_DBG_PHASE(label)                \
      do {                                       \
        if (blockIdx.x == 0 && threadIdx.x == 0) \
          printf("[MONO_DBG] === %s ===\n", label); \
      } while (0)

    // `MONO_DBG_VALS("label", ptr, n)` — print n values of a float-
    // convertible array (works for float / __nv_bfloat16 / fp8 via the
    // (float) cast) from block0/t0, on one line.
    #define MONO_DBG_VALS(label, ptr, n)                  \
      do {                                                \
        if (blockIdx.x == 0 && threadIdx.x == 0) {        \
          printf("[MONO_DBG] %s:", label);                \
          for (int _i = 0; _i < (int)(n); ++_i)           \
            printf(" %.5f", (float)((ptr)[_i]));          \
          printf("\n");                                   \
        }                                                 \
      } while (0)

    // `MONO_DBG_VALS_T(tid, "label", ptr, n)` — same but from a chosen
    // thread (e.g. a calc-warp lane that holds the value of interest).
    #define MONO_DBG_VALS_T(tid, label, ptr, n)              \
      do {                                                   \
        if (blockIdx.x == 0 && threadIdx.x == (tid)) {       \
          printf("[MONO_DBG] %s:", label);                   \
          for (int _i = 0; _i < (int)(n); ++_i)              \
            printf(" %.5f", (float)((ptr)[_i]));             \
          printf("\n");                                      \
        }                                                    \
      } while (0)

  #else

    #define DBG_B0T0(fmt, ...)
    #define DBG_B0(fmt, ...)
    #define DBG_B0_THREAD(tid, fmt, ...)
    #define MONO_DBG_PHASE(label)
    #define MONO_DBG_VALS(label, ptr, n)
    #define MONO_DBG_VALS_T(tid, label, ptr, n)

  #endif  // DEBUG_MOE_PRINT

#endif  // MOE_DEBUG_H
