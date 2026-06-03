
#pragma once
#ifndef MOE_PREPARE_CU
  #define MOE_PREPARE_CU

  #ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
    #error Do not include this file directly.
  #endif

  #include <cstdint>

  #include "moe_internal.h"

  #define FULL_MASK 0xFFFFFFFFU

// No BS64 prepare functions remain.  The BS8 prepare logic
// (`prepare_moe_topk_BS8`) lives in `moe_routing.cu`.

#endif
