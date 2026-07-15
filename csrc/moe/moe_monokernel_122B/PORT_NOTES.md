# Qwen3.5-122B MoE monokernel port — status & residuals

Fork of the Qwen3.5-35B BS8 TMA+WGMMA monokernel for the 122B shape
(`hidden_size = 3072`, `intermediate_size = 1024`). Dev copy lives here;
the original 35B kernel is unchanged at `csrc/moe/moe_monokernel/`.

## Shape deltas (35B → 122B)

| quantity            | 35B   | 122B  |
|---------------------|-------|-------|
| HIDDEN_STATES (K)   | 2048  | 3072  |
| N (intermediate)    | 512   | 1024  |
| UP_SCALE_COLS       | 16    | 24    |
| DOWN_COL_TILE       | 256   | 384   |
| DOWN_COL_HALVES     | 2     | 3     |
| UP_COL_HALVES       | 1     | 2     |
| DOWN_ACT_BLOCK_SIZE | 64    | 128   |
| UP_W_SLOTS          | 4     | 2     |
| K_TILES_UP          | 16    | 24    |
| K_STEP_UP/DOWN      | 128   | 128   |

## Done (compile-verified for sm_90a)

The probe `/tmp/probe_122b.cu` force-instantiates
`moe_kernel_topk<Dims_BS8_E256_Qwen3_5_122B_BlockFP8_WGMMA_TMA>` and its
SHM/spec types; every per-Dims `static_assert` is evaluated.

```
nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 \
     --expt-relaxed-constexpr -I. -c /tmp/probe_122b.cu -o /tmp/probe_122b.o
```

Exit 0 (a benign C7520 wgmma-serialization perf warning is the only output).

1. **Dims struct** (`moe_interface.h`):
   `Dims_BS8_E256_Qwen3_5_122B_BlockFP8_WGMMA_TMA` with N=1024, K=3072,
   `DOWN_COL_TILE=384`, `K_STEP_DOWN=128`, `K_STEP_UP=128`,
   `USE_PAIR_LAYOUT=true`.
2. **Parameterized internal.h**: `down_col_tile<Dims>` SFINAE, derived
   `UP_COL_HALVES`, `W_UP_TILE_EFFECTIVE`, `W_WGMMA_M_TOTAL=256`,
   `UP_W_SLOTS=2`, `post_silu_scratch[256][9]`,
   `w_down_scale[DOWN_COL_HALVES][2][cols]`, `DOWN_ACT_BLOCK_SIZE=128`.
   SHM budget = 96 KB (down weight tile dominator) < 224 KB cap.
3. **Up-projection fork** (`moe_up_projection.cu` →
   `moe_up_projection_BS8_122B_wgmma_tma`): two stacked 128-row M-atoms per
   block (`base_row_up = bid*128`, atom h at `base_row_up + h*64`),
   per-atom `final_d[2][4]` accumulators, 2-deep weight lookahead
   (`bar_w[s&1]` wait / `bar_w[(s+1)&1]` arm, single-slot pre-loop prime),
   deferred SiLU+fp8 writeback with a 128-feature (4-value/lane) reduce-max
   for the block-128 quant. Dispatched from `moe.cu` via
   `if constexpr (CoreDims::UP_COL_HALVES == 2u)`. The 35B function is
   untouched. Up-scale loader generalized to a 32-strided loop so the
   48-element (2×24) 122B scale tile loads fully (was a single
   `thread < TILE` guard that silently dropped lanes 32..47).
4. **Down-projection** (`moe_down_projection.cu`): `DOWN_COL_HALVES<=3`,
   per-half loop over weight atoms + scale-apply + `down_out` writeback
   (row_base `h*128+…` ≤ 343 < 384, in-bounds), `block_lo==block_hi` for
   the 128-element activation-quant block (lo/hi 64-K chunks share a scale).

## Build & test integration — DONE

The 122B variant is now built **alongside** the 35B kernel and exposed as a
torch op:

- **`moe_wrapper.cu`**: instantiates `Dims_..._122B_...` (was a verbatim 35B
  copy). Down-weight TMA descriptor built with **`row_box=128`** (not
  `DOWN_COL_TILE`) — see the down-weight TMA fix below.
- **Down-weight TMA fix** (`moe_down_projection.cu`): the TMA `boxDim`
  hardware cap is 256 rows, so `DOWN_COL_TILE=384` cannot be a single box.
  All three down-weight TMA issue sites (prime / intra-expert / inter-expert
  lookahead) now issue one 128-row TMA per `(kk, h)` atom at GM col
  `base_col + h*128` → SHM row `(kk*DOWN_COL_HALVES + h)*128`.
- **`moe_ops.h` + `torch_bindings.cpp`**: declare + `m.def`/`m.impl` the op
  `moe_monokernel_topk_BS8_E256_Qwen3_5_122B_BlockFP8_WGMMA_TMA`.
- **`CMakeLists.txt`**: adds `moe_monokernel_122B/moe_wrapper.cu` (gencode
  9.0a + `OBJECT_DEPENDS`). Does **NOT** add 122B's `moe_tma.cu` — it is
  byte-identical to 35B's and defines the same free `create_*` host symbols;
  the 122B wrapper links against 35B's `moe_tma.o`. Safe because the build
  is `-fno-gpu-rdc` (device/template symbols are per-TU) and the only
  potential host collisions are those `create_*` factories.
- **`test_monokernel_accuracy.py`**: added the `qwen3.5_122b` model entry
  (E=256, N_HALF=1024, K=3072) pointing `op_bs8` at the new op. Run with
  `--model qwen3.5_122b --batch-sizes 1 2 4 8` (M>8 raises by design).
  Both `qwen3.5` (35B) and `qwen3.5_122b` are registered side by side;
  default `--model` is still `qwen3.5`. The scratchpad sizing was
  shape-hardcoded to 35B (`DOWN_COL_TILE=256`, `DOWN_ACT_BLOCK_SIZE=64`) —
  now derived from the shape (`DOWN_COL_TILE = K/(GRID/16)`,
  `UP_COL_HALVES = 2*N_HALF*DOWN_COL_TILE/(128*K)`,
  `DOWN_ACT_BLOCK_SIZE = UP_COL_HALVES*64`) and `down_partial_out` corrected
  to the single `[BS][K]` buffer the kernel actually uses. The rest of the
  test (reference math, triton path, quant helpers) is parameterized by
  `N_HALF`/`K`/`E` and needs no change. The BS64-only debug SiLU reader
  (`temp_act_scale[...][N/64]`) is skipped on the BS8 path, so its 35B `/64`
  assumption never executes for 122B.

Build: `cmake --build --preset release --target install`.

### vLLM Python integration (fp8.py path)

The 122B kernel is reachable from real model serving, not just the test:

- **`vllm/_custom_ops.py`** (`moe_monokernel_topk`): the high-level op now
  dispatches by `(E, N, K)` shape — `E=256,N=1024,K=2048` → 35B `_moe_C` op;
  `E=256,N=2048,K=3072` → 122B `_moe_C` op. Both use the shape-generic
  `interleave_for_tma_wgmma_up_v2` repack. (`N` = fused gate+up = 2*N_half.)
- **`vllm/model_executor/layers/quantization/fp8.py`**
  (`process_weights_after_loading`): the `_use_moe_monokernel` eligibility
  gate now accepts both shapes (`w13_weight` = `[E, 2*N_half, K]`:
  1024/2048 for 35B, 2048/3072 for 122B). The 16 MB scratchpad and the
  load-time `interleave_for_tma_wgmma_up_v2` pre-pack already cover 122B
  (struct ~198 KB ≪ 16 MB; N_half=1024 % 64 == 0).

## Residuals — still to validate

- **Accuracy run on H200**: the kernel compiles for sm_90a but has NOT been
  run. Only `test_monokernel_accuracy.py --model qwen3.5_122b` against the
  triton/dense reference will catch a wrong register↔feature mapping in the
  up-proj fork or the down-proj 3-half generalization — the compile cannot.
- **Down interleave**: the up repack `interleave_for_tma_wgmma_up_v2`
  supports N=1024 (the test wires it). Down weights are passed RAW (no
  interleave), so `DOWN_COL_HALVES=3` needs no Python change — but confirm
  on the first run.
- **Profiling fields**: the `phase_timestamps` instrumentation in the fork
  was dropped for clarity (the 35B path keeps it). Re-add
  `MONO_PHASE_TIMESTAMP_IF` calls if you need the per-phase 122B timeline.

## Known design notes

- 122B pins `K_STEP_DOWN=128` (not 256): at 256 the down weight tile would
  be 192 KB and blow the 224 KB SHM budget.
- Both up-proj atoms of a block share the same gate/up weight-scale block
  (their gate rows fall in one `BLOCK_SCALE_ROW=128` block), so `up_scale`
  needs no per-half dimension.
