# MoE Monokernel — Runbook

How to build, tune, test, integrate, and profile the kernel.  For the
architecture itself see [DESIGN.md](DESIGN.md).

All commands run from the repo root (`/fsx/export/workspaces/ygkyle/vllm`)
with the venv active:

```bash
source venv_vllm/bin/activate
# cmake >= 3.26 must be first on PATH (see workspace CLAUDE.md for the path)
```

## 1. Build

Incremental compile + install of the CUDA extension (only changed TUs
rebuild, then the fresh `_moe_C*.so` is copied into `vllm/`):

```bash
cmake --build --preset release --target install
```

Do not pipe the build through `tail`/`grep` (the pipeline exit code masks
failures).  If the cached `build.ninja` points at a deleted cmake, re-run
`cmake --preset release` once to regenerate it.

Fast compile-check of kernel-only edits without the torch build:

```bash
cd csrc/moe/moe_monokernel/src
nvcc -gencode arch=compute_90a,code=sm_90a -std=c++17 --expt-relaxed-constexpr \
     -I. -c <probe>.cu -o /tmp/probe.o
```

(`-gencode ...sm_90a`, not `-arch=sm_90a` — the latter maps to plain sm_90
and fails on wgmma.)

## 2. Adding / editing a shape or config

1. Edit `shapes.json` (shape dims, config knob tuples; config 0 must be the
   shipped default).  `tools/enum_configs.py --shape <key>` enumerates the
   feasible knob space for a new shape (`--verify` cross-checks SHM size via
   an nvcc probe).
2. Regenerate: `python csrc/moe/moe_monokernel/tools/gen_shapes.py`
   (emits `generated/*.inc` + `vllm/model_executor/layers/fused_moe/monokernel_shapes.py`).
3. Rebuild (step 1).

## 3. Accuracy test

`test_monokernel_accuracy.py` compares the CUDA kernel against a dequantized
fp32 Python reference and vLLM's Triton `fused_experts` path, per batch
size, including intermediate checks read straight from the scratchpad:

```bash
python test_monokernel_accuracy.py --model qwen3.5           # 35B shape
python test_monokernel_accuracy.py --model qwen3.5_122b --batch-sizes 1 8
python test_monokernel_accuracy.py --list-models
```

To exercise a specific tuned config rather than the shipped default:

```bash
MONOKERNEL_CONFIG=2 python test_monokernel_accuracy.py --model qwen3.5_122b
```

To add a new model shape to the test: add an entry to the `MODELS` dict at
the top of `test_monokernel_accuracy.py` (E, N_HALF, K, default top_k, and
the op name — op names come from the generated registry, so after step 2
above the shape's `moe_monokernel_topk_BS8_E{E}_N{N}_K{K}_*` ops exist).
The harness builds random FP8 block-quantized weights, runs the kernel via
`torch.ops._moe_C.*`, and cosine-compares each stage.

Perf inside the same harness: `perf_test` times CUDA-graph latency vs
Triton.  `--route-capture dump.pt` replays real captured decode routing
(captured from a live server via the `MONOKERNEL_ROUTE_CAPTURE` env var)
instead of the synthetic uniform sweep — strongly preferred for ranking,
since uniform routing understates the monokernel.

## 4. Tuning

`tune_monokernel.py` sweeps every config id instantiated for a shape (no
rebuild between configs — selection rides the `MONOKERNEL_CONFIG` env var),
gates each on accuracy, then ranks by latency:

```bash
python tune_monokernel.py --model qwen3.5_122b --batch-sizes 1 2 4 8
python tune_monokernel.py --model qwen3.5_122b --route-capture dump.pt --json best.json
python tune_monokernel.py --model 35b --configs 0 2      # only these ids
```

- Accuracy gate: a config must clear `--acc-threshold` (default 0.999
  cosine) at every swept M to be ranked.
- With ≥ 2 GPUs the sweep shards by M over a Ray actor pool; otherwise it
  runs sequentially in-process.
- The summary tags each config `interleave` (UCH==1, needs the Python gate/up
  repack ⇒ a duplicate up-weight tensor in GM) vs `raw` (UCH≥2, no
  duplicate), and reports the best config overall and best per category, so
  the perf/memory trade is explicit.
- `--json` writes the winning config id per (shape, M); pin it at serve time
  with `MONOKERNEL_CONFIG`.

Runtime config selection (`MONOKERNEL_CONFIG`):

```bash
MONOKERNEL_CONFIG=2                    # config id 2 for any shape
MONOKERNEL_CONFIG=122b:1,35b:0         # per-shape by name/alias
MONOKERNEL_CONFIG=E256N2048K3072:1     # per-shape by fused E/N/K key
```

Unset or id < 0 ⇒ the shipped default (named op); config 0 of the tunable op
is byte-identical to that default.

## 5. vLLM integration

The serving path is already wired; the pieces, from bottom to top:

1. **Ops** — `moe_wrapper.cu` registers per-shape named + tunable ops in the
   `_moe_C` stable-ABI extension (generated `.inc` files; schemas in
   `csrc/libtorch_stable/moe/torch_bindings.cpp`).
2. **Dispatch op** — `vllm/_custom_ops.py::moe_monokernel_topk` is the
   single Python entry point: it looks up the shape row by
   `(E, N_fused, K)` in `monokernel_shapes.py`, resolves
   `MONOKERNEL_CONFIG`, interleaves the up weights when the selected config
   needs it (UCH==1), and calls the named or tunable op.  Registered as
   `torch.ops.vllm.moe_monokernel_topk` (with a fake impl for
   torch.compile).
3. **Layer hook** — `vllm/model_executor/layers/quantization/fp8.py`
   (`Fp8MoEMethod`):
   - `__init__` allocates the persistent zeroed scratchpad
     (`moe_monokernel_scratchpad`, sized `get_moe_max_scratchpad_size()`).
   - `process_weights_after_loading` sets `self._use_moe_monokernel` when
     the layer's `(E, 2*N, K)` is in the registry, quantization is FP8
     block-wise, top-k routing is plain (no expert grouping), and TP shape
     matches; it also pre-interleaves `w13` if the active config needs it
     and moves the scratchpad to the weight device.
   - `apply` routes eligible decode batches (M ≤ 8) to
     `torch.ops.vllm.moe_monokernel_topk`; everything else falls through to
     the Triton path.

So integrating a new model = declare the shape in `shapes.json` (+ regen +
rebuild) and confirm eligibility fires (look for the
"MoE monokernel fast path ENABLED" log line at startup).

## 6. Profiling

Driver scripts: `profile_monokernel.py` (workload driver; NVTX range per
(path, BS) pair) and `profile_monokernel.sh` (wraps it with ncu or nsys).

### nsys — timeline / end-to-end

```bash
./profile_monokernel.sh --profiler nsys --model qwen3.5_122b --path both --graph
# or manually:
nsys profile -o mono_timeline python profile_monokernel.py \
    --model qwen3.5_122b --bs8 --path both --graph
```

Filter the timeline by the NVTX ranges (`monokernel_<model>_bs<BS>`,
`triton_<model>_bs<BS>`).  `--graph` replays under a CUDA graph, matching
production dispatch (the kernel launches via plain `cudaLaunchKernel`, so
graph capture works).  nsys needs no counter permissions for the default
trace set; `--gpu-metrics-device=all` does.

### ncu — per-kernel SM metrics

```bash
./profile_monokernel.sh --model qwen3.5_122b --bs 8 --set detailed
# or manually:
ncu --nvtx --nvtx-include "monokernel_qwen3.5_122b_bs8/" \
    --kernel-name regex:moe_kernel_topk --set detailed \
    -o mono_ncu python profile_monokernel.py --model qwen3.5_122b --bs 8
```

All shapes share the templated `moe_kernel_topk` symbol, so one
`--kernel-name regex:moe_kernel_topk` filter matches any shape.  ncu kernel
replay requires GPU performance-counter permissions.

To isolate compute vs data movement, rebuild with one of the
`MONO_PROFILE_SKIP_*` flags (commented examples in `CMakeLists.txt`; use
`set_property(... APPEND PROPERTY COMPILE_DEFINITIONS ...)`, never
`set_source_files_properties`, which silently replaces earlier flags) and
compare two ncu runs.  Output is garbage under skip flags — timing only.

### Phase timing (in-kernel clock64 breakdown)

For a per-phase wall-clock breakdown finer than nsys/ncu can give, rebuild
with:

```cmake
set_property(SOURCE "csrc/moe/moe_monokernel/moe_wrapper.cu"
  APPEND PROPERTY COMPILE_DEFINITIONS "MONO_PROFILE_PHASE_TIMING")
```

Block 0 / thread 0 then writes clock64 timestamps at every phase boundary
(routing sub-phases, up-proj per-iter waits/computes, barrier sites,
down-proj prologue/K-loop/accumulate, Phase 5) into the
`phase_timestamps` struct at the tail of `MoEGemmSpec` in the scratchpad.
Kernel output stays correct; overhead is a handful of clock64 reads + GM
stores on one thread.

Readback: run any workload (e.g. the accuracy test), keep a reference to
the scratchpad tensor, and view its tail as int64:

```python
ts = scratchpad.view(torch.int64)[-N_FIELDS:]   # field order = struct order
deltas_us = (ts.diff() / SM_CLOCK_HZ * 1e6)
```

The field order is the declaration order of `phase_timestamps` in
`moe_internal.h`; anchor on `t_after_phase5` (always written last) when
auto-detecting the offset.
