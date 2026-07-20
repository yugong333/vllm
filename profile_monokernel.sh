#!/usr/bin/env bash
# Profile the MoE monokernel and / or the Triton FP8 baseline.
#
# Wraps `profile_monokernel.py` and dispatches to either Nsight Compute
# (per-kernel SM metrics) or Nsight Systems (end-to-end timeline).
# `--profiler ncu` (default) keeps the existing ncu-per-(path,BS)
# layout; `--profiler nsys` produces a single nsys report covering the
# full sweep.
#
# Usage:
#   # Default: ncu, monokernel, sweep BS={1,2,4,8} on qwen3.5
#   ./profile_monokernel.sh
#
#   # Nsight Systems timeline of the same sweep (one report file)
#   ./profile_monokernel.sh --profiler nsys
#
#   # Nsys, both paths back-to-back, with CUDA graph replay
#   ./profile_monokernel.sh --profiler nsys --path both --graph
#
#   # Profile the Qwen3.5-122B shape (E=256, N_HALF=1024, K=3072)
#   ./profile_monokernel.sh --model qwen3.5_122b --path both
#   ./profile_monokernel.sh --profiler nsys --model qwen3.5_122b --graph
#
#   # Override the BS sweep (comma- or space-separated)
#   ./profile_monokernel.sh --bs 1,2,4,8,16
#   BS_LIST="1 2 4 8" ./profile_monokernel.sh
#
#   # Nsight Compute section set (basic|default|detailed|full)
#   ./profile_monokernel.sh --set detailed
#
#   # Override output dir (default: ncu_profile/<model>/<ts> or
#   #                                nsys_profile/<model>/<ts>)
#   ./profile_monokernel.sh --out my_run
#
#   # Pass extra flags directly to the profiler binary (after --)
#   ./profile_monokernel.sh --bs 8 -- --kernel-name regex:moe_kernel_topk
#   ./profile_monokernel.sh --profiler nsys -- --gpu-metrics-device=all
#
# Models (--model):
#   qwen3.5       Qwen3.5-35B  block-wise FP8 (E=256, N_HALF=512,  K=2048) [default]
#   qwen3.5_122b  Qwen3.5-122B block-wise FP8 (E=256, N_HALF=1024, K=3072)
#   Both share the same templated device kernel `moe_kernel_topk`, so the
#   ncu `--kernel-name regex:moe_kernel_topk` filter matches either shape.
#
# Environment:
#   PYTHON   : python executable (default: python)
#   NCU      : ncu  executable (default: ncu,  fallback /usr/local/cuda/bin/ncu)
#   NSYS     : nsys executable (default: nsys, fallback /usr/local/cuda/bin/nsys)
#   WARMUP   : warmup iters per BS  (default: 3)
#   ITERS    : profiled iters per BS, replayed inside the NVTX range
#              (default: 10 for ncu, 50 for nsys)
#
# Notes:
#   * The monokernel uses standard `cudaLaunchKernel` (the software
#     `grid_barrier` replaces `cooperative_groups::this_grid().sync()`),
#     so CUDA graph capture is supported.
#   * ncu kernel replay needs GPU performance-counter permissions; nsys
#     does NOT for the default (CUDA API + NVTX + kernel launches)
#     trace set.  `--gpu-metrics-device=all` is opt-in and DOES need
#     counter permissions.
#   * `--set full` collects all ncu sections; use `--set basic` or
#     `--set default` for faster captures during iteration.

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
PROFILER="ncu"
MODEL="qwen3.5"
BS_LIST_DEFAULT="1 2 4 8"
BS_LIST="${BS_LIST:-$BS_LIST_DEFAULT}"
SET="full"
OUT_DIR=""               # filled in after profiler is finalized
WARMUP="${WARMUP:-3}"
ITERS=""                 # default depends on profiler
GRAPH=1
PATH_KIND="monokernel"   # one of: monokernel | triton | both
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/profile_monokernel.py"
PYTHON="${PYTHON:-python}"

# ── Parse args ──────────────────────────────────────────────────────────────
EXTRA_PROF_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profiler) PROFILER="$2";  shift 2 ;;
        --model)    MODEL="$2";     shift 2 ;;
        --bs)       BS_LIST="$2";   shift 2 ;;
        --set)      SET="$2";       shift 2 ;;
        --out)      OUT_DIR="$2";   shift 2 ;;
        --warmup)   WARMUP="$2";    shift 2 ;;
        --iters)    ITERS="$2";     shift 2 ;;
        --graph)    GRAPH=1;        shift ;;
        --no-graph) GRAPH=0;        shift ;;
        --path)     PATH_KIND="$2"; shift 2 ;;
        --)         shift; EXTRA_PROF_ARGS+=("$@"); break ;;
        -h|--help)
            sed -n '2,/^set -euo pipefail/p' "$0" | sed -E 's/^# ?//'
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'. See --help." >&2
            exit 2
            ;;
    esac
done

# ── Validate args ───────────────────────────────────────────────────────────
case "$PROFILER" in
    ncu|nsys) ;;
    *)
        echo "error: --profiler must be one of: ncu, nsys (got '$PROFILER')" >&2
        exit 2 ;;
esac

case "$PATH_KIND" in
    monokernel|triton|both) ;;
    *)
        echo "error: --path must be one of: monokernel, triton, both " \
             "(got '$PATH_KIND')" >&2
        exit 2
        ;;
esac
if [[ "$PATH_KIND" == "both" ]]; then
    PATHS=("monokernel" "triton")
else
    PATHS=("$PATH_KIND")
fi

# Export the env var so any vLLM code imported by the profiling script
# respects the backend selection (e.g. Fp8MoEMethod eligibility check).
case "$PATH_KIND" in
    monokernel) export VLLM_USE_MOE_MONOKERNEL=1 ;;
    triton)     export VLLM_USE_MOE_MONOKERNEL=0 ;;
    both)       ;; # set per-path inside the loop
esac

# Profiler-specific defaults (only fill if user did not override).
TS="$(date +%Y%m%d_%H%M%S)"
if [[ "$PROFILER" == "ncu" ]]; then
    : "${ITERS:=10}"
    OUT_DIR="${OUT_DIR:-ncu_profile/${MODEL}/${TS}}"
else
    : "${ITERS:=50}"
    OUT_DIR="${OUT_DIR:-nsys_profile/${MODEL}/${TS}}"
fi

# Normalize BS_LIST (accept commas or spaces)
BS_LIST="${BS_LIST//,/ }"

# ── Locate the profiler binary ──────────────────────────────────────────────
locate_tool() {
    local name="$1" var="$2" fallback="$3"
    local current="${!var:-$name}"
    if command -v "$current" >/dev/null 2>&1; then
        echo "$current"
    elif [[ -x "$fallback" ]]; then
        echo "$fallback"
    else
        echo "error: $name not found on PATH and $fallback does not exist." \
             "Set $var=/path/to/$name or install Nsight ${name^^}." >&2
        return 1
    fi
}

if [[ "$PROFILER" == "ncu" ]]; then
    NCU="$(locate_tool ncu NCU /usr/local/cuda/bin/ncu)" || exit 1
else
    NSYS="$(locate_tool nsys NSYS /usr/local/cuda/bin/nsys)" || exit 1
fi

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "error: cannot find $PY_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "────────────────────────────────────────────────────────────────"
echo " MoE profiling"
echo "   profiler    : $PROFILER"
echo "   model       : $MODEL"
echo "   path        : $PATH_KIND"
echo "   batch sizes : $BS_LIST"
[[ "$PROFILER" == "ncu" ]] && echo "   ncu set     : $SET"
echo "   warmup/iter : $WARMUP / $ITERS"
echo "   output dir  : $OUT_DIR"
echo "   python      : $PYTHON"
[[ "$PROFILER" == "ncu" ]] && echo "   ncu         : $NCU"
[[ "$PROFILER" == "nsys" ]] && echo "   nsys        : $NSYS"
echo "   graph       : $([[ $GRAPH -eq 1 ]] && echo yes || echo no)"
if [[ ${#EXTRA_PROF_ARGS[@]} -gt 0 ]]; then
    echo "   extra args  : ${EXTRA_PROF_ARGS[*]}"
fi
echo "────────────────────────────────────────────────────────────────"

if [[ $GRAPH -eq 1 ]]; then
    SUFFIX="_graph"
    PY_GRAPH_ARG="--graph"
else
    SUFFIX=""
    PY_GRAPH_ARG=""
fi

# Per-path kernel-name regex for ncu.  --kernel-name limits which
# kernels are *captured*; the rest still run.  For the monokernel
# there's exactly one kernel of interest (`moe_kernel_topk`, the same
# templated symbol for both the 35B and 122B shapes); for the Triton
# path the fused-experts pipeline has multiple Triton kernels
# (gemm_a8w8, silu_and_mul_quant_fp8, etc.) plus vLLM's fused_topk +
# align_block, so accept anything.
kernel_regex_for() {
    case "$1" in
        monokernel) echo "regex:moe_kernel_topk" ;;
        triton)     echo "regex:.*" ;;
    esac
}

# ── ncu: one report per (path, batch size) ─────────────────────────────────
run_ncu() {
    for PATH_NAME in "${PATHS[@]}"; do
        # Set the env var per-path so vLLM code respects the backend.
        case "$PATH_NAME" in
            monokernel) export VLLM_USE_MOE_MONOKERNEL=1 ;;
            triton)     export VLLM_USE_MOE_MONOKERNEL=0 ;;
        esac
        local KERNEL_REGEX="$(kernel_regex_for "$PATH_NAME")"
        for BS in $BS_LIST; do
            local REPORT_BASE="${OUT_DIR}/${PATH_NAME}_${MODEL}_bs${BS}${SUFFIX}"
            local RANGE="${PATH_NAME}_${MODEL}_bs${BS}${SUFFIX}"
            local LOG="${REPORT_BASE}.log"
            echo ">>> [ncu] path=$PATH_NAME BS=$BS  →  ${REPORT_BASE}.ncu-rep"
            "$NCU" \
                --target-processes application-only \
                --replay-mode kernel \
                --set "$SET" \
                --profile-from-start off \
                --nvtx \
                --nvtx-include "${RANGE}/" \
                --kernel-name "$KERNEL_REGEX" \
                --force-overwrite \
                -o "$REPORT_BASE" \
                "${EXTRA_PROF_ARGS[@]}" \
                "$PYTHON" "$PY_SCRIPT" \
                    --model "$MODEL" \
                    --path "$PATH_NAME" \
                    --bs "$BS" \
                    --warmup "$WARMUP" \
                    --iters "$ITERS" \
                    $PY_GRAPH_ARG \
                2>&1 | tee "$LOG"
        done
    done

    echo
    echo "done. reports in: $OUT_DIR"
    ls -1 "$OUT_DIR"/*.ncu-rep 2>/dev/null || true
}

# ── nsys: one report covering all paths × all batch sizes ──────────────────
run_nsys() {
    # For nsys with --path both, the Python driver iterates internally.
    # Set the env var to 1 (monokernel eligible) — the Python script's
    # --path flag controls which kernel is actually exercised.
    if [[ "$PATH_KIND" != "triton" ]]; then
        export VLLM_USE_MOE_MONOKERNEL=1
    else
        export VLLM_USE_MOE_MONOKERNEL=0
    fi

    # Build the --bs ... arg list once; the Python driver will then iterate
    # internally and emit one NVTX range per (path, BS).
    local BS_ARGS=()
    for BS in $BS_LIST; do
        BS_ARGS+=("--bs" "$BS")
    done

    local REPORT_BASE="${OUT_DIR}/${PATH_KIND}_${MODEL}${SUFFIX}"
    local LOG="${REPORT_BASE}.log"
    echo ">>> [nsys] path=$PATH_KIND BS={${BS_LIST}}  →  ${REPORT_BASE}.nsys-rep"

    # nsys arguments:
    #   --trace=cuda,nvtx,cudnn,cublas,osrt
    #       cuda  : runtime / driver API + kernel launches + memcpys
    #       nvtx  : per-(path,BS) NVTX ranges from the driver
    #       cudnn / cublas : helpful for the Triton path (Triton itself
    #               uses neither, but vLLM's fused_topk / align_block
    #               can hit cuBLAS).
    #       osrt  : OS runtime — small overhead, very useful for
    #               spotting host-side stalls.
    #   --capture-range=cudaProfilerApi
    #       The driver wraps profiled iterations in
    #       cudaProfilerStart/Stop, so nsys only records inside that
    #       window.  Warmup runs are excluded.
    #   --cuda-graph-trace=node
    #       Resolve each CUDA-graph kernel as its own timeline NODE
    #       (the monokernel launch, plus the Triton fused-experts
    #       kernels, are captured into a graph by --graph).  Without
    #       this nsys collapses a replay into a single opaque
    #       `cudaGraphLaunch` and you lose per-kernel timings.  Pairs
    #       with the driver capturing the graph INSIDE the profiler
    #       window so the node→kernel mapping is observable.
    #   --sample=none, --cpuctxsw=none
    #       Skip CPU sampling (we only care about GPU + CUDA API timeline).
    "$NSYS" profile \
        --output "$REPORT_BASE" \
        --force-overwrite=true \
        --trace=cuda,nvtx,cudnn,cublas,osrt \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-graph-trace=node \
        --sample=none \
        --cpuctxsw=none \
        --stats=false \
        "${EXTRA_PROF_ARGS[@]}" \
        "$PYTHON" "$PY_SCRIPT" \
            --model "$MODEL" \
            --path "$PATH_KIND" \
            "${BS_ARGS[@]}" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            $PY_GRAPH_ARG \
        2>&1 | tee "$LOG"

    echo
    echo "done. report:"
    ls -1 "${REPORT_BASE}".* 2>/dev/null || true
    echo
    echo "open in Nsight Systems GUI, or generate a CSV summary with:"
    echo "    $NSYS stats --report nvtxsum --format csv \\"
    echo "        --output - ${REPORT_BASE}.nsys-rep"
}

case "$PROFILER" in
    ncu)  run_ncu  ;;
    nsys) run_nsys ;;
esac
