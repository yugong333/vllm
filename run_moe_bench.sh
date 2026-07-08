#!/bin/bash
# =============================================================================
# Generic vLLM serve benchmark for ANY MoE model: triton vs monokernel.
#
# Parameterized version of run_GLM52_bench.sh / run_qwen35_bench.sh — the
# model, TP size, and extra serve args come from the environment/CLI instead
# of being hard-coded, so `onboard_model.py` (and humans) can benchmark a
# freshly onboarded model without writing a new script.
#
# Usage:
#   MODEL=<hf-id-or-path> TP=<n> ./run_moe_bench.sh [BACKEND] [DATASET]
#
#   BACKEND (optional, default: both)
#     monokernel  MoE monokernel fast path (VLLM_USE_MOE_MONOKERNEL=1)
#     triton      Standard Triton fused-MoE backend (VLLM_USE_MOE_MONOKERNEL=0)
#     both        Full sweep once per backend (triton then monokernel)
#     capture     Capture REAL decode routing on the Triton path (eager, BS=8)
#                 into <run_dir>/capture/route_capture.pt for the tuner's
#                 --route-capture mode.
#
#   DATASET (optional, default: sonnet): sonnet | gsm8k | sharegpt
#
#   Environment:
#     MODEL             HF id or local path (REQUIRED)
#     TP                tensor-parallel size (default 1)
#     MODEL_TAG         short tag for result paths (default: basename of MODEL)
#     MONOKERNEL_CONFIG tuned config selection, inherited by the server
#                       (e.g. "e256n2048k3072:1"); logged per run
#     SERVE_EXTRA_ARGS  extra `vllm serve` args as a single string
#                       (e.g. "--tool-call-parser glm47 --reasoning-parser glm45")
#     BATCH_SIZES       space-separated max-concurrency sweep (default "1 2 4 8")
#     NUM_PROMPTS       requests per batch-size run (default 50)
#     OUTPUT_LEN        output tokens per request (default 200)
#     PORT / HOST       server endpoint (default 8000 / 127.0.0.1)
#     GPU_MEM_UTIL      --gpu-memory-utilization (default 0.92)
#     MAX_MODEL_LEN     --max-model-len (default: model default; set to cap)
#
# Results:
#   result_outputs/vllm_bench/<MODEL_TAG>/<DATASET>_out_<OUT>/tp_<TP>/
#       run_<GPU>_<TS>/<backend>/batch_size_<bs>/...
# With BACKEND=both, summarize_bench.py prints the triton-vs-monokernel
# speedup table at the end.
# =============================================================================
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${VENV_ACTIVATE:-}" ]; then
    if [ -f "$REPO_DIR/venv_vllm/bin/activate" ]; then
        VENV_ACTIVATE="$REPO_DIR/venv_vllm/bin/activate"
    elif [ -f "/opt/vllm-venv/bin/activate" ]; then
        VENV_ACTIVATE="/opt/vllm-venv/bin/activate"
    else
        echo "[ERROR] No venv found (tried $REPO_DIR/venv_vllm and /opt/vllm-venv)."
        exit 1
    fi
fi

BACKEND="${1:-${BACKEND:-both}}"
DATASET="${2:-${DATASET:-sonnet}}"

MODEL_NAME="${MODEL:?Set MODEL=<hf-id-or-path> (e.g. MODEL=Qwen/Qwen3.5-35B-A3B-FP8)}"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_NAME")}"
TOKENIZER_NAME="${TOKENIZER:-$MODEL_NAME}"
TENSOR_PARALLEL_SIZE="${TP:-1}"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
OUTPUT_LEN="${OUTPUT_LEN:-200}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
read -r -a BATCH_SIZES <<< "${BATCH_SIZES:-1 2 4 8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SEED="${SEED:-11111}"

# Extra serve args (word-split intentionally).
read -r -a EXTRA_ARGS <<< "${SERVE_EXTRA_ARGS:-}"
if [ -n "${MAX_MODEL_LEN:-}" ]; then
    EXTRA_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi

# Dataset → vllm bench serve args.
DATASET_TAG="$DATASET"
DATASET_ARGS=()
case "$DATASET" in
    sonnet)
        INPUT_LEN="${INPUT_LEN:-1600}"
        SONNET_PREFIX_LEN=200
        DATASET_ARGS=(
            --dataset-name sonnet
            --dataset-path "$REPO_DIR/benchmarks/sonnet.txt"
            --sonnet-input-len "$INPUT_LEN"
            --sonnet-output-len "$OUTPUT_LEN"
            --sonnet-prefix-len "$SONNET_PREFIX_LEN"
        )
        DATASET_TAG="sonnet_${INPUT_LEN}"
        ;;
    gsm8k)
        DATASET_ARGS=(
            --dataset-name hf
            --dataset-path openai/gsm8k
            --hf-subset main
            --hf-split test
            --hf-output-len "$OUTPUT_LEN"
        )
        ;;
    sharegpt)
        SHAREGPT_PATH="${SHAREGPT_PATH:-Aeala/ShareGPT_Vicuna_unfiltered}"
        if [ -f "$SHAREGPT_PATH" ]; then
            DATASET_ARGS=(
                --dataset-name sharegpt
                --dataset-path "$SHAREGPT_PATH"
                --sharegpt-output-len "$OUTPUT_LEN"
            )
        else
            DATASET_ARGS=(
                --dataset-name hf
                --dataset-path "$SHAREGPT_PATH"
                --hf-output-len "$OUTPUT_LEN"
            )
        fi
        ;;
    *)
        echo "[ERROR] Unknown DATASET '$DATASET' (expected sonnet|gsm8k|sharegpt)."
        exit 1
        ;;
esac

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr -cd 'A-Za-z0-9_')"
GPU_NAME="${GPU_NAME:-GPU}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO_DIR/result_outputs/vllm_bench/${MODEL_TAG}/${DATASET_TAG}_out_${OUTPUT_LEN}/tp_${TENSOR_PARALLEL_SIZE}/run_${GPU_NAME}_${RUN_STAMP}}"

RESULTS_BASE_DIR=""
SERVER_LOG=""
VLLM_PID=""
CAPTURE_MODE=0
ROUTE_CAPTURE_PATH=""

configure_backend() {
    local backend="$1"
    CAPTURE_MODE=0
    unset MONOKERNEL_ROUTE_CAPTURE
    case "$backend" in
        monokernel)
            export VLLM_USE_MOE_MONOKERNEL=1
            echo "[INFO] MONOKERNEL_CONFIG = ${MONOKERNEL_CONFIG:-<default (id 0)>}"
            ;;
        triton)  export VLLM_USE_MOE_MONOKERNEL=0 ;;
        capture)
            export VLLM_USE_MOE_MONOKERNEL=0
            CAPTURE_MODE=1
            ;;
        *)
            echo "[ERROR] Unknown backend '$backend'."
            exit 1
            ;;
    esac
    RESULTS_BASE_DIR="$RUN_DIR/$backend"
    SERVER_LOG="$RESULTS_BASE_DIR/vllm_server.log"
    mkdir -p "$RESULTS_BASE_DIR"
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        ROUTE_CAPTURE_PATH="$RESULTS_BASE_DIR/route_capture.pt"
        export MONOKERNEL_ROUTE_CAPTURE="$ROUTE_CAPTURE_PATH"
        echo "[INFO] Routing capture ENABLED -> $ROUTE_CAPTURE_PATH"
    fi
}

wait_for_server() {
    local url="http://${HOST}:${PORT}"
    echo "[INFO] Waiting for server at $url ..."
    while true; do
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[ERROR] vLLM server process ($VLLM_PID) exited before ready."
            echo "[ERROR] See log: $SERVER_LOG"
            tail -30 "$SERVER_LOG" || true
            exit 1
        fi
        local response
        response=$(curl --write-out "%{http_code}" --silent --output /dev/null "$url/v1/models")
        if [[ "$response" -eq 200 ]]; then
            echo "[INFO] vLLM server is ready."
            break
        fi
        echo "[INFO] Not ready yet (status $response). Retrying in 10s..."
        sleep 10
    done
}

start_vllm_server() {
    echo "[INFO] ====================================================="
    echo "[INFO] Starting vLLM server"
    echo "[INFO]   Model           : $MODEL_NAME"
    echo "[INFO]   Tensor parallel : $TENSOR_PARALLEL_SIZE"
    echo "[INFO]   MoE backend     : $CURRENT_BACKEND (VLLM_USE_MOE_MONOKERNEL=$VLLM_USE_MOE_MONOKERNEL)"
    echo "[INFO]   Extra args      : ${EXTRA_ARGS[*]:-<none>}"
    echo "[INFO]   Server log      : $SERVER_LOG"
    echo "[INFO] ====================================================="

    local eager_flag=()
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        eager_flag=(--enforce-eager)
        echo "[INFO]   Eager mode      : ON (routing capture)"
    fi

    vllm serve "$MODEL_NAME" \
        --trust-remote-code \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-num-seqs 64 \
        --no-enable-prefix-caching \
        --enable-chunked-prefill \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        "${eager_flag[@]}" \
        "${EXTRA_ARGS[@]}" \
        &> "$SERVER_LOG" &
    VLLM_PID=$!
    echo "[INFO] vLLM server PID: $VLLM_PID"
    wait_for_server
}

kill_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[INFO] Stopping vLLM server (PID $VLLM_PID) ..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null
    fi
    VLLM_PID=""
}

run_benchmark() {
    local bs_list=("${BATCH_SIZES[@]}")
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        bs_list=(8)
        echo "[INFO] Capture mode: restricting sweep to BS=8."
    fi
    for bs in "${bs_list[@]}"; do
        local result_dir="$RESULTS_BASE_DIR/batch_size_${bs}"
        local result_filename="${MODEL_TAG}_${DATASET_TAG}_bs${bs}.json"
        mkdir -p "$result_dir"
        echo "[INFO] bench serve | backend=$CURRENT_BACKEND | dataset=$DATASET | max-concurrency=$bs"
        vllm bench serve \
            --backend openai \
            --host "$HOST" \
            --port "$PORT" \
            --model "$MODEL_NAME" \
            --tokenizer "$TOKENIZER_NAME" \
            "${DATASET_ARGS[@]}" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$bs" \
            --ignore-eos \
            --seed "$SEED" \
            --save-result \
            --result-dir "$result_dir" \
            --result-filename "$result_filename" \
            2>&1 | tee "$result_dir/run.log"
    done
}

run_one_backend() {
    CURRENT_BACKEND="$1"
    configure_backend "$CURRENT_BACKEND"
    echo "[INFO] ############ BACKEND RUN: $CURRENT_BACKEND ############"
    start_vllm_server
    run_benchmark
    kill_vllm
}

# -----------------------------
# Main
# -----------------------------
echo "[INFO] Activating venv: $VENV_ACTIVATE"
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# Disable incompatible third-party vLLM plugins by default (empty allowlist);
# override by exporting VLLM_PLUGINS.
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

if [ "$DATASET" = "sonnet" ] && [ ! -f "$REPO_DIR/benchmarks/sonnet.txt" ]; then
    echo "[ERROR] Sonnet dataset not found at: $REPO_DIR/benchmarks/sonnet.txt"
    exit 1
fi

mkdir -p "$RUN_DIR"
echo "[INFO] Model=$MODEL_NAME TP=$TENSOR_PARALLEL_SIZE dataset=$DATASET backend(s)=$BACKEND"
echo "[INFO] Run directory: $RUN_DIR"

trap kill_vllm EXIT INT TERM

case "$BACKEND" in
    both)
        run_one_backend triton
        run_one_backend monokernel
        ;;
    monokernel|triton|capture)
        run_one_backend "$BACKEND"
        ;;
    *)
        echo "[ERROR] Invalid BACKEND '$BACKEND'."
        exit 1
        ;;
esac

if [ "$BACKEND" = "capture" ] && [ -n "$ROUTE_CAPTURE_PATH" ]; then
    if [ -f "$ROUTE_CAPTURE_PATH" ]; then
        echo "[INFO] Routing capture written to: $ROUTE_CAPTURE_PATH"
    else
        echo "[WARN] Expected routing dump not found at: $ROUTE_CAPTURE_PATH"
    fi
fi

trap - EXIT INT TERM

if [ "$BACKEND" = "both" ] && [ -f "$REPO_DIR/summarize_bench.py" ]; then
    echo "[INFO] Comparing backends (triton vs monokernel) ..."
    python "$REPO_DIR/summarize_bench.py" "$RUN_DIR" || \
        echo "[WARN] summarize_bench.py failed; JSONs are under $RUN_DIR"
fi

echo "[INFO] Done. Results under: $RUN_DIR"
