#!/bin/bash
# =============================================================================
# vllm bench serve benchmark for Qwen/Qwen3.6-35B-A3B-FP8
#
# Dataset      : sonnet (sonnet.txt)
# Input length : 1600 tokens
# Output length: 200 tokens
# TP           : 1
# Num requests : 50 (per batch-size run)
# Batch sizes  : 1 2 4 8  (mapped to --max-concurrency)
#
# Qwen3.6-35B-A3B is a vision-language MoE model (arch
# Qwen3_5MoeForConditionalGeneration). These are TEXT-ONLY throughput
# benchmarks, so the server is started with --language-model-only to skip
# the vision encoder and free its KV cache. The MoE shape (E=256, top_k=8,
# moe_intermediate=512, hidden=2048) is identical to Qwen3.5-35B-A3B, so the
# monokernel fast path applies the same way.
#
# The script starts a vLLM OpenAI-compatible server, waits until it is ready,
# sweeps the batch sizes with `vllm bench serve`, then shuts the server down.
#
# Usage:
#   ./run_qwen36_bench.sh [BACKEND]
#
#   BACKEND (optional, default: monokernel)
#     monokernel  Use the MoE monokernel fast path (VLLM_USE_MOE_MONOKERNEL=1)
#     triton      Force the standard TRITON fused-MoE backend
#                 (VLLM_USE_MOE_MONOKERNEL=0)
#     both        Run the full sweep once per backend (triton then monokernel)
#
#   Can also be set via the BACKEND environment variable.
#
# Results are written under a per-run timestamped folder:
#   result_outputs/vllm_bench/<MODEL>/sonnet_<IN>_out_<OUT>/tp_<TP>/
#       run_<HOST>_<TIMESTAMP>/<backend>/batch_size_<bs>/...
# =============================================================================
set -uo pipefail

# -----------------------------
# Configuration
# -----------------------------
# Resolve the repo from this script's own location so the in-repo venv is used
# regardless of where the script is invoked from.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer the in-repo venv; allow VENV_ACTIVATE to override.
VENV_ACTIVATE="${VENV_ACTIVATE:-$REPO_DIR/venv_vllm/bin/activate}"

MODEL_NAME="Qwen/Qwen3.6-35B-A3B-FP8"
TOKENIZER_NAME="$MODEL_NAME"

PORT=8000
HOST="127.0.0.1"
TENSOR_PARALLEL_SIZE=1

# Dataset / request shape
DATASET_NAME="sonnet"
DATASET_PATH="$REPO_DIR/benchmarks/sonnet.txt"
INPUT_LEN=1600
OUTPUT_LEN=200
SONNET_PREFIX_LEN=200          # default for sonnet; must be < INPUT_LEN
NUM_PROMPTS=50                 # request number
BATCH_SIZES=(1 2 4 8)          # --max-concurrency sweep

SEED=11111

# Backend selection: monokernel | triton | both (default monokernel).
# CLI arg takes precedence over the BACKEND env var.
BACKEND="${1:-${BACKEND:-monokernel}}"

# Timestamped run folder so repeated runs never clobber each other.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr -cd 'A-Za-z0-9_')"
GPU_NAME="${GPU_NAME:-GPU}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_DIR/result_outputs/vllm_bench/${MODEL_NAME}/sonnet_${INPUT_LEN}_out_${OUTPUT_LEN}/tp_${TENSOR_PARALLEL_SIZE}/run_${GPU_NAME}_${RUN_STAMP}"

# Per-backend values, set by configure_backend().
RESULTS_BASE_DIR=""
SERVER_LOG=""
VLLM_PID=""

# -----------------------------
# Helpers
# -----------------------------
configure_backend() {
    # $1 = backend label (monokernel | triton)
    local backend="$1"
    case "$backend" in
        monokernel) export VLLM_USE_MOE_MONOKERNEL=1 ;;
        triton)     export VLLM_USE_MOE_MONOKERNEL=0 ;;
        *)
            echo "[ERROR] Unknown backend '$backend' (expected monokernel|triton|both)."
            exit 1
            ;;
    esac
    RESULTS_BASE_DIR="$RUN_DIR/$backend"
    SERVER_LOG="$RESULTS_BASE_DIR/vllm_server.log"
    mkdir -p "$RESULTS_BASE_DIR"
}

wait_for_server() {
    local url="http://${HOST}:${PORT}"
    echo "[INFO] Waiting for server at $url ..."
    while true; do
        # Bail out early if the server process has died.
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[ERROR] vLLM server process ($VLLM_PID) exited before becoming ready."
            echo "[ERROR] See log: $SERVER_LOG"
            exit 1
        fi
        local response
        response=$(curl --write-out "%{http_code}" --silent --output /dev/null "$url/v1/models")
        if [[ "$response" -eq 200 ]]; then
            echo "[INFO] vLLM server is ready."
            break
        fi
        echo "[INFO] Not ready yet (status code: $response). Retrying in 10s..."
        sleep 10
    done
}

start_vllm_server() {
    echo "[INFO] ====================================================="
    echo "[INFO] Starting vLLM server"
    echo "[INFO]   Model               : $MODEL_NAME"
    echo "[INFO]   Tensor parallel     : $TENSOR_PARALLEL_SIZE"
    echo "[INFO]   Port                : $PORT"
    echo "[INFO]   MoE backend         : $CURRENT_BACKEND (VLLM_USE_MOE_MONOKERNEL=$VLLM_USE_MOE_MONOKERNEL)"
    echo "[INFO]   Server log          : $SERVER_LOG"
    echo "[INFO] ====================================================="

    vllm serve "$MODEL_NAME" \
        --trust-remote-code \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-num-seqs 64 \
        --language-model-only \
        --no-enable-prefix-caching \
        --enable-chunked-prefill \
        --reasoning-parser qwen3 \
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
    echo "[INFO] ====================================================="
}

run_benchmark() {
    for bs in "${BATCH_SIZES[@]}"; do
        local result_dir="$RESULTS_BASE_DIR/batch_size_${bs}"
        local result_filename="qwen36_sonnet_bs${bs}.json"
        local run_log="$result_dir/run.log"
        mkdir -p "$result_dir"

        echo "[INFO] -----------------------------------------------------"
        echo "[INFO] Running vllm bench serve | backend=$CURRENT_BACKEND | batch size (max-concurrency) = $bs"
        echo "[INFO]   num-prompts = $NUM_PROMPTS | input-len = $INPUT_LEN | output-len = $OUTPUT_LEN"
        echo "[INFO]   results -> $result_dir/$result_filename"

        local start_time
        start_time=$(date +%s)

        vllm bench serve \
            --backend openai \
            --host "$HOST" \
            --port "$PORT" \
            --model "$MODEL_NAME" \
            --tokenizer "$TOKENIZER_NAME" \
            --dataset-name "$DATASET_NAME" \
            --dataset-path "$DATASET_PATH" \
            --sonnet-input-len "$INPUT_LEN" \
            --sonnet-output-len "$OUTPUT_LEN" \
            --sonnet-prefix-len "$SONNET_PREFIX_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$bs" \
            --ignore-eos \
            --seed "$SEED" \
            --save-result \
            --result-dir "$result_dir" \
            --result-filename "$result_filename" \
            2>&1 | tee "$run_log"

        local end_time
        end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        echo "[INFO] Finished batch size $bs in $((total_time / 60))m $((total_time % 60))s"
    done
    echo "[INFO] -----------------------------------------------------"
}

run_one_backend() {
    # $1 = backend label (monokernel | triton)
    CURRENT_BACKEND="$1"
    configure_backend "$CURRENT_BACKEND"
    echo "[INFO] #####################################################"
    echo "[INFO] BACKEND RUN: $CURRENT_BACKEND"
    echo "[INFO]   Results dir : $RESULTS_BASE_DIR"
    echo "[INFO] #####################################################"
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

# Disable incompatible third-party vLLM plugins (sfai_inference_lib, nova, ...).
# They were built against an older vLLM and fail to import/patch against this
# local dev checkout, crashing `vllm serve` at startup. The benchmark model
# Qwen3.6 is a native vLLM arch, so no plugins are needed. Empty allowlist =
# load none. Override by exporting VLLM_PLUGINS before invoking this script.
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

if [ ! -f "$DATASET_PATH" ]; then
    echo "[ERROR] Sonnet dataset not found at: $DATASET_PATH"
    exit 1
fi

mkdir -p "$RUN_DIR"
echo "[INFO] ====================================================="
echo "[INFO] Run directory : $RUN_DIR"
echo "[INFO] Selected backend(s) : $BACKEND"
echo "[INFO] Batch sizes   : ${BATCH_SIZES[*]}"
echo "[INFO] ====================================================="

# Ensure the server is always cleaned up, even on error / Ctrl-C.
trap kill_vllm EXIT INT TERM

case "$BACKEND" in
    both)
        run_one_backend triton
        run_one_backend monokernel
        ;;
    monokernel|triton)
        run_one_backend "$BACKEND"
        ;;
    *)
        echo "[ERROR] Invalid BACKEND '$BACKEND' (expected monokernel|triton|both)."
        exit 1
        ;;
esac

trap - EXIT INT TERM

echo "[INFO] All benchmarks completed. Results under: $RUN_DIR"
