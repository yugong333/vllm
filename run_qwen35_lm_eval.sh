#!/bin/bash
# =============================================================================
# End-to-end accuracy evaluation for the Qwen3.5 A3B FP8 MoE models
# (35B and 122B variants) using lm_eval
#
# Uses the lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness)
# against a vLLM OpenAI-compatible server to measure task accuracy (e.g. GSM8K).
#
# This validates that the monokernel produces correct end-to-end model outputs
# by comparing accuracy scores between the monokernel and triton backends.
# (Unlike `run_qwen35_bench.sh`, which runs with --ignore-eos and only measures
# latency/throughput, lm_eval inspects the generated text and scores it, so it
# is the path that actually catches garbage / NaN output.)
#
# Usage:
#   ./run_qwen35_lm_eval.sh [BACKEND] [MODEL] [TASK]
#
#   BACKEND (optional, default: monokernel)
#     monokernel  Use the MoE monokernel fast path (VLLM_USE_MOE_MONOKERNEL=1)
#     triton      Force the standard TRITON fused-MoE backend
#                 (VLLM_USE_MOE_MONOKERNEL=0)
#     both        Run evaluation once per backend for comparison
#
#   MODEL (optional, default: 35b)
#     35b         Qwen/Qwen3.5-35B-A3B-FP8   (E=256, N=512,  K=2048)
#     122b        Qwen/Qwen3.5-122B-A10B-FP8 (E=256, N=1024, K=3072)
#
#   TASK (optional, default: gsm8k)
#     Any lm_eval task name (e.g. gsm8k, mmlu, hellaswag, arc_challenge)
#     Multiple tasks can be comma-separated: gsm8k,hellaswag
#
#   BACKEND/MODEL/TASK can also be set via env vars; positional args take
#   precedence.  Examples:
#     ./run_qwen35_lm_eval.sh                      # monokernel, 35b, gsm8k
#     ./run_qwen35_lm_eval.sh both 122b            # both backends, 122b, gsm8k
#     ./run_qwen35_lm_eval.sh monokernel 122b mmlu # monokernel, 122b, mmlu
#
# Prerequisites:
#   pip install lm_eval   (lm-evaluation-harness)
#
# Results are written under:
#   result_outputs/lm_eval/<MODEL>/run_<GPU>_<TIMESTAMP>/<backend>/
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

PORT=8000
HOST="127.0.0.1"
TENSOR_PARALLEL_SIZE=1

# Backend selection: monokernel | triton | both (default monokernel).
BACKEND="${1:-${BACKEND:-monokernel}}"

# Model selection: 35b | 122b (default 35b).
# Positional arg ($2) takes precedence over the MODEL env var.
MODEL="${2:-${MODEL:-35b}}"

# lm_eval task(s) — comma-separated for multiple (default gsm8k).
TASK="${3:-${LM_EVAL_TASK:-gsm8k}}"
case "$MODEL" in
    35b)
        MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8"
        ;;
    122b)
        MODEL_NAME="Qwen/Qwen3.5-122B-A10B-FP8"
        ;;
    *)
        echo "[ERROR] Unknown MODEL '$MODEL' (expected 35b|122b)."
        exit 1
        ;;
esac

# Optional memory knobs, passed to `vllm serve` only when set.  The 122B
# variant is ~122 GB in FP8 and OOM'd at startup on a 140 GB H200 with vLLM's
# default gpu-memory-utilization; export these to give it headroom, e.g.
#   GPU_MEMORY_UTILIZATION=0.95 MAX_MODEL_LEN=8192 ./run_qwen35_lm_eval.sh ... 122b
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# Number of examples to evaluate (set to 0 or empty for full eval).
# For quick validation, use a small number like 100.
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
LIMIT="${LIMIT:-100}"

# Timestamped run folder.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr -cd 'A-Za-z0-9_')"
GPU_NAME="${GPU_NAME:-GPU}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_DIR/result_outputs/lm_eval/${MODEL_NAME}/run_${GPU_NAME}_${RUN_STAMP}"

# Per-backend state.
RESULTS_BASE_DIR=""
SERVER_LOG=""
VLLM_PID=""
CURRENT_BACKEND=""

# -----------------------------
# Helpers
# -----------------------------
configure_backend() {
    local backend="$1"
    case "$backend" in
        monokernel)
            export VLLM_USE_MOE_MONOKERNEL=1
            # MONOKERNEL_CONFIG (if exported by the caller) selects a tuned
            # KernelConfig id; inherited by `vllm serve` as-is.  Logged here
            # so the run output records which config was evaluated.
            echo "[INFO]   MONOKERNEL_CONFIG   : ${MONOKERNEL_CONFIG:-<default (id 0)>}"
            ;;
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
    echo "[INFO]   GPU mem util        : ${GPU_MEMORY_UTILIZATION:-<default>}"
    echo "[INFO]   Max model len       : ${MAX_MODEL_LEN:-<default>}"
    echo "[INFO]   Server log          : $SERVER_LOG"
    echo "[INFO] ====================================================="

    # Optional flags, added only when the corresponding env var is set.
    local extra_args=()
    if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
        extra_args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    fi
    if [ -n "$MAX_MODEL_LEN" ]; then
        extra_args+=(--max-model-len "$MAX_MODEL_LEN")
    fi

    vllm serve "$MODEL_NAME" \
        --trust-remote-code \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-num-seqs 64 \
        --no-enable-prefix-caching \
        --enable-chunked-prefill \
        --reasoning-parser qwen3 \
        "${extra_args[@]}" \
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

run_lm_eval() {
    local output_dir="$RESULTS_BASE_DIR/results"
    local eval_log="$RESULTS_BASE_DIR/lm_eval.log"
    mkdir -p "$output_dir"

    echo "[INFO] -----------------------------------------------------"
    echo "[INFO] Running lm_eval"
    echo "[INFO]   Backend     : $CURRENT_BACKEND"
    echo "[INFO]   Task(s)     : $TASK"
    echo "[INFO]   Num fewshot : $NUM_FEWSHOT"
    echo "[INFO]   Limit       : ${LIMIT:-full}"
    echo "[INFO]   Output dir  : $output_dir"
    echo "[INFO] -----------------------------------------------------"

    local lm_eval_args=(
        --model local-completions
        --model_args "model=${MODEL_NAME},base_url=http://${HOST}:${PORT}/v1/completions,tokenized_requests=False,num_concurrent=16"
        --tasks "$TASK"
        --num_fewshot "$NUM_FEWSHOT"
        --output_path "$output_dir"
        --log_samples
    )

    # Add --limit if set and non-zero.
    if [ -n "${LIMIT}" ] && [ "${LIMIT}" != "0" ]; then
        lm_eval_args+=(--limit "$LIMIT")
    fi

    local start_time
    start_time=$(date +%s)

    lm_eval "${lm_eval_args[@]}" 2>&1 | tee "$eval_log"

    local end_time
    end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    echo "[INFO] lm_eval completed in $((total_time / 60))m $((total_time % 60))s"
    echo "[INFO] Results saved to: $output_dir"
    echo "[INFO] -----------------------------------------------------"
}

run_one_backend() {
    CURRENT_BACKEND="$1"
    configure_backend "$CURRENT_BACKEND"
    echo "[INFO] #####################################################"
    echo "[INFO] BACKEND: $CURRENT_BACKEND"
    echo "[INFO]   Results dir : $RESULTS_BASE_DIR"
    echo "[INFO] #####################################################"
    start_vllm_server
    run_lm_eval
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
# local dev checkout, crashing `vllm serve` at startup. The eval model
# Qwen3.5 is a native vLLM arch, so no plugins are needed. Empty allowlist =
# load none. Override by exporting VLLM_PLUGINS before invoking this script.
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

# Check lm_eval is installed.
if ! command -v lm_eval &>/dev/null; then
    echo "[ERROR] lm_eval not found. Install it with:"
    echo "        pip install lm_eval"
    exit 1
fi

mkdir -p "$RUN_DIR"
echo "[INFO] ====================================================="
echo "[INFO] lm_eval accuracy evaluation"
echo "[INFO]   Model          : $MODEL_NAME  (MODEL=$MODEL)"
echo "[INFO]   Task(s)        : $TASK"
echo "[INFO]   Backend(s)     : $BACKEND"
echo "[INFO]   Limit          : ${LIMIT:-full}"
echo "[INFO]   Run directory  : $RUN_DIR"
echo "[INFO] ====================================================="

# Ensure the server is always cleaned up.
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

echo ""
echo "[INFO] ====================================================="
echo "[INFO] All evaluations completed."
echo "[INFO] Results under: $RUN_DIR"
echo "[INFO]"
echo "[INFO] To compare backends (if 'both' was used):"
echo "[INFO]   cat $RUN_DIR/triton/results/*/results.json"
echo "[INFO]   cat $RUN_DIR/monokernel/results/*/results.json"
echo "[INFO] ====================================================="
