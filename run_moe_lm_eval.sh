#!/bin/bash
# =============================================================================
# Unified end-to-end lm_eval accuracy runner for MoE FP8 models.
#
# Supported models:
#   35b, qwen35-35b       -> Qwen/Qwen3.5-35B-A3B-FP8
#   122b, qwen35-122b     -> Qwen/Qwen3.5-122B-A10B-FP8
#   qwen36, qwen36-35b    -> Qwen/Qwen3.6-35B-A3B-FP8
#   glm52, glm5.2         -> zai-org/GLM-5.2-FP8
#
# Uses lm-evaluation-harness against a vLLM OpenAI-compatible server to
# validate generated text accuracy. This catches end-to-end output issues that
# throughput-only benchmarks can miss.
#
# Usage:
#   ./run_moe_lm_eval.sh [BACKEND] [MODEL] [TASK]
#
#   BACKEND (optional, default: monokernel)
#     monokernel  Use the MoE monokernel fast path (VLLM_USE_MOE_MONOKERNEL=1)
#     triton      Force the standard TRITON fused-MoE backend
#     both        Run triton, then monokernel, for comparison
#
#   MODEL (optional, default: 35b)
#     35b | qwen35-35b | qwen3.5-35b
#     122b | qwen35-122b | qwen3.5-122b
#     qwen36 | qwen36-35b | qwen3.6-35b
#     glm52 | glm5.2 | glm-5.2
#     A full HF model name can also be passed for the supported models.
#
#   TASK (optional, default: gsm8k,humaneval)
#     Any lm_eval task name. Multiple tasks can be comma-separated.
#
# Examples:
#   ./run_moe_lm_eval.sh
#   ./run_moe_lm_eval.sh both 122b
#   ./run_moe_lm_eval.sh monokernel qwen36 humaneval
#   ./run_moe_lm_eval.sh both glm52 gsm8k,humaneval
#
# Useful env overrides:
#   VENV_ACTIVATE=/path/to/venv/bin/activate
#   BACKEND=triton MODEL=glm52 LM_EVAL_TASK=gsm8k LIMIT=0 ./run_moe_lm_eval.sh
#   BS=8 ./run_moe_lm_eval.sh both 122b gsm8k,humaneval
#   GPU_MEMORY_UTILIZATION=0.95 MAX_MODEL_LEN=8192 ./run_moe_lm_eval.sh ...
#
# Batch-size knobs:
#   BS_LIST                    Space-separated sweep; default: "1 8".
#   BS / BATCH_SIZE             Single-BS shortcut when BS_LIST is unset.
#   LM_EVAL_NUM_CONCURRENT      Direct override for all local-completions runs.
#   MAX_NUM_SEQS                vLLM server cap; defaults to max sweep BS.
#   LM_EVAL_BATCH_SIZE          Optional lm_eval --batch_size override.
#
# Results:
#   result_outputs/lm_eval/<MODEL>/run_<GPU>_<TIMESTAMP>/<backend>/
# =============================================================================
set -uo pipefail
# -----------------------------
# Configuration
# -----------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the venv: explicit VENV_ACTIVATE wins; otherwise prefer the in-repo
# venv_vllm and fall back to the historical /opt/vllm-venv.
if [ -z "${VENV_ACTIVATE:-}" ]; then
    if [ -f "$REPO_DIR/venv_vllm/bin/activate" ]; then
        VENV_ACTIVATE="$REPO_DIR/venv_vllm/bin/activate"
    elif [ -f "/opt/vllm-venv/bin/activate" ]; then
        VENV_ACTIVATE="/opt/vllm-venv/bin/activate"
    else
        echo "[ERROR] No venv found (tried $REPO_DIR/venv_vllm and /opt/vllm-venv)."
        echo "[ERROR] Set VENV_ACTIVATE=/path/to/venv/bin/activate and retry."
        exit 1
    fi
fi

BACKEND="${1:-${BACKEND:-monokernel}}"
MODEL="${2:-${MODEL:-35b}}"
TASK="${3:-${LM_EVAL_TASK:-gsm8k,humaneval}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# High-level batch-size sweep for API-backed lm_eval runs. For local-completions,
# each BS value is the request concurrency sent to the vLLM server. MAX_NUM_SEQS
# is the server-side cap, so keep it at least as large as the largest swept BS
# unless explicitly overridden.
if [ -n "${BS_LIST:-}" ]; then
    BS_SWEEP_STR="$BS_LIST"
elif [ -n "${BS:-}" ] || [ -n "${BATCH_SIZE:-}" ]; then
    BS_SWEEP_STR="${BS:-${BATCH_SIZE}}"
else
    BS_SWEEP_STR="1 8"
fi
# shellcheck disable=SC2206
BS_SWEEP=($BS_SWEEP_STR)
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
for bs in "${BS_SWEEP[@]}"; do
    if [ "$MAX_NUM_SEQS" -lt "$bs" ]; then
        MAX_NUM_SEQS="$bs"
    fi
done

# Number of examples to evaluate (set LIMIT=0 or LIMIT= for full eval).
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
LIMIT="${LIMIT:-100}"

MODEL_NAME=""
MODEL_LABEL=""
MODEL_FAMILY=""
DEFAULT_TENSOR_PARALLEL_SIZE="1"
DEFAULT_GPU_MEMORY_UTILIZATION=""
DEFAULT_MAX_MODEL_LEN=""
LANGUAGE_MODEL_ONLY=0
REASONING_PARSER=""
KV_CACHE_DTYPE=""
TOOL_CALL_PARSER=""
ENABLE_AUTO_TOOL_CHOICE=0
# -----------------------------
# Model resolution
# -----------------------------
resolve_model() {
    case "$MODEL" in
        35b|qwen35|qwen35-35b|qwen3.5|qwen3.5-35b|Qwen/Qwen3.5-35B-A3B-FP8)
            MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8"
            MODEL_LABEL="qwen35-35b"
            MODEL_FAMILY="qwen"
            DEFAULT_TENSOR_PARALLEL_SIZE="1"
            REASONING_PARSER="qwen3"
            ;;
        122b|qwen35-122b|qwen3.5-122b|Qwen/Qwen3.5-122B-A10B-FP8)
            MODEL_NAME="Qwen/Qwen3.5-122B-A10B-FP8"
            MODEL_LABEL="qwen35-122b"
            MODEL_FAMILY="qwen"
            DEFAULT_TENSOR_PARALLEL_SIZE="1"
            REASONING_PARSER="qwen3"
            ;;
        qwen36|qwen36-35b|qwen3.6|qwen3.6-35b|Qwen/Qwen3.6-35B-A3B-FP8)
            MODEL_NAME="Qwen/Qwen3.6-35B-A3B-FP8"
            MODEL_LABEL="qwen36-35b"
            MODEL_FAMILY="qwen"
            DEFAULT_TENSOR_PARALLEL_SIZE="1"
            LANGUAGE_MODEL_ONLY=1
            REASONING_PARSER="qwen3"
            ;;
        glm52|glm5.2|glm-5.2|zai-org/GLM-5.2-FP8)
            MODEL_NAME="zai-org/GLM-5.2-FP8"
            MODEL_LABEL="glm52"
            MODEL_FAMILY="glm"
            DEFAULT_TENSOR_PARALLEL_SIZE="8"
            DEFAULT_GPU_MEMORY_UTILIZATION="0.92"
            DEFAULT_MAX_MODEL_LEN="131072"
            REASONING_PARSER="glm45"
            KV_CACHE_DTYPE="fp8"
            TOOL_CALL_PARSER="glm47"
            ENABLE_AUTO_TOOL_CHOICE=1
            ;;
        *)
            echo "[ERROR] Unknown MODEL '$MODEL'."
            echo "[ERROR] Expected one of: 35b, 122b, qwen36, glm52."
            exit 1
            ;;
    esac
}

resolve_model

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TENSOR_PARALLEL_SIZE}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"

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
            echo "[INFO]   MONOKERNEL_CONFIG   : ${MONOKERNEL_CONFIG:-<default (id 0)>}"
            ;;
        triton)
            export VLLM_USE_MOE_MONOKERNEL=0
            ;;
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

build_vllm_args() {
    VLLM_ARGS=(
        serve "$MODEL_NAME"
        --trust-remote-code
        --host "$HOST"
        --port "$PORT"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --max-num-seqs "$MAX_NUM_SEQS"
        --no-enable-prefix-caching
        --enable-chunked-prefill
    )

    if [ "$LANGUAGE_MODEL_ONLY" -eq 1 ]; then
        VLLM_ARGS+=(--language-model-only)
    fi
    if [ -n "$REASONING_PARSER" ]; then
        VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
    fi
    if [ -n "$KV_CACHE_DTYPE" ]; then
        VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    fi
    if [ -n "$TOOL_CALL_PARSER" ]; then
        VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
    fi
    if [ "$ENABLE_AUTO_TOOL_CHOICE" -eq 1 ]; then
        VLLM_ARGS+=(--enable-auto-tool-choice)
    fi
    if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
        VLLM_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    fi
    if [ -n "$MAX_MODEL_LEN" ]; then
        VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    fi
}

start_vllm_server() {
    echo "[INFO] ====================================================="
    echo "[INFO] Starting vLLM server"
    echo "[INFO]   Model               : $MODEL_NAME"
    echo "[INFO]   Model label         : $MODEL_LABEL"
    echo "[INFO]   Tensor parallel     : $TENSOR_PARALLEL_SIZE"
    echo "[INFO]   Port                : $PORT"
    echo "[INFO]   MoE backend         : $CURRENT_BACKEND (VLLM_USE_MOE_MONOKERNEL=$VLLM_USE_MOE_MONOKERNEL)"
    echo "[INFO]   GPU mem util        : ${GPU_MEMORY_UTILIZATION:-<default>}"
    echo "[INFO]   Max model len       : ${MAX_MODEL_LEN:-<default>}"
    echo "[INFO]   Reasoning parser    : ${REASONING_PARSER:-<none>}"
    echo "[INFO]   Server log          : $SERVER_LOG"
    echo "[INFO] ====================================================="

    build_vllm_args
    vllm "${VLLM_ARGS[@]}" &> "$SERVER_LOG" &
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
    for bs in "${BS_SWEEP[@]}"; do
        local concurrent="${LM_EVAL_NUM_CONCURRENT:-$bs}"
        local output_dir="$RESULTS_BASE_DIR/bs_${bs}/results"
        local eval_log="$RESULTS_BASE_DIR/bs_${bs}/lm_eval.log"
        mkdir -p "$output_dir"

        echo "[INFO] -----------------------------------------------------"
        echo "[INFO] Running lm_eval"
        echo "[INFO]   Backend        : $CURRENT_BACKEND"
        echo "[INFO]   Model          : $MODEL_NAME"
        echo "[INFO]   Task(s)        : $TASK"
        echo "[INFO]   Num fewshot    : $NUM_FEWSHOT"
        echo "[INFO]   Limit          : ${LIMIT:-full}"
        echo "[INFO]   BS/concurrent  : $concurrent"
        echo "[INFO]   lm_eval batch  : ${LM_EVAL_BATCH_SIZE:-<default>}"
        echo "[INFO]   Output dir     : $output_dir"
        echo "[INFO] -----------------------------------------------------"

        local lm_eval_args=(
            --model local-completions
            --model_args "model=${MODEL_NAME},base_url=http://${HOST}:${PORT}/v1/completions,tokenized_requests=False,num_concurrent=${concurrent}"
            --tasks "$TASK"
            --num_fewshot "$NUM_FEWSHOT"
            --output_path "$output_dir"
            --log_samples
            --confirm_run_unsafe_code
        )

        if [ -n "${LIMIT}" ] && [ "${LIMIT}" != "0" ]; then
            lm_eval_args+=(--limit "$LIMIT")
        fi
        if [ -n "$LM_EVAL_BATCH_SIZE" ]; then
            lm_eval_args+=(--batch_size "$LM_EVAL_BATCH_SIZE")
        fi

        local start_time
        start_time=$(date +%s)
        lm_eval "${lm_eval_args[@]}" 2>&1 | tee "$eval_log"

        local end_time
        end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        echo "[INFO] lm_eval BS=$bs completed in $((total_time / 60))m $((total_time % 60))s"
        echo "[INFO] Results saved to: $output_dir"
        echo "[INFO] -----------------------------------------------------"
    done
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
# These models are native vLLM architectures, so no plugins are needed.
# Empty allowlist = load none. Override by exporting VLLM_PLUGINS before use.
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

# HumanEval/code-execution tasks have two explicit opt-in gates:
#   1. HF evaluate's metric import gate.
#   2. lm_eval's own unsafe-code confirmation CLI flag.
# NOTE: HumanEval completions execute generated Python on this host unsandboxed.
export HF_ALLOW_CODE_EVAL=1

if ! command -v vllm &>/dev/null; then
    echo "[ERROR] vllm command not found after activating: $VENV_ACTIVATE"
    exit 1
fi
if ! command -v lm_eval &>/dev/null; then
    echo "[ERROR] lm_eval not found after activating: $VENV_ACTIVATE"
    echo "[ERROR] Install it with: pip install lm_eval"
    exit 1
fi

mkdir -p "$RUN_DIR"
echo "[INFO] ====================================================="
echo "[INFO] lm_eval accuracy evaluation"
echo "[INFO]   Model selector : $MODEL"
echo "[INFO]   Model          : $MODEL_NAME"
echo "[INFO]   Model family   : $MODEL_FAMILY"
echo "[INFO]   Task(s)        : $TASK"
echo "[INFO]   Backend(s)     : $BACKEND"
echo "[INFO]   Limit          : ${LIMIT:-full}"
echo "[INFO]   BS/concurrent  : ${LM_EVAL_NUM_CONCURRENT:-<per-BS sweep: ${BS_SWEEP[*]}>}"
echo "[INFO]   Max num seqs   : $MAX_NUM_SEQS"
echo "[INFO]   Run directory  : $RUN_DIR"
echo "[INFO] ====================================================="

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
