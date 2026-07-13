#!/bin/bash
# =============================================================================
# Unified vllm bench serve runner for MoE FP8 models.
#
# Usage:
#   ./run_moe_bench.sh [BACKEND] [MODEL] [DATASET]
#
#   BACKEND (optional, default: monokernel)
#     monokernel  Use the MoE monokernel fast path (VLLM_USE_MOE_MONOKERNEL=1)
#     triton      Force the standard TRITON fused-MoE backend
#     both        Run triton, then monokernel, for comparison
#     capture     Capture real decode-time MoE routing on the Triton path,
#                 in eager mode, BS=8 only, to <run_dir>/capture/route_capture.pt
#
#   MODEL (optional, default: 35b)
#     35b | qwen35-35b | qwen3.5-35b       -> Qwen/Qwen3.5-35B-A3B-FP8
#     122b | qwen35-122b | qwen3.5-122b    -> Qwen/Qwen3.5-122B-A10B-FP8
#     qwen36 | qwen36-35b | qwen3.6-35b    -> Qwen/Qwen3.6-35B-A3B-FP8
#     glm52 | glm5.2 | glm-5.2             -> zai-org/GLM-5.2-FP8
#
#   DATASET (optional, default: sonnet)
#     sonnet      Fixed-shape sonnet.txt prompts (default IN=1600, OUT=200)
#     gsm8k       openai/gsm8k via the HF loader
#     sharegpt    ShareGPT-style prompts; override with SHAREGPT_PATH
#
# Examples:
#   ./run_moe_bench.sh
#   ./run_moe_bench.sh both 122b gsm8k
#   ./run_moe_bench.sh monokernel qwen36 sonnet
#   ./run_moe_bench.sh capture glm52 sharegpt
#
# Results:
#   result_outputs/vllm_bench/<MODEL>/<DATASET>_out_<OUT>/tp_<TP>/
#       run_<GPU>_<TIMESTAMP>/<backend>/batch_size_<bs>/...
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
DATASET="${3:-${DATASET:-sonnet}}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

OUTPUT_LEN="${OUTPUT_LEN:-200}"
INPUT_LEN="${INPUT_LEN:-1600}"
SONNET_PREFIX_LEN="${SONNET_PREFIX_LEN:-200}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
BATCH_SIZES_STR="${BATCH_SIZES:-1 2 4 8}"
# shellcheck disable=SC2206
BATCH_SIZES=($BATCH_SIZES_STR)
SEED="${SEED:-11111}"

MODEL_NAME=""
MODEL_TAG=""
MODEL_FAMILY=""
TOKENIZER_NAME=""
DEFAULT_TENSOR_PARALLEL_SIZE="1"
DEFAULT_GPU_MEMORY_UTILIZATION=""
DEFAULT_MAX_MODEL_LEN=""
LANGUAGE_MODEL_ONLY=0
REASONING_PARSER=""
KV_CACHE_DTYPE=""
TOOL_CALL_PARSER=""
ENABLE_AUTO_TOOL_CHOICE=0
# -----------------------------
# Model and dataset resolution
# -----------------------------
resolve_model() {
    case "$MODEL" in
        35b|qwen35|qwen35-35b|qwen3.5|qwen3.5-35b|Qwen/Qwen3.5-35B-A3B-FP8)
            MODEL_NAME="Qwen/Qwen3.5-35B-A3B-FP8"
            MODEL_TAG="qwen35"
            MODEL_FAMILY="qwen"
            DEFAULT_TENSOR_PARALLEL_SIZE="1"
            REASONING_PARSER="qwen3"
            ;;
        122b|qwen35-122b|qwen3.5-122b|Qwen/Qwen3.5-122B-A10B-FP8)
            MODEL_NAME="Qwen/Qwen3.5-122B-A10B-FP8"
            MODEL_TAG="qwen35_122b"
            MODEL_FAMILY="qwen"
            DEFAULT_TENSOR_PARALLEL_SIZE="1"
            REASONING_PARSER="qwen3"
            ;;
        qwen36|qwen36-35b|qwen3.6|qwen3.6-35b|Qwen/Qwen3.6-35B-A3B-FP8)
            MODEL_NAME="Qwen/Qwen3.6-35B-A3B-FP8"
            MODEL_TAG="qwen36"
            MODEL_FAMILY="qwen"
            DEFAULT_TENSOR_PARALLEL_SIZE="1"
            LANGUAGE_MODEL_ONLY=1
            REASONING_PARSER="qwen3"
            ;;
        glm52|glm5.2|glm-5.2|zai-org/GLM-5.2-FP8)
            MODEL_NAME="zai-org/GLM-5.2-FP8"
            MODEL_TAG="glm52"
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
            echo "[ERROR] Unknown MODEL '$MODEL' (expected 35b|122b|qwen36|glm52)."
            exit 1
            ;;
    esac
    TOKENIZER_NAME="$MODEL_NAME"
}

resolve_dataset() {
    DATASET_TAG="$DATASET"
    FILENAME_DATASET_TAG="$DATASET"
    DATASET_ARGS=()
    case "$DATASET" in
        sonnet)
            DATASET_ARGS=(
                --dataset-name sonnet
                --dataset-path "$REPO_DIR/benchmarks/sonnet.txt"
                --sonnet-input-len "$INPUT_LEN"
                --sonnet-output-len "$OUTPUT_LEN"
                --sonnet-prefix-len "$SONNET_PREFIX_LEN"
            )
            DATASET_TAG="sonnet_${INPUT_LEN}"
            # Preserve the old Qwen3.6 filename style for the default sonnet run.
            if [ "$MODEL_TAG" = "qwen36" ]; then
                FILENAME_DATASET_TAG="sonnet"
            else
                FILENAME_DATASET_TAG="$DATASET_TAG"
            fi
            ;;
        gsm8k)
            DATASET_ARGS=(
                --dataset-name hf
                --dataset-path openai/gsm8k
                --hf-subset main
                --hf-split test
                --hf-output-len "$OUTPUT_LEN"
            )
            DATASET_TAG="gsm8k"
            FILENAME_DATASET_TAG="gsm8k"
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
            DATASET_TAG="sharegpt"
            FILENAME_DATASET_TAG="sharegpt"
            ;;
        *)
            echo "[ERROR] Unknown DATASET '$DATASET' (expected sonnet|gsm8k|sharegpt)."
            exit 1
            ;;
    esac
}
resolve_model
resolve_dataset

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$DEFAULT_TENSOR_PARALLEL_SIZE}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$DEFAULT_GPU_MEMORY_UTILIZATION}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr -cd 'A-Za-z0-9_')"
GPU_NAME="${GPU_NAME:-GPU}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_DIR/result_outputs/vllm_bench/${MODEL_NAME}/${DATASET_TAG}_out_${OUTPUT_LEN}/tp_${TENSOR_PARALLEL_SIZE}/run_${GPU_NAME}_${RUN_STAMP}"

RESULTS_BASE_DIR=""
SERVER_LOG=""
VLLM_PID=""
CURRENT_BACKEND=""
CAPTURE_MODE=0
ROUTE_CAPTURE_PATH=""

# -----------------------------
# Helpers
# -----------------------------
configure_backend() {
    local backend="$1"
    CAPTURE_MODE=0
    unset MONOKERNEL_ROUTE_CAPTURE
    case "$backend" in
        monokernel)
            export VLLM_USE_MOE_MONOKERNEL=1
            echo "[INFO] MONOKERNEL_CONFIG = ${MONOKERNEL_CONFIG:-<default (id 0)>}"
            ;;
        triton)
            export VLLM_USE_MOE_MONOKERNEL=0
            ;;
        capture)
            export VLLM_USE_MOE_MONOKERNEL=0
            CAPTURE_MODE=1
            ;;
        *)
            echo "[ERROR] Unknown backend '$backend' (expected monokernel|triton|capture|both)."
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
    if [ -n "$KV_CACHE_DTYPE" ]; then
        VLLM_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    fi
    if [ -n "$TOOL_CALL_PARSER" ]; then
        VLLM_ARGS+=(--tool-call-parser "$TOOL_CALL_PARSER")
    fi
    if [ "$ENABLE_AUTO_TOOL_CHOICE" -eq 1 ]; then
        VLLM_ARGS+=(--enable-auto-tool-choice)
    fi
    if [ -n "$REASONING_PARSER" ]; then
        VLLM_ARGS+=(--reasoning-parser "$REASONING_PARSER")
    fi
    if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
        VLLM_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    fi
    if [ -n "$MAX_MODEL_LEN" ]; then
        VLLM_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        VLLM_ARGS+=(--enforce-eager)
    fi
}

start_vllm_server() {
    echo "[INFO] ====================================================="
    echo "[INFO] Starting vLLM server"
    echo "[INFO]   Model               : $MODEL_NAME"
    echo "[INFO]   Model tag           : $MODEL_TAG"
    echo "[INFO]   Tensor parallel     : $TENSOR_PARALLEL_SIZE"
    echo "[INFO]   Port                : $PORT"
    echo "[INFO]   MoE backend         : $CURRENT_BACKEND (VLLM_USE_MOE_MONOKERNEL=$VLLM_USE_MOE_MONOKERNEL)"
    echo "[INFO]   GPU mem util        : ${GPU_MEMORY_UTILIZATION:-<default>}"
    echo "[INFO]   Max model len       : ${MAX_MODEL_LEN:-<default>}"
    echo "[INFO]   Reasoning parser    : ${REASONING_PARSER:-<none>}"
    echo "[INFO]   Server log          : $SERVER_LOG"
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        echo "[INFO]   Eager mode          : ON (routing capture)"
    fi
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
run_benchmark() {
    local bs_list=("${BATCH_SIZES[@]}")
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        bs_list=(8)
        echo "[INFO] Capture mode: restricting batch-size sweep to BS=8 only."
    fi

    for bs in "${bs_list[@]}"; do
        local result_dir="$RESULTS_BASE_DIR/batch_size_${bs}"
        local result_filename="${MODEL_TAG}_${FILENAME_DATASET_TAG}_bs${bs}.json"
        local run_log="$result_dir/run.log"
        mkdir -p "$result_dir"

        echo "[INFO] -----------------------------------------------------"
        echo "[INFO] Running vllm bench serve | backend=$CURRENT_BACKEND | model=$MODEL_TAG | dataset=$DATASET | batch size (max-concurrency) = $bs"
        echo "[INFO]   num-prompts = $NUM_PROMPTS | output-len = $OUTPUT_LEN"
        echo "[INFO]   dataset args = ${DATASET_ARGS[*]}"
        echo "[INFO]   results -> $result_dir/$result_filename"

        local start_time
        start_time=$(date +%s)

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
            2>&1 | tee "$run_log"

        local end_time
        end_time=$(date +%s)
        local total_time=$((end_time - start_time))
        echo "[INFO] Finished batch size $bs in $((total_time / 60))m $((total_time % 60))s"
    done
    echo "[INFO] -----------------------------------------------------"
}

run_one_backend() {
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

print_capture_result() {
    if [ "$BACKEND" != "capture" ] || [ -z "$ROUTE_CAPTURE_PATH" ]; then
        return
    fi
    echo "[INFO] ====================================================="
    if [ -f "$ROUTE_CAPTURE_PATH" ]; then
        echo "[INFO] Routing capture written to: $ROUTE_CAPTURE_PATH"
        local bench_model_key=""
        case "$MODEL_TAG" in
            qwen35) bench_model_key="qwen3.5" ;;
            qwen35_122b) bench_model_key="qwen3.5_122b" ;;
            *) bench_model_key="" ;;
        esac
        if [ -n "$bench_model_key" ]; then
            echo "[INFO] Replay it into the kernel benchmark with:"
            echo "[INFO]   python test_monokernel_accuracy.py --model $bench_model_key \\" 
            echo "[INFO]       --route-capture $ROUTE_CAPTURE_PATH"
        fi
    else
        echo "[WARN] Expected routing dump not found at: $ROUTE_CAPTURE_PATH"
        echo "[WARN] Check the server log for the [route_capture] line: $SERVER_LOG"
    fi
}
run_summary_if_requested() {
    if [ "$BACKEND" != "both" ]; then
        return
    fi
    local summarize="$REPO_DIR/summarize_bench.py"
    if [ -f "$summarize" ]; then
        echo "[INFO] ====================================================="
        echo "[INFO] Comparing backends (triton vs monokernel) ..."
        python "$summarize" "$RUN_DIR" || \
            echo "[WARN] summarize_bench.py failed; JSONs are under $RUN_DIR"
    else
        echo "[WARN] $summarize not found; skipping auto-comparison."
    fi
}

# -----------------------------
# Main
# -----------------------------
echo "[INFO] Activating venv: $VENV_ACTIVATE"
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# Disable incompatible third-party vLLM plugins (sfai_inference_lib, nova, ...).
# These benchmark models are native vLLM architectures, so no plugins are needed.
# Empty allowlist = load none. Override by exporting VLLM_PLUGINS before use.
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

if ! command -v vllm &>/dev/null; then
    echo "[ERROR] vllm command not found after activating: $VENV_ACTIVATE"
    exit 1
fi

# Sonnet is the only file-based dataset; gsm8k / sharegpt(HF) are pulled from
# the HF hub by the loader, so only validate the local path for sonnet.
if [ "$DATASET" = "sonnet" ] && [ ! -f "$REPO_DIR/benchmarks/sonnet.txt" ]; then
    echo "[ERROR] Sonnet dataset not found at: $REPO_DIR/benchmarks/sonnet.txt"
    exit 1
fi

mkdir -p "$RUN_DIR"
echo "[INFO] ====================================================="
echo "[INFO] vLLM MoE benchmark"
echo "[INFO]   Model selector : $MODEL"
echo "[INFO]   Model          : $MODEL_NAME"
echo "[INFO]   Model tag      : $MODEL_TAG"
echo "[INFO]   Dataset        : $DATASET  (args: ${DATASET_ARGS[*]})"
echo "[INFO]   Run directory  : $RUN_DIR"
echo "[INFO]   Backend(s)     : $BACKEND"
echo "[INFO]   Batch sizes    : ${BATCH_SIZES[*]}"
echo "[INFO] ====================================================="

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
        echo "[ERROR] Invalid BACKEND '$BACKEND' (expected monokernel|triton|capture|both)."
        exit 1
        ;;
esac

print_capture_result

trap - EXIT INT TERM

run_summary_if_requested

echo "[INFO] All benchmarks completed. Results under: $RUN_DIR"
