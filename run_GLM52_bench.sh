#!/bin/bash
# =============================================================================
# vllm bench serve benchmark for the GLM-5.2-FP8 model (zai-org/GLM-5.2-FP8).
#
# Dataset      : sonnet (default) | gsm8k | sharegpt  (see DATASET below)
# Output length: 200 tokens
# TP           : 1
# Num requests : 50 (per batch-size run)
# Batch sizes  : 1 2 4 8  (mapped to --max-concurrency)
#
# The script starts a vLLM OpenAI-compatible server, waits until it is ready,
# sweeps the batch sizes with `vllm bench serve`, then shuts the server down.
#
# Usage:
#   ./run_GLM52_bench.sh [BACKEND] [DATASET]
#
#   BACKEND (optional, default: monokernel)
#     monokernel  Use the MoE monokernel fast path (VLLM_USE_MOE_MONOKERNEL=1)
#     triton      Force the standard TRITON fused-MoE backend
#                 (VLLM_USE_MOE_MONOKERNEL=0)
#     both        Run the full sweep once per backend (triton then monokernel)
#     capture     Capture REAL decode-time MoE routing for offline kernel
#                 benchmarking. Runs the TRITON backend in EAGER mode at
#                 BS=8 only, with MONOKERNEL_ROUTE_CAPTURE pointed at a .pt
#                 dump under the run dir.
#
#   DATASET (optional, default: sonnet)
#     sonnet      Fixed-shape sonnet.txt prompts (INPUT_LEN=1600, OUT=200).
#                 Deterministic input length — best for apples-to-apples perf.
#     gsm8k       openai/gsm8k (grade-school math) via the HF loader. Short,
#                 natural prompts; OUT capped at 256. Routing reflects real
#                 reasoning-style traffic.
#     sharegpt    Aeala/ShareGPT_Vicuna_unfiltered chat logs via the HF loader
#                 ("gptchat"-style conversational prompts). Variable length.
#                 Override the HF id with SHAREGPT_PATH=<hf-id-or-local.json>.
#
#   Both can also be set via the BACKEND / DATASET environment variables;
#   the positional args take precedence.  Examples:
#     ./run_GLM52_bench.sh                  # monokernel, sonnet
#     ./run_GLM52_bench.sh both gsm8k       # triton+mono, gsm8k
#     ./run_GLM52_bench.sh capture sharegpt # routing capture on sharegpt
#     BACKEND=monokernel DATASET=gsm8k ./run_GLM52_bench.sh
#
# Results are written under a per-run timestamped folder:
#   result_outputs/vllm_bench/<MODEL>/<DATASET>_<IN>_out_<OUT>/tp_<TP>/
#       run_<HOST>_<TIMESTAMP>/<backend>/batch_size_<bs>/...
# =============================================================================
set -uo pipefail

# -----------------------------
# Configuration
# -----------------------------
# Resolve the repo from this script's own location so the in-repo venv is used
# regardless of where the script is invoked from.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the venv: prefer the in-repo venv_vllm. An explicit VENV_ACTIVATE env
# var overrides; fall back to the historical /opt/vllm-venv if present.
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

# Backend selection: monokernel | triton | capture | both (default monokernel).
# Dataset selection: sonnet | gsm8k | sharegpt (default sonnet).
# Positional args take precedence over the env vars:
#   $1 = BACKEND, $2 = DATASET.
BACKEND="${1:-${BACKEND:-monokernel}}"
DATASET="${2:-${DATASET:-sonnet}}"

# GLM-5.2-FP8 — a single fixed model + a short tag used in result paths.
MODEL_NAME="zai-org/GLM-5.2-FP8"
MODEL_TAG="glm52"
TOKENIZER_NAME="$MODEL_NAME"

PORT=8000
HOST="127.0.0.1"
TENSOR_PARALLEL_SIZE=8

# Dataset / request shape
OUTPUT_LEN=200
NUM_PROMPTS=50                 # request number
BATCH_SIZES=(1 2 4 8)          # --max-concurrency sweep

SEED=11111

# Map the DATASET selector to `vllm bench serve` args. sonnet is fixed-shape
# (deterministic input length); gsm8k / sharegpt are real variable-length
# prompts loaded via the HF dataset loader, so they have no --*-input-len and
# their output length is bounded by --*-output-len (=OUTPUT_LEN here).
# DATASET_ARGS is expanded verbatim into the `vllm bench serve` invocation.
DATASET_TAG="$DATASET"         # used in the result path + filenames
DATASET_ARGS=()
case "$DATASET" in
    sonnet)
        INPUT_LEN=1600
        SONNET_PREFIX_LEN=200  # default for sonnet; must be < INPUT_LEN
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
        # openai/gsm8k via the HF loader (subset "main", split "test").
        DATASET_ARGS=(
            --dataset-name hf
            --dataset-path openai/gsm8k
            --hf-subset main
            --hf-split test
            --hf-output-len "$OUTPUT_LEN"
        )
        DATASET_TAG="gsm8k"
        ;;
    sharegpt)
        # "gptchat"-style conversational prompts. Defaults to the canonical HF
        # ShareGPT mirror; override with SHAREGPT_PATH (HF id or local .json).
        SHAREGPT_PATH="${SHAREGPT_PATH:-Aeala/ShareGPT_Vicuna_unfiltered}"
        if [ -f "$SHAREGPT_PATH" ]; then
            # Local JSON file -> the native sharegpt loader.
            DATASET_ARGS=(
                --dataset-name sharegpt
                --dataset-path "$SHAREGPT_PATH"
                --sharegpt-output-len "$OUTPUT_LEN"
            )
        else
            # HF dataset id -> the HF loader.
            DATASET_ARGS=(
                --dataset-name hf
                --dataset-path "$SHAREGPT_PATH"
                --hf-output-len "$OUTPUT_LEN"
            )
        fi
        DATASET_TAG="sharegpt"
        ;;
    *)
        echo "[ERROR] Unknown DATASET '$DATASET' (expected sonnet|gsm8k|sharegpt)."
        exit 1
        ;;
esac

# Timestamped run folder so repeated runs never clobber each other.
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' | tr -cd 'A-Za-z0-9_')"
GPU_NAME="${GPU_NAME:-GPU}"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_DIR/result_outputs/vllm_bench/${MODEL_NAME}/${DATASET_TAG}_out_${OUTPUT_LEN}/tp_${TENSOR_PARALLEL_SIZE}/run_${GPU_NAME}_${RUN_STAMP}"

# Per-backend values, set by configure_backend().
RESULTS_BASE_DIR=""
SERVER_LOG=""
VLLM_PID=""

# -----------------------------
# Helpers
# -----------------------------
# Per-backend capture state, set by configure_backend(): when CAPTURE_MODE=1
# the server runs eager + restricts the sweep to BS=8 and writes a routing
# dump for offline kernel benchmarking.
CAPTURE_MODE=0
ROUTE_CAPTURE_PATH=""

configure_backend() {
    # $1 = backend label (monokernel | triton | capture)
    local backend="$1"
    CAPTURE_MODE=0
    unset MONOKERNEL_ROUTE_CAPTURE
    case "$backend" in
        monokernel)
            export VLLM_USE_MOE_MONOKERNEL=1
            # MONOKERNEL_CONFIG (if exported by the caller) selects a tuned
            # KernelConfig id; inherited by `vllm serve`.  Logged for the record.
            echo "[INFO] MONOKERNEL_CONFIG = ${MONOKERNEL_CONFIG:-<default (id 0)>}"
            ;;
        triton)     export VLLM_USE_MOE_MONOKERNEL=0 ;;
        capture)
            # Capture real routing on the TRITON path (matches the kernel
            # benchmark's Triton baseline). The capture hook lives in the
            # common MoE runner path, so it fires regardless of backend.
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

    # In capture mode run eager (no CUDA graphs) so the routing-capture hook
    # observes every decode step and its atexit flush runs cleanly on shutdown.
    local eager_flag=()
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        eager_flag=(--enforce-eager)
        echo "[INFO]   Eager mode          : ON (routing capture)"
    fi

    vllm serve "$MODEL_NAME" \
        --trust-remote-code \
        --host "$HOST" \
        --port "$PORT" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-num-seqs 64 \
        --no-enable-prefix-caching \
        --enable-chunked-prefill \
        --kv-cache-dtype fp8 \
        --tool-call-parser glm47 \
        --enable-auto-tool-choice \
        --reasoning-parser glm45 \
        --gpu-memory-utilization 0.92 \
        --max-model-len 131072 \
        "${eager_flag[@]}" \
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
    # Capture mode only needs the BS=8 decode shape (M<=8 is the monokernel
    # fast-path regime we're characterising), so restrict the sweep.
    local bs_list=("${BATCH_SIZES[@]}")
    if [ "$CAPTURE_MODE" -eq 1 ]; then
        bs_list=(8)
        echo "[INFO] Capture mode: restricting batch-size sweep to BS=8 only."
    fi
    for bs in "${bs_list[@]}"; do
        local result_dir="$RESULTS_BASE_DIR/batch_size_${bs}"
        local result_filename="${MODEL_TAG}_${DATASET_TAG}_bs${bs}.json"
        local run_log="$result_dir/run.log"
        mkdir -p "$result_dir"

        echo "[INFO] -----------------------------------------------------"
        echo "[INFO] Running vllm bench serve | backend=$CURRENT_BACKEND | dataset=$DATASET | batch size (max-concurrency) = $bs"
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
# GLM-5.2 is a native vLLM arch, so no plugins are needed. Empty allowlist =
# load none. Override by exporting VLLM_PLUGINS before invoking this script.
export VLLM_PLUGINS="${VLLM_PLUGINS-}"

# sonnet is the only file-based dataset; gsm8k / sharegpt(HF) are pulled from
# the HF hub by the loader, so only validate the local path for sonnet.
if [ "$DATASET" = "sonnet" ] && [ ! -f "$REPO_DIR/benchmarks/sonnet.txt" ]; then
    echo "[ERROR] Sonnet dataset not found at: $REPO_DIR/benchmarks/sonnet.txt"
    exit 1
fi

mkdir -p "$RUN_DIR"
echo "[INFO] ====================================================="
echo "[INFO] Model         : $MODEL_NAME"
echo "[INFO] Dataset       : $DATASET  (args: ${DATASET_ARGS[*]})"
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
    monokernel|triton|capture)
        run_one_backend "$BACKEND"
        ;;
    *)
        echo "[ERROR] Invalid BACKEND '$BACKEND' (expected monokernel|triton|capture|both)."
        exit 1
        ;;
esac

# Capture mode: surface the routing-dump path.
if [ "$BACKEND" = "capture" ] && [ -n "$ROUTE_CAPTURE_PATH" ]; then
    echo "[INFO] ====================================================="
    if [ -f "$ROUTE_CAPTURE_PATH" ]; then
        echo "[INFO] Routing capture written to: $ROUTE_CAPTURE_PATH"
    else
        echo "[WARN] Expected routing dump not found at: $ROUTE_CAPTURE_PATH"
        echo "[WARN] Check the server log for the [route_capture] line: $SERVER_LOG"
    fi
fi

trap - EXIT INT TERM

# When both backends ran, auto-compare them (triton vs monokernel) so the
# speedup table prints right after the sweep. summarize_bench.py reads the
# per-backend JSONs under $RUN_DIR.
if [ "$BACKEND" = "both" ]; then
    SUMMARIZE="$REPO_DIR/summarize_bench.py"
    if [ -f "$SUMMARIZE" ]; then
        echo "[INFO] ====================================================="
        echo "[INFO] Comparing backends (triton vs monokernel) ..."
        python "$SUMMARIZE" "$RUN_DIR" || \
            echo "[WARN] summarize_bench.py failed; JSONs are under $RUN_DIR"
    else
        echo "[WARN] $SUMMARIZE not found; skipping auto-comparison."
    fi
fi

echo "[INFO] All benchmarks completed. Results under: $RUN_DIR"
