#!/usr/bin/env bash
# =============================================================================
# Profile the full vLLM decode phase (OFFLINE mode) with Nsight Systems.
#
# Qwen3.5-122B-A10B-FP8 variant.  Uses vLLM's offline LLM API (no server) —
# simpler, faster startup, and the entire process is wrapped under nsys so all
# GPU activity is captured.
#
# Usage:
#   ./profile_vllm_decode_offline_122b.sh [OPTIONS]
#
# Options:
#   --path monokernel|triton   Backend to profile (default: monokernel)
#   --bs 1,2,4,8              Decode batch sizes (comma-separated, default: 1)
#                              Each BS gets its own nsys report.
#   --input-len N             Input tokens per request (default: 1600)
#   --max-tokens N            Decode tokens per request (default: 200)
#   --out DIR                 Output directory (default: nsys_vllm_offline_122b/<ts>)
#   --gpu-mem-util F          gpu_memory_utilization for vLLM (default: 0.95)
#   --max-model-len N         max_model_len for vLLM (default: 8192)
#
# Examples:
#   ./profile_vllm_decode_offline_122b.sh --path monokernel --bs 1,2,4,8
#   ./profile_vllm_decode_offline_122b.sh --path triton --bs 4
#
# Output:
#   <out>/<path>_bs<N>.nsys-rep  — open in Nsight Systems GUI
#   <out>/<path>_bs<N>.log       — stdout/stderr from the run
#
# Notes:
#   * Offline mode uses vLLM's LLM class directly (no HTTP server overhead).
#   * The Python driver script generates prompts of the specified input length,
#     submits them as a batch, and generates max_tokens output tokens.
#   * Prefill processes all BS prompts, then decode runs BS tokens/step.
#   * In the nsys timeline: skip model loading, look for the repeating decode
#     pattern after the initial prefill burst.
#   * The 122B variant is ~122 GB in FP8 and OOM'd at startup on a 140 GB H200
#     with vLLM's default gpu_memory_utilization.  This script defaults to
#     0.95 + max_model_len 8192 to give it headroom; tune via the flags above.
#   * Only the decode steps (M<=8) run on the monokernel; prefill of a
#     1600-token prompt uses the Triton fallback in BOTH backends (the M>8
#     gate in fp8.py), so the monokernel-vs-triton delta lives in decode.
# =============================================================================
set -uo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
PATH_KIND="monokernel"
MODEL_NAME="Qwen/Qwen3.5-122B-A10B-FP8"
BS_LIST="1"
MAX_TOKENS=200
INPUT_LEN=1600
OUT_DIR=""
GPU_MEM_UTIL="0.95"
MAX_MODEL_LEN="8192"
# Repo-local venv (the working env for this checkout).  Resolve relative to
# this script so it works regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${VENV_ACTIVATE:-${SCRIPT_DIR}/venv_vllm/bin/activate}"

NSYS="${NSYS:-}"

# ── Parse args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)          PATH_KIND="$2";     shift 2 ;;
        --bs)            BS_LIST="$2";       shift 2 ;;
        --max-tokens)    MAX_TOKENS="$2";    shift 2 ;;
        --input-len)     INPUT_LEN="$2";     shift 2 ;;
        --out)           OUT_DIR="$2";       shift 2 ;;
        --gpu-mem-util)  GPU_MEM_UTIL="$2";  shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -uo pipefail/p' "$0" | sed -E 's/^# ?//'
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'. See --help." >&2
            exit 2
            ;;
    esac
done

# ── Validate ────────────────────────────────────────────────────────────────
case "$PATH_KIND" in
    monokernel) export VLLM_USE_MOE_MONOKERNEL=1 ;;
    triton)     export VLLM_USE_MOE_MONOKERNEL=0 ;;
    *)
        echo "error: --path must be monokernel or triton (got '$PATH_KIND')" >&2
        exit 2
        ;;
esac

BS_LIST="${BS_LIST//,/ }"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-nsys_vllm_offline_122b/${PATH_KIND}_${TS}}"
mkdir -p "$OUT_DIR"

# Locate nsys
if [[ -z "$NSYS" ]]; then
    if command -v nsys >/dev/null 2>&1; then
        NSYS="nsys"
    elif [[ -x /usr/local/cuda/bin/nsys ]]; then
        NSYS="/usr/local/cuda/bin/nsys"
    else
        echo "error: nsys not found. Set NSYS=/path/to/nsys." >&2
        exit 1
    fi
fi

echo "════════════════════════════════════════════════════════════════"
echo " vLLM OFFLINE decode profiling (Qwen3.5-122B)"
echo "   Backend         : $PATH_KIND (VLLM_USE_MOE_MONOKERNEL=$VLLM_USE_MOE_MONOKERNEL)"
echo "   Model           : $MODEL_NAME"
echo "   Decode BS       : $BS_LIST"
echo "   Input len       : $INPUT_LEN"
echo "   Max tokens      : $MAX_TOKENS"
echo "   GPU mem util    : $GPU_MEM_UTIL"
echo "   Max model len   : $MAX_MODEL_LEN"
echo "   Output          : $OUT_DIR"
echo "   nsys            : $NSYS"
echo "   venv            : $VENV_ACTIVATE"
echo "════════════════════════════════════════════════════════════════"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "error: venv activate not found at '$VENV_ACTIVATE'." \
         "Set VENV_ACTIVATE=/path/to/venv/bin/activate." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV_ACTIVATE"

# ── Generate the inline Python driver ───────────────────────────────────────
DRIVER_SCRIPT="${OUT_DIR}/_offline_driver.py"
cat > "$DRIVER_SCRIPT" << 'PYEOF'
"""Offline vLLM profiling driver.

Loads the model once, then for each batch size:
  1. Generates BS prompts of the specified input length.
  2. Runs vLLM generate (prefill + decode).
  3. Uses torch.cuda.cudart().cudaProfilerStart/Stop to bracket the
     generation so nsys --capture-range=cudaProfilerApi captures only
     the inference, not model loading.
"""
import argparse
import torch
from vllm import LLM, SamplingParams


def make_prompts(tokenizer, input_len: int, batch_size: int) -> list[str]:
    """Create batch_size prompts each approximately input_len tokens."""
    # Use a repeated phrase to hit the target length reliably.
    base = "The quick brown fox jumps over the lazy dog. "
    # Tokenize the base to know tokens per repetition.
    base_ids = tokenizer.encode(base)
    reps = max(1, input_len // len(base_ids))
    long_text = base * reps
    # Truncate to exactly input_len tokens.
    ids = tokenizer.encode(long_text)[:input_len]
    prompt = tokenizer.decode(ids)
    return [prompt] * batch_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bs", type=int, nargs="+", required=True)
    parser.add_argument("--input-len", type=int, default=1600)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    # Load model once.  The 122B variant needs more memory headroom than
    # vLLM's default gpu_memory_utilization (it OOM'd at startup on a 140 GB
    # H200 otherwise), so both knobs are passed explicitly.
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        max_num_seqs=64,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )

    # Warmup (outside profiler window).
    warmup_prompts = make_prompts(tokenizer, args.input_len, 1)
    llm.generate(warmup_prompts, sampling_params)
    torch.cuda.synchronize()

    # Profile each batch size.
    for bs in args.bs:
        prompts = make_prompts(tokenizer, args.input_len, bs)
        torch.cuda.synchronize()

        # Start profiling (nsys captures from here).
        torch.cuda.cudart().cudaProfilerStart()

        outputs = llm.generate(prompts, sampling_params)

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        # Sanity check.
        for i, out in enumerate(outputs):
            gen_len = len(out.outputs[0].token_ids)
            print(f"  [BS={bs}] request {i}: generated {gen_len} tokens")

    print("Done.")


if __name__ == "__main__":
    main()
PYEOF

# ── Profile each batch size ─────────────────────────────────────────────────
for BS in $BS_LIST; do
    REPORT_BASE="${OUT_DIR}/${PATH_KIND}_bs${BS}"
    RUN_LOG="${OUT_DIR}/${PATH_KIND}_bs${BS}.log"

    echo ""
    echo "[INFO] ═══════════════════════════════════════════════════════"
    echo "[INFO] Profiling offline decode BS=$BS"
    echo "[INFO]   Report: ${REPORT_BASE}.nsys-rep"
    echo "[INFO] ═══════════════════════════════════════════════════════"

    "$NSYS" profile \
        --output "$REPORT_BASE" \
        --force-overwrite=true \
        --trace=cuda,nvtx,cublas,osrt \
        --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --sample=none \
        --cpuctxsw=none \
        --stats=false \
        python "$DRIVER_SCRIPT" \
            --model "$MODEL_NAME" \
            --bs "$BS" \
            --input-len "$INPUT_LEN" \
            --max-tokens "$MAX_TOKENS" \
            --gpu-mem-util "$GPU_MEM_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
        2>&1 | tee "$RUN_LOG"

    if [[ -f "${REPORT_BASE}.nsys-rep" ]]; then
        echo "[INFO] ✓ Report saved: ${REPORT_BASE}.nsys-rep"
    else
        echo "[WARN] Report not found. Check log: $RUN_LOG"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo " Profiling complete."
echo "   Reports in: $OUT_DIR"
echo ""
ls -1 "$OUT_DIR"/*.nsys-rep 2>/dev/null || echo "   (no .nsys-rep files found)"
echo ""
echo " Open in Nsight Systems GUI:"
echo "   nsys-ui <report>.nsys-rep"
echo ""
echo " Generate kernel summary:"
echo "   $NSYS stats --report gpukernsum --format csv <report>.nsys-rep"
echo ""
echo " The trace captures ONLY the generate() call (not model loading)"
echo " thanks to cudaProfilerStart/Stop bracketing."
echo "   - Prefill: initial burst (attention + MoE on $INPUT_LEN tokens)"
echo "   - Decode: repeating pattern ($MAX_TOKENS iterations, BS tokens/step)"
echo "   - Monokernel appears as 'moe_kernel_topk' in decode"
echo "════════════════════════════════════════════════════════════════"
