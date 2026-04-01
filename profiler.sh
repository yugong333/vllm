#!/bin/bash

# export VLLM_TUNED_CONFIG_FOLDER=/home/ubuntu/KernelTuner/Triton/multi_lora/configs/test
# export VLLM_TUNED_CONFIG_FOLDER=/home/ubuntu/KernelTuner/Triton/multi_lora/configs/nemotron

# MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
# export VLLM_MOE_FORCE_TRITON=1
# export VLLM_USE_FLASHINFER_MOE_FP8=0
# export VLLM_DISABLE_CUTLASS_MOE=1
# export VLLM_USE_DEEP_GEMM=0
# export VLLM_MOE_USE_DEEP_GEMM=0

MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
LORA_PATH="/home/ubuntu/adapter_training/pissa-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8-fp16"
TP_SIZE=8
# Profiler: nsys | torch | ncu
PROFILER="nsys"

# Sanitize model name for use in directory paths (replace / with _)
MODEL_NAME_SAFE="${MODEL_NAME//\//_}"

MAX_LORAS=(0)
CONCURRENCY_LEVELS=(1)

for MAX_LORA in "${MAX_LORAS[@]}"; do
    for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)

        # Build python args once — shared across all profilers
        PY_ARGS=(
            --model-name "${MODEL_NAME}"
            --concurrency "${CONCURRENCY}"
            --max-loras "${MAX_LORA}"
            --lora-path "${LORA_PATH}"
            --tp-size "${TP_SIZE}"
        )
        [ "$MAX_LORA" -gt 0 ] && PY_ARGS+=(--use-lora)

        if [ "$PROFILER" = "nsys" ]; then
            DES_DIR="/home/ubuntu/vllm/nsys_profile/${MODEL_NAME_SAFE}/max_lora_${MAX_LORA}/concurrency_${CONCURRENCY}/${TIMESTAMP}"
            mkdir -p "$DES_DIR"
            nsys profile \
                -t nvtx,cuda \
                --cudabacktrace=all \
                --cuda-graph-trace=node \
                --wait all \
                --capture-range cudaProfilerApi \
                --capture-range-end=stop \
                --trace-fork-before-exec=true \
                -o "${DES_DIR}/profile" \
                --force-overwrite=true \
                python nsys_profile.py "${PY_ARGS[@]}" --nsys-profile

        elif [ "$PROFILER" = "torch" ]; then
            DES_DIR="/home/ubuntu/vllm/torch_profile/${MODEL_NAME_SAFE}/max_lora_${MAX_LORA}/concurrency_${CONCURRENCY}/${TIMESTAMP}"
            mkdir -p "$DES_DIR"
            export VLLM_TORCH_PROFILER_DIR="$DES_DIR"
            python nsys_profile.py "${PY_ARGS[@]}" --torch-profile

        elif [ "$PROFILER" = "ncu" ]; then
            DES_DIR="/home/ubuntu/vllm/ncu_profile/${MODEL_NAME_SAFE}/max_lora_${MAX_LORA}/concurrency_${CONCURRENCY}/${TIMESTAMP}"
            mkdir -p "$DES_DIR"
            ncu \
                --target-processes all \
                --replay-mode kernel \
                --launch-skip 50 --launch-count 20 \
                --kernel-name "::regex:moe_kernel" \
                --set basic \
                --section WarpStateStats \
                --section SchedulerStats \
                --section ComputeWorkloadAnalysis \
                --section MemoryWorkloadAnalysis \
                --section SpeedOfLight \
                -o "${DES_DIR}/profile" \
                -f \
                python nsys_profile.py "${PY_ARGS[@]}"

        else
            echo "[ERROR] Unknown profiler: $PROFILER. Choose from: nsys, torch, ncu"
            exit 1
        fi

        echo "[INFO] Done: PROFILER=${PROFILER} MAX_LORA=${MAX_LORA} CONCURRENCY=${CONCURRENCY}"
        echo "[INFO] Output: ${DES_DIR}"

    done
done

# --launch-skip 20 --launch-count 4 \
# --kernel-name ::regex:".*_fused_moe_lora_kernel.*" \
# --profile-from-start off \
