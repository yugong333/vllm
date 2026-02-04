#!/bin/bash

MAX_LORAS=(0 8)
CONCURRENCIES=(4)

for MAX_LORA in "${MAX_LORAS[@]}"; do
    for CONCURRENCY in "${CONCURRENCIES[@]}"; do
        # ## torch profiler
        # export VLLM_TORCH_PROFILER_DIR=/home/ubuntu/vllm_main_pr/torch_profile/torch_profiler_max_loras_${MAX_LORA}_concurrency_${CONCURRENCY}
        
        # if [ "$MAX_LORA" -eq 0 ]; then
        #     python nsys_profile.py --concurrency ${CONCURRENCY}
        # else
        #     python nsys_profile.py --max-loras ${MAX_LORA} --use-lora --concurrency ${CONCURRENCY}
        # fi


        ## nsys profiler
        if [ "$MAX_LORA" -eq 0 ]; then

            nsys profile \
                -t nvtx,cuda \
                --cudabacktrace=all \
                --cuda-graph-trace=node \
                --wait all \
                --capture-range cudaProfilerApi \
                --capture-range-end=stop \
                --trace-fork-before-exec=true \
                -o nsys_profile_max_loras_${MAX_LORA} \
                --force-overwrite=true \
                python nsys_profile.py --concurrency ${CONCURRENCY}
        else
            nsys profile \
                -t nvtx,cuda \
                --cudabacktrace=all \
                --cuda-graph-trace=node \
                --wait all \
                --capture-range cudaProfilerApi \
                --capture-range-end=stop \
                --trace-fork-before-exec=true \
                -o nsys_profile_max_loras_${MAX_LORA} \
                --force-overwrite=true \
                python nsys_profile.py --max-loras ${MAX_LORA} --use-lora --concurrency ${CONCURRENCY}
        fi
    done
done
