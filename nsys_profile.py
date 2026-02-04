# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

import torch

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

LORA_PATH = "/home/ubuntu/adapter_training/pissa-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8-fp16-zero"


def llm_profile(args):
    prompts = [
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is 5+5?",
        "What is 6+6?",
        "What is 7+7?",
        "What is 8+8?",
        "What is 9+9?",
    ]

    max_num_seqs = 8

    # Set to false for baseline profiling w/out lora
    max_loras = args.max_loras
    use_lora = args.use_lora
    concurrency = args.concurrency
    if use_lora:
        llm = LLM(
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=max_num_seqs,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=32,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            compilation_config={"compile_sizes": [1, 2, 4, 8, 16, 32, 64]},
        )
        lora_requests = []
        for i in range(concurrency):
            lora_req = LoRARequest(
                lora_name=f"lora{i + 1}", lora_int_id=i + 1, lora_path=LORA_PATH
            )
            llm.llm_engine.add_lora(lora_req)
            lora_requests.append(lora_req)
    else:
        llm = LLM(
            model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            compilation_config={"compile_sizes": [1, 2, 4, 8, 16, 32, 64]},
        )
        lora_req = None

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=600)

    # warmup
    if lora_req:
        warmup_prompts = prompts[:concurrency]
        # Map each prompt to a LoRA in round-robin fashion
        warmup_lora_mapping = [
            lora_requests[i % len(lora_requests)] for i in range(len(warmup_prompts))
        ]
        llm.generate(warmup_prompts, sampling_params, lora_request=warmup_lora_mapping)
    else:
        llm.generate(prompts[:max_num_seqs], sampling_params)

    llm.reset_prefix_cache()

    torch.cuda.cudart().cudaProfilerStart()
    # llm.start_profile()
    if lora_req:
        request_prompts = prompts[:concurrency]
        # Map each prompt to a LoRA in round-robin fashion
        request_lora_mapping = [
            lora_requests[i % len(lora_requests)] for i in range(len(request_prompts))
        ]
        llm.generate(
            request_prompts, sampling_params, lora_request=request_lora_mapping
        )
    else:
        llm.generate(prompts[:max_num_seqs], sampling_params)
    # llm.stop_profile()
    # torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--max-loras", type=int, default=8, help="max loras to use"
    )
    args_parser.add_argument(
        "--use-lora", action="store_true", help="whether to use lora"
    )
    args_parser.add_argument(
        "--concurrency", type=int, default=8, help="concurrency level"
    )
    args = args_parser.parse_args()
    print(f"Running with max_loras = {args.max_loras}")
    llm_profile(args)
