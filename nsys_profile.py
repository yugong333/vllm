import sys
sys.path.insert(0, "/home/ubuntu/vllm/vllm")

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
import nvtx


def build_prompts(sonnet_path: str, target_input_len: int, num_prompts: int) -> list[str]:
    """
    Load prompts from sonnet.txt, tiling to approximately target_input_len tokens
    (~4 chars/token). Falls back to random word generation if file not found.
    """
    target_chars = target_input_len * 4

    try:
        with open(sonnet_path, "r") as f:
            sonnet_text = f.read()
        tiled = sonnet_text * ((target_chars // len(sonnet_text)) + 2)
        base_prompt = tiled[:target_chars]
        prompts = [base_prompt] * num_prompts
        print(f"[INFO] Loaded sonnet prompts (~{target_input_len} tokens each) from {sonnet_path}")
    except FileNotFoundError:
        import random, string
        print(f"[WARN] {sonnet_path} not found — generating random prompts (~{target_input_len} tokens each)")
        words = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
            for _ in range(target_input_len * num_prompts)
        ]
        prompts = [' '.join(words[i * target_input_len:(i + 1) * target_input_len]) for i in range(num_prompts)]

    return prompts


def llm_profile(args):
    max_loras = args.max_loras
    use_lora = args.use_lora
    concurrency = args.concurrency
    model_name = args.model_name
    torch_profile = args.torch_profile
    nsys_profile = args.nsys_profile
    tp = args.tp_size
    max_num_seqs = max(concurrency, 8)

    prompts = build_prompts(args.sonnet_path, args.input_len, max_num_seqs)

    # Build profiler config for LLM constructor
    # profiler_kwargs = {}
    # if torch_profile:
    #     profiler_kwargs["profiler_config"] = {
    #         "profiler": "torch",
    #         "torch_profiler_dir": args.torch_profile_dir,
    #     }

    if use_lora:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            max_num_seqs=max_num_seqs,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=32,
            compilation_config={"compile_sizes": [1,2,4,8,16,32,64]},
            # **profiler_kwargs,
        )
        lora_requests = []
        for i in range(concurrency):
            lora_req = LoRARequest(
                lora_name=f"lora{i+1}",
                lora_int_id=i+1,
                lora_path=args.lora_path,
            )
            llm.llm_engine.add_lora(lora_req)
            lora_requests.append(lora_req)
    else:
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp,
            max_num_seqs=max_num_seqs,
            compilation_config={"compile_sizes": [1,2,4,8,16,32,64]},
            # **profiler_kwargs,
        )
        lora_requests = None

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
    batch_size = concurrency

    def lora_mapping(n):
        return [lora_requests[i % len(lora_requests)] for i in range(n)]

    # warmup
    if lora_requests:
        llm.generate(prompts[:batch_size], sampling_params, lora_request=lora_mapping(batch_size))
    else:
        llm.generate(prompts[:batch_size], sampling_params)

    llm.reset_prefix_cache()

    # profiled run
    if nsys_profile:
        assert not torch_profile
        torch.cuda.cudart().cudaProfilerStart()
    elif torch_profile:
        llm.start_profile()

    if lora_requests:
        llm.generate(prompts[:batch_size], sampling_params, lora_request=lora_mapping(batch_size))
    else:
        llm.generate(prompts[:batch_size], sampling_params)

    if nsys_profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    elif torch_profile:
        llm.stop_profile()
    # torch profiling: vLLM handles start/stop internally via profiler_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, help="model name to use")
    parser.add_argument("--max-loras", type=int, default=8, help="max loras to use")
    parser.add_argument("--use-lora", action='store_true', help="whether to use lora")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrency level")
    parser.add_argument("--tp-size", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--lora-path", type=str,
                        default="/home/ubuntu/adapter_training/pissa-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8-fp16",
                        help="path to LoRA adapter")
    parser.add_argument("--sonnet-path", type=str, default="/home/ubuntu/vllm/sonnet.txt",
                        help="path to sonnet.txt for input prompts")
    parser.add_argument("--input-len", type=int, default=1600,
                        help="approximate input token length per prompt")
    parser.add_argument("--torch-profile", action='store_true', help="use torch profiler")
    parser.add_argument("--torch-profile-dir", type=str, default="",
                        help="output directory for torch profiler traces")
    parser.add_argument("--nsys-profile", action='store_true', help="use nsys profiler")
    args = parser.parse_args()
    print(f"Running: model={args.model_name}, max_loras={args.max_loras}, "
          f"concurrency={args.concurrency}, tp={args.tp_size}, input_len={args.input_len}")
    llm_profile(args)
