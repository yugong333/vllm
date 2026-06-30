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
