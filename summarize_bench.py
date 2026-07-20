#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize a run_qwen35_bench.sh run: triton vs monokernel across batch sizes.

Usage:
    python summarize_bench.py <run_dir>
    # <run_dir> is the timestamped folder that contains the per-backend
    # subdirs, e.g.
    #   result_outputs/vllm_bench/<MODEL>/<dataset>_out_<OUT>/tp_1/run_<...>

Prints, per batch size, the decode-relevant metrics (TPOT, output throughput,
TTFT) for each backend and the monokernel-vs-triton speedup. If only one
backend is present, it just prints that backend's numbers.
"""

import glob
import json
import os
import sys


def load_results(run_dir):
    """run_dir/<backend>/batch_size_<bs>/*.json -> {backend: {bs: metrics}}."""
    out = {}
    for backend in ("triton", "monokernel"):
        bdir = os.path.join(run_dir, backend)
        if not os.path.isdir(bdir):
            continue
        out[backend] = {}
        for jpath in glob.glob(os.path.join(bdir, "batch_size_*", "*.json")):
            bs = int(os.path.basename(os.path.dirname(jpath)).split("_")[-1])
            with open(jpath) as f:
                out[backend][bs] = json.load(f)
    return out


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = sys.argv[1]
    res = load_results(run_dir)
    if not res:
        print(f"No results found under {run_dir}")
        sys.exit(1)

    backends = [b for b in ("triton", "monokernel") if b in res]
    all_bs = sorted({bs for b in backends for bs in res[b]})

    print(f"\nRun: {run_dir}")
    print(f"Backends: {', '.join(backends)}\n")

    # Per-backend table.
    hdr = (
        f"  {'backend':>10s}  {'bs':>3s}  {'TPOT(ms)':>9s}  "
        f"{'out_tok/s':>10s}  {'TTFT(ms)':>9s}  {'dur(s)':>7s}  {'ok':>3s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in backends:
        for bs in all_bs:
            d = res[b].get(bs)
            if not d:
                continue
            print(
                f"  {b:>10s}  {bs:>3d}  {fmt(d['mean_tpot_ms']):>9s}  "
                f"{fmt(d['output_throughput']):>10s}  "
                f"{fmt(d['mean_ttft_ms']):>9s}  {fmt(d['duration']):>7s}  "
                f"{d['completed']:>3d}"
            )

    # Speedup table (monokernel vs triton) if both present.
    if "triton" in res and "monokernel" in res:
        print(
            f"\n  {'bs':>3s}  {'TPOT speedup':>13s}  {'throughput speedup':>18s}"
        )
        print("  " + "-" * 38)
        for bs in all_bs:
            t = res["triton"].get(bs)
            m = res["monokernel"].get(bs)
            if not t or not m:
                continue
            # TPOT: lower is better, so triton/mono > 1 means mono faster.
            tpot_su = t["mean_tpot_ms"] / m["mean_tpot_ms"]
            # throughput: higher is better, mono/triton.
            thr_su = m["output_throughput"] / t["output_throughput"]
            print(f"  {bs:>3d}  {tpot_su:>12.3f}x  {thr_su:>17.3f}x")
        print(
            "\n  (speedup > 1.0 => monokernel faster; TPOT is the decode-phase "
            "per-token latency)"
        )


if __name__ == "__main__":
    main()
