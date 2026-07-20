#!/usr/bin/env python3
"""Monokernel autotuner — sweep instantiated KernelConfig variants, gate on
accuracy, rank by performance, emit the best-config JSON.

For each config_id instantiated in the C++ X-macro tables
(`MONO_CONFIGS_35B` / `MONO_CONFIGS_122B` in
csrc/moe/moe_monokernel/moe_wrapper.cu), this:

  1. ACCURACY GATE — runs the per-M CUDA-vs-Py cosine check
     (test_monokernel_accuracy.accuracy_test) at every requested M and
     rejects the config if any falls below --acc-threshold.
  2. PERF — times CUDA-graph latency vs the Triton baseline, either on the
     synthetic random sweep (perf_test) or, with --route-capture, on REAL
     captured decode routing (route_capture_perf machinery).  Real routing is
     STRONGLY preferred: the synthetic uniform sweep understates the
     monokernel (≈1.04x synthetic vs ≈1.66x real on gsm8k).
  3. RANK + EMIT — prints a ranked table and writes the winning config_id
     per (shape, M) to a JSON the serving path / MONOKERNEL_CONFIG can pin.
     Each config is tagged interleave (UCH==1, the 64-feature single-WGMMA
     up-proj that needs a Python gate/up repack ⇒ a duplicate up-weight
     tensor in GM) vs raw (UCH>=2, two-TMA pure-half up-proj, no duplicate).
     The summary reports the interleave memory overhead and, per M, the best
     config OVERALL plus the best WITH and WITHOUT interleave, so the
     perf/memory trade is explicit.  (For shapes whose interleave layout is
     not coupled-feasible — e.g. 122B — there will be no UCH==1 config until
     the decoupled up/down grid lands; the summary says so.)

Config selection rides the MONOKERNEL_CONFIG env var (resolved fresh per op
call by vllm._custom_ops), so the sweep is in-process — no rebuild between
configs (all candidates are already instantiated in the .so).

Multi-GPU sharding (Ray):
  The sweep is sharded by batch size M — each GPU owns one M at a time and
  sweeps ALL configs for it in latency isolation (timing two workloads on one
  GPU would contend and corrupt both numbers, so M is the unit of GPU
  ownership).  M values are pulled from a dynamic Ray ActorPool queue, so a
  free GPU grabs the next M (good balance when M costs differ, e.g. M=8 ≫
  M=1).  With <2 visible GPUs it falls back to an in-process sequential sweep
  — same results, no Ray overhead.  Gating semantics are unchanged from the
  single-process tuner: a config must clear the accuracy threshold at EVERY
  swept M to be ranked.

Usage:
  tune_monokernel.py --model qwen3.5_122b --batch-sizes 1 2 4 8
  tune_monokernel.py --model qwen3.5_122b --route-capture dump.pt --json best.json
  tune_monokernel.py --model qwen3.5 --configs 0 2   # only these ids
"""
import argparse
import json
import os
import re
import statistics
import sys

# Import the existing harness (weights/quant/timing/accuracy live there).
import test_monokernel_accuracy as H

# Generated shape registry (source of truth = csrc/moe/moe_monokernel/
# shapes.json via tools/gen_shapes.py).  Carries per-shape (E,N,K), op names,
# default top_k, and the full per-config knob table — so the tuner no longer
# re-parses the C++ X-macro tables.
from vllm.model_executor.layers.fused_moe import monokernel_shapes as REG


def shape_row(model_key):
    """Registry row for a tuner --model key (matches the shape key OR any
    alias, e.g. 'qwen3.5', '35b', 'qwen3.5_122b', '122b')."""
    r = REG.BY_NAME.get(model_key)
    if r is None:
        raise SystemExit(f"unknown --model {model_key!r}; choices: "
                         f"{sorted(REG.BY_NAME)}")
    return r


def config_table_for(model_key):
    """{id: dict(grid,down_col_tile,k_step_up,k_step_down,up_w_slots,
    up_col_halves)} for a shape, straight from the registry (int-keyed)."""
    return {int(cid): dict(knobs)
            for cid, knobs in shape_row(model_key)["configs"].items()}


def set_config_env(shape_key, cid):
    os.environ["MONOKERNEL_CONFIG"] = f"{shape_key}:{cid}"


def config_uch(model_cfg, knobs):
    """Replicate the C++ ``UP_COL_HALVES`` derivation for a config's knobs.

    Mirrors ``MoECoreDims::UP_COL_HALVES`` /
    ``MoEGemmSpec::UP_COL_HALVES_LOCAL`` in moe_internal.h:

        UCH = max(1, (2*N * DOWN_COL_TILE) // (128 * HIDDEN_STATES))

    where ``HIDDEN_STATES == K`` for both instantiated shapes.  UCH==1 is the
    64-feature interleaved single-WGMMA up-proj (needs the Python gate/up
    repack ⇒ a duplicate weight tensor in GM); UCH>=2 is the raw two-TMA
    pure-half up-proj (no interleave, no duplicate tensor).

    DECOUPLED shapes pin UCH explicitly (the DCT identity has no integer
    solution for them and would floor to 1 here, mislabeling the config as
    interleave); the registry carries the authoritative per-config value in
    knobs["up_col_halves"], so prefer it when present.
    """
    if knobs.get("up_col_halves"):
        return int(knobs["up_col_halves"])
    n = model_cfg["N_HALF"]
    hidden = model_cfg["K"]
    dct = knobs["down_col_tile"]
    v = (2 * n * dct) // (128 * hidden)
    return v if v > 0 else 1


def config_is_interleave(model_cfg, knobs):
    """True if this config uses the interleaved (UCH==1) up-proj layout."""
    return config_uch(model_cfg, knobs) == 1


def interleave_overhead_mb(model_cfg):
    """Extra GM footprint (MiB) of an interleave config: the kernel keeps a
    repacked DUPLICATE of the up-weight tensor [E, 2*N, K] fp8 (1 byte/elem)
    alongside the raw weights.  Raw (UCH>=2) configs carry NO such duplicate.
    """
    e, n, k = model_cfg["E"], model_cfg["N_HALF"], model_cfg["K"]
    return (e * 2 * n * k) / (2 ** 20)


# Multiple seeds per M: a single seed can miss (or single-handedly trip) an
# input-specific accuracy failure.  Gating across several seeds makes the
# verdict robust and surfaces seed-specific bugs (e.g. the M=8 seed=42 UP-row
# collapse this tuner found, which is config-independent — see task 9).
GATE_SEEDS = [42, 123, 300, 7, 80]


def perf_synthetic_M(model_cfg, M, top_k):
    """Synthetic-sweep speedup (tri_ms / cuda_ms) for the CURRENT config at one
    batch size M.  NaN if the timing produced no usable cuda latency."""
    cuda_ms, tri_ms = H.perf_test(model_cfg, M, top_k)
    return (tri_ms / cuda_ms) if (cuda_ms and cuda_ms == cuda_ms) else float("nan")


def build_route_weights(model_cfg):
    """Build + block-quantize the (shape-only) expert weights and scratchpad
    used by the route-capture perf timing.  The tensors depend solely on
    (E, N, K) — constant across M and config — so a worker builds them ONCE
    and reuses them for every (M, config) it times, rather than re-allocating
    per call (the per-config re-alloc the old all-M helper paid)."""
    import torch

    E, N_HALF, K = model_cfg["E"], model_cfg["N_HALF"], model_cfg["K"]
    torch.manual_seed(42)
    w13_f = torch.randn(E, 2 * N_HALF, K, device=H.DEV) * 0.1
    w2_f = torch.randn(E, K, N_HALF, device=H.DEV) * 0.1
    w13c, s13c = H.quant_fp8_block_wise(w13_f)
    w13c, s13c = w13c.contiguous(), s13c.contiguous()
    w2c, s2c = H.quant_fp8_block_wise(w2_f)
    w2c, s2c = w2c.contiguous(), s2c.contiguous()
    scratchpad = torch.zeros(1024, 4096, dtype=torch.float32, device=H.DEV)
    del w13_f, w2_f
    torch.cuda.empty_cache()
    return dict(w13c=w13c, s13c=s13c, w2c=w2c, s2c=s2c, scratchpad=scratchpad)


def perf_route_capture_M(model_cfg, snapshots, M, weights):
    """Geomean speedup over the captured routing snapshots WHOSE batch size is
    M, for the CURRENT config.  `weights` is a `build_route_weights` dict
    reused across calls.  Returns the geomean (or None if no snapshot at M
    produced a usable timing)."""
    import torch

    E, K = model_cfg["E"], model_cfg["K"]
    w13c, s13c = weights["w13c"], weights["s13c"]
    w2c, s2c = weights["w2c"], weights["s2c"]
    scratchpad = weights["scratchpad"]

    ratios = []
    for snap in snapshots:
        if snap["M"] != M:
            continue
        top_k = snap["top_k"]
        logits = snap["router_logits"].to(H.DEV, dtype=torch.bfloat16)
        if logits.shape != (M, E):
            continue
        kernel_op = H.get_model_op(model_cfg, M)
        x = torch.randn(M, K, device=H.DEV, dtype=torch.bfloat16)

        def cuda_fn():
            return kernel_op(x, logits, w13c, s13c, w2c, s2c, scratchpad,
                             top_k=top_k, scoring_func="softmax", renormalize=True)

        def triton_fn():
            return H._triton_e2e(x, logits, w13c, s13c, w2c, s2c, top_k)

        try:
            c = H._bench_cudagraph(cuda_fn)
            t = H._bench_cudagraph(triton_fn)
        except Exception:
            continue
        if c and c == c and t == t and c > 0:
            ratios.append(t / c)
    return statistics.geometric_mean(ratios) if ratios else None


def sweep_one_M(model_cfg, shape_key, M, top_k, config_ids, acc_threshold,
                skip_accuracy, snapshots, weights, seeds=GATE_SEEDS):
    """Sweep ALL `config_ids` for a SINGLE batch size M, in this process's GPU.

    This is the unit of GPU ownership in the Ray sharding: one M is timed in
    isolation (concurrent timing on one GPU would contend and corrupt both
    latencies).  Returns ``{cid: {"cos": worst_cos_at_M, "passed_M": bool,
    "speedup": float|None}}``; `speedup` is None when the config failed the
    accuracy gate at this M (mirroring the single-process tuner, which skipped
    perf on an accuracy reject).  The cross-M ``passed`` verdict is recombined
    by the caller from every M's ``passed_M``."""
    out = {}
    for cid in config_ids:
        set_config_env(shape_key, cid)
        if skip_accuracy:
            passed_M, cos = True, float("nan")
        else:
            cos = 1.0
            for seed in seeds:
                res = H.accuracy_test(model_cfg, M, top_k, seed=seed)
                cos = min(cos, res["cuda_py"])
            passed_M = cos >= acc_threshold
        speedup = None
        if passed_M:
            if snapshots is not None:
                speedup = perf_route_capture_M(model_cfg, snapshots, M, weights)
            else:
                speedup = perf_synthetic_M(model_cfg, M, top_k)
        out[cid] = dict(cos=cos, passed_M=passed_M, speedup=speedup)
    os.environ.pop("MONOKERNEL_CONFIG", None)
    return out


def _gpu_worker_class():
    """Build the Ray remote TunerWorker class lazily (only import ray when we
    actually shard).  Each actor reserves one GPU; Ray pins its
    CUDA_VISIBLE_DEVICES BEFORE this process touches torch, so the harness's
    hardcoded ``DEV = "cuda"`` resolves to the actor's single assigned GPU —
    no device-string surgery needed."""
    import ray

    @ray.remote(num_gpus=1)
    class TunerWorker:
        def __init__(self, model_cfg, shape_key, top_k, acc_threshold,
                     skip_accuracy, snapshots):
            # Import the harness AFTER Ray has pinned this actor's GPU.
            self.model_cfg = model_cfg
            self.shape_key = shape_key
            self.top_k = top_k
            self.acc_threshold = acc_threshold
            self.skip_accuracy = skip_accuracy
            self.snapshots = snapshots
            self.weights = (build_route_weights(model_cfg)
                            if snapshots is not None else None)

        def sweep_M(self, M, config_ids):
            res = sweep_one_M(
                self.model_cfg, self.shape_key, M, self.top_k, config_ids,
                self.acc_threshold, self.skip_accuracy, self.snapshots,
                self.weights)
            return M, res

    return TunerWorker


def run_sweep(model_cfg, shape_key, batch_sizes, top_k, ids, acc_threshold,
              skip_accuracy, snapshots):
    """Drive the full (M × config) sweep and return ``per_M_results[M] = {cid:
    {"cos","passed_M","speedup"}}``.

    Default: shard by M across one-GPU Ray actors via a dynamic ActorPool
    queue (a free GPU pulls the next M).  With <2 GPUs visible it falls back
    to an in-process sequential M loop.  Both produce identical per-M result
    dicts."""
    try:
        import torch
        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0

    if num_gpus < 2:
        print(f"# sweep: IN-PROCESS sequential (only {num_gpus} GPU visible)",
              flush=True)
        weights = (build_route_weights(model_cfg)
                   if snapshots is not None else None)
        per_M = {}
        for M in batch_sizes:
            print(f"# --- sweeping M={M} ---", flush=True)
            per_M[M] = sweep_one_M(
                model_cfg, shape_key, M, top_k, ids, acc_threshold,
                skip_accuracy, snapshots, weights)
        return per_M

    import ray
    from ray.util.actor_pool import ActorPool

    if not ray.is_initialized():
        ray.init()
    n_actors = min(num_gpus, len(batch_sizes))
    print(f"# sweep: RAY-SHARDED by M across {n_actors} GPU actor(s) "
          f"({num_gpus} GPUs visible, {len(batch_sizes)} batch sizes); "
          f"dynamic queue", flush=True)
    Worker = _gpu_worker_class()
    actors = [Worker.remote(model_cfg, shape_key, top_k, acc_threshold,
                            skip_accuracy, snapshots)
              for _ in range(n_actors)]
    pool = ActorPool(actors)
    # ActorPool dispatches each M to whichever actor is free → dynamic
    # load balancing (M=8 may cost ≫ M=1, so static round-robin would idle
    # the GPU that drew the cheap M's).
    per_M = {}
    for M, res in pool.map_unordered(
            lambda a, M: a.sweep_M.remote(M, ids), list(batch_sizes)):
        print(f"# M={M} done ({sum(1 for r in res.values() if r['passed_M'])}"
              f"/{len(res)} configs passed accuracy)", flush=True)
        per_M[M] = res
    for a in actors:
        ray.kill(a)
    return per_M


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="shape key or alias from the registry, e.g. "
                         "'qwen3.5'/'35b', 'qwen3.5_122b'/'122b'")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--top-k", type=int, default=None,
                    help="top_k to tune at (1..8). Default: the shape's "
                         "default_top_k. top_k is a runtime arg, so any value "
                         "in [1,8] works without a rebuild.")
    ap.add_argument("--configs", type=int, nargs="+", default=None,
                    help="Only sweep these config ids (default: all in the table)")
    ap.add_argument("--acc-threshold", type=float, default=0.999,
                    help="Reject a config if any per-M CUDA-vs-Py cosine < this")
    ap.add_argument("--skip-accuracy", action="store_true",
                    help="Skip the accuracy gate and rank ALL configs by perf "
                         "only.  Use for perf investigation when a known "
                         "input-specific accuracy bug (task 9) would otherwise "
                         "reject every config.  NEVER ship a config tuned this "
                         "way without a separate accuracy pass.")
    ap.add_argument("--route-capture", metavar="DUMP.pt",
                    help="Use REAL captured routing for perf (preferred over "
                         "the synthetic sweep)")
    ap.add_argument("--route-limit", type=int, default=None)
    ap.add_argument("--json", help="write best-config-per-M selection here")
    args = ap.parse_args()

    row = shape_row(args.model)
    shape_key = row["key"]
    # The accuracy/timing harness is keyed by its own MODELS dict; map via the
    # registry key or any alias the harness happens to use.
    model_cfg = H.MODELS.get(shape_key)
    if model_cfg is None:
        for a in row["aliases"]:
            if a in H.MODELS:
                model_cfg = H.MODELS[a]
                break
    if model_cfg is None:
        raise SystemExit(f"shape {shape_key!r} not in test_monokernel_accuracy."
                         f"MODELS (have {sorted(H.MODELS)}); add it there.")
    top_k = args.top_k if args.top_k is not None else row["default_top_k"]
    if not (1 <= top_k <= 8):
        raise SystemExit(f"--top-k must be in [1,8], got {top_k}")
    table = config_table_for(args.model)
    ids = sorted(args.configs) if args.configs else sorted(table)
    for cid in ids:
        if cid not in table:
            raise SystemExit(f"config_id {cid} not in shape {shape_key} "
                             f"(have {sorted(table)})")

    snapshots = None
    if args.route_capture:
        snaps, _meta = H.load_route_capture(args.route_capture)
        if args.route_limit:
            snaps = snaps[: args.route_limit]
        snapshots = snaps
        print(f"# perf source: REAL captured routing "
              f"({len(snapshots)} snapshots) from {args.route_capture}")
    else:
        print("# perf source: SYNTHETIC random sweep "
              "(understates the monokernel; prefer --route-capture)")

    print(f"# model={args.model} shape={shape_key} top_k={top_k} "
          f"acc_threshold={args.acc_threshold}")
    print(f"# sweeping config ids {ids} from {row['config_macro']}\n")

    # Run the (M × config) sweep — Ray-sharded by M across GPUs, or in-process.
    # Returns per_M[M] = {cid: {"cos","passed_M","speedup"}}.
    per_M_results = run_sweep(
        model_cfg, shape_key, args.batch_sizes, top_k, ids, args.acc_threshold,
        args.skip_accuracy, snapshots)

    # ── Recombine per-M shards into the per-config result rows the summary
    #    consumes: (cid, knobs, passed, worst_cos, perf_by_M).  Gating is
    #    cross-M (a config must clear the threshold at EVERY swept M), exactly
    #    as the single-process tuner did; a config that fails any M is marked
    #    rejected with an empty perf map and dropped from the ranking.
    results = []
    for cid in ids:
        knobs = table[cid]
        uch = config_uch(model_cfg, knobs)
        layout = "interleave(UCH=1)" if uch == 1 else f"raw(UCH={uch})"
        ktag = (f"grid={knobs['grid']} DCT={knobs['down_col_tile']} "
                f"KUP={knobs['k_step_up']} KDN={knobs['k_step_down']} "
                f"SLOTS={knobs['up_w_slots']} {layout}")
        per_cid = {M: per_M_results[M][cid] for M in args.batch_sizes
                   if M in per_M_results and cid in per_M_results[M]}
        worst = min((r["cos"] for r in per_cid.values()), default=float("nan"))
        passed = bool(per_cid) and all(r["passed_M"] for r in per_cid.values())
        print(f"=== config {cid}: {ktag} ===")
        if uch == 1:
            print(f"  up-proj layout: INTERLEAVE — "
                  f"+{interleave_overhead_mb(model_cfg):.0f} MB duplicate "
                  f"up-weight tensor")
        else:
            print("  up-proj layout: RAW two-TMA — no duplicate weight tensor")
        if args.skip_accuracy:
            print("  accuracy gate SKIPPED (--skip-accuracy)")
        elif not passed:
            print(f"  ACCURACY FAIL (worst CUDA-vs-Py cos={worst:.6f} "
                  f"< {args.acc_threshold}) — rejected\n", flush=True)
            results.append((cid, knobs, False, worst, {}))
            continue
        else:
            print(f"  accuracy OK (worst cos={worst:.6f})")
        # Perf map: only M whose timing produced a usable speedup.
        perf = {M: r["speedup"] for M, r in per_cid.items()
                if r["speedup"] is not None and r["speedup"] == r["speedup"]}
        for M in sorted(perf):
            print(f"    M={M}: speedup vs Triton = {perf[M]:.3f}x")
        print(flush=True)
        results.append((cid, knobs, True, worst, perf))

    # ── Ranked summary + best-per-M selection ───────────────────────────────
    print("\n" + "#" * 72)
    print(f"# TUNING SUMMARY — {model_cfg['display_name']} "
          f"({'real routing' if snapshots is not None else 'synthetic'})")
    print("#" * 72)
    all_M = sorted({M for *_x, perf in results for M in perf})
    overhead_mb = interleave_overhead_mb(model_cfg)
    have_interleave = any(config_is_interleave(model_cfg, table[cid])
                          for cid, _k, ok, _w, _p in results if ok)
    have_raw = any(not config_is_interleave(model_cfg, table[cid])
                   for cid, _k, ok, _w, _p in results if ok)
    print(f"# interleave duplicate up-weight tensor overhead: "
          f"+{overhead_mb:.0f} MB (paid only by UCH==1 configs)")
    if not have_interleave:
        print("# NOTE: no interleave (UCH==1) config in this sweep — for "
              "this shape\n#       the interleave layout may need the "
              "decoupled up/down grid (task 10).")
    if not have_raw:
        print("# NOTE: no raw (UCH>=2) config in this sweep.")
    print("#" * 72)

    def best_in(M, pred):
        """Highest-speedup (cid, speedup) among accuracy-passing configs at M
        whose interleave-ness satisfies ``pred(is_interleave)``."""
        ranked = sorted(
            [(cid, perf[M]) for cid, _k, ok, _w, perf in results
             if ok and M in perf
             and pred(config_is_interleave(model_cfg, table[cid]))],
            key=lambda t: -t[1])
        return ranked[0] if ranked else None

    def entry(M, pick):
        cid, sp = pick
        return dict(config_id=cid, speedup=sp, knobs=table[cid],
                    interleave=config_is_interleave(model_cfg, table[cid]),
                    up_col_halves=config_uch(model_cfg, table[cid]))

    best_per_M = {}            # best OVERALL (any layout)
    best_interleave_per_M = {}  # best among UCH==1 configs
    best_no_interleave_per_M = {}  # best among UCH>=2 configs
    for M in all_M:
        ranked = sorted(
            [(cid, perf[M]) for cid, _k, ok, _w, perf in results
             if ok and M in perf],
            key=lambda t: -t[1])
        if not ranked:
            continue
        best_cid, best_sp = ranked[0]
        best_per_M[M] = entry(M, (best_cid, best_sp))
        bi = best_in(M, lambda il: il)
        bn = best_in(M, lambda il: not il)
        if bi:
            best_interleave_per_M[M] = entry(M, bi)
        if bn:
            best_no_interleave_per_M[M] = entry(M, bn)
        order = "  ".join(
            f"cfg{c}{'*' if config_is_interleave(model_cfg, table[c]) else ''}"
            f"={s:.3f}x" for c, s in ranked)
        # '*' marks interleave configs in the ranked order.
        line = f"  M={M}: BEST cfg{best_cid} ({best_sp:.3f}x) | {order}"
        if bi and bn:
            # Both layouts present → show the interleave perf premium and
            # whether it is worth the +overhead_mb memory.
            ic, isp = bi
            nc, nsp = bn
            delta = (isp / nsp - 1.0) * 100.0
            line += (f"\n         interleave best=cfg{ic}({isp:.3f}x) "
                     f"vs no-interleave best=cfg{nc}({nsp:.3f}x) "
                     f"→ interleave {delta:+.1f}% for +{overhead_mb:.0f} MB")
        print(line)

    if args.json and best_per_M:
        out = dict(model=args.model, shape=shape_key, top_k=top_k,
                   acc_threshold=args.acc_threshold,
                   perf_source=("route_capture" if snapshots is not None
                                else "synthetic"),
                   interleave_overhead_mb=round(overhead_mb, 1),
                   best_per_M={str(M): v for M, v in best_per_M.items()},
                   best_interleave_per_M={
                       str(M): v for M, v in best_interleave_per_M.items()},
                   best_no_interleave_per_M={
                       str(M): v for M, v in best_no_interleave_per_M.items()},
                   config_table={
                       cid: dict(k, interleave=config_is_interleave(model_cfg, k),
                                 up_col_halves=config_uch(model_cfg, k))
                       for cid, k in table.items()})
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n# wrote best-config selection -> {args.json}")

    # Clean up env so a follow-on import doesn't inherit a pinned config.
    os.environ.pop("MONOKERNEL_CONFIG", None)


if __name__ == "__main__":
    main()
