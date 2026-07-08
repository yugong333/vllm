#!/usr/bin/env python3
"""Onboard a new MoE model onto the monokernel, end to end.

Given an HF model id (or local model path), this pipeline:

  1. inspect    — read config.json (+ safetensors metadata), extract the MoE
                  geometry: E, moe_intermediate_size (N_half), hidden_size (K),
                  top_k, scoring/bias/scaling routing config, quantization.
  2. tp         — decide the TP size on the target GPU (H200 by default):
                  smallest power-of-two TP such that weights + KV/activation
                  headroom fit, subject to divisibility (KV heads, N_half
                  blocks) — overridable with --tp.
  3. shape      — derive the PER-GPU monokernel shape (N_half/TP, K) and
                  validate it against the kernel constraints (E%32, N%128,
                  K%128, FP8 block-wise 128x128 quant, top_k<=8).
  4. configs    — enumerate feasible KernelConfig candidates with
                  tools/enum_configs.py, pick a small diverse candidate set,
                  and append the new shape to shapes.json (config 0 = the
                  heuristic default).
  5. build      — regenerate (tools/gen_shapes.py) + incremental cmake build.
  6. tune       — run tune_monokernel.py over the candidates (accuracy gate +
                  perf rank) and write best_config.json.
  7. pin        — print/emit the MONOKERNEL_CONFIG selection for serving.
  8. bench      — end-to-end `vllm serve` benchmark, triton vs monokernel,
                  via run_moe_bench.sh (BACKEND=both).

Each step is resumable/skippable: --steps inspect,tp,shape (comma list),
--dry-run stops after printing what WOULD change (no file edits, no build).

Usage:
  python onboard_model.py Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
  python onboard_model.py zai-org/GLM-5.2-FP8 --tp 8
  python onboard_model.py /models/foo --steps inspect,tp,shape --dry-run
  python onboard_model.py <model> --steps tune,pin,bench   # after a build
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
SHAPES_JSON = os.path.join(REPO, "csrc", "moe", "moe_monokernel", "shapes.json")
TOOLS = os.path.join(REPO, "csrc", "moe", "moe_monokernel", "tools")
PYTHON = sys.executable

ALL_STEPS = ["inspect", "tp", "shape", "configs", "build", "tune", "pin", "bench"]

# ── GPU targets ─────────────────────────────────────────────────────────────
GPUS = {
    "h200": dict(mem_gib=141, sms=132),
    "h100": dict(mem_gib=80, sms=132),
}
# Fraction of GPU memory the weights may occupy; the rest is KV cache,
# activations, CUDA graphs, fragmentation.  Matches practical vLLM serving
# headroom at --gpu-memory-utilization 0.92.
WEIGHT_MEM_FRACTION = 0.75


def sh(cmd, **kw):
    print(f"[onboard] $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


# ═════════════════════════════════════════════════════════════════════════
# Step 1 — inspect
# ═════════════════════════════════════════════════════════════════════════

def _load_json_from_model(model, filename):
    """config.json / index.json from a local path or the HF hub."""
    local = os.path.join(model, filename)
    if os.path.isdir(model):
        if os.path.exists(local):
            with open(local) as f:
                return json.load(f)
        return None
    import urllib.request
    url = f"https://huggingface.co/{model}/resolve/main/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def _hf_api_info(model):
    if os.path.isdir(model):
        return None
    import urllib.request
    try:
        url = f"https://huggingface.co/api/models/{model}"
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except Exception:
        return None


def _local_weight_bytes(model):
    total = 0
    for root, _dirs, files in os.walk(model):
        for f in files:
            if f.endswith((".safetensors", ".bin")):
                total += os.path.getsize(os.path.join(root, f))
    return total or None


def _first(cfg, *keys, default=None):
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return v
    return default


def inspect_model(model):
    """Extract the MoE + quant geometry from the model config."""
    cfg = _load_json_from_model(model, "config.json")
    if cfg is None:
        raise SystemExit(f"[onboard] cannot read config.json for {model!r} "
                         f"(bad HF id / path, or no network?)")
    # Some models nest the LLM config (e.g. multimodal wrappers).
    for sub in ("text_config", "llm_config", "language_config"):
        if sub in cfg and isinstance(cfg[sub], dict) and \
                "hidden_size" in cfg[sub]:
            cfg = {**cfg, **cfg[sub]}

    info = dict(
        model=model,
        model_type=cfg.get("model_type"),
        E=_first(cfg, "n_routed_experts", "num_experts", "num_local_experts"),
        N_half_full=_first(cfg, "moe_intermediate_size"),
        K=_first(cfg, "hidden_size"),
        top_k=_first(cfg, "num_experts_per_tok", "moe_top_k", "moe_topk",
                     default=8),
        num_kv_heads=_first(cfg, "num_key_value_heads",
                            "num_attention_heads"),
        num_layers=_first(cfg, "num_hidden_layers"),
        n_shared_experts=_first(cfg, "n_shared_experts",
                                "num_shared_experts", default=0),
        norm_topk_prob=cfg.get("norm_topk_prob"),
        scoring_func=cfg.get("scoring_func"),           # deepseek/glm style
        topk_method=cfg.get("topk_method"),
        routed_scaling_factor=cfg.get("routed_scaling_factor"),
        use_grouped_topk=bool(cfg.get("n_group") and cfg.get("n_group") > 1),
    )

    # Quantization method + block size.
    q = cfg.get("quantization_config") or {}
    info["quant_method"] = q.get("quant_method")
    info["weight_block_size"] = q.get("weight_block_size")
    if info["quant_method"] == "compressed-tensors":
        # compressed-tensors: dig the weight scheme out of config_groups.
        groups = q.get("config_groups") or {}
        for g in groups.values():
            w = (g or {}).get("weights") or {}
            if w:
                info["weight_block_size"] = w.get("block_structure")
                info["ct_weight_bits"] = w.get("num_bits")
                info["ct_weight_type"] = w.get("type")
                info["ct_weight_strategy"] = w.get("strategy")
                break

    # Weight bytes: HF API safetensors param counts (bytes ~= fp8 params * 1
    # + bf16 params * 2), else the index total_size, else local file sizes.
    wb = None
    api = _hf_api_info(model)
    if api and api.get("safetensors"):
        params = api["safetensors"].get("parameters") or {}
        BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
                 "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}
        wb = sum(n * BYTES.get(dt, 2) for dt, n in params.items())
    if not wb:
        idx = _load_json_from_model(model, "model.safetensors.index.json")
        if idx and idx.get("metadata", {}).get("total_size"):
            wb = idx["metadata"]["total_size"]
    if not wb and os.path.isdir(model):
        wb = _local_weight_bytes(model)
    info["weight_bytes"] = wb

    missing = [k for k in ("E", "N_half_full", "K") if not info[k]]
    if missing:
        raise SystemExit(f"[onboard] config.json is missing MoE fields "
                         f"{missing} — is this an MoE model? cfg keys: "
                         f"{sorted(cfg)[:40]}")
    return info


# ═════════════════════════════════════════════════════════════════════════
# Step 2 — TP decision
# ═════════════════════════════════════════════════════════════════════════

def decide_tp(info, gpu, forced_tp=None, max_tp=8):
    """Smallest power-of-two TP such that per-GPU weights fit in
    WEIGHT_MEM_FRACTION of the GPU, subject to divisibility:
      * num_kv_heads % TP == 0 (vLLM attention sharding), and
      * the sharded N_half stays a multiple of 128 (FP8 block + kernel atom).
    """
    mem_bytes = GPUS[gpu]["mem_gib"] * (1 << 30)
    budget = mem_bytes * WEIGHT_MEM_FRACTION
    wb = info["weight_bytes"]
    kv = info["num_kv_heads"] or max_tp
    n_half = info["N_half_full"]

    candidates = [t for t in (1, 2, 4, 8, 16) if t <= max_tp]

    def valid(t):
        return kv % t == 0 and (n_half // t) % 128 == 0 and n_half % t == 0

    if forced_tp is not None:
        if not valid(forced_tp):
            print(f"[onboard] WARNING: --tp {forced_tp} violates divisibility "
                  f"(kv_heads={kv}, N_half={n_half}); proceeding anyway.")
        return forced_tp, "forced by --tp"

    if wb is None:
        print("[onboard] WARNING: could not determine weight size; "
              "defaulting TP=1 (override with --tp).")
        return 1, "unknown weight size"

    for t in candidates:
        if not valid(t):
            continue
        if wb / t <= budget:
            reason = (f"weights {wb / 2**30:.0f} GiB / TP{t} = "
                      f"{wb / t / 2**30:.0f} GiB <= "
                      f"{budget / 2**30:.0f} GiB budget "
                      f"({WEIGHT_MEM_FRACTION:.0%} of {gpu.upper()} "
                      f"{GPUS[gpu]['mem_gib']} GiB)")
            return t, reason
    raise SystemExit(f"[onboard] no valid TP <= {max_tp} fits "
                     f"{wb / 2**30:.0f} GiB on {gpu.upper()} "
                     f"(kv_heads={kv}, N_half={n_half}).")


# ═════════════════════════════════════════════════════════════════════════
# Step 3 — per-GPU shape + kernel validation
# ═════════════════════════════════════════════════════════════════════════

def derive_shape(info, tp):
    n_half = info["N_half_full"] // tp
    shape = dict(E=info["E"], N=n_half, K=info["K"],
                 top_k=min(int(info["top_k"]), 8))

    problems = []
    if info["E"] % 32:
        problems.append(f"E={info['E']} not a multiple of 32")
    if n_half % 128:
        problems.append(f"sharded N_half={n_half} not a multiple of 128")
    if info["K"] % 128:
        problems.append(f"K={info['K']} not a multiple of 128")
    if int(info["top_k"]) > 8:
        problems.append(f"top_k={info['top_k']} > 8 (kernel MAX_TOPK)")
    if info["E"] * 8 > 64 * 8:  # informational only; pairs cap is BS*top_k<=64
        pass
    blk = info.get("weight_block_size")
    if blk and list(blk) != [128, 128] and blk != "128x128":
        problems.append(f"weight_block_size={blk} != [128, 128] "
                        f"(kernel assumes 128x128 block-wise FP8)")
    qm = info.get("quant_method")
    if qm == "fp8" and not blk:
        problems.append("fp8 per-tensor quant (no weight_block_size) — the "
                        "kernel needs BLOCK-wise 128x128 scales")
    if qm == "compressed-tensors":
        if info.get("ct_weight_type") != "float" or \
                info.get("ct_weight_bits") != 8:
            problems.append(f"compressed-tensors weights are not fp8 "
                            f"(bits={info.get('ct_weight_bits')}, "
                            f"type={info.get('ct_weight_type')})")
        if info.get("ct_weight_strategy") not in ("block",) and \
                info.get("weight_block_size") not in ([128, 128], "128x128"):
            problems.append(f"compressed-tensors weight strategy="
                            f"{info.get('ct_weight_strategy')} block="
                            f"{info.get('weight_block_size')} — need 128x128 "
                            f"block scales")
    if info.get("use_grouped_topk"):
        problems.append("model uses grouped top-k (n_group > 1) — the "
                        "monokernel routes over ALL experts (biased top-k "
                        "only); the fp8.py eligibility gate will reject it")

    shape["problems"] = problems
    # Routing metadata for shapes.json (only recorded when non-default).
    routing = {}
    if info.get("scoring_func") == "sigmoid" or info.get("topk_method"):
        routing["scoring_func"] = "sigmoid"
    if info.get("topk_method") == "noaux_tc":
        routing["use_expert_bias"] = True
    rsf = info.get("routed_scaling_factor")
    if rsf and float(rsf) != 1.0:
        routing["routed_scaling_factor"] = float(rsf)
    shape["routing"] = routing
    return shape


# ═════════════════════════════════════════════════════════════════════════
# Step 4 — candidate configs + shapes.json update
# ═════════════════════════════════════════════════════════════════════════

def existing_shape_row(E, N, K):
    with open(SHAPES_JSON) as f:
        d = json.load(f)
    for s in d["shapes"]:
        if (s["E"], s["N"], s["K"]) == (E, N, K):
            return s
    return None


def enumerate_candidates(E, N, K, verify=False):
    """Run tools/enum_configs.py and return the feasible config list."""
    out_json = f"/tmp/onboard_enum_E{E}_N{N}_K{K}.json"
    cmd = [PYTHON, os.path.join(TOOLS, "enum_configs.py"),
           "--N", str(N), "--K", str(K), "--E", str(E),
           "--mode", "both", "--json", out_json]
    if verify:
        cmd.append("--verify")
    sh(cmd)
    with open(out_json) as f:
        return json.load(f)["configs"]


def pick_candidates(cands, sms, max_configs=6):
    """Down-select a diverse, high-utilization candidate set.

    Heuristics distilled from the 35B/122B/GLM 5.2 tunings:
      * prefer the largest grid (highest SM utilization; grid ≈ SMs);
      * prefer coupled over decoupled at equal grid (cheaper barrier);
      * KDN=128 first (activation slab stays 1 KB), KUP∈{256,128};
      * SLOTS 2 and 4; dedupe by (dct, kup, kdn, slots, uch, grid).
    Config 0 (the default) = the best-ranked candidate.
    """
    def rank(c):
        coupled = 0 if c["mode"] == "coupled" else 1
        # Decoupled + interleaved (UCH=1) has never been exercised; the
        # validated decoupled path is raw UCH=2 (GLM 5.2), so prefer it.
        unproven = 1 if (c["mode"] == "decoupled"
                         and c["up_col_halves"] != 2) else 0
        return (
            -c["grid"],                     # more SMs used first
            coupled,                        # coupled preferred
            unproven,                       # validated layout combos first
            abs(c["k_step_up"] - 256),      # KUP=256 sweet spot
            c["k_step_down"] != 128,        # KDN=128 preferred
            c["up_w_slots"] != 2,           # SLOTS=2 preferred (less SHM)
            c["down_col_tile"] != 384,      # mid DCT preferred
        )

    ranked = sorted(cands, key=rank)
    # A shape is either coupled or decoupled-with-one-pinned-UCH (the
    # DimsTunable knobs can't mix modes in one shape); lock the mode to the
    # best-ranked candidate's BEFORE down-selecting so the shape gets the
    # full candidate budget.
    best = ranked[0]
    if best["mode"] == "coupled":
        ranked = [c for c in ranked if c["mode"] == "coupled"]
    else:
        ranked = [c for c in ranked
                  if c["mode"] == "decoupled"
                  and c["up_col_halves"] == best["up_col_halves"]]
    # Two passes for tuning diversity: first the best config of each
    # distinct carve (grid, DCT), then fill remaining slots with the
    # next-best K-step/slot variants overall.
    seen, picked, carves = set(), [], set()

    def take(c):
        key = (c["grid"], c["down_col_tile"], c["k_step_up"],
               c["k_step_down"], c["up_w_slots"])
        if key in seen:
            return
        seen.add(key)
        picked.append(c)

    for c in ranked:
        carve = (c["down_col_tile"], c["up_col_halves"])
        if carve in carves:
            continue
        carves.add(carve)
        take(c)
        if len(picked) >= max_configs:
            return picked
    for c in ranked:
        take(c)
        if len(picked) >= max_configs:
            break
    return picked


def add_shape_to_json(model_tag, shape, cands, dry_run):
    """Append the new shape (with its candidate configs) to shapes.json."""
    E, N, K = shape["E"], shape["N"], shape["K"]
    key = f"e{E}_n{N}_k{K}"
    with open(SHAPES_JSON) as f:
        raw = f.read()
    d = json.loads(raw)

    entry = {
        "_comment": f"Onboarded via onboard_model.py for {model_tag}. "
                    f"Candidate configs from enum_configs; config 0 is the "
                    f"heuristic default — retune with tune_monokernel.py.",
        "key": key,
        "aliases": [model_tag] if model_tag != key else [],
        "display_name": f"{model_tag} E{E} N_half{N} K{K} block-wise FP8",
        "E": E, "N": N, "K": K,
        "default_top_k": shape["top_k"],
    }
    # Decoupled shapes pin UCH on the base Dims (pick_candidates already
    # locked all candidates to one mode/UCH).
    if cands[0]["mode"] == "decoupled":
        entry["up_col_halves"] = cands[0]["up_col_halves"]
    entry.update(shape.get("routing", {}))
    entry["configs"] = [
        dict(id=i, grid=c["grid"], dct=c["down_col_tile"],
             kup=c["k_step_up"], kdn=c["k_step_down"], slots=c["up_w_slots"],
             note=("heuristic default (untuned)" if i == 0 else
                   f"{c['mode']} candidate, SHM~{c['shm_est'] // 1024}KB"))
        for i, c in enumerate(cands)
    ]

    if dry_run:
        print("[onboard] DRY RUN — would append to shapes.json:")
        print(json.dumps(entry, indent=2))
        return key, entry

    # Surgical text edit: insert before the final "  ]\n}" so the existing
    # entries' formatting is untouched.  Configs render one per line to
    # match the file's compact style.
    cfg_lines = ",\n        ".join(
        json.dumps(c, separators=(", ", ": ")) for c in entry["configs"])
    head = {k: v for k, v in entry.items() if k != "configs"}
    head_body = json.dumps(head, indent=6, separators=(",", ": "))
    head_body = head_body[1:-1].rstrip()  # strip outer braces
    block = ("    {" + head_body + ",\n"
             '      "configs": [\n        ' + cfg_lines + "\n      ]\n    }")
    close = raw.rindex("\n  ]\n}")
    new_raw = raw[:close] + ",\n" + block + raw[close:]
    json.loads(new_raw)  # validate before writing
    with open(SHAPES_JSON, "w") as f:
        f.write(new_raw)
    print(f"[onboard] appended shape '{key}' ({len(cands)} configs) "
          f"to shapes.json")
    return key, entry


# ═════════════════════════════════════════════════════════════════════════
# Steps 5–8 — build / tune / pin / bench
# ═════════════════════════════════════════════════════════════════════════

def regen_and_build():
    sh([PYTHON, os.path.join(TOOLS, "gen_shapes.py")])
    env = os.environ.copy()
    # cmake >= 3.26 must be found; keep the caller's PATH (per workspace
    # setup the right cmake is already first on PATH when building).
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("[onboard] cmake not on PATH")
    # /opt/vllm-venv symlink is required by the preset and gets wiped.
    if not os.path.exists("/opt/vllm-venv"):
        try:
            os.symlink(os.path.join(REPO, "venv_vllm"), "/opt/vllm-venv")
        except OSError as e:
            print(f"[onboard] WARNING: cannot restore /opt/vllm-venv: {e}")
    sh(["cmake", "--build", "--preset", "release", "--target", "install"],
       cwd=REPO, env=env)


def run_tune(shape_key, batch_sizes, out_json, route_capture=None):
    cmd = [PYTHON, os.path.join(REPO, "tune_monokernel.py"),
           "--model", shape_key,
           "--batch-sizes"] + [str(b) for b in batch_sizes] + [
           "--json", out_json]
    if route_capture:
        cmd += ["--route-capture", route_capture]
    sh(cmd)


def pin_from_tuning(out_json, shape_key):
    """Pick one config id for serving: the id that wins the most Ms
    (ties → higher speedup at the largest M)."""
    with open(out_json) as f:
        best = json.load(f)
    per_m = best.get("best_per_M") or {}
    if not per_m:
        print("[onboard] tuning produced no passing config; keeping default 0")
        return 0
    from collections import Counter
    votes = Counter(v["config_id"] for v in per_m.values())
    top = votes.most_common()
    best_ids = [cid for cid, n in top if n == top[0][1]]
    if len(best_ids) > 1:
        largest_m = str(max(int(m) for m in per_m))
        pref = per_m[largest_m]["config_id"]
        cid = pref if pref in best_ids else best_ids[0]
    else:
        cid = best_ids[0]
    print(f"[onboard] pinned config: MONOKERNEL_CONFIG={shape_key}:{cid}")
    for m, v in sorted(per_m.items(), key=lambda kv: int(kv[0])):
        print(f"  M={m}: best cfg{v['config_id']} {v['speedup']:.3f}x")
    return cid


def run_bench(model, tp, shape_key, cid, dataset, backend="both",
              serve_extra=""):
    env = os.environ.copy()
    env["MODEL"] = model
    env["TP"] = str(tp)
    if cid is not None and cid >= 0:
        env["MONOKERNEL_CONFIG"] = f"{shape_key}:{cid}"
    if serve_extra:
        env["SERVE_EXTRA_ARGS"] = serve_extra
    sh(["bash", os.path.join(REPO, "run_moe_bench.sh"), backend, dataset],
       env=env)


# ═════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="HF model id or local model path")
    ap.add_argument("--gpu", choices=sorted(GPUS), default="h200")
    ap.add_argument("--tp", type=int, default=None,
                    help="force the TP size instead of auto-deciding")
    ap.add_argument("--max-tp", type=int, default=8)
    ap.add_argument("--steps", default=",".join(ALL_STEPS),
                    help=f"comma list from {ALL_STEPS}")
    ap.add_argument("--dry-run", action="store_true",
                    help="stop before editing files / building / running")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-configs", type=int, default=6,
                    help="candidate configs to instantiate for tuning")
    ap.add_argument("--verify-shm", action="store_true",
                    help="nvcc-verify exact SHM during enumeration (slow)")
    ap.add_argument("--route-capture", metavar="DUMP.pt", default=None,
                    help="tune on captured real routing instead of synthetic")
    ap.add_argument("--dataset", default="sonnet",
                    choices=["sonnet", "gsm8k", "sharegpt"])
    ap.add_argument("--serve-extra-args", default="",
                    help="extra `vllm serve` args for the e2e bench")
    ap.add_argument("--model-tag", default=None,
                    help="short alias for the shape (default: from model name)")
    args = ap.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        raise SystemExit(f"unknown steps {bad}; choose from {ALL_STEPS}")

    tag = args.model_tag or re.sub(
        r"[^a-z0-9]+", "_",
        os.path.basename(args.model.rstrip("/")).lower()).strip("_")

    state_path = f"/tmp/onboard_{tag}.json"
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"[onboard] resuming from {state_path}")

    def save_state():
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    # ── 1. inspect ─────────────────────────────────────────────────────
    if "inspect" in steps or "info" not in state:
        info = inspect_model(args.model)
        state["info"] = info
        save_state()
        print("[onboard] model geometry:")
        for k in ("model_type", "E", "N_half_full", "K", "top_k",
                  "num_kv_heads", "num_layers", "quant_method",
                  "weight_block_size", "scoring_func", "topk_method",
                  "routed_scaling_factor"):
            print(f"    {k:<24} = {info.get(k)}")
        wb = info.get("weight_bytes")
        print(f"    {'weight_bytes':<24} = "
              f"{wb / 2**30:.1f} GiB" if wb else
              "    weight_bytes             = <unknown>")
    info = state["info"]

    # ── 2. TP ──────────────────────────────────────────────────────────
    if "tp" in steps or "tp" not in state:
        tp, reason = decide_tp(info, args.gpu, args.tp, args.max_tp)
        state["tp"] = tp
        save_state()
        print(f"[onboard] TP on {args.gpu.upper()}: {tp}  ({reason})")
    tp = state["tp"]

    # ── 3. shape ───────────────────────────────────────────────────────
    if "shape" in steps or "shape" not in state:
        shape = derive_shape(info, tp)
        state["shape"] = shape
        save_state()
        print(f"[onboard] per-GPU monokernel shape: E={shape['E']} "
              f"N_half={shape['N']} K={shape['K']} top_k={shape['top_k']} "
              f"routing={shape['routing'] or 'softmax+renorm (default)'}")
        if shape["problems"]:
            print("[onboard] BLOCKERS:")
            for p in shape["problems"]:
                print(f"    ✗ {p}")
            raise SystemExit("[onboard] shape is not monokernel-compatible; "
                             "fix the blockers (or pick another TP with --tp) "
                             "and rerun.")
        print("[onboard] shape is monokernel-compatible ✓")
    shape = state["shape"]
    E, N, K = shape["E"], shape["N"], shape["K"]

    row = existing_shape_row(E, N, K)
    if row is not None:
        state["shape_key"] = row["key"]
        save_state()
        print(f"[onboard] shape already in shapes.json as "
              f"'{row['key']}' — skipping configs step "
              f"({len(row['configs'])} configs present).")

    # ── 4. configs ─────────────────────────────────────────────────────
    if "configs" in steps and row is None:
        cands = enumerate_candidates(E, N, K, verify=args.verify_shm)
        if not cands:
            raise SystemExit(
                "[onboard] enum_configs found no feasible KernelConfig for "
                f"this shape (N_half={N}, K={K}).\n"
                "  The kernel needs an up-grid carve with UP_COL_HALVES <= 2 "
                "(2N/128 must have a divisor yielding a 128- or 256-row "
                "up-tile)\n  and a down carve with DOWN_COL_TILE <= 512 "
                "(K/DCT integer), with grid <= SM count.\n"
                "  Options: pick a different TP (--tp) so the sharded N_half "
                "changes, or extend the kernel (UCH>2 / DCT>384 support).")
        sms = GPUS[args.gpu]["sms"]
        picked = pick_candidates(cands, sms, args.max_configs)
        print(f"[onboard] {len(cands)} feasible configs; instantiating "
              f"{len(picked)} candidates:")
        for i, c in enumerate(picked):
            print(f"    cfg{i}: {c['mode']} grid={c['grid']} "
                  f"DCT={c['down_col_tile']} KUP={c['k_step_up']} "
                  f"KDN={c['k_step_down']} SLOTS={c['up_w_slots']} "
                  f"UCH={c['up_col_halves']} SHM~{c['shm_est'] // 1024}KB")
        key, _entry = add_shape_to_json(tag, shape, picked, args.dry_run)
        state["shape_key"] = key
        save_state()
    shape_key = state.get("shape_key")

    if args.dry_run:
        print("[onboard] DRY RUN — stopping before build/tune/bench.")
        return

    # ── 5. build ───────────────────────────────────────────────────────
    if "build" in steps:
        regen_and_build()

    # ── 6. tune ────────────────────────────────────────────────────────
    tune_json = f"/tmp/onboard_{tag}_best.json"
    if "tune" in steps:
        if shape_key is None:
            raise SystemExit("[onboard] no shape key (run the configs step)")
        run_tune(shape_key, args.batch_sizes, tune_json, args.route_capture)
        state["tune_json"] = tune_json
        save_state()

    # ── 7. pin ─────────────────────────────────────────────────────────
    cid = state.get("pinned_config")
    if "pin" in steps:
        tj = state.get("tune_json", tune_json)
        if os.path.exists(tj):
            cid = pin_from_tuning(tj, shape_key)
        else:
            print("[onboard] no tuning result; serving with default config 0")
            cid = 0
        state["pinned_config"] = cid
        save_state()

    # ── 8. bench ───────────────────────────────────────────────────────
    if "bench" in steps:
        run_bench(args.model, tp, shape_key, cid, args.dataset,
                  serve_extra=args.serve_extra_args)

    print(f"[onboard] DONE.  Serving recipe:")
    print(f"    VLLM_USE_MOE_MONOKERNEL=1 \\")
    if cid is not None and shape_key:
        print(f"    MONOKERNEL_CONFIG={shape_key}:{cid} \\")
    print(f"    vllm serve {args.model} --tensor-parallel-size {tp} ...")


if __name__ == "__main__":
    main()
