---
name: monokernel-onboard
description: Onboard a new MoE model (HF id or local path) onto the MoE monokernel end to end — decide TP on H200, derive the per-GPU kernel shape, add + tune KernelConfigs, and run the triton-vs-monokernel e2e serve benchmark. Use when the user gives a model link/path and wants it running on the monokernel, wants the TP/shape feasibility checked, or wants the full tune+benchmark pipeline.
---

# Monokernel model onboarding

The pipeline is automated by `onboard_model.py` (repo root).  It drives:
inspect → tp → shape → configs → build → tune → pin → bench.

```bash
source venv_vllm/bin/activate
# cmake >= 3.26 first on PATH (needed by the build step; see workspace CLAUDE.md)
export PATH="/fsx/export/workspaces/ygkyle/.cache/uv/archive-v0/nE_A-UMAhcWw_e9n/cmake/data/bin:$PATH"

python onboard_model.py <HF-id-or-path>            # full pipeline
python onboard_model.py <model> --dry-run          # feasibility check only
python onboard_model.py <model> --tp 8             # force TP
python onboard_model.py <model> --steps tune,pin,bench   # resume later steps
```

State persists in `/tmp/onboard_<tag>.json`, so steps are resumable; the
tuning result lands in `/tmp/onboard_<tag>_best.json`.

## Recommended flow (agent-driven)

1. **Feasibility first, always** — run with
   `--steps inspect,tp,shape,configs --dry-run` and show the user:
   - the extracted geometry (E, N_half, K, top_k, quant, routing),
   - the TP decision + reason (weights vs 75% of H200 141 GiB, KV-head and
     N_half%128 divisibility),
   - blockers if any (E%32, N%128, K%128, non-128×128-block FP8, top_k>8,
     grouped top-k),
   - the candidate KernelConfigs it would add to shapes.json.
   If the shape already exists in shapes.json the configs step is skipped —
   the model can reuse the existing tuned configs directly.

2. **Confirm with the user before mutating** — the non-dry run appends to
   `csrc/moe/moe_monokernel/shapes.json`, regenerates `generated/*.inc` +
   `monokernel_shapes.py` (gen_shapes.py), and runs the incremental cmake
   build (~minutes).  Back up shapes.json if experimenting.

3. **Build + tune** — `--steps build,tune,pin`.  Tuning gates each config on
   accuracy (cosine vs the Python reference, 5 seeds) then ranks by
   CUDA-graph latency vs Triton.  For rankings that match production traffic,
   first capture real routing:
   ```bash
   MODEL=<model> TP=<tp> ./run_moe_bench.sh capture sonnet
   python onboard_model.py <model> --steps tune,pin \
       --route-capture <run_dir>/capture/route_capture.pt
   ```

4. **E2E benchmark** — `--steps bench` (or directly):
   ```bash
   MODEL=<model> TP=<tp> MONOKERNEL_CONFIG=<key>:<cid> \
       ./run_moe_bench.sh both sonnet     # or gsm8k / sharegpt
   ```
   Runs `vllm serve` once per backend (triton, then monokernel) and prints
   the speedup table via summarize_bench.py.  Models needing special serve
   flags (tool parsers, kv-cache dtype, max-model-len): pass
   `--serve-extra-args "--kv-cache-dtype fp8 ..."` to onboard_model.py or
   `SERVE_EXTRA_ARGS=...` to run_moe_bench.sh.

## Key constraints (why a shape gets rejected)

- Kernel: E % 32 == 0, sharded N_half % 128 == 0, K % 128 == 0, top_k ≤ 8,
  FP8 block-wise 128×128 weight scales, BS ≤ 8 (decode fast path only).
- Grid carve: enum_configs must find UP_COL_HALVES ≤ 2 and
  DOWN_COL_TILE ≤ 512 with grid ≤ SM count — some (N_half, K) pairs have
  no feasible carve at a given TP; try another `--tp`.
- vLLM eligibility (fp8.py): plain or biased top-k over all experts only
  (no grouped top-k), FP8 block quant, shape present in the registry.
- After the pipeline, verify the server log shows
  "MoE monokernel fast path ENABLED" — if not, the eligibility gate
  rejected it at runtime; check the log line above it for the reason.

## Files touched

| file | role |
|---|---|
| `onboard_model.py` | the orchestrator (this pipeline) |
| `csrc/moe/moe_monokernel/shapes.json` | shape + config source of truth |
| `csrc/moe/moe_monokernel/tools/enum_configs.py` | feasible-config enumeration |
| `csrc/moe/moe_monokernel/tools/gen_shapes.py` | code/registry generation |
| `tune_monokernel.py` | accuracy-gated config sweep |
| `test_monokernel_accuracy.py` | harness (auto-registers registry shapes) |
| `run_moe_bench.sh` | generic serve benchmark (triton vs monokernel) |

Docs: `csrc/moe/moe_monokernel/DESIGN.md` (kernel design + config tables),
`csrc/moe/moe_monokernel/README.md` (build/tune/profile runbook).
