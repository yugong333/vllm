# Down-Projection DRAM Throughput — Investigation Notes

Status: investigation record, 2026-07-15 .. 2026-07-17.
Shape under study: Qwen3.5-35B (E=256, N_half=512, K=2048), BS16 path,
H200 (sm_90a). Baseline timings quoted as M9/M12/M16 CUDA-graph replay.

This documents (1) why the down projection's DRAM throughput is low and
comb-shaped while the up projection sits on a high plateau, (2) every
mitigation we tried and its measured result, (3) why the up projection is
structurally immune, and (4) how DeepGEMM's SM100 mega-MoE kernel handles
the same problem.

---

## 1. Why down-projection DRAM throughput is low

### The duty-cycle model

DRAM utilization is `bytes / (peak_bw x elapsed_time)`. Bytes are fixed by
the weights, so low utilization means elapsed time contains intervals that
issue no DRAM demand. In the down phase those intervals are the
**per-expert boundary (the "seam")**: after the last WGMMA of expert `e`,
the calc warps run a serial tail before expert `e+1`'s WGMMAs start:

```text
wgmma_wait -> scale-apply -> drain final_d -> __syncthreads -> next-expert
                             (registers      (publish        scales/gating
                              -> down_out)    down_out)
```

The tail is paid **once per expert regardless of K-steps**, and the 35B
down phase has only `K_TILES_DOWN = N/KDN = 512/256 = 2` K-steps per
expert. Toy numbers (1 us stream per tile, 1 us tail):

```text
down (2 tiles/expert):  |w0|w1|....|w0|w1|....|      duty = 2/3  ~ 67%
up   (8 tiles/expert):  |w0|..|w7|.|w0|..|w7|.|      duty = 8/9  ~ 89%
```

### Why it shows as a comb (and why "more K" flattens it)

The NCU trace is the sum of per-group demand square waves (16 blocks per
expert group run in lockstep; ~8-16 independent phases). With a 33%-off
wave, group tails frequently coincide -> deep synchronized dips (comb).
With an 11%-off wave the dips are narrow, rare, dephased, and averaged
away by the profiler's sampling window -> a flat line. **Flatness is
cosmetic; the tail cost lives in the plateau's height.** The synthetic
8-step shape (below) plateaus at ~67-68% of peak, not ~95% — amortized,
still paid.

### Sizing the seam

Per expert, per down block at BS16: weights = 128 KB (2 x 64 KB tiles),
activations = 8 KB. A block's DRAM share is ~4.8 TB/s / 128 ~ 37 GB/s, so
a ~1 us seam swallows ~37 KB of potential transfer. Only weights have the
mass to fill it; activations are a rounding error (see section 2, probes
5-6).

### What the seam is NOT

* Not a data wait: at the next expert's `bar_w` the measured residual wait
  is **~0.10 us** (weight TMA delivery itself is ~2.27 us — i.e. the
  lookahead already hid it). Activation exposed latency: **~0.06 us**.
* Not pipeline depth: DPD=4 measured flat (probe 1).
* Not the SHM->GM accumulate plumbing: worth 0.1-0.6 us total (probe 9).
* It IS the calc-warp register drain + block sync: `final_d` lives in the
  WGMMA accumulator registers, which the next expert's WGMMA needs back;
  no other warp can read registers, and prefetch by construction cannot
  hide compute that runs on the warps that would issue the next WGMMA.

---

## 2. What we tried

Production baseline during this work: M9/M12/M16 = 0.085/0.092/0.107 ms
(later re-measured 0.0868/0.1023/0.1184 on the post-cleanup binary).

| # | Experiment | Result | Verdict |
| --- | --- | --- | --- |
| 1 | 4-deep down TMA ring (config 7, KDN=128/DPD=4) | flat vs config 0 | Pipeline depth is not the limiter; no 3rd tile of the same expert exists to look ahead to |
| 2 | `%globaltimer` boundary timing | weight issue->`bar_w` 2.27 us; next-expert `bar_w` wait 0.10 us; act exposure 0.06 us | Data arrives before it is needed; seam is compute/sync |
| 3 | Compile out down epilogue (SKIP profile) | 136.256 -> 122.496 us, DRAM 50.4% -> 56.1%, identical read volume | Upper bound for all seam work: ~13.8 us (~10%) |
| 4 | Synthetic `e256_n2048_k2048` (8 down K-steps, same DCT/KDN/DOWN_GROUPS) | DRAM ~67-68%, flat plateau; boundary share 38% -> 14%; DPD=4 still flat there | Confirms amortization model; K-step count drives duty cycle |
| 5 | Raw-epilogue handoff (spill WGMMA fragments, PF warps apply scales) | 0.091/0.099/0.116 vs 0.085/0.092/0.107 — regression | Handoff (spill+sync+reload) costs more than the ~0.05 us of math it moves; drain size is set by output tile, not K |
| 6 | Early weight arming (signal launcher right after last `wgmma_wait`) | 0.087/0.092/0.107 — flat | The wait was already ~0.10 us; nothing to recover |
| 7 | Opportunistic next-expert activation stitch (poll publication in the tail) | 0.135-0.149 vs 0.109 — large regression | Publication scanning costs more than the 0.06 us residual |
| 8 | Deferred accumulate + `slot_to_token` (LANDED) | production behavior | `out_accum += down_out` moved to PF warps, sliced across next expert's K-steps; per-rank iteration cuts PF work ~8x at BS16 |
| 9 | Independent 4-slot weight ring, config 8 (DCT=128, KDN=256, DWS=4/DPD=2) (LANDED, opt-in) | M16 0.105 vs 0.110 (~4.5%); SHM 200,704 B; 110 reg/thread; cosine 0.999731 | Streams next expert's weight tiles through the seam — DRAM stays fed during the tail. Slower at M4/M8, so opt-in |
| 10 | Skip out_accum zero-fill + final atomicAdd pass (probe, garbage output) | M9/M12/M16: -0.6/-0.4/-0.1 us | Kills the disjoint-output rewrite: what it would remove is already ~free |

Analysis-only (rejected without implementation, with reasons):

* **More warps for the epilogue** — data is trapped in calc-warp
  registers; parallelizing needs spill+`__syncthreads`+reload (~0.75 us
  fixed cost against ~0.05 us of work). Superset measured as probe 5.
* **Fetch all activations up front / bigger activation fetch** — 8 KB per
  expert vs a ~37 KB seam; already issued maximally early (expert-top +
  s=0); next expert's are publication-gated (probe 7 measured the cost of
  beating the gate).
* **Disjoint per-(token,expert) output rows + separate reduce** (mirror of
  `temp_fp8`) — legal (routing weights are baked in upstream), but probe
  10 shows the removable plumbing is 0.1-0.6 us while the scheme adds ~2 MB
  traffic and a fatter Phase 5. The drain + publication sync survive in
  either scheme.
* **Register-resident cross-expert accumulation** — rank->token mapping is
  dynamic per expert; dynamic register indexing spills to local memory.
  This is exactly why `down_out` (SHM, dynamically indexable) exists.
* **Spill k0's partial early / use both of e+1's K-steps for e's
  accumulate** — drain size is K-invariant (8-16 floats/lane either way);
  the accumulate already fits in a fraction of one WGMMA window
  (routed_count is 1-2 for most experts).
* **Block redesign / scheduling** — per-expert cost is routing-independent
  (fixed GEMM shape), so static round-robin is balanced to +-1 expert.
  2 blocks/SM needs <=116 KB/block: forces KDN=128-class geometries that
  measured slower. Grid 132 doesn't divide the 35B carve. Cluster
  multicast of activations caps at <1% (6% of traffic / 8).
* **Wave scheduling (DeepGEMM-style up/down interleaving)** — each phase
  flip re-pays ring re-prime (~2.3 us), breaks the cross-expert stitches,
  and forces the up deferred epilogue to flush (post_silu_scratch aliases
  down_out). ~14 flips cost 30-50 us against a ~10 us DRAM-mixing ceiling.
  Minimal 2-half variant estimated break-even.

Scoped but unbuilt (the remaining candidate with positive expected value):

* **Merged expert pairs** — run 2 experts in one fused 4-step K-loop with
  *separate* `final_d` sets. Sync count is invariant (per-step syncs scale
  with K-steps), and with a 2-slot ring the fetch cadence is identical to
  today's lookahead — the unique win is that e0's drain executes in e1's
  WGMMA shadow, since the register-reuse dependency is broken. Cost:
  +16 regs/lane (109 -> ~125), +16 KB SHM (down_out x2, 216.7 KB fits),
  odd-count tail path, loop restructure. Best evaluated on top of
  config 8. This is the software imitation of SM100 TMEM (section 4).

---

## 3. Why the up projection does not have this problem

Not just "more K-steps" (8 vs 2) — every per-expert seam cost is absent,
prefetched, stitched, or deferred:

| Seam cost | Down phase | Up phase |
| --- | --- | --- |
| Activation fetch | per-expert TMA, publication-gated sentinel poll | absent — `fp8_act_full` loaded once in Phase 2, resident all phase |
| Activation scales | per-expert SHM load at expert top | absent — `act_scale` written once in Phase 2 |
| Weight scales | per-expert load at expert top | cp.async-prefetched one full K-loop ahead (ping-pong slot) |
| Weight stream | 1-tile lookahead (cfg 0) / continuous (cfg 8) | continuous stitch: launcher arms `s + ARM_DISTANCE`, wrapping into the next expert — the stream never stops |
| Epilogue | `final_d -> down_out` drain on calc warps + accumulate ordering | calc warps only combine -> `post_silu_scratch`; SiLU + reduce-max + fp8 quant + GM store + scale publish run on PF warps during the *next* expert's K-loop |
| Output collisions | top-k experts add into the same `out_accum[token]` rows -> needs `down_out` staging + publication sync | none — each routed (token,expert) pair owns its own `temp_fp8` row; writeback is fire-and-forget |

Two of these are structural asymmetries the down phase cannot copy:

1. **Dynamic activations**: up's B operand is the same tokens for every
   expert (known at launch); down's B operand is produced per expert by
   Phase 3 in *other blocks* and cannot exist in SHM before publication.
2. **Colliding outputs**: down performs a cross-expert reduction per
   token. (Though per probe 10, the reduction plumbing itself is cheap —
   the expensive residue is the drain + sync, which up also pays but
   amortizes over 4x more streaming.)

What was copyable has been copied: config 8's continuous weight stream is
the up-projection stitch transplanted onto the down weight ring; the
deferred accumulate is the down analog of up's deferred writeback.

---

## 4. What DeepGEMM does (SM100 mega-MoE, for comparison)

Source: `DeepGEMM/deep_gemm/include/deep_gemm/{scheduler,impls}/mega_moe.cuh`,
`csrc/jit_kernels/heuristics/mega_moe.hpp`. Blackwell (sm_100), EP
multi-rank, fused dispatch + L1 (gate/up) + L2 (down) + combine.

Same skeleton as our monokernel:

* Persistent CTAs, grid = `num_sms`, exactly 1 CTA/SM (`__launch_bounds__`
    * SHM sized to fill the SM; `num_stages` maximized). 2-CTA clusters for
  the 2-SM UMMA.
* **Software grid barrier** (atomic counter + 0x80000000 finish tag,
  acq/rel — same idea as our `grid_barrier`), used **only at communication
  edges** (dispatch, NVLink barriers, pre-combine), *not* for L1->L2.
* **L1->L2 handoff via data-path readiness**: L1's epilogue TMA-stores its
  output then `red.or.release` into a per-token-block arrival bitmask; the
  L2 activation loader spins `ld.acquire` on the mask. Functionally our
  site-#2 sentinel scheme. Their comment records that on-demand L1/L2
  overlap was tried and **removed as a negative** — the same conclusion as
  our probe 7.
* **Static strided scheduling, no work stealing**: the flattened
  (expert, m_block, n_block) tile space is consumed with
  `block_idx += kNumSMs`. Unlike ours, tile counts are
  token-proportional (`num_m_blocks = ceil(tokens_e / BLOCK_M)`,
  BLOCK_M 16-192 chosen from expected tokens/expert), because their EP
  batch sizes make per-expert M large and variable. A wave state machine
  alternates L1 and L2 in chunks of `num_experts_per_wave`, sized so one
  wave's L1 block count ~ 2x num_sms — a structural guarantee that L1 is
  done before its wave's L2 starts (their substitute for polling slack).

The two things that actually neutralize the seam for them:

1. **TMEM staged accumulators (hardware)**. On SM100 the accumulator lives
   in tensor memory with `tmem_full/empty`-tracked stages: the MMA warp
   issues the next tile's UMMA into stage k+1 while epilogue warpgroups
   drain stage k. The register-drain dependency that defines our seam
   does not exist on that silicon. Our merged-pair proposal is the SM90
   software imitation.
2. **Disjoint outputs + separate combine**. L2 writes BF16 results to
   per-(token, topk) slots in remote NVLink buffers; a later combine phase
   reduces top-k per token (~3 us/token, plus a ~4 us grid-sync barrier).
   This pays for them only because tokens must cross NVLink and be
   re-gathered anyway; probe 10 shows the equivalent trade is a net loss
   in our single-GPU kernel.

### Takeaway

DeepGEMM independently converges on the same architecture (persistent
1-CTA/SM blocks, software grid sync at phase edges only, publication-flag
handoff, static strided work, no on-demand overlap). Its immunity to the
down-projection seam comes from SM100 TMEM, not from scheduling. On
Hopper, the remaining levers are, in order of expected value: merged
expert pairs (drain hiding), seam micro-costs (scale loads / sync
placement), and nothing else with measured support.
