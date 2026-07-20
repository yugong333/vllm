#!/usr/bin/env python3
"""
Diff per-kernel GPU invocation counts between two nsys reports
(the Triton FP8 baseline vs the MoE monokernel) for the SAME model + workload.

This is a transparent diff, NOT a pass/fail assertion.  It answers:
  "Which kernels are launched a different number of times in the two runs,
   and by how much?"

Why a plain diff (and not "everything except MoE is identical"):
  The monokernel ONLY replaces the MoE *decode* path (the M<=8 gate in
  fp8.py).  PREFILL (M>8) falls back to the Triton fused-experts kernels in
  BOTH runs, so the monokernel run STILL launches `fused_moe_kernel`,
  `per_token_group_quant_8bit_kernel`, `act_and_mul_kernel`, etc. — just far
  fewer times (only for the prefill forwards, not the 199 decode steps).
  The routing kernel also differs by phase: the Triton path uses the fused
  `topkGating` for both prefill and decode, while the monokernel path's
  prefill routing goes through `select_experts` (softmax_warp_forward +
  sbtopk::gatherTopK) and its decode routing is fused inside moe_kernel_topk.

So rather than claim "identical outside MoE", we just print the full delta
table and tag which kernels are MoE-related, letting the numbers speak.

Counts decompose cleanly for this workload (48 MoE layers, 1 warmup + 1
profiled prefill forward, 199 decode steps):
    decode-MoE invocations  = 48 * 199 = 9552   (== moe_kernel_topk count)
    prefill-MoE invocations = 48 * 2   = 96      (warmup + profiled prefill)

Usage:
    python compare_moe_kernels.py \
        --triton nsys_vllm_offline_122b/triton_.../triton_bs8.nsys-rep \
        --mono   nsys_vllm_offline_122b/monokernel_.../monokernel_bs8.nsys-rep

    # Show only MoE-tagged kernels:
    python compare_moe_kernels.py --triton a.sqlite --mono b.sqlite --moe-only

    # Show every kernel (including count==count matches), not just diffs:
    python compare_moe_kernels.py --triton a.sqlite --mono b.sqlite --all
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys

# ── MoE-related kernel patterns ──────────────────────────────────────────────
# Tagging only — purely informational.  A kernel whose demangled name CONTAINS
# any of these is labelled "MoE" in the diff so you can see at a glance which
# deltas come from the routed-expert path vs everything else.  Some of these
# names (per_token_group_quant, act_and_mul, the BFloat16 reduce) are also the
# ones the Triton prefill MoE uses in BOTH runs, which is exactly the point.
MOE_PATTERNS = [
    "moe_monokernel::moe_kernel_topk",          # monokernel (decode, fused)
    "vllm::moe::topkGating",                     # triton routing (prefill+decode)
    "softmax_warp_forward",                      # select_experts routing (mono prefill)
    "sbtopk::gatherTopK",                        # select_experts topk (mono prefill)
    "fused_moe_kernel",                          # triton GEMM1 / GEMM2
    "per_token_group_quant_8bit_kernel",         # triton activation quant
    "vllm::act_and_mul_kernel",                  # triton SiLU
    "vllm::moe::moe_align_block_size_kernel",    # triton expert align
    "vllm::moe::count_and_sort_expert_tokens",   # triton expert sort
    "ReduceOp<c10::BFloat16",                    # triton moe_sum (top_k reduce)
]


def moe_tag(name: str) -> str:
    return "MoE" if any(p in name for p in MOE_PATTERNS) else ""


# ── nsys report -> {kernel_name: count} ───────────────────────────────────────
def _ensure_sqlite(report: str) -> str:
    """Return a SQLite DB path for `report`, exporting it from a .nsys-rep
    if needed (reusing an up-to-date existing export)."""
    if report.endswith(".sqlite"):
        if not os.path.isfile(report):
            sys.exit(f"error: sqlite not found: {report}")
        return report
    if not report.endswith(".nsys-rep"):
        sys.exit(f"error: expected a .nsys-rep or .sqlite file, got: {report}")
    if not os.path.isfile(report):
        sys.exit(f"error: report not found: {report}")

    sqlite_path = report[: -len(".nsys-rep")] + ".sqlite"
    if os.path.isfile(sqlite_path) and \
            os.path.getmtime(sqlite_path) >= os.path.getmtime(report):
        return sqlite_path

    nsys = os.environ.get("NSYS") or shutil.which("nsys") or "/usr/local/cuda/bin/nsys"
    if not (os.path.isabs(nsys) and os.path.isfile(nsys)) and not shutil.which(nsys):
        sys.exit("error: nsys not found to export sqlite. Set NSYS=/path/to/nsys "
                 "or pass an already-exported .sqlite file.")
    print(f"[exporting] {report} -> {sqlite_path}")
    subprocess.run(
        [nsys, "export", "--type", "sqlite", "--force-overwrite=true",
         "--output", sqlite_path, report],
        check=True, stdout=subprocess.DEVNULL,
    )
    return sqlite_path


def kernel_counts(report: str) -> dict[str, int]:
    """Return {demangled_kernel_name: launch_count} for an nsys report."""
    db = sqlite3.connect(_ensure_sqlite(report))
    try:
        rows = db.execute(
            """
            SELECT s.value AS name, COUNT(*) AS n
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id
            GROUP BY name
            """
        ).fetchall()
    finally:
        db.close()
    return {name: n for name, n in rows}


def _short(name: str, width: int = 74) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def _print_table(rows, title):
    """rows: list of (name, triton, mono).  Prints a delta table."""
    print(f"\n{'-' * 104}")
    print(f" {title}  ({len(rows)} kernels)")
    print("-" * 104)
    print(f"   {'triton':>9}  {'mono':>9}  {'delta':>9}  {'tag':>4}   kernel")
    for name, t, m in rows:
        print(f"   {t:>9}  {m:>9}  {t - m:>+9}  {moe_tag(name):>4}   {_short(name)}")


def main():
    ap = argparse.ArgumentParser(
        description="Diff per-kernel GPU invocation counts between two nsys "
                    "reports (Triton vs monokernel).")
    ap.add_argument("--triton", required=True,
                    help="Triton-path nsys report (.nsys-rep or .sqlite).")
    ap.add_argument("--mono", required=True,
                    help="Monokernel-path nsys report (.nsys-rep or .sqlite).")
    ap.add_argument("--all", action="store_true",
                    help="Show every kernel, including those with equal counts "
                    "(default: only kernels whose counts differ).")
    ap.add_argument("--moe-only", action="store_true",
                    help="Restrict the listing to MoE-tagged kernels.")
    args = ap.parse_args()

    tri = kernel_counts(args.triton)
    mono = kernel_counts(args.mono)
    names = sorted(set(tri) | set(mono))

    bar = "=" * 104
    print(bar)
    print(" Per-kernel invocation diff:  Triton baseline  vs  MoE monokernel")
    print(bar)
    print(f"   triton report : {args.triton}")
    print(f"   mono   report : {args.mono}")
    print(f"   total launches: triton={sum(tri.values()):>9}   "
          f"mono={sum(mono.values()):>9}   "
          f"delta={sum(tri.values()) - sum(mono.values()):>+9}")
    print(f"   distinct kerns: triton={len(tri):>9}   mono={len(mono):>9}")

    # Build rows, optionally filtered to MoE, optionally only diffs.
    rows = [(n, tri.get(n, 0), mono.get(n, 0)) for n in names]
    if args.moe_only:
        rows = [r for r in rows if moe_tag(r[0])]
    diff_rows = [r for r in rows if r[1] != r[2]]
    same_rows = [r for r in rows if r[1] == r[2]]

    # ── MoE-tagged kernels: always listed in full (the heart of the diff) ───
    moe_rows = sorted((r for r in rows if moe_tag(r[0])),
                      key=lambda r: -(r[1] + r[2]))
    if not args.moe_only:
        _print_table(moe_rows, "MoE-LAYER kernels (decode fused into 1 launch; "
                               "prefill still uses the Triton kernels in BOTH runs)")
        moe_t = sum(r[1] for r in moe_rows)
        moe_m = sum(r[2] for r in moe_rows)
        print(f"   {'-' * 9}  {'-' * 9}  {'-' * 9}")
        print(f"   {moe_t:>9}  {moe_m:>9}  {moe_t - moe_m:>+9}   TOTAL MoE-layer launches")
        if moe_m:
            print(f"   monokernel: {moe_t} -> {moe_m} MoE launches "
                  f"({moe_t / moe_m:.1f}x fewer)")

    # ── All differing kernels, sorted by magnitude ─────────────────────────
    listing = (diff_rows if not args.all else rows)
    listing = sorted(listing, key=lambda r: -abs(r[1] - r[2]))
    title = ("ALL kernels (sorted by |delta|)" if args.all
             else "KERNELS THAT DIFFER (sorted by |delta|)")
    _print_table(listing, title)

    if not args.all and not args.moe_only:
        print(f"\n   ({len(same_rows)} more kernels have identical counts in both "
              f"runs — pass --all to list them.)")

    print(f"\n{bar}")
    print(" Note: this is a descriptive diff, not a pass/fail check. The MoE-layer")
    print(" rows above are the expected difference; non-MoE rows that differ are")
    print(" typically tiny one-off / cuBLAS-heuristic / allocator jitter (|delta|")
    print(" not proportional to the 9552 decode-step count).")
    print(bar)


if __name__ == "__main__":
    main()
