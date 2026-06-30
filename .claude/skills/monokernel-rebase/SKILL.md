---
name: monokernel-rebase
description: Rebase the Qwen3.5 MoE-monokernel feature branch onto a newer vLLM base (release tag or upstream/main) quickly and correctly. Use when the user wants to rebase/update the monokernel branch onto a new vLLM version, fix a "not rebased well" file, or verify the rebase didn't regress the monokernel. Covers the keep-our-files / overlay-edited-files / verify-accuracy workflow.
---

# Rebasing the MoE Monokernel onto a new vLLM base

The monokernel branch (`qwen3.5_122b`) is a large feature on top of vLLM. When the
base moves (e.g. a new release tag `v0.X.0`), the rebase splits cleanly into three
file classes. Get the class right per file and the rebase is fast and correct.

## Mental model: every changed file is in exactly one class

1. **OURS-ONLY** — files that do NOT exist in the new base (the whole
   `csrc/moe/moe_monokernel*/**` tree, `monokernel_shapes.py`, `route_capture.py`,
   `moe_monokernel_interleave.py`, `tune_monokernel.py`, `test_monokernel_accuracy.py`,
   the `run_*` scripts, etc.). → **Keep verbatim from the feature tip.** No merge.
2. **INTEGRATION** — files that exist in the base AND we edited, almost always to
   wire in the monokernel (`vllm/_custom_ops.py`, `vllm/model_executor/layers/
   quantization/fp8.py`, `.../fused_moe/layer.py`, `.../fused_moe/runner/moe_runner.py`,
   `csrc/libtorch_stable/moe/torch_bindings.cpp`, `CMakeLists.txt`, `vllm/envs.py`,
   `benchmarks/kernels/benchmark_moe.py`). → **Take the new base's file and re-apply
   ONLY our monokernel additions on top (a pure-addition overlay).**
3. **CRUFT** — tracked-but-unrelated junk the branch carries (`venv_vllm/**`,
   `result_outputs/**`, `nsys_profile/**`, `*.nsys-rep`, `*.sqlite`, `*.rej`,
   `*_old.py`, stale `csrc/moe/marlin_moe_wna16/` + `csrc/quantization/marlin/`
   migration leftovers). → **Keep as-is** (the branch has always tracked them; do
   not let them block or pollute the rebase). They never affect rebase correctness.

The golden invariant: **a correct rebase result differs from the new base in EXACTLY
the feature-branch's file set, and each INTEGRATION file is a pure-addition overlay
(+N / −0) — no reworking of the base's own code.**

## Procedure

### 0. Set variables and back up
```bash
NEWBASE=v0.24.0            # the target tag/commit
OLDBASE=$(git merge-base HEAD "$NEWBASE")   # where our branch forked from the base line
git branch -f qwen3.5_122b_prerebase HEAD   # reversible backup
```
If `NEWBASE` is a divergent release tag (NOT an ancestor of `upstream/main`) and
`OLDBASE..HEAD` contains hundreds of base-line commits, a per-commit
`git rebase --onto` is conflict-hell. Use the **squash overlay** below instead.

### 1. Categorize the changed files (no truncation!)
```bash
git diff --name-only "$OLDBASE" HEAD > /tmp/delta.txt   # the feature's file set
: > /tmp/ours.txt; : > /tmp/shared.txt
while read f; do
  git cat-file -e "$NEWBASE:$f" 2>/dev/null && echo "$f" >> /tmp/shared.txt || echo "$f" >> /tmp/ours.txt
done < /tmp/delta.txt
```
`ours.txt` = OURS-ONLY + CRUFT (keep verbatim). `shared.txt` = INTEGRATION + a few
CRUFT-that-also-exist-in-base. NEVER `head -N` these lists — truncation silently
drops files (this bites: monokernel_shapes.py etc. landed past the first 60).

### 2. Build the result on the new base
```bash
git checkout -b qwen3.5_122b_rebased "$NEWBASE"
# OURS-ONLY + CRUFT: copy verbatim from the backup tip
while read f; do
  git cat-file -e "qwen3.5_122b_prerebase:$f" 2>/dev/null &&
    { mkdir -p "$(dirname "$f")"; git show "qwen3.5_122b_prerebase:$f" > "$f"; git add "$f"; }
done < /tmp/ours.txt
# SHARED: 3-way merge (base=OLDBASE, ours=NEWBASE, theirs=feature tip)
while read f; do
  git show "$OLDBASE:$f"  > /tmp/b 2>/dev/null || : > /tmp/b
  git show "$NEWBASE:$f"  > /tmp/o 2>/dev/null || : > /tmp/o
  git show "qwen3.5_122b_prerebase:$f" > /tmp/t 2>/dev/null || : > /tmp/t
  git merge-file -p /tmp/o /tmp/b /tmp/t > "$f"; git add "$f"   # leaves <<< markers on conflict
done < /tmp/shared.txt
```

### 3. CRITICAL — fix every INTEGRATION file to be a clean overlay
The 3-way merge **auto-merges by text** and will happily rework the base's own code
when our edits were authored against the OLD structure. For each INTEGRATION file:
```bash
git diff --numstat "$NEWBASE" -- <file>     # want +N -0; any -M>0 = it reworked base code
```
If `-M > 0` (or it looks wrong), do the **manual overlay** — this is the reliable fix:
```bash
git checkout "$NEWBASE" -- <file>                 # restore pristine base
git diff "$OLDBASE" qwen3.5_122b_prerebase -- <file> > /tmp/ours.diff   # our original additions
# re-apply each hunk by hand onto the base's current structure (Edit tool),
# keeping ONLY our monokernel additions.
```
**Stale-API hazard (this WILL happen):** our additions may call base symbols that the
new base renamed/removed. A pure-addition overlay can still crash. After re-applying,
for every `super().X`, cross-module import, and `layer.`/`self.` attribute our code
touches, confirm it still exists in the new base. Real example from v0.24.0:
fp8.py's `mk_owns_shared_expert` override called `super().mk_owns_shared_expert`, but
v0.24.0 removed that property (the runner now runs shared experts unconditionally via
`_maybe_apply_shared_experts`) → the override had to be DROPPED, not ported. When a
base API is gone, adapt or drop our edit; don't carry a dead `super()` call.

Verify each integration `.py` imports against the new base:
```bash
venv_vllm/bin/python -c "import <module>; print('OK')"
```

### 4. Commit, and confirm the invariant
```bash
git diff --name-only "$NEWBASE" | sort > /tmp/diff_now.txt
comm -23 /tmp/diff_now.txt <(sort /tmp/delta.txt)   # MUST be empty (no stray changes vs base)
```

### 5. Rebuild and run the accuracy + performance tests (no-regression gate)
A new base changes shared headers/ABI; the monokernel C++ may need adapting (e.g. the
v0.24.0 stable-ABI `_moe_C_stable_libtorch` port: `torch::Tensor`→`torch::stable::Tensor`,
`TORCH_CHECK`→`STD_TORCH_CHECK`, `TORCH_BOX` registration). Build, then test BOTH shapes:
```bash
# build (see ../../CLAUDE.md "Building the monokernel"; or full reconcile after a base bump:)
#   uv pip install -e . --torch-backend=auto      # isolated full build, reconciles deps
venv_vllm/bin/python test_monokernel_accuracy.py --model qwen3.5      --batch-sizes 1 2 4 8
venv_vllm/bin/python test_monokernel_accuracy.py --model qwen3.5_122b --batch-sizes 1 2 4 8
```
PASS = CUDA-vs-Py cosine > 0.99 at every M (expect ~0.9997 for 35B, ~0.9999 for 122B).
The perf table prints speedup vs Triton; record it and compare to the prior base to
confirm no regression. Synthetic routing understates the win — use `--route-capture
DUMP.pt` for production-representative numbers if available.

## Gotchas (learned the hard way)
- **The build / accuracy harness needs a working `venv_vllm/bin/python`.** The venv is
  tracked in git with its python launchers stored as plain files (not symlinks), so
  `git checkout`/`reset` during a rebase clobbers them → "Permission denied". Repair:
  `cp /usr/local/bin/python3.12 venv_vllm/bin/python{,3,3.12}; chmod +x`. Then
  `git update-index --assume-unchanged venv_vllm/bin/python*` so the 31 MB binaries
  aren't committed.
- **`/opt/vllm-venv` symlink** is referenced by the cmake cache and gets deleted
  periodically; restore with `ln -sfn $PWD/venv_vllm /opt/vllm-venv` if a build no-ops
  or "Unable to find python". `CMakeUserPresets.json` now points at the in-repo venv.
- **`git rebase --onto` on git 2.34** can bail with "local changes would be
  overwritten" even on a clean tree (stale stat / mid-pick partial). `git reset --hard
  HEAD` then retry; or just use the squash overlay above.
- **Don't push without confirming the target.** The branch history is rewritten by a
  rebase → `git push --force-with-lease origin qwen3.5_122b` (after the user confirms).
