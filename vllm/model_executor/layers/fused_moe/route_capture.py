# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight MoE routing capture for offline kernel benchmarking.

Real decode-time routing is *imbalanced*: a handful of "hot" experts receive
most of the tokens while the rest receive none.  The kernel micro-benchmark
(``test_monokernel_accuracy.py``), by contrast, synthesises routing from
``torch.randn(M, E)`` logits, which spreads ~M*top_k assignments almost
uniformly across the experts.  That difference changes the per-expert GEMM
batching and explains why the monokernel's *end-to-end* speedup (1.5x) is far
larger than its *kernel-level* speedup (~1.04x) measured on synthetic routing.

This module captures, per MoE layer, the **number of tokens** and the **router
logits** of decode-shaped forward calls during a real ``vllm serve`` run.  The
captured logits are later replayed into the kernel benchmark so both the CUDA
monokernel and the Triton baseline route exactly as production does.

Enable by setting ``MONOKERNEL_ROUTE_CAPTURE=/path/to/dump.pt`` before starting
the server.  Capture is otherwise a no-op (the env var is read once at import).

Knobs (all optional):
  MONOKERNEL_ROUTE_CAPTURE        output .pt path; presence enables capture
  MONOKERNEL_ROUTE_CAPTURE_MAXM   max M treated as "decode" (default 8)
  MONOKERNEL_ROUTE_CAPTURE_MAX    total snapshot cap across all layers
                                  (default 1024)
  MONOKERNEL_ROUTE_CAPTURE_PERLAYER  per-layer snapshot cap (default 8)

The dump is a torch.save of::

    {
      "meta": {"max_m": int, "captured": int},
      "snapshots": [
          {"layer_id": int, "layer_name": str, "M": int,
           "top_k": int, "scoring_func": str, "renormalize": bool,
           "router_logits": Tensor[M, E] (cpu, float16)},
          ...
      ],
    }
"""

import atexit
import os

# Read config once at import.  An unset MONOKERNEL_ROUTE_CAPTURE disables the
# whole module: capture_routing() returns immediately on the first guard.
_PATH = os.environ.get("MONOKERNEL_ROUTE_CAPTURE")
_MAX_M = int(os.environ.get("MONOKERNEL_ROUTE_CAPTURE_MAXM", "8"))
_MAX_TOTAL = int(os.environ.get("MONOKERNEL_ROUTE_CAPTURE_MAX", "1024"))
_MAX_PER_LAYER = int(os.environ.get("MONOKERNEL_ROUTE_CAPTURE_PERLAYER", "8"))

_ENABLED = _PATH is not None

# How many new snapshots to accumulate between incremental disk flushes. The
# model runs in a worker subprocess that `vllm serve` stops with SIGTERM, and
# atexit handlers do NOT run on signal termination — so we cannot rely on a
# single flush at exit. The data is tiny (~4KB/snapshot), so we re-save the
# whole dump every FLUSH_EVERY snapshots; the file is always up to date when
# the process is killed.
_FLUSH_EVERY = int(os.environ.get("MONOKERNEL_ROUTE_CAPTURE_FLUSH_EVERY", "16"))

# Accumulated snapshots (CPU tensors) and a per-layer counter.
_snapshots: list[dict] = []
_per_layer_count: dict[int, int] = {}
_armed_logged = False


def is_enabled() -> bool:
    return _ENABLED


def capture_routing(
    layer,
    x,
    router_logits,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
) -> None:
    """Record one decode-shaped routing snapshot.

    Cheap and best-effort: any failure is swallowed so capture can never break
    a serving run.  Only ``M <= MAXM`` (decode-shaped) calls are recorded, up
    to the global / per-layer caps.  The dump is re-written to disk every
    ``_FLUSH_EVERY`` snapshots so it survives the server's SIGTERM shutdown.
    """
    global _armed_logged
    if not _ENABLED:
        return
    try:
        if not _armed_logged:
            _armed_logged = True
            print(
                f"[route_capture] ARMED (max_m={_MAX_M}, per_layer={_MAX_PER_LAYER}, "
                f"total={_MAX_TOTAL}) -> {_PATH}",
                flush=True,
            )

        M = x.size(0)
        if M <= 0 or M > _MAX_M:
            return
        if len(_snapshots) >= _MAX_TOTAL:
            return

        # extract_layer_index-based id; fall back to -1 if the property is
        # unavailable (keeps per-layer accounting working as a single bucket).
        try:
            layer_id = int(layer.layer_id)
        except Exception:
            layer_id = -1
        layer_name = getattr(layer, "layer_name", "")

        if _per_layer_count.get(layer_id, 0) >= _MAX_PER_LAYER:
            return

        # Detach to CPU as float16: [M, E] is tiny (8*256*2B = 4KB) and float16
        # round-trips the argmax/top-k routing faithfully for benchmarking.
        import torch

        logits_cpu = router_logits.detach().to("cpu", dtype=torch.float16)

        _snapshots.append(
            {
                "layer_id": layer_id,
                "layer_name": layer_name,
                "M": int(M),
                "top_k": int(top_k),
                "scoring_func": str(scoring_func),
                "renormalize": bool(renormalize),
                "router_logits": logits_cpu,
            }
        )
        _per_layer_count[layer_id] = _per_layer_count.get(layer_id, 0) + 1

        # Incremental flush so the dump survives a SIGTERM (no atexit on kill).
        if len(_snapshots) % _FLUSH_EVERY == 0:
            _save()
    except Exception:
        # Never let instrumentation crash the server.
        pass


def _save() -> None:
    """Write the current snapshot buffer to disk. Idempotent; safe to call
    repeatedly (incrementally) and again at exit."""
    if not _ENABLED or not _snapshots:
        return
    try:
        import torch

        os.makedirs(os.path.dirname(os.path.abspath(_PATH)) or ".", exist_ok=True)
        # Write to a temp path then rename, so a kill mid-write can't leave a
        # truncated/corrupt dump.
        tmp = _PATH + ".tmp"
        torch.save(
            {
                "meta": {"max_m": _MAX_M, "captured": len(_snapshots)},
                "snapshots": _snapshots,
            },
            tmp,
        )
        os.replace(tmp, _PATH)
        n_layers = len({s["layer_id"] for s in _snapshots})
        print(
            f"[route_capture] wrote {len(_snapshots)} routing snapshots "
            f"across {n_layers} layers -> {_PATH}",
            flush=True,
        )
    except Exception as e:  # pragma: no cover - best effort
        print(f"[route_capture] failed to write dump: {e}", flush=True)


if _ENABLED:
    # Final flush on a clean exit; the incremental writes cover SIGTERM kills.
    atexit.register(_save)
