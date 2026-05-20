"""
SA polish for legal placements.

PURPOSE
=======
Take a *legal* placement (zero hard-macro overlaps) and search nearby
legal configurations for a lower proxy cost. This is the last stage of
the standard hybrid macro-placement pipeline:

    GP (DREAMPlace-style) -> legalize -> SA polish

The legalizer makes the placement legal but typically perturbs macros
from their GP-optimal positions, hurting wirelength. SA polish recovers
that wirelength damage by exploring small displacements that preserve
legality.

DIFFERENCE FROM EXPLORATION-MODE SA
====================================
This polisher is intentionally simpler than `sa_placer_fast.py`:

  * STRICT legality:  any move that creates an overlap is rejected
    immediately. No overlap penalty term — we never trade legality
    for proxy.
  * Local moves only: displacement only, no swap, no teleport. The
    input is already near-optimal; large jumps would mostly hurt.
  * Low temperature, small sigma: scaled so even uphill moves stay
    near the current state.
  * Time-budgeted: stops when wall-clock budget is hit, not after
    a fixed sweep count.

DESIGN DECISIONS
================
- Uses the incremental-cost API (`compute_proxy_cost_incremental` +
  `_set_placement_fast_moved`) for ~10x speedup vs full-cost path.
- Tracks best legal proxy separately from current proxy; we may walk
  uphill briefly to escape a shallow basin.
- No trajectory logging — that's debug-mode SA, not polish.
- Only HARD movable macros move. Soft macros are locked by challenge
  rules; the incremental cost cache also requires moved slots to be
  in [0, num_hard_macros).

USAGE
=====
    from sa_polisher import polish

    polished = polish(
        legal_placement,
        benchmark,
        plc,                       # already-loaded plc object (reuse from GP)
        time_budget_seconds=180,
    )
"""

import math
import random
import time
from pathlib import Path
import sys

import torch

from macro_place.benchmark import Benchmark

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    _set_placement_fast,
    _set_placement_fast_moved,
    compute_proxy_cost_incremental,
    validate_placement,
)


# Internal legality gap — must match (or be tighter than) the legalizer's
# GAP so we don't drift into proxy-detectable overlaps during polish.
GAP = 1e-3


def polish(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    time_budget_seconds: float = 180.0,
    T0: float = 0.005,
    T_min: float = 1e-7,
    alpha: float = 0.95,
    sigma_frac: float = 0.005,
    inner_per_macro: int = 3,
    verbose: bool = True,
) -> torch.Tensor:
    """
    SA polish a legal placement; returns a legal placement with same-or-better proxy.

    Args:
        placement:           [num_macros, 2] LEGAL input placement.
        benchmark:           Benchmark; provides sizes, masks, canvas.
        plc:                 PlacementCostAccelerated, already loaded for
                             this benchmark (pass from GP to avoid re-load).
        time_budget_seconds: wall-clock budget. Polish stops early when hit.
        T0:                  initial temperature. 0.005 is very low — only
                             tiny worsening moves get accepted.
        T_min:               stop cooling at this T (then keep doing greedy
                             accepts until budget expires).
        alpha:               geometric cooling rate per sweep.
        sigma_frac:          fraction of min(canvas) for the base displace
                             sigma at T=T0. Sigma scales with T/T0.
        inner_per_macro:     moves proposed per movable macro per sweep.
        verbose:             print start/end summary.

    Returns:
        torch.Tensor [num_macros, 2] — same-or-better legal placement.
    """
    pos = placement.clone().cpu()
    sizes = benchmark.macro_sizes  # [num_macros, 2] on CPU
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    # Only hard MOVABLE macros are candidates. Soft macros and fixed
    # macros stay put. The incremental cost path requires moved slots in
    # [0, num_hard_macros), so this also matches what the kernels expect.
    movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    movable_idx = torch.where(movable_mask)[0].tolist()
    if not movable_idx:
        if verbose:
            print("[polish] no movable hard macros — returning input unchanged")
        return pos

    # ------------------------------------------------------------------
    # Sync plc state to the input placement, then prime the incremental
    # cost caches. The full _set_placement_fast also invalidates all
    # caches, so the first incremental call rebuilds from scratch.
    # ------------------------------------------------------------------
    _set_placement_fast(plc, pos, benchmark)
    pc0 = compute_proxy_cost_incremental(pos, benchmark, plc, moved_slots=None)

    initial_cost = float(pc0["proxy_cost"])
    initial_overlaps = pc0["overlap_metrics"]["overlap_count"]
    if initial_overlaps > 0:
        # Polish requires a legal start. Refuse to operate on illegal
        # input — the caller is responsible for legalization upstream.
        if verbose:
            print(f"[polish] WARNING: input has {initial_overlaps} overlaps; "
                  f"polish requires legal input. Returning unchanged.")
        return pos

    cur_cost = initial_cost
    best_cost = initial_cost
    best_pos = pos.clone()

    # Sigma at T=T0. Scales with T/T0 so it shrinks as the schedule cools.
    sigma_base = max(0.1, sigma_frac * min(canvas_w, canvas_h))

    n_inner = max(50, inner_per_macro * len(movable_idx))

    moves_total = 0
    moves_accepted = 0
    moves_improved = 0
    moves_rejected_overlap = 0
    sweeps = 0

    start_time = time.time()
    T = T0

    # Convert sizes to a plain list once to skip torch overhead per move.
    sizes_list = [(float(sizes[i, 0]), float(sizes[i, 1])) for i in range(len(sizes))]

    if verbose:
        print(f"[polish] start: cost={initial_cost:.4f}, "
              f"{len(movable_idx)} movable, "
              f"T0={T0}, sigma_base={sigma_base:.3f}, "
              f"budget={time_budget_seconds:.0f}s")

    while time.time() - start_time < time_budget_seconds:
        sigma = max(0.01, sigma_base * (T / T0))

        sweep_accepted = 0
        sweep_improved = 0

        for _ in range(n_inner):
            # Budget check inside the inner loop too — sweep can be long
            # on big benchmarks.
            if time.time() - start_time >= time_budget_seconds:
                break

            # ---- propose: pick a random movable, gaussian displace ----
            idx = movable_idx[random.randrange(len(movable_idx))]
            w, h = sizes_list[idx]
            old_x = float(pos[idx, 0])
            old_y = float(pos[idx, 1])

            new_x = old_x + random.gauss(0.0, sigma)
            new_y = old_y + random.gauss(0.0, sigma)

            # Clamp inside canvas with the legality gap. Clipping here
            # (rather than rejecting) reduces wasted proposals near walls.
            new_x = max(0.5 * w + GAP, min(canvas_w - 0.5 * w - GAP, new_x))
            new_y = max(0.5 * h + GAP, min(canvas_h - 0.5 * h - GAP, new_y))

            # Zero-move sentinel (sigma can collapse very small at low T)
            if new_x == old_x and new_y == old_y:
                continue

            # ---- apply to torch tensor AND to plc state ----
            pos[idx, 0] = new_x
            pos[idx, 1] = new_y
            _set_placement_fast_moved(plc, [idx], [(new_x, new_y)])

            pc = compute_proxy_cost_incremental(
                pos, benchmark, plc, moved_slots=[idx],
            )
            overlaps = pc["overlap_metrics"]["overlap_count"]

            # ---- STRICT legality: any overlap => reject and roll back ----
            if overlaps > 0:
                pos[idx, 0] = old_x
                pos[idx, 1] = old_y
                _set_placement_fast_moved(plc, [idx], [(old_x, old_y)])
                # Reverse-apply once with the same slot to roll the
                # incremental cache back symmetrically. The kernels'
                # subtract/add structure means a same-slot symmetric
                # update undoes the previous delta.
                compute_proxy_cost_incremental(
                    pos, benchmark, plc, moved_slots=[idx],
                )
                moves_rejected_overlap += 1
                moves_total += 1
                continue

            # ---- Metropolis on proxy cost ----
            new_cost = float(pc["proxy_cost"])
            delta = new_cost - cur_cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
                # Accept.
                cur_cost = new_cost
                sweep_accepted += 1
                if delta < 0:
                    sweep_improved += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_pos = pos.clone()
            else:
                # Reject. Same revert-and-resync pattern as overlap reject.
                pos[idx, 0] = old_x
                pos[idx, 1] = old_y
                _set_placement_fast_moved(plc, [idx], [(old_x, old_y)])
                compute_proxy_cost_incremental(
                    pos, benchmark, plc, moved_slots=[idx],
                )

            moves_total += 1

        moves_accepted += sweep_accepted
        moves_improved += sweep_improved
        sweeps += 1

        # Cool. When we hit T_min, keep iterating but with frozen T so
        # we squeeze the last greedy improvements before the budget runs out.
        if T > T_min:
            T *= alpha
            if T < T_min:
                T = T_min  # don't go below — greedy-mode

    elapsed = time.time() - start_time

    if verbose:
        accept_pct = 100.0 * moves_accepted / max(1, moves_total)
        improve_pct = 100.0 * moves_improved / max(1, moves_total)
        reject_overlap_pct = 100.0 * moves_rejected_overlap / max(1, moves_total)
        print(f"[polish] done in {elapsed:.1f}s, {sweeps} sweeps, "
              f"{moves_total} moves "
              f"(accept={accept_pct:.1f}%, improve={improve_pct:.1f}%, "
              f"overlap-reject={reject_overlap_pct:.1f}%)")
        print(f"[polish] cost: start={initial_cost:.4f} -> "
              f"best={best_cost:.4f}  "
              f"(delta={best_cost - initial_cost:+.4f})")

    # Final paranoia check — make sure best_pos is still legal.
    is_valid, violations = validate_placement(best_pos, benchmark)
    if not is_valid and verbose:
        print(f"[polish] WARNING: best_pos failed validate_placement: "
              f"{violations[:3]}")
        # Fall back to the input — at least it was provably legal.
        return placement.clone().cpu()

    return best_pos
