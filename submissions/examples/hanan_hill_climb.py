"""
Deterministic Hanan-grid hill climb for legalized placements.

PURPOSE
=======
Take a legal placement, and for each movable hard macro try alternative
legal positions from the Hanan grid (positions flush with neighbor
edges). Accept any move that improves the EXACT proxy cost. Repeat
until no improvement in a round, or the time budget runs out.

This is a deterministic counterpart to SA polish:
  - No temperature, no Metropolis — strict greedy: only accept
    improvements.
  - No random gaussian proposals — deterministic enumeration of
    structured candidate positions.
  - Lower overhead per move because candidates are bounded (~K per
    macro, K from k_neighbors).
  - No teleport / no swap; only displace to a candidate slot.

WHY THIS HELPS AFTER LEGALIZATION
==================================
The legalizer minimizes DISPLACEMENT, not proxy. Often a Hanan candidate
just one neighbor over has lower proxy than the legalizer's nearest
choice. This hill climb catches those cases.

DESIGN
======
- Reuses legalizer.py's candidate generators (_hanan_xs/_ys) and
  legality check (_legal_at).
- Uses utils.compute_proxy_cost_incremental for sub-ms per-move
  evaluation (orders of magnitude faster than full proxy recompute).
- Tracks plc state via _set_placement_fast_moved.
- "Revert after each candidate, reapply best at end" pattern: simpler
  state management than tracking the running winner.

USAGE
=====
    polished = hill_climb(legal_pos, benchmark, plc,
                          max_rounds=5, time_budget_seconds=60)
"""

import time
from pathlib import Path
import sys

import numpy as np
import torch

from macro_place.benchmark import Benchmark

sys.path.insert(0, str(Path(__file__).parent))
from utils import (  # noqa: E402
    _set_placement_fast,
    _set_placement_fast_moved,
    compute_proxy_cost_incremental,
    validate_placement,
)
from legalizer import (  # noqa: E402
    _hanan_xs, _hanan_ys, _nearest_neighbors, _legal_at,
    GAP,
)


def hill_climb(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc,
    max_rounds: int = 5,
    time_budget_seconds: float = 60.0,
    k_neighbors: int = 10,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Greedy Hanan-grid hill climb on a legal placement.

    Args:
        placement:           [num_macros, 2] LEGAL input placement.
        benchmark:           the Benchmark.
        plc:                 PlacementCostAccelerated (already loaded
                             for this benchmark).
        max_rounds:          maximum hill-climb rounds. A round = one
                             pass over all movable macros.
        time_budget_seconds: wall-clock budget. Stops mid-round when hit.
        k_neighbors:         k for Hanan-grid neighbor selection. Smaller
                             = faster, fewer candidates per macro.
        verbose:             print per-round summary.

    Returns:
        torch.Tensor [num_macros, 2] same-or-better legal placement.
    """
    pos = placement.clone().cpu()
    sizes_np = benchmark.macro_sizes.numpy().astype(np.float64)

    hard_mask = benchmark.get_hard_macro_mask().numpy()
    movable_mask = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).numpy()
    hard_ids = np.where(hard_mask)[0]
    movable_ids = np.where(movable_mask)[0]
    if len(movable_ids) == 0:
        if verbose:
            print("[hill_climb] no movable hard macros — returning input unchanged")
        return pos

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    # Numpy mirror of pos for cheap indexed reads inside the candidate
    # generators (which expect numpy arrays).
    pos_np = pos.numpy().astype(np.float64)

    # Sync plc state and prime incremental cost caches. The first call
    # with moved_slots=None forces a full rebuild.
    _set_placement_fast(plc, pos, benchmark)
    pc = compute_proxy_cost_incremental(pos, benchmark, plc, moved_slots=None)
    initial_score = float(pc["proxy_cost"])
    initial_overlaps = pc["overlap_metrics"]["overlap_count"]

    if initial_overlaps > 0:
        if verbose:
            print(f"[hill_climb] WARNING: input has {initial_overlaps} overlaps; "
                  f"hill_climb requires legal input. Returning unchanged.")
        return pos

    current_score = initial_score

    if verbose:
        print(f"[hill_climb] start: score={initial_score:.4f}, "
              f"{len(movable_ids)} movable, max_rounds={max_rounds}, "
              f"budget={time_budget_seconds:.0f}s, k_neighbors={k_neighbors}")

    start_time = time.time()
    total_visited = 0
    total_improvements = 0

    for round_idx in range(max_rounds):
        if time.time() - start_time >= time_budget_seconds:
            if verbose:
                print(f"[hill_climb] time budget hit before round {round_idx}")
            break

        round_improvements = 0
        round_delta = 0.0
        round_visited = 0

        # Random order each round so we don't bias toward early-id macros.
        # Larger-impact macros tend to benefit from being processed first,
        # but randomization helps avoid local-minimum traps.
        order = list(movable_ids)
        np.random.shuffle(order)

        for macro_idx in order:
            if time.time() - start_time >= time_budget_seconds:
                break
            round_visited += 1
            total_visited += 1

            old_x = float(pos_np[macro_idx, 0])
            old_y = float(pos_np[macro_idx, 1])

            # Build Hanan candidates from k nearest hard-macro neighbors.
            neighbor_ids = _nearest_neighbors(macro_idx, pos_np, hard_ids, k_neighbors)
            xs = _hanan_xs(macro_idx, pos_np, sizes_np, neighbor_ids, canvas_w, GAP)
            ys = _hanan_ys(macro_idx, pos_np, sizes_np, neighbor_ids, canvas_h, GAP)

            best_cand = None
            best_score = current_score

            for cand_x in xs:
                for cand_y in ys:
                    # Skip identity-move
                    if cand_x == old_x and cand_y == old_y:
                        continue
                    # Skip illegal candidates (cheap check before cost
                    # evaluation, which is the expensive part)
                    if not _legal_at(macro_idx, cand_x, cand_y,
                                     pos_np, sizes_np, hard_ids, GAP):
                        continue

                    # Apply trial move.
                    pos[macro_idx, 0] = cand_x
                    pos[macro_idx, 1] = cand_y
                    _set_placement_fast_moved(plc, [macro_idx], [(cand_x, cand_y)])

                    trial_pc = compute_proxy_cost_incremental(
                        pos, benchmark, plc, moved_slots=[macro_idx],
                    )
                    # Defensive: should always be 0 since we passed _legal_at,
                    # but the proxy threshold differs slightly from GAP.
                    if trial_pc["overlap_metrics"]["overlap_count"] == 0:
                        trial_score = float(trial_pc["proxy_cost"])
                        if trial_score < best_score:
                            best_score = trial_score
                            best_cand = (cand_x, cand_y)

                    # Revert. Always — we'll reapply the best at the end.
                    pos[macro_idx, 0] = old_x
                    pos[macro_idx, 1] = old_y
                    _set_placement_fast_moved(plc, [macro_idx], [(old_x, old_y)])
                    compute_proxy_cost_incremental(
                        pos, benchmark, plc, moved_slots=[macro_idx],
                    )

            # If a winner was found, commit it.
            if best_cand is not None:
                pos[macro_idx, 0] = best_cand[0]
                pos[macro_idx, 1] = best_cand[1]
                pos_np[macro_idx, 0] = best_cand[0]
                pos_np[macro_idx, 1] = best_cand[1]
                _set_placement_fast_moved(plc, [macro_idx], [best_cand])
                compute_proxy_cost_incremental(
                    pos, benchmark, plc, moved_slots=[macro_idx],
                )
                round_delta += (current_score - best_score)
                current_score = best_score
                round_improvements += 1
                total_improvements += 1

        if verbose:
            print(f"[hill_climb] round {round_idx}: "
                  f"visited={round_visited}, improvements={round_improvements}, "
                  f"score={current_score:.4f}, "
                  f"delta this round={-round_delta:+.4f}")

        if round_improvements == 0:
            if verbose:
                print(f"[hill_climb] converged at round {round_idx} "
                      f"(no improvements found in a full pass)")
            break

    elapsed = time.time() - start_time
    if verbose:
        print(f"[hill_climb] done: {elapsed:.1f}s, "
              f"{total_visited} macros visited, "
              f"{total_improvements} improvements, "
              f"score: {initial_score:.4f} -> {current_score:.4f} "
              f"({current_score - initial_score:+.4f})")

    # Paranoia: confirm output is still legal under validate_placement.
    is_valid, violations = validate_placement(pos, benchmark)
    if not is_valid:
        if verbose:
            print(f"[hill_climb] WARNING: output failed validate_placement: "
                  f"{violations[:3]}; falling back to input")
        return placement.clone().cpu()

    return pos
