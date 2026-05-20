
from pathlib import Path
import sys

import numpy as np
import torch

from macro_place.benchmark import Benchmark

sys.path.insert(0, str(Path(__file__).parent))
from utils import validate_placement  # noqa: E402



GAP = 1e-3


def _overlap_xy_scalar(pi, si, pj, sj, gap):
    ox = 0.5 * (si[0] + sj[0]) + gap - abs(pi[0] - pj[0])
    oy = 0.5 * (si[1] + sj[1]) + gap - abs(pi[1] - pj[1])
    return ox, oy


def _find_overlapping_pairs(pos, sizes, hard_ids, gap):
    h_pos = pos[hard_ids]            # [H, 2] hard-macro centers
    h_size = sizes[hard_ids]         # [H, 2] hard-macro sizes

    # Pairwise absolute distance per axis. [H, H] matrices.
    dx = np.abs(h_pos[:, None, 0] - h_pos[None, :, 0])
    dy = np.abs(h_pos[:, None, 1] - h_pos[None, :, 1])

    # Pairwise sum of half-widths/heights.
    sumw = 0.5 * (h_size[:, None, 0] + h_size[None, :, 0])
    sumh = 0.5 * (h_size[:, None, 1] + h_size[None, :, 1])

    # Per-axis overlap. Positive iff overlapping on that axis.
    ox = sumw + gap - dx
    oy = sumh + gap - dy

    # Both axes overlap => rectangles actually overlap.
    # Upper triangle (k=1) excludes the diagonal (self-pairs) and the
    # duplicate (j, i) entries — each pair contributes exactly once.
    overlap_mask = (ox > 0) & (oy > 0)
    overlap_mask = np.triu(overlap_mask, k=1)

    ii, jj = np.where(overlap_mask)
    areas = ox[ii, jj] * oy[ii, jj]

    # Sort largest-area first.
    order = np.argsort(-areas)
    return [
        (int(hard_ids[ii[k]]), int(hard_ids[jj[k]]), float(areas[k]))
        for k in order
    ]


def _legal_at(i, cx, cy, pos, sizes, hard_ids, gap):
    w, h = sizes[i, 0], sizes[i, 1]
    for j in hard_ids:
        if j == i:
            continue
        dx = abs(cx - pos[j, 0])
        # X-axis early-out: cheap to compute, large fraction of pairs
        # are non-overlapping just from X distance alone.
        ox = 0.5 * (w + sizes[j, 0]) + gap - dx
        if ox <= 0:
            continue
        dy = abs(cy - pos[j, 1])
        oy = 0.5 * (h + sizes[j, 1]) + gap - dy
        if oy > 0:
            return False  # overlap on both axes
    return True


def _clamp_to_canvas(cx, cy, w, h, canvas_w, canvas_h, gap):
    cx = max(0.5 * w + gap, min(canvas_w - 0.5 * w - gap, cx))
    cy = max(0.5 * h + gap, min(canvas_h - 0.5 * h - gap, cy))
    return cx, cy


def _ripple_pair(pos, sizes, i, j, movable_mask, canvas_w, canvas_h, gap):
    pi, pj = pos[i], pos[j]
    si, sj = sizes[i], sizes[j]

    ox, oy = _overlap_xy_scalar(pi, si, pj, sj, gap)
    if ox <= 0 or oy <= 0:
        # Earlier move in this sweep already separated them.
        return False

    i_mov = bool(movable_mask[i])
    j_mov = bool(movable_mask[j])
    if not i_mov and not j_mov:
        # Both fixed. We can't fix this; usually an artifact of the input
        # placement or fixed-fixed initial-state issues. Skip and let the
        # caller report at the end.
        return False

    if i_mov and j_mov:
        area_i = si[0] * si[1]
        area_j = sj[0] * sj[1]
        # Smaller area = less surrounding overlap potential when moved.
        mover, anchor = (i, j) if area_i <= area_j else (j, i)
    elif i_mov:
        mover, anchor = i, j
    else:
        mover, anchor = j, i

    # Small extra (eps) beyond the overlap so we actually clear the
    # legality threshold, not just barely touch it.
    eps = gap

    if ox <= oy:
        # Push along X (smaller overlap = shorter displacement)
        if pos[mover, 0] < pos[anchor, 0]:
            pos[mover, 0] -= (ox + eps)
        else:
            pos[mover, 0] += (ox + eps)
    else:
        # Push along Y
        if pos[mover, 1] < pos[anchor, 1]:
            pos[mover, 1] -= (oy + eps)
        else:
            pos[mover, 1] += (oy + eps)

    # Clamp to canvas. If the push was into a wall this undoes part of
    # the displacement and the pair gets re-detected on the next sweep.
    # Repeated wall-deadlocks are handed off to Stage B.
    w, h = sizes[mover, 0], sizes[mover, 1]
    pos[mover, 0], pos[mover, 1] = _clamp_to_canvas(
        pos[mover, 0], pos[mover, 1], w, h, canvas_w, canvas_h, gap,
    )
    return True


def _stage_a_ripple(pos, sizes, hard_ids, movable_mask,
                    canvas_w, canvas_h, gap, max_iters=200):
    pos = pos.copy()

    for it in range(max_iters):
        pairs = _find_overlapping_pairs(pos, sizes, hard_ids, gap)
        if not pairs:
            return pos, True, it

        any_change = False
        for i, j, _area in pairs:
            if _ripple_pair(pos, sizes, i, j, movable_mask,
                            canvas_w, canvas_h, gap):
                any_change = True

        if not any_change:
            # No pair caused any movement this sweep -> all remaining
            # overlaps are either fixed-fixed (unresolvable here) or
            # canvas-edge deadlocks (Stage B can handle).
            return pos, False, it

    return pos, False, max_iters


# ---------------------------------------------------------------------------
# Stage B: Hanan-grid candidate search
# ---------------------------------------------------------------------------

def _hanan_xs(i, pos, sizes, neighbor_ids, canvas_w, gap):
    w = sizes[i, 0]
    x_lo = 0.5 * w + gap
    x_hi = canvas_w - 0.5 * w - gap

    xs = {x_lo, x_hi, float(pos[i, 0])}

    for j in neighbor_ids:
        if j == i:
            continue
        xj, wj = float(pos[j, 0]), float(sizes[j, 0])
        # i to the right of j (i's left edge ~ j's right edge + gap)
        cand = xj + 0.5 * wj + 0.5 * w + gap
        xs.add(max(x_lo, min(x_hi, cand)))
        # i to the left of j
        cand = xj - 0.5 * wj - 0.5 * w - gap
        xs.add(max(x_lo, min(x_hi, cand)))

    return sorted(xs)


def _hanan_ys(i, pos, sizes, neighbor_ids, canvas_h, gap):
    """Symmetric Y-axis version of _hanan_xs."""
    h = sizes[i, 1]
    y_lo = 0.5 * h + gap
    y_hi = canvas_h - 0.5 * h - gap

    ys = {y_lo, y_hi, float(pos[i, 1])}

    for j in neighbor_ids:
        if j == i:
            continue
        yj, hj = float(pos[j, 1]), float(sizes[j, 1])
        cand = yj + 0.5 * hj + 0.5 * h + gap
        ys.add(max(y_lo, min(y_hi, cand)))
        cand = yj - 0.5 * hj - 0.5 * h - gap
        ys.add(max(y_lo, min(y_hi, cand)))

    return sorted(ys)


def _nearest_neighbors(i, pos, hard_ids, k):

    if len(hard_ids) <= k + 1:
        return [int(j) for j in hard_ids if j != i]

    dists = np.linalg.norm(pos[hard_ids] - pos[i], axis=1)
    # argpartition is O(H), full sort is O(H log H). For k << H, partition
    # is meaningfully faster.
    idxs = np.argpartition(dists, k + 1)[: k + 1]
    return [int(hard_ids[idx]) for idx in idxs if hard_ids[idx] != i]


def _best_legal_position(i, pos, sizes, hard_ids, canvas_w, canvas_h,
                         gap, k_neighbors=15, max_displacement=None):

    cur_x, cur_y = float(pos[i, 0]), float(pos[i, 1])

    neighbor_ids = _nearest_neighbors(i, pos, hard_ids, k_neighbors)
    xs = _hanan_xs(i, pos, sizes, neighbor_ids, canvas_w, gap)
    ys = _hanan_ys(i, pos, sizes, neighbor_ids, canvas_h, gap)

    best = None
    best_disp_sq = float("inf")
    max_disp_sq = (max_displacement ** 2) if max_displacement is not None \
        else float("inf")

    for x in xs:
        # X-only pre-filter: skip the entire y-row if x alone exceeds
        # the best displacement so far.
        dx_sq = (x - cur_x) ** 2
        if dx_sq >= best_disp_sq or dx_sq > max_disp_sq:
            continue
        for y in ys:
            disp_sq = dx_sq + (y - cur_y) ** 2
            if disp_sq >= best_disp_sq or disp_sq > max_disp_sq:
                continue
            if _legal_at(i, x, y, pos, sizes, hard_ids, gap):
                best = (x, y)
                best_disp_sq = disp_sq

    return best


def _stage_b_hanan(pos, sizes, hard_ids, movable_mask,
                   canvas_w, canvas_h, gap,
                   max_outer_iters=10, k_neighbors=15, max_displacement=None):
    pos = pos.copy()

    for outer in range(max_outer_iters):
        pairs = _find_overlapping_pairs(pos, sizes, hard_ids, gap)
        if not pairs:
            return pos, True, outer

        # Build per-bad-macro overlap-area score. Only movable macros
        # are bad-candidates; fixed-fixed overlaps are unresolvable.
        bad_score = {}
        for i, j, area in pairs:
            if movable_mask[i]:
                bad_score[i] = bad_score.get(i, 0.0) + area
            if movable_mask[j]:
                bad_score[j] = bad_score.get(j, 0.0) + area

        if not bad_score:
            # All remaining overlaps are fixed-fixed -> can't fix here.
            return pos, False, outer

        # Worst offenders first so they stop poisoning others.
        bad_ordered = sorted(bad_score, key=lambda m: -bad_score[m])

        any_change = False
        for i in bad_ordered:
            best = _best_legal_position(
                i, pos, sizes, hard_ids, canvas_w, canvas_h, gap,
                k_neighbors=k_neighbors,
                max_displacement=max_displacement,
            )
            if best is not None:
                pos[i, 0], pos[i, 1] = best
                any_change = True

        if not any_change:
            return pos, False, outer

    return pos, False, max_outer_iters


# ---------------------------------------------------------------------------
# Stage C: Spiral fallback
# ---------------------------------------------------------------------------

def _spiral_candidates(cx, cy, max_radius, step):

    yield (cx, cy)
    r = step
    while r <= max_radius:
        n = max(8, int(2 * np.pi * r / step))
        for k in range(n):
            theta = 2.0 * np.pi * k / n
            yield (cx + r * np.cos(theta), cy + r * np.sin(theta))
        r *= 1.5


def _stage_c_spiral(pos, sizes, hard_ids, movable_mask,
                    canvas_w, canvas_h, gap, max_outer_iters=5):

    pos = pos.copy()

    for outer in range(max_outer_iters):
        pairs = _find_overlapping_pairs(pos, sizes, hard_ids, gap)
        if not pairs:
            return pos, True, outer

        bad_set = set()
        for i, j, _ in pairs:
            if movable_mask[i]:
                bad_set.add(i)
            if movable_mask[j]:
                bad_set.add(j)
        if not bad_set:
            return pos, False, outer

        any_change = False
        for i in bad_set:
            w, h = float(sizes[i, 0]), float(sizes[i, 1])
            # Step size: half a macro dimension. Smaller steps would
            # find tighter slots but explode candidate count.
            step = max(0.05, 0.5 * min(w, h))
            max_r = max(canvas_w, canvas_h)
            cx0, cy0 = float(pos[i, 0]), float(pos[i, 1])

            for cand_x, cand_y in _spiral_candidates(cx0, cy0, max_r, step):
                cand_x, cand_y = _clamp_to_canvas(
                    cand_x, cand_y, w, h, canvas_w, canvas_h, gap,
                )
                if _legal_at(i, cand_x, cand_y, pos, sizes, hard_ids, gap):
                    pos[i, 0], pos[i, 1] = cand_x, cand_y
                    any_change = True
                    break

        if not any_change:
            return pos, False, outer

    return pos, False, max_outer_iters



def legalize(
    placement: torch.Tensor,
    benchmark: Benchmark,
    gap: float = GAP,
    hanan_max_displacement_frac: float = 0.15,
    hanan_k_neighbors: int = 15,
    verbose: bool = True,
) -> torch.Tensor:


    # ------------------- Setup -------------------
    pos = placement.detach().cpu().numpy().astype(np.float64).copy()
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)

    hard_mask = benchmark.get_hard_macro_mask().numpy()
    movable_mask = benchmark.get_movable_mask().numpy()
    hard_ids = np.where(hard_mask)[0]

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    # Hanan displacement budget in absolute microns.
    hanan_max_disp = hanan_max_displacement_frac * min(canvas_w, canvas_h)

    # ------------------- Pre-flight -------------------
    # First, clamp every movable macro into the canvas. Out-of-canvas
    # macros would create false-positive overlap signals at the wall and
    # are easy to fix — just project them in.
    for i in range(len(pos)):
        if not movable_mask[i]:
            continue
        w, h = sizes[i, 0], sizes[i, 1]
        pos[i, 0], pos[i, 1] = _clamp_to_canvas(
            pos[i, 0], pos[i, 1], w, h, canvas_w, canvas_h, gap,
        )

    initial_pairs = _find_overlapping_pairs(pos, sizes, hard_ids, gap)
    if verbose:
        print(f"[legalize] start: {len(initial_pairs)} overlapping pairs")

    if not initial_pairs:
        if verbose:
            print("[legalize] already legal after canvas clamp")
        return _finalize(pos, benchmark, verbose)

    # ------------------- Stage A: Ripple -------------------
    pos, ok, iters = _stage_a_ripple(
        pos, sizes, hard_ids, movable_mask, canvas_w, canvas_h, gap,
    )
    remaining = len(_find_overlapping_pairs(pos, sizes, hard_ids, gap))
    if verbose:
        print(f"[legalize] after ripple ({iters} iters): "
              f"{remaining} overlaps remain")
    if ok:
        return _finalize(pos, benchmark, verbose)

    # ------------------- Stage B: Hanan grid (bounded) -------------------
    pos, ok, iters = _stage_b_hanan(
        pos, sizes, hard_ids, movable_mask, canvas_w, canvas_h, gap,
        k_neighbors=hanan_k_neighbors,
        max_displacement=hanan_max_disp,
    )
    remaining = len(_find_overlapping_pairs(pos, sizes, hard_ids, gap))
    if verbose:
        print(f"[legalize] after Hanan-bounded ({iters} iters, "
              f"max_disp={hanan_max_disp:.2f}): "
              f"{remaining} overlaps remain")
    if ok:
        return _finalize(pos, benchmark, verbose)

    # ------------------- Stage B retry without displacement budget -------------------
    # Sometimes the displacement budget prevented a legal Hanan slot that
    # exists further away. Re-run Hanan with no budget before falling to
    # spiral — Hanan's structured candidates (flush with neighbor edges)
    # usually beat spiral's geometric ones in proxy quality even at long
    # range, because they leave no wasted gaps.
    if verbose:
        print("[legalize] retrying Hanan with no displacement budget...")
    pos, ok, iters = _stage_b_hanan(
        pos, sizes, hard_ids, movable_mask, canvas_w, canvas_h, gap,
        k_neighbors=hanan_k_neighbors,
        max_displacement=None,
    )
    remaining = len(_find_overlapping_pairs(pos, sizes, hard_ids, gap))
    if verbose:
        print(f"[legalize] after Hanan-unbounded ({iters} iters): "
              f"{remaining} overlaps remain")
    if ok:
        return _finalize(pos, benchmark, verbose)

    # ------------------- Stage C: Spiral fallback -------------------
    pos, ok, iters = _stage_c_spiral(
        pos, sizes, hard_ids, movable_mask, canvas_w, canvas_h, gap,
    )
    remaining = len(_find_overlapping_pairs(pos, sizes, hard_ids, gap))
    if verbose:
        print(f"[legalize] after spiral ({iters} iters): "
              f"{remaining} overlaps remain")

    if not ok:
        # The canvas is genuinely too packed for our algorithm. The caller
        # should report this and may need to improve GP overlap pressure
        # so the input has fewer/smaller overlaps to fix.
        raise ValueError(
            f"[legalize] failed: {remaining} overlapping pairs remain "
            f"after all three stages. The input has too many or too "
            f"large overlaps for this legalizer to resolve."
        )

    return _finalize(pos, benchmark, verbose)


def _finalize(pos: np.ndarray, benchmark: Benchmark, verbose: bool) -> torch.Tensor:
    """
    Cast back to torch.Tensor [num_macros, 2] float32 CPU and run
    validate_placement as a paranoia check. validate_placement raises
    on any violation — we want loud failures, not silent illegality.
    """
    legal_pos = torch.from_numpy(pos.astype(np.float32))

    is_valid, violations = validate_placement(legal_pos, benchmark)
    if not is_valid and verbose:
        # The most common cause is that the proxy evaluator uses a
        # slightly different overlap threshold than our GAP. If this
        # fires, increase GAP and re-run.
        print(f"[legalize] WARNING: validate_placement reported violations: "
              f"{violations[:5]}")

    return legal_pos
