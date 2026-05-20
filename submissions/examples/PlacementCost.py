"""Open-Sourced PlacementCost client class."""
from ast import Assert
import os, io
import re
import math
from typing import Text, Tuple, overload
from absl import logging
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import traceback, sys
import random
from collections import defaultdict
import numba

"""
    Numpy/numba accelerated version of the array

    there is a dict of idx -> name for getting it back
    
    then, there is a connectivity list stored in the CSR format, where we have:
    
    [net1_sink1, net1_sink2, net1_sink3, net2_sink1, net2_sink2, ...] 
    and pointer list [0, 3, 5, ...]

    then there are two position numpy arrays 
    [net1_x, net2_x, ...]
    [net1_y, net2_y, ...]

"""

Block = namedtuple('Block', 'x_max y_max x_min y_min')

@numba.njit(cache=True, fastmath=True, inline="always")
def _sort3_by_col(rl, cl, rm, cm, rh, ch):
    #ascending sort network on (col, row)
    
    if cl > cm or (cl == cm and rl > rm): rl, rm, cl, cm = rm, rl, cm, cl 
    if cl > ch or (cl == ch and rl > rh): rl, rh, cl, ch = rh, rl, ch, cl
    if cm > ch or (cm == ch and rm > rh): rm, rh, cm, ch = rh, rm, ch, cm
    
    return rl, cl, rm, cm, rh, ch

@numba.njit(cache=True, fastmath=True)
def __t_routing(col_lo, col_mid, col_hi, row_lo, row_mid, row_hi, weight, H, V, grid_cols):

    # execute the node_gcells.sort() 
    col_lo, row_lo, col_mid, row_mid, col_hi, row_hi = _sort3_by_col(col_lo, row_lo, col_mid, row_mid, col_hi, row_hi)
    

    xmin = min(col_lo, col_mid, col_hi)
    xmax = max(col_lo, col_mid, col_hi)
    
    # H routing (xmin, y2) to (xmax, y2)
    for col in range(xmin, xmax):
        row = row_mid 
        H[row * grid_cols + col] += weight
        
    # V routing (x1, y1) to (x1, y2)
    for row in range(min(row_lo, row_mid), max(row_lo, row_mid)):
        col = col_lo 
        V[row * grid_cols + col] += weight

    # V routing (x3, y3) to (x3, y2)
    for row in range(min(row_mid, row_hi), max(row_mid, row_hi)):
        col = col_hi 
        V[row * grid_cols + col] += weight

@numba.njit(cache=True, fastmath=True, inline="always")
def _l_route_driver_to_sink(drow, dcol, sr, sc, w, H, V, grid_cols):
    """Single L-route: horizontal leg at driver's row, vertical leg at sink's col."""
    if dcol <= sc:
        c_lo, c_hi = dcol, sc
    else:
        c_lo, c_hi = sc, dcol
    for c in range(c_lo, c_hi):
        H[drow * grid_cols + c] += w

    if drow <= sr:
        r_lo, r_hi = drow, sr
    else:
        r_lo, r_hi = sr, drow
    for r in range(r_lo, r_hi):
        V[r * grid_cols + sc] += w

@numba.njit(cache=True, fastmath=True)
def accumulate_net_routing(pin_row: np.ndarray,
                           pin_col: np.ndarray,
                           driver_pin: np.ndarray,
                           sink_offsets: np.ndarray,
                           sink_pinks: np.ndarray,
                           weights: np.ndarray,
                           H: np.ndarray,
                           V: np.ndarray,
                           grid_cols: int):
    # Pre-size scratch buffers once (numba hoists this outside the per-net loop).
    # Large enough for the biggest net in this benchmark.
    max_sinks = 0
    for n in range(driver_pin.shape[0]):
        k = sink_offsets[n + 1] - sink_offsets[n]
        if k > max_sinks:
            max_sinks = k
    uniq_r = np.empty(max_sinks + 1, dtype=np.int32)
    uniq_c = np.empty(max_sinks + 1, dtype=np.int32)

    for n in range(driver_pin.shape[0]):
        dpin = driver_pin[n]
        drow = pin_row[dpin]
        dcol = pin_col[dpin]
        w = weights[n]
        s0 = sink_offsets[n]
        s1 = sink_offsets[n + 1]

        # Build list of unique sink gcells, excluding any sink that lands in
        # the driver's own cell (matches the reference's set semantics, which
        # treats the driver and a colocated sink as one element).
        k = 0
        for s in range(s0, s1):
            sp = sink_pinks[s]
            sr = pin_row[sp]
            sc = pin_col[sp]
            if sr == drow and sc == dcol:
                continue
            dup = False
            for j in range(k):
                if uniq_r[j] == sr and uniq_c[j] == sc:
                    dup = True
                    break
            if not dup:
                uniq_r[k] = sr
                uniq_c[k] = sc
                k += 1

        # Branch on number of UNIQUE non-driver cells, matching reference semantics.
        #   k == 0  → all pins in driver's cell, nothing to route
        #   k == 1  → 2 unique cells total, one L-route
        #   k == 2  → 3 unique cells total, 3-pin T/L special
        #   k >= 3  → split into driver→each-unique-sink L-routes
        if k == 0:
            continue

        if k == 1:
            _l_route_driver_to_sink(drow, dcol, uniq_r[0], uniq_c[0], w, H, V, grid_cols)
            continue

        if k == 2:
            row_lo, col_lo = drow, dcol
            row_mid, col_mid = uniq_r[0], uniq_c[0]
            row_hi, col_hi = uniq_r[1], uniq_c[1]

            row_lo, col_lo, row_mid, col_mid, row_hi, col_hi = _sort3_by_col(
                row_lo, col_lo, row_mid, col_mid, row_hi, col_hi
            )

            # L-routing: x1 < x2 < x3 and min(y1,y3) < y2 < max(y1,y3)
            if (col_lo < col_mid and col_mid < col_hi
                    and min(row_lo, row_hi) < row_mid
                    and max(row_lo, row_hi) > row_mid):
                for col in range(col_lo, col_mid):
                    H[row_lo * grid_cols + col] += w
                for col in range(col_mid, col_hi):
                    H[row_mid * grid_cols + col] += w
                for row in range(min(row_lo, row_mid), max(row_lo, row_mid)):
                    V[row * grid_cols + col_mid] += w
                for row in range(min(row_mid, row_hi), max(row_mid, row_hi)):
                    V[row * grid_cols + col_hi] += w

            # S-shape: x2 == x3 and x1 < x2 and y1 < min(y2, y3)
            elif col_mid == col_hi and col_lo < col_mid and row_lo < min(row_mid, row_hi):
                for col in range(col_lo, col_mid):
                    H[row_lo * grid_cols + col] += w
                for row in range(row_lo, max(row_mid, row_hi)):
                    V[row * grid_cols + col_mid] += w

            # horizontal extension: y2 == y3
            elif row_mid == row_hi:
                for col in range(col_lo, col_mid):
                    H[row_lo * grid_cols + col] += w
                for col in range(col_mid, col_hi):
                    H[row_mid * grid_cols + col] += w
                for row in range(min(row_lo, row_hi), max(row_lo, row_hi)):
                    V[row * grid_cols + col_mid] += w

            # fallback: T-routing
            else:
                __t_routing(col_lo, col_mid, col_hi, row_lo, row_mid, row_hi, w, H, V, grid_cols)

            continue

        # k >= 3: driver → each unique sink L-route
        for j in range(k):
            _l_route_driver_to_sink(drow, dcol, uniq_r[j], uniq_c[j], w, H, V, grid_cols)


@numba.njit(cache=True, fastmath=True)
def accumulate_net_routing_subset(affected_nets: np.ndarray,
                                  pin_row: np.ndarray,
                                  pin_col: np.ndarray,
                                  driver_pin: np.ndarray,
                                  sink_offsets: np.ndarray,
                                  sink_pinks: np.ndarray,
                                  weights: np.ndarray,
                                  H: np.ndarray,
                                  V: np.ndarray,
                                  grid_cols: int,
                                  sign: float,
                                  max_sinks_cap: int):
    """
    Same routing semantics as accumulate_net_routing, but:
      * Only processes net indices in `affected_nets`.
      * Each accumulation into H/V is multiplied by `sign`
        (use +1.0 to add, -1.0 to subtract).
      * `max_sinks_cap` is the allocation hint for the local dedup scratch
        (pre-computed once at build time by the caller).
    """
    uniq_r = np.empty(max_sinks_cap + 1, dtype=np.int32)
    uniq_c = np.empty(max_sinks_cap + 1, dtype=np.int32)

    for ai in range(affected_nets.shape[0]):
        n = affected_nets[ai]
        dpin = driver_pin[n]
        drow = pin_row[dpin]
        dcol = pin_col[dpin]
        w = weights[n] * sign
        s0 = sink_offsets[n]
        s1 = sink_offsets[n + 1]

        k = 0
        for s in range(s0, s1):
            sp = sink_pinks[s]
            sr = pin_row[sp]
            sc = pin_col[sp]
            if sr == drow and sc == dcol:
                continue
            dup = False
            for j in range(k):
                if uniq_r[j] == sr and uniq_c[j] == sc:
                    dup = True
                    break
            if not dup:
                uniq_r[k] = sr
                uniq_c[k] = sc
                k += 1

        if k == 0:
            continue

        if k == 1:
            _l_route_driver_to_sink(drow, dcol, uniq_r[0], uniq_c[0], w, H, V, grid_cols)
            continue

        if k == 2:
            row_lo, col_lo = drow, dcol
            row_mid, col_mid = uniq_r[0], uniq_c[0]
            row_hi, col_hi = uniq_r[1], uniq_c[1]

            row_lo, col_lo, row_mid, col_mid, row_hi, col_hi = _sort3_by_col(
                row_lo, col_lo, row_mid, col_mid, row_hi, col_hi
            )

            if (col_lo < col_mid and col_mid < col_hi
                    and min(row_lo, row_hi) < row_mid
                    and max(row_lo, row_hi) > row_mid):
                for col in range(col_lo, col_mid):
                    H[row_lo * grid_cols + col] += w
                for col in range(col_mid, col_hi):
                    H[row_mid * grid_cols + col] += w
                for row in range(min(row_lo, row_mid), max(row_lo, row_mid)):
                    V[row * grid_cols + col_mid] += w
                for row in range(min(row_mid, row_hi), max(row_mid, row_hi)):
                    V[row * grid_cols + col_hi] += w

            elif col_mid == col_hi and col_lo < col_mid and row_lo < min(row_mid, row_hi):
                for col in range(col_lo, col_mid):
                    H[row_lo * grid_cols + col] += w
                for row in range(row_lo, max(row_mid, row_hi)):
                    V[row * grid_cols + col_mid] += w

            elif row_mid == row_hi:
                for col in range(col_lo, col_mid):
                    H[row_lo * grid_cols + col] += w
                for col in range(col_mid, col_hi):
                    H[row_mid * grid_cols + col] += w
                for row in range(min(row_lo, row_hi), max(row_lo, row_hi)):
                    V[row * grid_cols + col_mid] += w

            else:
                __t_routing(col_lo, col_mid, col_hi, row_lo, row_mid, row_hi, w, H, V, grid_cols)

            continue

        for j in range(k):
            _l_route_driver_to_sink(drow, dcol, uniq_r[j], uniq_c[j], w, H, V, grid_cols)


@numba.njit(cache=True, fastmath=True)
def accumulate_macro_blockage(macro_x, macro_y, macro_w, macro_h,
                              grid_w, grid_h, grid_rows, grid_cols,
                              hrouting_alloc, vrouting_alloc, H_macro, V_macro):

    EPS = 1e-5

    for i in range(len(macro_x)):
        mx_lo = macro_x[i] - macro_w[i] / 2
        mx_hi = macro_x[i] + macro_w[i] / 2
        my_lo = macro_y[i] - macro_h[i] / 2
        my_hi = macro_y[i] + macro_h[i] / 2

        # macro bounding box in grid coords (clamped)
        r_lo = max(0, int(my_lo / grid_h))
        r_hi = min(grid_rows - 1, int(my_hi / grid_h))
        c_lo = max(0, int(mx_lo / grid_w))
        c_hi = min(grid_cols - 1, int(mx_hi / grid_w))

        partial_v = False   # at least one boundary row has oy != grid_h
        partial_h = False   # at least one boundary col has ox != grid_w

        # ---- main accumulation pass ----
        for r in range(r_lo, r_hi + 1):
            cell_y_lo = r * grid_h
            cell_y_hi = (r + 1) * grid_h
            oy = min(cell_y_hi, my_hi) - max(cell_y_lo, my_lo)

            if r_hi != r_lo:
                if (r == r_lo or r == r_hi) and abs(oy - grid_h) > EPS:
                    partial_v = True

            for c in range(c_lo, c_hi + 1):
                cell_x_lo = c * grid_w
                cell_x_hi = (c + 1) * grid_w
                ox = min(cell_x_hi, mx_hi) - max(cell_x_lo, mx_lo)

                if c_hi != c_lo:
                    if (c == c_lo or c == c_hi) and abs(ox - grid_w) > EPS:
                        partial_h = True

                V_macro[r * grid_cols + c] += ox * vrouting_alloc
                H_macro[r * grid_cols + c] += oy * hrouting_alloc

        # ---- partial-overlap correction: zero top row's V contribution ----
        if partial_v:
            r = r_hi
            for c in range(c_lo, c_hi + 1):
                cell_x_lo = c * grid_w
                cell_x_hi = (c + 1) * grid_w
                ox = min(cell_x_hi, mx_hi) - max(cell_x_lo, mx_lo)
                V_macro[r * grid_cols + c] -= ox * vrouting_alloc

        # ---- partial-overlap correction: zero right col's H contribution ----
        if partial_h:
            c = c_hi
            for r in range(r_lo, r_hi + 1):
                cell_y_lo = r * grid_h
                cell_y_hi = (r + 1) * grid_h
                oy = min(cell_y_hi, my_hi) - max(cell_y_lo, my_lo)
                H_macro[r * grid_cols + c] -= oy * hrouting_alloc

@numba.njit(cache=True, fastmath=True)
def smooth_routing_cong_fast(grid_col : int, 
                               grid_row : int,
                               smooth_range : int, 
                               V_routing_cong : np.ndarray, 
                               H_routing_cong : np.ndarray):
    
    temp_V_routing_cong = np.zeros(grid_col * grid_row, dtype=np.float32)
    temp_H_routing_cong = np.zeros(grid_col * grid_row, dtype=np.float32)
    
    for row in range(grid_row):
        for col in range(grid_col):
            lp = max(col - smooth_range, 0)
            rp = min(col + smooth_range, grid_col - 1)
            
            
            gcell_cnt = rp - lp + 1
            
            val = V_routing_cong[row * grid_col + col] / gcell_cnt
            
            for ptr in range(lp, rp + 1, 1):
                temp_V_routing_cong[row * grid_col + ptr] += val

    V_routing_cong[:] = temp_V_routing_cong[:]

    # h routing congestion — smooth along rows (axis 0)
    for row in range(grid_row):
        for col in range(grid_col):

            lp = max(row - smooth_range, 0)
            up = min(row + smooth_range, grid_row - 1)
            
            gcell_cnt = up - lp + 1
            
            val = H_routing_cong[row * grid_col + col] / gcell_cnt
            
            for ptr in range(lp, up + 1, 1):
                temp_H_routing_cong[ptr * grid_col + col] += val
                
    H_routing_cong[:] = temp_H_routing_cong[:]


@numba.njit(cache=True, fastmath=True)
def accumulate_single_macro_blockage(mx: float, my: float, mw: float, mh: float,
                                     grid_w: float, grid_h: float,
                                     grid_rows: int, grid_cols: int,
                                     hrouting_alloc: float, vrouting_alloc: float,
                                     H_macro: np.ndarray, V_macro: np.ndarray,
                                     sign: float):
    """
    Apply (+sign × blockage) for ONE macro. Semantics identical to
    accumulate_macro_blockage for a single macro, including the
    partial-overlap correction pass. Pass sign=+1 to add, -1 to subtract.
    """
    EPS = 1e-5

    mx_lo = mx - mw / 2
    mx_hi = mx + mw / 2
    my_lo = my - mh / 2
    my_hi = my + mh / 2

    r_lo = max(0, int(my_lo / grid_h))
    r_hi = min(grid_rows - 1, int(my_hi / grid_h))
    c_lo = max(0, int(mx_lo / grid_w))
    c_hi = min(grid_cols - 1, int(mx_hi / grid_w))

    partial_v = False
    partial_h = False

    # ---- main accumulation pass ----
    for r in range(r_lo, r_hi + 1):
        cell_y_lo = r * grid_h
        cell_y_hi = (r + 1) * grid_h
        oy = min(cell_y_hi, my_hi) - max(cell_y_lo, my_lo)

        if r_hi != r_lo:
            if (r == r_lo or r == r_hi) and abs(oy - grid_h) > EPS:
                partial_v = True

        for c in range(c_lo, c_hi + 1):
            cell_x_lo = c * grid_w
            cell_x_hi = (c + 1) * grid_w
            ox = min(cell_x_hi, mx_hi) - max(cell_x_lo, mx_lo)

            if c_hi != c_lo:
                if (c == c_lo or c == c_hi) and abs(ox - grid_w) > EPS:
                    partial_h = True

            V_macro[r * grid_cols + c] += sign * ox * vrouting_alloc
            H_macro[r * grid_cols + c] += sign * oy * hrouting_alloc

    # ---- partial-overlap correction: zero top row's V contribution ----
    if partial_v:
        r = r_hi
        for c in range(c_lo, c_hi + 1):
            cell_x_lo = c * grid_w
            cell_x_hi = (c + 1) * grid_w
            ox = min(cell_x_hi, mx_hi) - max(cell_x_lo, mx_lo)
            V_macro[r * grid_cols + c] -= sign * ox * vrouting_alloc

    # ---- partial-overlap correction: zero right col's H contribution ----
    if partial_h:
        c = c_hi
        for r in range(r_lo, r_hi + 1):
            cell_y_lo = r * grid_h
            cell_y_hi = (r + 1) * grid_h
            oy = min(cell_y_hi, my_hi) - max(cell_y_lo, my_lo)
            H_macro[r * grid_cols + c] -= sign * oy * hrouting_alloc


# -------------------------------------------------------------------
# Density kernels.
# Density is the per-grid-cell sum of macro-body area intersection with the
# cell. ABU(10%) of normalized values, then × 0.5 — same semantics as the
# reference get_density_cost. These kernels are shared by the full-rebuild
# and incremental paths.
# -------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def accumulate_single_macro_density(mx: float, my: float, mw: float, mh: float,
                                    grid_w: float, grid_h: float,
                                    grid_rows: int, grid_cols: int,
                                    grid_occupied: np.ndarray, sign: float):
    """Add sign × (macro-cell overlap area) into grid_occupied for every cell
    the macro touches. Same geometry as reference `__add_module_to_grid_cells`;
    out-of-canvas portions are clipped to the grid boundary."""
    mx_lo = mx - mw / 2
    mx_hi = mx + mw / 2
    my_lo = my - mh / 2
    my_hi = my + mh / 2

    # Early-out for macros entirely outside canvas.
    if mx_hi <= 0 or my_hi <= 0:
        return

    # Bbox in grid coords, clipped to grid bounds.
    r_lo = int(my_lo / grid_h)
    r_hi = int(my_hi / grid_h)
    c_lo = int(mx_lo / grid_w)
    c_hi = int(mx_hi / grid_w)
    if r_lo < 0: r_lo = 0
    if c_lo < 0: c_lo = 0
    if r_hi > grid_rows - 1: r_hi = grid_rows - 1
    if c_hi > grid_cols - 1: c_hi = grid_cols - 1
    if r_lo > r_hi or c_lo > c_hi:
        return

    for r in range(r_lo, r_hi + 1):
        cell_y_lo = r * grid_h
        cell_y_hi = (r + 1) * grid_h
        oy = min(cell_y_hi, my_hi) - max(cell_y_lo, my_lo)
        if oy <= 0:
            continue
        for c in range(c_lo, c_hi + 1):
            cell_x_lo = c * grid_w
            cell_x_hi = (c + 1) * grid_w
            ox = min(cell_x_hi, mx_hi) - max(cell_x_lo, mx_lo)
            if ox <= 0:
                continue
            grid_occupied[r * grid_cols + c] += sign * ox * oy


@numba.njit(cache=True, fastmath=True)
def accumulate_all_macros_density(macro_x: np.ndarray, macro_y: np.ndarray,
                                  macro_w: np.ndarray, macro_h: np.ndarray,
                                  grid_w: float, grid_h: float,
                                  grid_rows: int, grid_cols: int,
                                  grid_occupied: np.ndarray):
    """Zero grid_occupied and splat every macro's overlap area into it.
    Used by the full rebuild path (and the fast non-incremental version)."""
    grid_occupied[:] = 0.0
    for i in range(macro_x.shape[0]):
        accumulate_single_macro_density(
            macro_x[i], macro_y[i], macro_w[i], macro_h[i],
            grid_w, grid_h, grid_rows, grid_cols,
            grid_occupied, 1.0,
        )


# -------------------------------------------------------------------
# Incremental HPWL kernels.
# -------------------------------------------------------------------

@numba.njit(cache=True, fastmath=True)
def hpwl_all_nets(x_pos, y_pos,
                  net_driver_pin, net_sink_offsets, net_sink_pins,
                  net_weights, net_hpwl_cache):
    """Recompute per-net HPWL from scratch. Writes into net_hpwl_cache (f32),
    returns the total accumulated in f64 to avoid drift over many ops."""
    total = np.float64(0.0)
    for n in range(net_driver_pin.shape[0]):
        d = net_driver_pin[n]
        xmin = xmax = x_pos[d]
        ymin = ymax = y_pos[d]
        for k in range(net_sink_offsets[n], net_sink_offsets[n + 1]):
            p = net_sink_pins[k]
            x = x_pos[p]; y = y_pos[p]
            if x < xmin: xmin = x
            if x > xmax: xmax = x
            if y < ymin: ymin = y
            if y > ymax: ymax = y
        h = net_weights[n] * ((xmax - xmin) + (ymax - ymin))
        net_hpwl_cache[n] = h
        total += np.float64(h)
    return total


@numba.njit(cache=True, fastmath=True)
def hpwl_update_nets(affected_nets, x_pos, y_pos,
                     net_driver_pin, net_sink_offsets, net_sink_pins,
                     net_weights, net_hpwl_cache):
    """Recompute HPWL for only the given nets. Updates net_hpwl_cache in place
    and returns (new_sum - old_sum) in f64."""
    delta = np.float64(0.0)
    for i in range(affected_nets.shape[0]):
        n = affected_nets[i]
        d = net_driver_pin[n]
        xmin = xmax = x_pos[d]
        ymin = ymax = y_pos[d]
        for k in range(net_sink_offsets[n], net_sink_offsets[n + 1]):
            p = net_sink_pins[k]
            x = x_pos[p]; y = y_pos[p]
            if x < xmin: xmin = x
            if x > xmax: xmax = x
            if y < ymin: ymin = y
            if y > ymax: ymax = y
        new_h = net_weights[n] * ((xmax - xmin) + (ymax - ymin))
        delta += np.float64(new_h) - np.float64(net_hpwl_cache[n])
        net_hpwl_cache[n] = new_h
    return delta


@numba.njit(cache=True, fastmath=True)
def collect_affected_nets(moved_slots, macro_to_nets_offsets, macro_to_nets_flat,
                          mark_buffer, out_buffer):
    """Gather the union of nets touching any of the moved macro slots.

    `mark_buffer` is a bool array of length num_nets, must be all False on entry
    and is left all False on exit (caller-owned scratch). Returns the number
    of affected nets written into `out_buffer`.
    """
    k = 0
    # Mark
    for s_i in range(moved_slots.shape[0]):
        s = moved_slots[s_i]
        start = macro_to_nets_offsets[s]
        end = macro_to_nets_offsets[s + 1]
        for t in range(start, end):
            net = macro_to_nets_flat[t]
            if not mark_buffer[net]:
                mark_buffer[net] = True
                out_buffer[k] = net
                k += 1
    # Unmark for reuse
    for i in range(k):
        mark_buffer[out_buffer[i]] = False
    return k


class PlacementCostAccelerated(object):

    def __init__(self,
                netlist_file: Text,
                macro_macro_x_spacing: float = 0.0,
                macro_macro_y_spacing: float = 0.0) -> None:
        """
        Creates a PlacementCost object.
        """
        self.netlist_file = netlist_file
        self.macro_macro_x_spacing = macro_macro_x_spacing
        self.macro_macro_y_spacing = macro_macro_y_spacing

        # Update flags
        self.FLAG_UPDATE_WIRELENGTH = True
        self.FLAG_UPDATE_DENSITY = True
        self.FLAG_UPDATE_CONGESTION = True
        self.FLAG_UPDATE_MACRO_ADJ = True
        self.FLAG_UPDATE_MACRO_AND_CLUSTERED_PORT_ADJ = True
        self.FLAG_UPDATE_NODE_MASK = True

        # Check netlist existance
        assert os.path.isfile(self.netlist_file)

        # [Experimental] Net Data Structure
        # nets[driver] => [list of sinks]
        self.nets = {}
        
        # Set meta information
        self.init_plc = None
        self.project_name = "circuit_training"
        self.block_name = netlist_file.rsplit('/', -1)[-2]
        self.hroutes_per_micron = 0.0
        self.vroutes_per_micron = 0.0
        self.smooth_range = 0.0
        self.overlap_thres = 0.0
        self.hrouting_alloc = 0.0
        self.vrouting_alloc = 0.0
        self.macro_horizontal_routing_allocation = 0.0
        self.macro_vertical_routing_allocation = 0.0
        self.canvas_boundary_check = True

        # net information
        self.net_cnt = 0

        # All modules look-up table
        self.modules = []
        self.modules_w_pins = []

        # modules to index look-up table
        self.indices_to_mod_name = {}
        self.mod_name_to_indices = {}

        # indices storage
        self.port_indices = []
        self.hard_macro_indices = []
        self.hard_macro_pin_indices = []
        self.soft_macro_indices = []
        self.soft_macro_pin_indices = []

        # macro to pins look-up table: [MACRO_NAME] => [PIN_NAME]
        self.hard_macros_to_inpins = {}
        self.soft_macros_to_inpins = {}

        # Placed macro
        self.placed_macro = []

        # not used
        self.use_incremental_cost = False
        # blockage
        self.blockages = []
        # read netlist
        self.__read_protobuf()

        # default canvas width/height based on cell area
        self.width = math.sqrt(self.get_area()/0.6)
        self.height = math.sqrt(self.get_area()/0.6)

        # default gridding
        self.grid_col = 10
        self.grid_row = 10

        # initialize congestion map
        self.V_routing_cong = [0] * (self.grid_col * self.grid_row)
        self.H_routing_cong = [0] * (self.grid_col * self.grid_row)
        self.V_macro_routing_cong = [0] * (self.grid_col * self.grid_row)
        self.H_macro_routing_cong = [0] * (self.grid_col * self.grid_row)
        # initial grid mask, flatten before output
        self.node_mask = np.array([1] * (self.grid_col * self.grid_row))\
            .reshape(self.grid_row, self.grid_col)
        
        # store module/component count
        self.ports_cnt = len(self.port_indices)
        self.hard_macro_cnt = len(self.hard_macro_indices)
        self.hard_macro_pins_cnt = len(self.hard_macro_pin_indices)
        self.soft_macros_cnt = len(self.soft_macro_indices)
        self.soft_macro_pins_cnt = len(self.soft_macro_pin_indices)
        self.module_cnt = self.hard_macro_cnt + self.soft_macros_cnt + self.ports_cnt

        # assert module and pin count are correct
        assert (len(self.modules)) == self.module_cnt
        assert (len(self.modules_w_pins) - \
            self.hard_macro_pins_cnt - self.soft_macro_pins_cnt) \
                == self.module_cnt
                
                
        
        self.hard_macros_idxs         = np.array(self.hard_macro_indices, dtype=np.int32)
        self.soft_macros_idxs         = np.array(self.soft_macro_indices, dtype=np.int32)
        
        self.fwd_connectivity_CSR     = []
        self.start_idx                = []
        self.fwd_start_offsets        = []
        self.fwd_sizes                = []
        self.weights_fast             = []
        
        self.bckwd_connectivity_CSR   = []
        self.bckwd_start_offsets      = []
        self.bckwd_sizes              = []
        
        self.hard_macro_idx           = []
        
        self.x_pos                    = []
        self.y_pos                    = []  
         
        self.net_driver_pin           = []
        self.net_sink_offsets         = []
        self.net_sink_pins            = []
        self.net_weights              = [] 

        # variables for the routing implementation
        self.grid_width = float(self.width / self.grid_col)
        self.grid_height = float(self.height / self.grid_row)       
        self.hmacro_indices_np = np.array(self.hard_macro_indices, dtype=np.int32)
        self.hmacro_widths = []
        self.hmacro_heights = []   

        self.build_fast_representation()    
    
    def build_fast_representation(self):
        """
            Create a multiple arrays : 
            
            firstly we need to create a global list of positions : it will be an array of positions where 
            index corresponds to the module index in the modules list 
            
            then we need to create a forward and backward connectivity list in the CSR format
        """
        
        # get the hard macro widths and heights for the congestion update
        for macro_idx in self.hmacro_indices_np:
            module = self.modules_w_pins[macro_idx]
            
            self.hmacro_widths.append(module.get_width())
            self.hmacro_heights.append(module.get_height())
        
        
        # build global list of positions 
        for i in range(len(self.modules_w_pins)):
            
            type = self.modules_w_pins[i].get_type() 
            if type == 'MACRO_PIN' or type == 'PORT':            
                x, y = self.__get_pin_position(i)
                self.x_pos.append(x)
                self.y_pos.append(y)
            elif type in ("MACRO", "SOFT_MACRO"):
                x, y = self.modules_w_pins[i].get_pos()
                self.x_pos.append(x)
                self.y_pos.append(y)
            
            else:
                self.x_pos.append(-1)
                self.y_pos.append(-1)
            
        idx = 0
        off = 0
        off_sinks = 0
        
        bckwd = defaultdict(list)
        
        fwd = []
        sinks = []

        # create forward CSR and backwards adjacency list (for congestion update)
        for idx, driver_pin_name in enumerate(self.nets.keys()):

            driver_pin_idx = self.mod_name_to_indices[driver_pin_name]
            driver_pin = self.modules_w_pins[driver_pin_idx]
            
            weight_fact = driver_pin.get_weight()
            self.weights_fast.append(weight_fact)
            
            self.fwd_start_offsets.append(off)
            self.net_sink_offsets.append(off_sinks)
            
            # needed for the routing update
            self.net_driver_pin.append(driver_pin_idx)
            self.net_weights.append(weight_fact)

            fwd.append(driver_pin_idx)
            self.fwd_connectivity_CSR.append(driver_pin_idx)
            off += 1    

            for sink_pin_name in self.nets[driver_pin_name]:
                sink_pin_idx = self.mod_name_to_indices[sink_pin_name]
                
                # for routing update
                self.net_sink_pins.append(sink_pin_idx)
                
                self.fwd_connectivity_CSR.append(sink_pin_idx)
                self.start_idx.append(driver_pin_idx)
                off += 1   
                off_sinks += 1             
                
                bckwd[sink_pin_idx].append(driver_pin_idx)
                sinks.append(sink_pin_idx)

            self.fwd_sizes.append(1 + len(self.nets[driver_pin_name]))
                         
        #self.fwd_start_offsets.append(off) # append sentinel        
        self.net_sink_offsets.append(off_sinks) # append sentinel            
                         
        # cast everything to numpy for vectorized computation
        self.fwd_connectivity_CSR = np.array(self.fwd_connectivity_CSR, dtype=np.int32)
        self.fwd_start_offsets = np.array(self.fwd_start_offsets, dtype=np.int32)
        self.x_pos = np.array(self.x_pos, dtype=np.float32)
        self.y_pos = np.array(self.y_pos, dtype=np.float32)
        self.start_idx = np.array(self.start_idx, dtype=np.int32) 

        self.fwd_sizes = np.array(self.fwd_sizes, dtype=np.int32)
        self.weights_fast = np.array(self.weights_fast, dtype=np.float32)
        
        self.net_driver_pin = np.array(self.net_driver_pin, dtype=np.int32)
        self.net_sink_pins = np.array(self.net_sink_pins, dtype=np.int32)
        self.net_sink_offsets = np.array(self.net_sink_offsets, dtype=np.int32)
        self.net_weights = np.array(self.net_weights, dtype=np.float32)
        
        # cast hard macro widths and heights to numpy
        self.hmacro_widths = np.array(self.hmacro_widths, dtype=np.float32)
        self.hmacro_heights = np.array(self.hmacro_heights, dtype=np.float32)

        # -------------------------------------------------------------------
        # Static arrays for the fast _set_placement path.
        #
        # Layout plan:
        #   placement[:num_hard]        -> slot into hmacro_indices_np
        #   placement[num_hard:]        -> slot into smacro_indices_np
        #   hard macro pin at slot s:   plc.x_pos[hard_pin_indices[k]]
        #                               = placement[s, 0] + hard_pin_offset_x[k]
        #   soft macro pin at slot s:   plc.x_pos[soft_pin_indices[k]]
        #                               = placement[num_hard + s, 0]   (offset is 0)
        # -------------------------------------------------------------------
        self.smacro_indices_np = np.array(self.soft_macro_indices, dtype=np.int32)

        # Map macro name -> slot in placement tensor.
        hard_name_to_slot = {
            self.modules_w_pins[m].get_name(): s
            for s, m in enumerate(self.hard_macro_indices)
        }
        soft_name_to_slot = {
            self.modules_w_pins[m].get_name(): s
            for s, m in enumerate(self.soft_macro_indices)
        }

        # Walk every MACRO_PIN once; bin into hard / soft buckets.
        hard_pin_indices = []
        hard_pin_macro_slot = []
        hard_pin_offset_x = []
        hard_pin_offset_y = []
        soft_pin_indices = []
        soft_pin_macro_slot = []

        for pin_idx in self.hard_macro_pin_indices:
            pin = self.modules_w_pins[pin_idx]
            slot = hard_name_to_slot.get(pin.get_macro_name())
            if slot is None:
                continue  # orphan pin (shouldn't happen in a well-formed netlist)
            hard_pin_indices.append(pin_idx)
            hard_pin_macro_slot.append(slot)
            hard_pin_offset_x.append(pin.x_offset)
            hard_pin_offset_y.append(pin.y_offset)

        for pin_idx in self.soft_macro_pin_indices:
            pin = self.modules_w_pins[pin_idx]
            slot = soft_name_to_slot.get(pin.get_macro_name())
            if slot is None:
                continue
            soft_pin_indices.append(pin_idx)
            soft_pin_macro_slot.append(slot)

        self.hard_pin_indices    = np.array(hard_pin_indices, dtype=np.int32)
        self.hard_pin_macro_slot = np.array(hard_pin_macro_slot, dtype=np.int32)
        self.hard_pin_offset_x   = np.array(hard_pin_offset_x, dtype=np.float32)
        self.hard_pin_offset_y   = np.array(hard_pin_offset_y, dtype=np.float32)

        self.soft_pin_indices    = np.array(soft_pin_indices, dtype=np.int32)
        self.soft_pin_macro_slot = np.array(soft_pin_macro_slot, dtype=np.int32)

        # -------------------------------------------------------------------
        # CSR of macro_slot -> its pins, for the moved-only placement update.
        #
        # When hard-macro slot s moves to (x, y), the pins to refresh are:
        #   a, b = hard_slot_pin_offsets[s], hard_slot_pin_offsets[s + 1]
        #   plc.x_pos[hard_slot_pin_indices[a:b]] = x + hard_slot_pin_offset_x[a:b]
        #   plc.y_pos[hard_slot_pin_indices[a:b]] = y + hard_slot_pin_offset_y[a:b]
        #
        # Soft macros: pin offsets are 0, so the soft-slot CSR only carries
        # pin indices.
        # -------------------------------------------------------------------
        num_hard_slots = len(self.hard_macro_indices)
        num_soft_slots = len(self.soft_macro_indices)

        if len(hard_pin_indices) > 0:
            hp_idx = np.asarray(hard_pin_indices, dtype=np.int32)
            hp_slot = np.asarray(hard_pin_macro_slot, dtype=np.int32)
            hp_ox = np.asarray(hard_pin_offset_x, dtype=np.float32)
            hp_oy = np.asarray(hard_pin_offset_y, dtype=np.float32)

            # Stable sort by slot so all pins of slot s land contiguously.
            order = np.argsort(hp_slot, kind="stable")
            self.hard_slot_pin_indices  = hp_idx[order]
            self.hard_slot_pin_offset_x = hp_ox[order]
            self.hard_slot_pin_offset_y = hp_oy[order]

            counts = np.bincount(hp_slot, minlength=num_hard_slots).astype(np.int32)
            offsets = np.empty(num_hard_slots + 1, dtype=np.int32)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            self.hard_slot_pin_offsets = offsets
        else:
            self.hard_slot_pin_indices  = np.zeros(0, dtype=np.int32)
            self.hard_slot_pin_offset_x = np.zeros(0, dtype=np.float32)
            self.hard_slot_pin_offset_y = np.zeros(0, dtype=np.float32)
            self.hard_slot_pin_offsets  = np.zeros(num_hard_slots + 1, dtype=np.int32)

        if len(soft_pin_indices) > 0:
            sp_idx = np.asarray(soft_pin_indices, dtype=np.int32)
            sp_slot = np.asarray(soft_pin_macro_slot, dtype=np.int32)

            order = np.argsort(sp_slot, kind="stable")
            self.soft_slot_pin_indices = sp_idx[order]

            counts = np.bincount(sp_slot, minlength=num_soft_slots).astype(np.int32)
            offsets = np.empty(num_soft_slots + 1, dtype=np.int32)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            self.soft_slot_pin_offsets = offsets
        else:
            self.soft_slot_pin_indices = np.zeros(0, dtype=np.int32)
            self.soft_slot_pin_offsets = np.zeros(num_soft_slots + 1, dtype=np.int32)

        # -------------------------------------------------------------------
        # CSR of macro_slot -> nets touching that macro. Used by the incremental
        # HPWL path: when macro s moves, `macro_to_nets_flat[offsets[s]:offsets[s+1]]`
        # are the net indices whose HPWL must be recomputed.
        # -------------------------------------------------------------------
        pin_to_macro_slot = {}   # plc pin index -> hard macro slot
        for pin_idx, slot in zip(hard_pin_indices, hard_pin_macro_slot):
            pin_to_macro_slot[int(pin_idx)] = int(slot)

        num_hard_slots = len(self.hard_macro_indices)
        macro_nets_sets = [set() for _ in range(num_hard_slots)]
        num_nets = len(self.net_driver_pin)
        for net_idx in range(num_nets):
            d = int(self.net_driver_pin[net_idx])
            s = pin_to_macro_slot.get(d)
            if s is not None:
                macro_nets_sets[s].add(net_idx)
            s0 = int(self.net_sink_offsets[net_idx])
            s1 = int(self.net_sink_offsets[net_idx + 1])
            for k in range(s0, s1):
                p = int(self.net_sink_pins[k])
                ms = pin_to_macro_slot.get(p)
                if ms is not None:
                    macro_nets_sets[ms].add(net_idx)

        macro_to_nets_offsets = [0]
        macro_to_nets_flat = []
        for s in range(num_hard_slots):
            net_list = sorted(macro_nets_sets[s])
            macro_to_nets_flat.extend(net_list)
            macro_to_nets_offsets.append(len(macro_to_nets_flat))

        self.macro_to_nets_offsets = np.array(macro_to_nets_offsets, dtype=np.int32)
        self.macro_to_nets_flat    = np.array(macro_to_nets_flat, dtype=np.int32)

        # -------------------------------------------------------------------
        # Incremental HPWL state. Lazily initialized on the first call to
        # get_wirelength_incremental (requires positions to be set first).
        # -------------------------------------------------------------------
        self.net_hpwl_cache = np.zeros(num_nets, dtype=np.float32)
        self.total_hpwl = 0.0
        self._hpwl_incremental_ready = False

        # Max sinks-per-net — precomputed once for scratch allocation in the
        # incremental routing kernel.
        max_sinks = 0
        for n in range(num_nets):
            k = int(self.net_sink_offsets[n + 1]) - int(self.net_sink_offsets[n])
            if k > max_sinks:
                max_sinks = k
        self._max_sinks_per_net = int(max_sinks)

        # -------------------------------------------------------------------
        # Incremental congestion state. Lazily initialized on first call to
        # get_congestion_cost_incremental.
        #
        # Maintains raw (pre-normalization, pre-smooth) grids plus caches of
        # pin-gcells and macro centers that reflect the state-of-those-grids.
        # -------------------------------------------------------------------
        self.H_net_raw = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.V_net_raw = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.H_macro_raw = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.V_macro_raw = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.pin_row_cache = np.zeros(0, dtype=np.int32)
        self.pin_col_cache = np.zeros(0, dtype=np.int32)
        self.macro_x_cache = np.zeros(len(self.hard_macro_indices), dtype=np.float32)
        self.macro_y_cache = np.zeros(len(self.hard_macro_indices), dtype=np.float32)
        self._congestion_incremental_ready = False

        # -------------------------------------------------------------------
        # Incremental density state. Grid_occupied_raw is the running splat
        # of macro-body overlap area into the density grid (pre-normalization
        # and pre-ABU). density_all_* hold the snapshot of positions/sizes
        # the raw grid was built with, across ALL macros (hard + soft).
        # -------------------------------------------------------------------
        num_all_macros = len(self.hard_macro_indices) + len(self.soft_macro_indices)
        self.grid_occupied_raw = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.density_all_x = np.zeros(num_all_macros, dtype=np.float32)
        self.density_all_y = np.zeros(num_all_macros, dtype=np.float32)
        self.density_all_w = np.zeros(num_all_macros, dtype=np.float32)
        self.density_all_h = np.zeros(num_all_macros, dtype=np.float32)
        self._density_incremental_ready = False


    def __peek(self, f:io.TextIOWrapper):
        """
        Return String next line by peeking into the next line without moving file descriptor
        """
        pos = f.tell()
        t_line = f.readline()
        f.seek(pos)
        return t_line

    def __read_protobuf(self):
        """
        private function: Protobuf Netlist Parser
        """
        print("#[INFO] Reading from " + self.netlist_file)
        with open(self.netlist_file) as fp:
            line = fp.readline()
            node_cnt = 0

            while line:
                line_item = re.findall(r'\w+', line)

                # skip empty lines
                if len(line_item) == 0:
                    # advance ptr
                    line = fp.readline()
                    continue

                # skip comments
                if re.search(r"\S", line)[0] == '#':
                    # advance ptr
                    line = fp.readline()
                    continue

                # node found
                if line_item[0] == 'node':
                    node_name = ''
                    input_list = []

                    # advance ptr
                    line = fp.readline()
                    line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)
                    # retrieve node name
                    if line_item[0] == 'name':
                        node_name = line_item[1]
                        # skip metadata header
                        if node_name == "__metadata__":
                            pass
                        else:
                            node_cnt += 1
                    else:
                        node_name = 'N/A name'
                        

                    # advance ptr
                    line = fp.readline()
                    line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)
                    # retrieve node input
                    if line_item[0] == 'input':
                        input_list.append(line_item[1])

                        while re.findall(r'\w+[^\:\n\\{\}\s"]*', self.__peek(fp))[0] == 'input':
                            line = fp.readline()
                            line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)
                            input_list.append(line_item[1])

                        line = fp.readline()
                        line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)
                    else:
                        input_list = None

                    # advance, expect multiple attributes
                    attr_dict = {}
                    while len(line_item) != 0 and line_item[0] == 'attr':

                        # advance, expect key
                        line = fp.readline()
                        line_item = re.findall(r'\w+', line)
                        key = line_item[1]

                        if key == "macro_name":
                             # advance, expect value
                            line = fp.readline()
                            line_item = re.findall(r'\w+', line)

                            # advance, expect value item
                            line = fp.readline()
                            line_item = re.findall(r'\w+[^\:\n\\{\}\s"]*', line)

                            attr_dict[key] = line_item

                            line = fp.readline()
                            line = fp.readline()
                            line = fp.readline()

                            line_item = re.findall(r'\w+', line)
                        else:
                            # advance, expect value
                            line = fp.readline()
                            line_item = re.findall(r'\w+', line)

                            # advance, expect value item
                            line = fp.readline()
                            # Fixed regex to handle scientific notation (e.g., 1.42109e-16)
                            # Pattern matches: scientific notation OR identifiers/paths/numbers
                            line_item = re.findall(r'[-+]?\d+\.?\d*[eE][-+]?\d+|[-]?\w+\.?[\w/]*', line)

                            attr_dict[key] = line_item

                            line = fp.readline()
                            line = fp.readline()
                            line = fp.readline()

                            line_item = re.findall(r'\w+', line)

                    # putting info into data structure
                    if node_name == "__metadata__":
                        # skipping metadata header
                        logging.info('[INFO NETLIST PARSER] skipping invalid net input')
                        
                    elif attr_dict['type'][1] == 'macro':
                        # soft macro
                        # check if all required information is obtained
                        try:
                            assert 'x' in attr_dict.keys()
                        except AssertionError:
                            logging.warning('[ERROR NETLIST PARSER] x is not defined')

                        try:
                            assert 'y' in attr_dict.keys()
                        except AssertionError:
                            logging.warning('[ERROR NETLIST PARSER] y is not defined')

                        soft_macro = self.SoftMacro(name=node_name, width=attr_dict['width'][1],
                                                    height = attr_dict['height'][1],
                                                    x = attr_dict['x'][1], y = attr_dict['y'][1])
                        self.modules_w_pins.append(soft_macro)
                        self.modules.append(soft_macro)
                        # mapping node_name ==> node idx
                        self.mod_name_to_indices[node_name] = node_cnt-1
                        # mapping node idx ==> node_name
                        self.indices_to_mod_name[node_cnt-1] = node_name
                        # store current node indx
                        self.soft_macro_indices.append(node_cnt-1)

                    elif attr_dict['type'][1] == 'macro_pin':
                        # [MACRO_NAME]/[PIN_NAME]
                        soft_macro_name = node_name.rsplit('/', 1)[0]
                        # soft macro pin
                        soft_macro_pin = self.SoftMacroPin(name=node_name,ref_id=None,
                                                           x = attr_dict['x'][1],
                                                           y = attr_dict['y'][1],
                                                           macro_name = attr_dict['macro_name'][1])

                        if 'weight' in attr_dict.keys():
                            soft_macro_pin.set_weight(float(attr_dict['weight'][1]))
                        
                        # if pin has net info
                        if input_list:
                            # net count should be factored by net weight
                            if 'weight' in attr_dict.keys():
                                self.net_cnt += 1 * float(attr_dict['weight'][1])
                            else:
                                self.net_cnt += 1
                            soft_macro_pin.add_sinks(input_list)
                            # add net
                            self.nets[node_name] = input_list

                        self.modules_w_pins.append(soft_macro_pin)
                        # mapping node_name ==> node idx
                        self.mod_name_to_indices[node_name] = node_cnt-1
                        # mapping node idx ==> node_name
                        self.indices_to_mod_name[node_cnt-1] = node_name
                        # store current node indx
                        self.soft_macro_pin_indices.append(node_cnt-1)

                        if soft_macro_name in self.soft_macros_to_inpins.keys():
                            self.soft_macros_to_inpins[soft_macro_name]\
                                .append(soft_macro_pin.get_name())
                        else:
                            self.soft_macros_to_inpins[soft_macro_name]\
                                = [soft_macro_pin.get_name()]

                    elif attr_dict['type'][1] == 'MACRO':
                        # hard macro
                        hard_macro = self.HardMacro(name=node_name,
                                                    width=attr_dict['width'][1],
                                                    height = attr_dict['height'][1],
                                                    x = attr_dict['x'][1],
                                                    y = attr_dict['y'][1],
                                                    orientation = attr_dict['orientation'][1])

                        self.modules_w_pins.append(hard_macro)
                        self.modules.append(hard_macro)
                        # mapping node_name ==> node idx
                        self.mod_name_to_indices[node_name] = node_cnt-1
                        # mapping node idx ==> node_name
                        self.indices_to_mod_name[node_cnt-1] = node_name
                        # store current node indx
                        self.hard_macro_indices.append(node_cnt-1)

                    elif attr_dict['type'][1] == 'MACRO_PIN':
                        # [MACRO_NAME]/[PIN_NAME]
                        hard_macro_name = node_name.rsplit('/', 1)[0]
                        # hard macro pin
                        hard_macro_pin = self.HardMacroPin(name=node_name,ref_id=None,
                                                        x = attr_dict['x'][1],
                                                        y = attr_dict['y'][1],
                                                        x_offset = attr_dict['x_offset'][1],
                                                        y_offset = attr_dict['y_offset'][1],
                                                        macro_name = attr_dict['macro_name'][1])
                        
                        # if net weight is defined, set weight
                        if 'weight' in attr_dict.keys():
                            hard_macro_pin.set_weight(float(attr_dict['weight'][1]))

                        # if pin has net info
                        if input_list:
                            # net count should be factored by net weight
                            if 'weight' in attr_dict.keys():
                                self.net_cnt += 1 * float(attr_dict['weight'][1])
                            else:
                                self.net_cnt += 1
                            hard_macro_pin.add_sinks(input_list)
                            self.nets[node_name] = input_list

                        self.modules_w_pins.append(hard_macro_pin)
                        # mapping node_name ==> node idx
                        self.mod_name_to_indices[node_name] = node_cnt-1
                        # mapping node idx ==> node_name
                        self.indices_to_mod_name[node_cnt-1] = node_name
                        # store current node indx
                        self.hard_macro_pin_indices.append(node_cnt-1)

                        # add to dict
                        if hard_macro_name in self.hard_macros_to_inpins.keys():
                            self.hard_macros_to_inpins[hard_macro_name]\
                                .append(hard_macro_pin.get_name())
                        else:
                            self.hard_macros_to_inpins[hard_macro_name]\
                                 = [hard_macro_pin.get_name()]

                    elif attr_dict['type'][1] == 'PORT':
                        # port
                        port = self.Port(name= node_name,
                                        x = attr_dict['x'][1],
                                        y = attr_dict['y'][1],
                                        side = attr_dict['side'][1])

                        # if pin has net info
                        if input_list:
                            self.net_cnt += 1
                            port.add_sinks(input_list)
                            # ports does not have pins so update connection immediately
                            port.add_connections(input_list)
                            self.nets[node_name] = input_list

                        self.modules_w_pins.append(port)
                        self.modules.append(port)
                        # mapping node_name ==> node idx
                        self.mod_name_to_indices[node_name] = node_cnt-1
                        # mapping node idx ==> node_name
                        self.indices_to_mod_name[node_cnt-1] = node_name
                        # store current node indx
                        self.port_indices.append(node_cnt-1)

        # 1. mapping connection degree to each macros
        # 2. update offset based on Hard macro orientation
        self.__update_connection()

        # all hard macros are placed on canvas initially
        self.__update_init_placed_node()

    def __read_plc(self, plc_pth: str):
        """
        Plc file Parser
        """
        # meta information
        _columns = 0
        _rows = 0
        _width = 0.0
        _height = 0.0
        _area = 0.0
        _block = None
        _routes_per_micron_hor = 0.0
        _routes_per_micron_ver = 0.0
        _routes_used_by_macros_hor = 0.0
        _routes_used_by_macros_ver = 0.0
        _smoothing_factor = 0
        _overlap_threshold = 0.0

        # node information
        _hard_macros_cnt = 0
        _hard_macro_pins_cnt = 0
        _macros_cnt = 0
        _macro_pin_cnt = 0
        _ports_cnt = 0
        _soft_macros_cnt = 0
        _soft_macro_pins_cnt = 0
        _stdcells_cnt = 0

        # node placement
        _node_plc = {}

        for cnt, line in enumerate(open(plc_pth, 'r')):
            line_item = re.findall(r'[0-9A-Za-z\.\-]+', line)

            # skip empty lines
            if len(line_item) == 0:
                continue

            if 'Columns' in line_item and 'Rows' in line_item:
                # Columns and Rows should be defined on the same one-line
                _columns = int(line_item[1])
                _rows = int(line_item[3])
            elif "Width" in line_item and "Height" in line_item:
                # Width and Height should be defined on the same one-line
                _width = float(line_item[1])
                _height = float(line_item[3])
            elif all(it in line_item for it in ['Area', 'stdcell', 'macros']):
                # Total core area of modules
                _area = float(line_item[3])
            elif "Area" in line_item:
                # Total core area of modules
                _area = float(line_item[1])
            elif "Block" in line_item:
                # The block name of the testcase
                _block = str(line_item[1])
            elif all(it in line_item for it in\
                ['Routes', 'per', 'micron', 'hor', 'ver']):
                # For routing congestion computation
                _routes_per_micron_hor = float(line_item[4])
                _routes_per_micron_ver = float(line_item[6])
            elif all(it in line_item for it in\
                    ['Routes', 'used', 'by', 'macros', 'hor', 'ver']):
                # For MACRO congestion computation
                _routes_used_by_macros_hor = float(line_item[5])
                _routes_used_by_macros_ver = float(line_item[7])
            elif all(it in line_item for it in ['Smoothing', 'factor']):
                # smoothing factor for routing congestion
                _smoothing_factor = int(line_item[2])
            elif all(it in line_item for it in ['Overlap', 'threshold']):
                # overlap
                _overlap_threshold = float(line_item[2])
            elif all(it in line_item for it in ['HARD', 'MACROs'])\
                and len(line_item) == 3:
                _hard_macros_cnt = int(line_item[2])
            elif all(it in line_item for it in ['HARD', 'MACRO', 'PINs'])\
                and len(line_item) == 4:
                _hard_macro_pins_cnt = int(line_item[3])
            elif all(it in line_item for it in ['PORTs'])\
                and len(line_item) == 2:
                _ports_cnt = int(line_item[1])
            elif all(it in line_item for it in ['SOFT', 'MACROs'])\
                and len(line_item) == 3:
                _soft_macros_cnt = int(line_item[2])
            elif all(it in line_item for it in ['SOFT', 'MACRO', 'PINs'])\
                and len(line_item) == 4:
                _soft_macro_pins_cnt = int(line_item[3])
            elif all(it in line_item for it in ['STDCELLs'])\
                and len(line_item) == 2:
                _stdcells_cnt = int(line_item[1])
            elif all(it in line_item for it in ['MACROs'])\
                and len(line_item) == 2:
                _macros_cnt = int(line_item[1])
            elif all(re.match(r'[0-9FNEWS\.\-]+', it) for it in line_item)\
                and len(line_item) == 5:
                # [node_index] [x] [y] [orientation] [fixed]
                _node_plc[int(line_item[0])] = line_item[1:]
        
        # return as dictionary
        info_dict = {   "columns":_columns, 
                        "rows":_rows,
                        "width":_width,
                        "height":_height,
                        "area":_area,
                        "block":_block,
                        "routes_per_micron_hor":_routes_per_micron_hor,
                        "routes_per_micron_ver":_routes_per_micron_ver,
                        "routes_used_by_macros_hor":_routes_used_by_macros_hor,
                        "routes_used_by_macros_ver":_routes_used_by_macros_ver,
                        "smoothing_factor":_smoothing_factor,
                        "overlap_threshold":_overlap_threshold,
                        "hard_macros_cnt":_hard_macros_cnt,
                        "hard_macro_pins_cnt":_hard_macro_pins_cnt,
                        "macros_cnt":_macros_cnt,
                        "macro_pin_cnt":_macro_pin_cnt,
                        "ports_cnt":_ports_cnt,
                        "soft_macros_cnt":_soft_macros_cnt,
                        "soft_macro_pins_cnt":_soft_macro_pins_cnt,
                        "stdcells_cnt":_stdcells_cnt,
                        "node_plc":_node_plc
                    }

        return info_dict

    def restore_placement(self, plc_pth: str, ifInital=True, ifValidate=False, ifReadComment = False):
        """
            Read and retrieve .plc file information
            NOTE: DO NOT always set self.init_plc because this function is also 
            used to read final placement file.

            ifReadComment: By default, Google's plc_client does not extract 
            information from .plc comment. This is purely done in 
            placement_util.py. For purpose of testing, we included this option.
        """
        # if plc is an initial placement
        if ifInital:
            self.init_plc = plc_pth
        
        # recompute cost from new location
        self.FLAG_UPDATE_CONGESTION = True
        self.FLAG_UPDATE_DENSITY = True
        self.FLAG_UPDATE_WIRELENGTH = True
        
        self.FLAG_UPDATE_NODE_MASK = True
        
        # extracted information from .plc file
        info_dict = self.__read_plc(plc_pth)

        # validate netlist.pb.txt is on par with .plc
        if ifValidate:
            try:
                assert(self.hard_macro_cnt == info_dict['hard_macros_cnt'])
                assert(self.hard_macro_pins_cnt == info_dict['hard_macro_pins_cnt'])
                assert(self.soft_macros_cnt == info_dict['soft_macros_cnt'])
                assert(self.soft_macro_pins_cnt == info_dict['soft_macro_pins_cnt'])
                assert(self.ports_cnt == info_dict['ports_cnt'])
            except AssertionError:
                _, _, tb = sys.exc_info()
                traceback.print_tb(tb)
                tb_info = traceback.extract_tb(tb)
                _, line, _, text = tb_info[-1]
                print('[ERROR NETLIST/PLC MISMATCH] at line {} in statement {}'\
                    .format(line, text))
                exit(1)
        
        # restore placement for each module
        try:
            # print(sorted(list(info_dict['node_plc'].keys())))
            assert sorted(self.port_indices +\
                self.hard_macro_indices +\
                self.soft_macro_indices) == sorted(list(info_dict['node_plc'].keys()))
        except AssertionError:
            print('[ERROR PLC INDICES MISMATCH]', len(sorted(self.port_indices +\
                self.hard_macro_indices +\
                self.soft_macro_indices)), len(list(info_dict['node_plc'].keys())))
            exit(1)
        
        for mod_idx in info_dict['node_plc'].keys():
            mod_x = mod_y = mod_orient = mod_ifFixed = None
            try:
                mod_x = float(info_dict['node_plc'][mod_idx][0])
                mod_y = float(info_dict['node_plc'][mod_idx][1])
                mod_orient = info_dict['node_plc'][mod_idx][2]
                mod_ifFixed = int(info_dict['node_plc'][mod_idx][3])
                
            except Exception as e:
                print('[ERROR PLC PARSER] %s' % str(e))

            #TODO ValueError: Error in calling RestorePlacement with ('./Plc_client/test/ariane/initial.plc',): Can't place macro i_ariane/i_frontend/i_icache/sram_block_3__tag_sram/mem/mem_inst_mem_256x45_256x16_0x0 at (341.75, 8.8835). Exceeds the boundaries of the placement area..

            self.modules_w_pins[mod_idx].set_pos(mod_x, mod_y)
            
            if mod_orient and mod_orient != '-':
                self.modules_w_pins[mod_idx].set_orientation(mod_orient)
            
            if mod_ifFixed == 0:
                self.modules_w_pins[mod_idx].set_fix_flag(False)
            elif mod_ifFixed == 1:
                self.modules_w_pins[mod_idx].set_fix_flag(True)
        
        # set meta information
        if ifReadComment:
            print("[INFO] Retrieving Meta information from .plc comments")
            self.set_canvas_size(info_dict['width'], info_dict['height'])
            self.set_placement_grid(info_dict['columns'], info_dict['rows'])
            self.set_block_name(info_dict['block'])
            self.set_routes_per_micron(
                info_dict['routes_per_micron_hor'],
                info_dict['routes_per_micron_ver']
                )
            self.set_macro_routing_allocation(
                info_dict['routes_used_by_macros_hor'],
                info_dict['routes_used_by_macros_ver']
                )
            self.set_congestion_smooth_range(info_dict['smoothing_factor'])
            self.set_overlap_threshold(info_dict['overlap_threshold'])

    def __update_connection(self):
        """
        Update connection degree for each macro pin
        """
        for macro_idx in (self.hard_macro_indices + self.soft_macro_indices):
            macro = self.modules_w_pins[macro_idx]
            macro_name = macro.get_name()

            # Hard macro
            if not self.is_node_soft_macro(macro_idx):
                if macro_name in self.hard_macros_to_inpins.keys():
                    pin_names = self.hard_macros_to_inpins[macro_name]
                else:
                    print("[ERROR UPDATE CONNECTION] MACRO pins not found")
                    continue

                # also update pin offset based on macro orientation
                orientation = macro.get_orientation()
                self.update_macro_orientation(macro_idx, orientation)

            # Soft macro
            elif self.is_node_soft_macro(macro_idx):
                if macro_name in self.soft_macros_to_inpins.keys():
                    pin_names = self.soft_macros_to_inpins[macro_name]
                else:
                    print("[ERROR UPDATE CONNECTION] macro pins not found")
                    continue

            for pin_name in pin_names:
                pin = self.modules_w_pins[self.mod_name_to_indices[pin_name]]
                inputs = pin.get_sink()

                if inputs:
                    for k in inputs.keys():
                        if self.get_node_type(macro_idx) == "MACRO":
                            weight = pin.get_weight()
                            macro.add_connections(inputs[k], weight)
    
    def __update_init_placed_node(self):
        """
        Place all hard macros on node mask initially
        """
        for macro_idx in (self.hard_macro_indices  + self.soft_macro_indices):
            self.placed_macro.append(macro_idx)

    def get_cost(self) -> float:
        """
        Compute wirelength cost from wirelength
        """
        if self.net_cnt == 0:
            self.net_cnt = 1
        
        if self.FLAG_UPDATE_WIRELENGTH:
            self.FLAG_UPDATE_WIRELENGTH = False
        return self.get_wirelength() / ((self.get_canvas_width_height()[0]\
            + self.get_canvas_width_height()[1]) * self.net_cnt)

    def get_cost_fast(self) -> float:
        """
        Compute wirelength cost from wirelength
        """
        if self.net_cnt == 0:
            self.net_cnt = 1
        
        if self.FLAG_UPDATE_WIRELENGTH:
            self.FLAG_UPDATE_WIRELENGTH = False
            
        return self.get_wirelength_fast() / ((self.get_canvas_width_height()[0]\
            + self.get_canvas_width_height()[1]) * self.net_cnt)

    def get_area(self) -> float:
        """
        Compute Total Module Area
        """
        total_area = 0.0
        for mod in self.modules_w_pins:
            if hasattr(mod, 'get_area'):
                total_area += mod.get_area()
        return total_area

    def get_hard_macros_count(self) -> int:
        return self.hard_macro_cnt

    def get_ports_count(self) -> int:
        return self.ports_cnt

    def get_soft_macros_count(self) -> int:
        return self.soft_macros_cnt

    def get_hard_macro_pins_count(self) -> int:
        return self.hard_macro_pins_cnt

    def get_soft_macro_pins_count(self) -> int:
        return self.soft_macro_pins_cnt

    def __get_pin_position(self, pin_idx):
        """
        private function for getting pin location
            * PORT = its own position
            * HARD MACRO PIN = ref position + offset position
            * SOFT MACRO PIN = ref position
        """
        try:
            assert (self.modules_w_pins[pin_idx].get_type() == 'MACRO_PIN' or\
                self.modules_w_pins[pin_idx].get_type() == 'PORT')
        except Exception:
            print(self.modules_w_pins[pin_idx].get_type())
            print("[ERROR PIN POSITION] Not a MACRO PIN", self.modules_w_pins[pin_idx].get_name())
            exit(1)

        # PORT pin pos is itself
        if self.modules_w_pins[pin_idx].get_type() == 'PORT':
            return self.modules_w_pins[pin_idx].get_pos()

        # Retrieve node that this pin instantiated on
        ref_node_idx = self.get_ref_node_id(pin_idx)

        if ref_node_idx == -1:  
            print("[ERROR PIN POSITION] Parent Node Not Found.")
            exit(1)

        # Parent node
        ref_node = self.modules_w_pins[ref_node_idx]
        ref_node_x, ref_node_y = ref_node.get_pos()

        # Retrieve current pin node position
        pin_node = self.modules_w_pins[pin_idx]
        pin_node_x_offset, pin_node_y_offset = pin_node.get_offset()
        # Google's Plc client DOES NOT compute (node_position + pin_offset) when reading input
        return (ref_node_x + pin_node_x_offset, ref_node_y + pin_node_y_offset)
        # return pin_node.get_pos()

    def get_wirelength(self) -> float:
        # Proxy HPWL computation w/ [Experimental] net

        total_hpwl = 0.0
        for driver_pin_name in self.nets.keys():
            
            weight_fact = 1.0
            x_coord = []
            y_coord = []

            # extract driver pin
            driver_pin_idx = self.mod_name_to_indices[driver_pin_name]
            driver_pin = self.modules_w_pins[driver_pin_idx]
                        
            # extract net weight
            weight_fact = driver_pin.get_weight()

            x_coord.append(self.__get_pin_position(driver_pin_idx)[0])
            y_coord.append(self.__get_pin_position(driver_pin_idx)[1])

            # iterate through each sink
            for sink_pin_name in self.nets[driver_pin_name]:
                sink_pin_idx = self.mod_name_to_indices[sink_pin_name]

                x_coord.append(self.__get_pin_position(sink_pin_idx)[0])
                y_coord.append(self.__get_pin_position(sink_pin_idx)[1])
                
            if x_coord:
                total_hpwl += weight_fact * \
                    (abs(max(x_coord) - min(x_coord)) + \
                        abs(max(y_coord) - min(y_coord)))
        return total_hpwl

    def get_wirelength_fast(self) -> float:
        x_positions = self.x_pos[self.fwd_connectivity_CSR]
        y_positions = self.y_pos[self.fwd_connectivity_CSR]

        x_max = np.maximum.reduceat(x_positions, self.fwd_start_offsets)
        x_min = np.minimum.reduceat(x_positions, self.fwd_start_offsets)

        y_max = np.maximum.reduceat(y_positions, self.fwd_start_offsets)
        y_min = np.minimum.reduceat(y_positions, self.fwd_start_offsets)

        return np.sum(self.weights_fast * ((x_max - x_min) + (y_max - y_min)))


    # --------------------------------------------------------------
    # Incremental HPWL API
    # --------------------------------------------------------------
    def init_hpwl_incremental(self):
        """Full rebuild of the per-net HPWL cache from current x_pos/y_pos.
        Must be called once before the first get_wirelength_incremental,
        OR whenever the cache is invalidated (e.g. after a bulk repositioning)."""
        self.total_hpwl = float(hpwl_all_nets(
            self.x_pos, self.y_pos,
            self.net_driver_pin, self.net_sink_offsets, self.net_sink_pins,
            self.net_weights, self.net_hpwl_cache,
        ))
        # scratch buffers for the update kernel (allocated once, reused forever)
        num_nets = self.net_driver_pin.shape[0]
        self._hpwl_mark = np.zeros(num_nets, dtype=np.bool_)
        self._hpwl_out  = np.empty(num_nets, dtype=np.int32)
        self._hpwl_incremental_ready = True

    def invalidate_hpwl_incremental(self):
        """Force a full rebuild on next incremental call (use after any bulk
        repositioning that didn't go through moved_slots)."""
        self._hpwl_incremental_ready = False

    @staticmethod
    def _as_int32_array(moved_slots) -> np.ndarray:
        """Coerce a Python list or numpy array of macro slots to int32 ndarray.
        Numba kernels require ndarray with typed dtype."""
        if isinstance(moved_slots, np.ndarray):
            return moved_slots if moved_slots.dtype == np.int32 else moved_slots.astype(np.int32)
        return np.asarray(list(moved_slots), dtype=np.int32)

    def get_wirelength_incremental(self, moved_slots) -> float:
        """Return raw HPWL after updating only the nets touching the given
        hard-macro slots. Positions must already be written into x_pos/y_pos.

        moved_slots: iterable of hard-macro slot indices (0..num_hard). May be
        a Python list (from SA's propose_move) or an int32 ndarray.
        """
        if not self._hpwl_incremental_ready:
            self.init_hpwl_incremental()
            return self.total_hpwl

        moved_slots = self._as_int32_array(moved_slots)

        if moved_slots.shape[0] > 0:
            num_hard = int(self.hmacro_indices_np.shape[0])
            if moved_slots.max() >= num_hard or moved_slots.min() < 0:
                raise IndexError(
                    f"incremental HPWL: moved_slots out of range for hard-macro "
                    f"cache (num_hard={num_hard}, got min={int(moved_slots.min())}, "
                    f"max={int(moved_slots.max())}). Pass only hard-macro slots."
                )

        k = collect_affected_nets(
            moved_slots, self.macro_to_nets_offsets, self.macro_to_nets_flat,
            self._hpwl_mark, self._hpwl_out,
        )
        if k == 0:
            return self.total_hpwl

        delta = hpwl_update_nets(
            self._hpwl_out[:k], self.x_pos, self.y_pos,
            self.net_driver_pin, self.net_sink_offsets, self.net_sink_pins,
            self.net_weights, self.net_hpwl_cache,
        )
        self.total_hpwl += float(delta)
        return self.total_hpwl

    def get_cost_incremental(self, moved_slots: np.ndarray) -> float:
        """Normalized wirelength cost (same normalization as get_cost / get_cost_fast)."""
        raw = self.get_wirelength_incremental(moved_slots)
        denom = (self.width + self.height) * max(self.net_cnt, 1)
        return raw / denom


    # --------------------------------------------------------------
    # Incremental congestion API
    # --------------------------------------------------------------
    def _compute_pin_gcells(self):
        """Vectorized pin → gcell conversion from current x_pos / y_pos."""
        pin_row = np.clip(
            np.floor_divide(self.y_pos, self.grid_height).astype(np.int32),
            0, self.grid_row - 1,
        )
        pin_col = np.clip(
            np.floor_divide(self.x_pos, self.grid_width).astype(np.int32),
            0, self.grid_col - 1,
        )
        return pin_row, pin_col

    def init_congestion_incremental(self):
        """Full rebuild of the raw H/V grids from current x_pos/y_pos.
        Must be called once before the first get_congestion_cost_incremental,
        and whenever incremental state is invalidated."""
        # Geometry (may have changed since __init__ — restore_placement runs
        # AFTER build_fast_representation and can resize the grid).
        self.grid_width = float(self.width / self.grid_col)
        self.grid_height = float(self.height / self.grid_row)
        self.grid_v_routes = self.grid_width * self.vroutes_per_micron
        self.grid_h_routes = self.grid_height * self.hroutes_per_micron

        # (Re)allocate persistent raw grids if the current grid size differs
        # from whatever was pre-sized at build time. On a real benchmark load
        # this branch always fires once (default 10×10 → actual e.g. 45×41).
        num_cells = self.grid_row * self.grid_col
        if self.H_net_raw.shape[0] != num_cells:
            self.H_net_raw = np.zeros(num_cells, dtype=np.float32)
            self.V_net_raw = np.zeros(num_cells, dtype=np.float32)
            self.H_macro_raw = np.zeros(num_cells, dtype=np.float32)
            self.V_macro_raw = np.zeros(num_cells, dtype=np.float32)
        else:
            self.H_net_raw[:] = 0.0
            self.V_net_raw[:] = 0.0
            self.H_macro_raw[:] = 0.0
            self.V_macro_raw[:] = 0.0

        # Build pin-gcell cache from current positions.
        self.pin_row_cache, self.pin_col_cache = self._compute_pin_gcells()

        # Route all nets from scratch.
        accumulate_net_routing(
            pin_row=self.pin_row_cache, pin_col=self.pin_col_cache,
            driver_pin=self.net_driver_pin,
            sink_offsets=self.net_sink_offsets,
            sink_pinks=self.net_sink_pins,
            weights=self.net_weights,
            H=self.H_net_raw, V=self.V_net_raw,
            grid_cols=self.grid_col,
        )

        # Splat all macros and populate macro-center cache.
        macro_x = self.x_pos[self.hmacro_indices_np].astype(np.float32)
        macro_y = self.y_pos[self.hmacro_indices_np].astype(np.float32)
        accumulate_macro_blockage(
            macro_x=macro_x, macro_y=macro_y,
            macro_w=self.hmacro_widths, macro_h=self.hmacro_heights,
            grid_w=self.grid_width, grid_h=self.grid_height,
            grid_rows=self.grid_row, grid_cols=self.grid_col,
            hrouting_alloc=self.hrouting_alloc,
            vrouting_alloc=self.vrouting_alloc,
            H_macro=self.H_macro_raw, V_macro=self.V_macro_raw,
        )
        self.macro_x_cache = macro_x.copy()
        self.macro_y_cache = macro_y.copy()

        # Reuse HPWL scratch for affected-net collection if available.
        num_nets = int(self.net_driver_pin.shape[0])
        if not hasattr(self, "_hpwl_mark") or self._hpwl_mark.shape[0] != num_nets:
            self._hpwl_mark = np.zeros(num_nets, dtype=np.bool_)
            self._hpwl_out  = np.empty(num_nets, dtype=np.int32)

        self._congestion_incremental_ready = True

    def invalidate_congestion_incremental(self):
        self._congestion_incremental_ready = False

    def _finalize_abu(self) -> float:
        """Normalize (out-of-place), smooth net grids, combine with macro
        grids, and compute ABU(5%). Raw grids are left untouched so future
        incremental updates remain valid."""
        inv_h = 1.0 / self.grid_h_routes
        inv_v = 1.0 / self.grid_v_routes

        H_net = self.H_net_raw * inv_h
        V_net = self.V_net_raw * inv_v

        smooth_routing_cong_fast(
            grid_col=self.grid_col, grid_row=self.grid_row,
            smooth_range=int(self.smooth_range),
            V_routing_cong=V_net, H_routing_cong=H_net,
        )

        H_total = H_net + self.H_macro_raw * inv_h
        V_total = V_net + self.V_macro_raw * inv_v
        return float(self.abu_fast(H_total, V_total, 0.05))

    def get_congestion_cost_incremental(self, moved_slots) -> float:
        """
        Incremental congestion cost. Contract:

          * Caller must have updated plc.x_pos / plc.y_pos for the moved
            macros BEFORE calling this.
          * moved_slots is an iterable of HARD-MACRO slot indices whose
            positions have changed.
          * On the first call (or after invalidation), does a full rebuild.
        """
        if not self._congestion_incremental_ready:
            self.init_congestion_incremental()
            return self._finalize_abu()

        moved_slots = self._as_int32_array(moved_slots)
        if moved_slots.shape[0] == 0:
            return self._finalize_abu()

        # Bounds guard: incremental state is hard-macro only.
        num_hard = int(self.hmacro_indices_np.shape[0])
        if moved_slots.max() >= num_hard or moved_slots.min() < 0:
            raise IndexError(
                f"incremental congestion: moved_slots out of range for hard-macro "
                f"cache (num_hard={num_hard}, got min={int(moved_slots.min())}, "
                f"max={int(moved_slots.max())}). If you're moving soft macros, "
                f"call the full-compute path or extend the hard-only caches."
            )

        # 1. Collect affected nets (nets touching any moved macro slot).
        k = collect_affected_nets(
            moved_slots, self.macro_to_nets_offsets, self.macro_to_nets_flat,
            self._hpwl_mark, self._hpwl_out,
        )
        affected = self._hpwl_out[:k] if k > 0 else np.zeros(0, dtype=np.int32)

        # 2. Subtract old net routing using CACHED pin gcells.
        if k > 0:
            accumulate_net_routing_subset(
                affected_nets=affected,
                pin_row=self.pin_row_cache, pin_col=self.pin_col_cache,
                driver_pin=self.net_driver_pin,
                sink_offsets=self.net_sink_offsets,
                sink_pinks=self.net_sink_pins,
                weights=self.net_weights,
                H=self.H_net_raw, V=self.V_net_raw,
                grid_cols=self.grid_col,
                sign=-1.0, max_sinks_cap=self._max_sinks_per_net,
            )

        # 3. Subtract old macro blockage using cached centers.
        for si in range(moved_slots.shape[0]):
            slot = int(moved_slots[si])
            accumulate_single_macro_blockage(
                mx=float(self.macro_x_cache[slot]),
                my=float(self.macro_y_cache[slot]),
                mw=float(self.hmacro_widths[slot]),
                mh=float(self.hmacro_heights[slot]),
                grid_w=self.grid_width, grid_h=self.grid_height,
                grid_rows=self.grid_row, grid_cols=self.grid_col,
                hrouting_alloc=self.hrouting_alloc,
                vrouting_alloc=self.vrouting_alloc,
                H_macro=self.H_macro_raw, V_macro=self.V_macro_raw,
                sign=-1.0,
            )

        # 4. Refresh pin-gcell cache from current x_pos / y_pos.
        #    Only pins on moved macros actually changed; a full vector
        #    recompute is microseconds, so don't bother being clever.
        self.pin_row_cache, self.pin_col_cache = self._compute_pin_gcells()

        # 5. Add new net routing using fresh pin gcells.
        if k > 0:
            accumulate_net_routing_subset(
                affected_nets=affected,
                pin_row=self.pin_row_cache, pin_col=self.pin_col_cache,
                driver_pin=self.net_driver_pin,
                sink_offsets=self.net_sink_offsets,
                sink_pinks=self.net_sink_pins,
                weights=self.net_weights,
                H=self.H_net_raw, V=self.V_net_raw,
                grid_cols=self.grid_col,
                sign=+1.0, max_sinks_cap=self._max_sinks_per_net,
            )

        # 6. Add new macro blockage, and refresh macro-center cache.
        for si in range(moved_slots.shape[0]):
            slot = int(moved_slots[si])
            mx_new = float(self.x_pos[self.hmacro_indices_np[slot]])
            my_new = float(self.y_pos[self.hmacro_indices_np[slot]])
            accumulate_single_macro_blockage(
                mx=mx_new, my=my_new,
                mw=float(self.hmacro_widths[slot]),
                mh=float(self.hmacro_heights[slot]),
                grid_w=self.grid_width, grid_h=self.grid_height,
                grid_rows=self.grid_row, grid_cols=self.grid_col,
                hrouting_alloc=self.hrouting_alloc,
                vrouting_alloc=self.vrouting_alloc,
                H_macro=self.H_macro_raw, V_macro=self.V_macro_raw,
                sign=+1.0,
            )
            self.macro_x_cache[slot] = mx_new
            self.macro_y_cache[slot] = my_new

        # 7. Normalize + smooth + combine + ABU (cheap, O(cells)).
        return self._finalize_abu()


    # --------------------------------------------------------------
    # Incremental density API
    # --------------------------------------------------------------
    def _density_abu(self) -> float:
        """Top-10% ABU on the normalized density grid, × 0.5.
        Matches reference `get_density_cost` semantics."""
        grid_area = self.grid_width * self.grid_height
        if grid_area <= 0.0:
            return 0.0
        grid_cells = self.grid_occupied_raw / grid_area
        num_cells = grid_cells.size

        # Reference: density_cnt = floor(len(grid_cells) * 0.1); sum top
        # density_cnt of the non-zero values, divide by density_cnt.
        top_n = int(math.floor(num_cells * 0.1))

        if num_cells < 10:
            nz = grid_cells[grid_cells > 0.0]
            if nz.size == 0:
                return 0.0
            return 0.5 * float(nz.mean())

        if top_n == 0:
            return 0.0

        nz = grid_cells[grid_cells > 0.0]
        if nz.size >= top_n:
            # top_n largest values
            top = np.partition(nz, -top_n)[-top_n:]
            return 0.5 * float(top.sum() / top_n)
        # fewer non-zero than top_n: reference sums ALL non-zero, still
        # divides by top_n (density_cnt in the reference).
        return 0.5 * float(nz.sum() / top_n)

    def _build_density_macro_snapshot(self):
        """Populate density_all_{x,y,w,h} from current state.
        Order matches placement: hard macros [0, num_hard), then soft."""
        num_hard = len(self.hard_macro_indices)
        modules = self.modules_w_pins

        for i, macro_idx in enumerate(self.hard_macro_indices):
            mod = modules[macro_idx]
            self.density_all_x[i] = self.x_pos[macro_idx]
            self.density_all_y[i] = self.y_pos[macro_idx]
            self.density_all_w[i] = mod.get_width()
            self.density_all_h[i] = mod.get_height()

        for i, macro_idx in enumerate(self.soft_macro_indices):
            slot = num_hard + i
            mod = modules[macro_idx]
            self.density_all_x[slot] = self.x_pos[macro_idx]
            self.density_all_y[slot] = self.y_pos[macro_idx]
            self.density_all_w[slot] = mod.get_width()
            self.density_all_h[slot] = mod.get_height()

    def init_density_incremental(self):
        """Full rebuild of the raw density grid from current x_pos/y_pos.
        Handles the init-order problem the same way congestion does
        (build_fast_representation pre-sizes with the 10×10 default grid,
        restore_placement changes it; we reallocate here on first call)."""
        self.grid_width = float(self.width / self.grid_col)
        self.grid_height = float(self.height / self.grid_row)

        num_cells = self.grid_row * self.grid_col
        if self.grid_occupied_raw.shape[0] != num_cells:
            self.grid_occupied_raw = np.zeros(num_cells, dtype=np.float32)
        else:
            self.grid_occupied_raw[:] = 0.0

        self._build_density_macro_snapshot()

        accumulate_all_macros_density(
            self.density_all_x, self.density_all_y,
            self.density_all_w, self.density_all_h,
            self.grid_width, self.grid_height,
            self.grid_row, self.grid_col,
            self.grid_occupied_raw,
        )
        self._density_incremental_ready = True

    def invalidate_density_incremental(self):
        self._density_incremental_ready = False

    def get_density_cost_fast(self) -> float:
        """Full-grid numba rebuild, no caching. Matches reference semantics,
        ~1-2 ms on ibm01 vs ~25 ms for the reference Python loop."""
        self.grid_width = float(self.width / self.grid_col)
        self.grid_height = float(self.height / self.grid_row)

        num_cells = self.grid_row * self.grid_col
        if self.grid_occupied_raw.shape[0] != num_cells:
            self.grid_occupied_raw = np.zeros(num_cells, dtype=np.float32)

        self._build_density_macro_snapshot()

        accumulate_all_macros_density(
            self.density_all_x, self.density_all_y,
            self.density_all_w, self.density_all_h,
            self.grid_width, self.grid_height,
            self.grid_row, self.grid_col,
            self.grid_occupied_raw,
        )
        # Full rebuild invalidates any differential state; mark ready so
        # future incremental calls can pick up from this snapshot.
        self._density_incremental_ready = True
        return self._density_abu()

    def get_density_cost_incremental(self, moved_slots) -> float:
        """
        Incremental density cost. Contract:

          * Caller must have updated plc.x_pos / plc.y_pos for the moved
            macros BEFORE calling this (typically via _set_placement_fast).
          * moved_slots is an iterable of HARD-MACRO slot indices in
            [0, num_hard_macros); we don't support moving soft macros
            incrementally (they're assumed fixed during SA).
          * First call (or after invalidation) does a full rebuild.
        """
        if not self._density_incremental_ready:
            self.init_density_incremental()
            return self._density_abu()

        moved_slots = self._as_int32_array(moved_slots)
        if moved_slots.shape[0] == 0:
            return self._density_abu()

        num_hard = int(self.hmacro_indices_np.shape[0])
        if moved_slots.max() >= num_hard or moved_slots.min() < 0:
            raise IndexError(
                f"incremental density: moved_slots out of range for hard-macro "
                f"snapshot (num_hard={num_hard}, got min={int(moved_slots.min())}, "
                f"max={int(moved_slots.max())}). Pass only hard-macro slots."
            )

        for si in range(moved_slots.shape[0]):
            slot = int(moved_slots[si])

            # 1. Subtract old contribution using cached position.
            accumulate_single_macro_density(
                mx=float(self.density_all_x[slot]),
                my=float(self.density_all_y[slot]),
                mw=float(self.density_all_w[slot]),
                mh=float(self.density_all_h[slot]),
                grid_w=self.grid_width, grid_h=self.grid_height,
                grid_rows=self.grid_row, grid_cols=self.grid_col,
                grid_occupied=self.grid_occupied_raw, sign=-1.0,
            )

            # 2. Add new contribution at current position.
            mx_new = float(self.x_pos[self.hmacro_indices_np[slot]])
            my_new = float(self.y_pos[self.hmacro_indices_np[slot]])
            accumulate_single_macro_density(
                mx=mx_new, my=my_new,
                mw=float(self.density_all_w[slot]),
                mh=float(self.density_all_h[slot]),
                grid_w=self.grid_width, grid_h=self.grid_height,
                grid_rows=self.grid_row, grid_cols=self.grid_col,
                grid_occupied=self.grid_occupied_raw, sign=+1.0,
            )

            # 3. Update cache.
            self.density_all_x[slot] = mx_new
            self.density_all_y[slot] = my_new

        return self._density_abu()


    def _get_wirelength(self) -> float:
        """
        Proxy HPWL computation
        """
        # NOTE: in pb.txt, netlist input count exceed certain threshold will be ommitted
        total_hpwl = 0.0

        for mod_idx, mod in enumerate(self.modules_w_pins):
            norm_fact = 1.0
            curr_type = mod.get_type()
            # bounding box data structure
            x_coord = []
            y_coord = []

            # default value of weight
            weight_fact = 1.0

            # NOTE: connection only defined on PORT, soft/hard macro pins
            if curr_type == "PORT" and mod.get_sink():
                # add source position
                x_coord.append(mod.get_pos()[0])
                y_coord.append(mod.get_pos()[1])
                # get sink 
                for sink_name in mod.get_sink():
                    for sink_pin in mod.get_sink()[sink_name]:
                        # retrieve indx in modules_w_pins
                        sink_idx = self.mod_name_to_indices[sink_pin]
                        # retrieve sink object
                        sink = self.modules_w_pins[sink_idx]
                        # only consider placed sink
                        # ref_sink = self.modules_w_pins[self.get_ref_node_id(sink_idx)]
                        # if not placed, skip this edge
                        # if not ref_sink.get_placed_flag():
                        #     x_coord.append(0)
                        #     y_coord.append(0)
                        # else:# retrieve location
                        x_coord.append(self.__get_pin_position(sink_idx)[0])
                        y_coord.append(self.__get_pin_position(sink_idx)[1])

            elif curr_type == "MACRO_PIN":
                ref_mod = self.modules_w_pins[self.get_ref_node_id(mod_idx)]
                # # if not placed, skip this edge
                # if not ref_mod.get_placed_flag():
                #     continue
                # get pin weight
                weight_fact = mod.get_weight()
                # add source position
                x_coord.append(self.__get_pin_position(mod_idx)[0])
                y_coord.append(self.__get_pin_position(mod_idx)[1])

                if mod.get_sink():
                    for input_list in mod.get_sink().values():
                        for sink_name in input_list:
                            # retrieve indx in modules_w_pins
                            input_idx = self.mod_name_to_indices[sink_name]

                            # sink_ref_mod = self.modules_w_pins[self.get_ref_node_id(mod_idx)]
                            # if not placed, skip this edge
                            # if not sink_ref_mod.get_placed_flag():
                            #     x_coord.append(0)
                            #     y_coord.append(0)
                            # else:
                            # retrieve location
                            x_coord.append(self.__get_pin_position(input_idx)[0])
                            y_coord.append(self.__get_pin_position(input_idx)[1])

            if x_coord:
                total_hpwl += weight_fact * \
                    (abs(max(x_coord) - min(x_coord)) + \
                        abs(max(y_coord) - min(y_coord)))
    
        return total_hpwl

    def abu(self, xx, n = 0.1):
        xxs = sorted(xx, reverse = True)
        cnt = math.floor(len(xxs)*n)
        if cnt == 0:
            return max(xxs)
        return sum(xxs[0:cnt])/cnt
    
    def abu_fast(self, H, V, percent=0.1):
        """Mean of the top `percent` (e.g. 0.05) cell values across H and V.

        Matches reference semantics: `math.floor(len * percent)` cells. If
        that count is 0 (tiny grid), returns the single maximum.
        """
        combined = np.concatenate((H.ravel(), V.ravel()))
        n = combined.size

        k = int(math.floor(n * percent))
        if k == 0:
            return float(combined.max())

        idx = n - k                              # partition point for top-k
        part = np.partition(combined, idx)       # values ≥ part[idx] are in [idx:]
        return float(part[idx:].mean())
    
    def get_V_congestion_cost(self) -> float:
        """
        compute average of top 10% of grid cell cong and take half of it
        """
        occupied_cells = sorted([gc for gc in self.V_routing_cong if gc != 0.0], reverse=True)
        cong_cost = 0.0

        # take top 10%
        cong_cnt = math.floor(len(self.V_routing_cong) * 0.1)

        # if grid cell smaller than 10, take the average over occupied cells
        if len(self.V_routing_cong) < 10:
            cong_cost = float(sum(occupied_cells) / len(occupied_cells))
            return cong_cost

        idx = 0
        sum_cong = 0
        # take top 10%
        while idx < cong_cnt and idx < len(occupied_cells):
            sum_cong += occupied_cells[idx]
            idx += 1

        return float(sum_cong / cong_cnt)
    
    def get_H_congestion_cost(self) -> float:
        """
        compute average of top 10% of grid cell cong and take half of it
        """
        occupied_cells = sorted([gc for gc in self.H_routing_cong if gc != 0.0], reverse=True)
        cong_cost = 0.0

        # take top 10%
        cong_cnt = math.floor(len(self.H_routing_cong) * 0.1)

        # if grid cell smaller than 10, take the average over occupied cells
        if len(self.H_routing_cong) < 10:
            cong_cost = float(sum(occupied_cells) / len(occupied_cells))
            return cong_cost

        idx = 0
        sum_cong = 0
        # take top 10%
        while idx < cong_cnt and idx < len(occupied_cells):
            sum_cong += occupied_cells[idx]
            idx += 1

        return float(sum_cong / cong_cnt)
    
    def get_congestion_cost(self):
        """
        Return congestion cost based on routing and macro placement
        """
        if self.FLAG_UPDATE_CONGESTION:
            self.get_routing()

        return self.abu(self.V_routing_cong + self.H_routing_cong, 0.05)
    
    def get_congestion_cost_fast(self):
        self.get_routing_fast()
        return self.abu_fast(self.H_routing_cong_fast, self.V_routing_cong_fast, 0.05)

    def __get_grid_cell_location(self, x_pos, y_pos):
        """
        private function: for getting grid cell row/col ranging from 0...N
        """
        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)
        row = math.floor(y_pos / self.grid_height)
        col = math.floor(x_pos / self.grid_width)
        # Clamp to valid grid range: pins with offsets can fall outside the
        # canvas when their owning macro sits near the edge.
        if row < 0: row = 0
        elif row > self.grid_row - 1: row = self.grid_row - 1
        if col < 0: col = 0
        elif col > self.grid_col - 1: col = self.grid_col - 1
        return row, col

    def __get_grid_cell_location_fast(self, x_pos, y_pos):
        """ In vectorized way, clip the values to be in the range 

        Args:
            x_pos (np.ndarray[np.float32]): array of x positions 
            y_pos (np.ndarray[np.float32]): array of y positions
        """
        
        row = np.floor_divide(x_pos, self.grid_width)
        col = np.floor_divide(y_pos, self.grid_height)
        
        row = np.clip(row, 0, self.grid_row -1)
        col = np.clip(col, 0, self.grid_col -1)
        
        return row.astype(np.int32), col.astype(np.int32)

    def __get_grid_location_position(self, col:int, row:int):
        """
        private function: for getting x y coord from grid cell row/col
        """
        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)
        x_pos = self.grid_width * col + self.grid_width / 2
        y_pos = self.grid_height * row + self.grid_height / 2

        return x_pos, y_pos
    
    def __get_grid_cell_position(self, grid_cell_idx:int):
        """
        private function: for getting x y coord from grid cell row/col
        """
        row = grid_cell_idx // self.grid_col
        col = grid_cell_idx % self.grid_col
        assert row * self.grid_col + col == grid_cell_idx

        return self.__get_grid_location_position(col, row)
    
    def __place_node_mask(self, 
                            grid_cell_idx:int,
                            mod_width:float,
                            mod_height:float
                        ):
        """
        private function: for updating node mask after a placement
        """
        row = grid_cell_idx // self.grid_col
        col = grid_cell_idx % self.grid_col
        assert row * self.grid_col + col == grid_cell_idx

        hor_pad, ver_pad = self.__node_pad_cell(mod_width=mod_width,
                                                mod_height=mod_height)

        self.node_mask[ row - ver_pad:row + ver_pad + 1, 
                        col - hor_pad:col + hor_pad + 1] = 0

    def __overlap_area(self, block_i, block_j, return_pos=False):
        """
        private function: for computing block overlapping
        """
        x_min_max = min(block_i.x_max, block_j.x_max)
        x_max_min = max(block_i.x_min, block_j.x_min)
        y_min_max = min(block_i.y_max, block_j.y_max)
        y_max_min = max(block_i.y_min, block_j.y_min)

        x_diff = x_min_max - x_max_min
        y_diff = y_min_max - y_max_min
        if x_diff >= 0 and y_diff >= 0:
            if return_pos:
                return x_diff * y_diff, (x_min_max, y_min_max), (x_max_min, y_max_min)
            else:
                return x_diff * y_diff
        return 0
    
    def __overlap_dist(self, block_i, block_j):
        """
        private function: for computing block overlapping
        """
        x_diff = min(block_i.x_max, block_j.x_max) - max(block_i.x_min, block_j.x_min)
        y_diff = min(block_i.y_max, block_j.y_max) - max(block_i.y_min, block_j.y_min)
        if x_diff > 0 and y_diff > 0:
            return x_diff, y_diff
        return 0, 0

    def __add_module_to_grid_cells(self, mod_x, mod_y, mod_w, mod_h):
        """
        private function: for add module to grid cells
        """
        # Two corners
        ur = (mod_x + (mod_w/2), mod_y + (mod_h/2))
        bl = (mod_x - (mod_w/2), mod_y - (mod_h/2))

        # construct block based on current module
        module_block = Block(
                            x_max=mod_x + (mod_w/2),
                            y_max=mod_y + (mod_h/2),
                            x_min=mod_x - (mod_w/2),
                            y_min=mod_y - (mod_h/2)
                            )

        # Only need two corners of a grid cell
        ur_row, ur_col = self.__get_grid_cell_location(*ur)
        bl_row, bl_col = self.__get_grid_cell_location(*bl)

        # check if out of bound
        if ur_row >= 0 and ur_col >= 0:
            if bl_row < 0:
                bl_row = 0

            if bl_col < 0:
                bl_col = 0
        else:
            # OOB, skip module
            return

        if bl_row >= 0 and bl_col >= 0:
            if ur_row > self.grid_row - 1:
                ur_row = self.grid_row - 1

            if ur_col > self.grid_col - 1:
                ur_col = self.grid_col - 1
        else:
            # OOB, skip module
            return

        for r_i in range(bl_row, ur_row + 1):
            for c_i in range(bl_col, ur_col + 1):
                # construct block based on current cell row/col
                grid_cell_block = Block(
                                        x_max= (c_i + 1) * self.grid_width,
                                        y_max= (r_i + 1) * self.grid_height,
                                        x_min= c_i * self.grid_width,
                                        y_min= r_i * self.grid_height
                                        )

                self.grid_occupied[self.grid_col * r_i + c_i] += \
                    self.__overlap_area(grid_cell_block, module_block)             

    def get_grid_cells_density(self):
        """
        compute density for all grid cells
        """
        # by default grid row/col is 10/10
        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)

        grid_area = self.grid_width * self.grid_height
        self.grid_occupied = [0] * (self.grid_col * self.grid_row)
        self.grid_cells = [0] * (self.grid_col * self.grid_row)

        for module_idx in (self.soft_macro_indices + self.hard_macro_indices):
            # extract module information
            module = self.modules_w_pins[module_idx]

            # skipping unplaced module
            # if not module.get_placed_flag():
            #     continue

            module_h = module.get_height()
            module_w = module.get_width()
            module_x, module_y = module.get_pos()

            self.__add_module_to_grid_cells(
                                            mod_x=module_x,
                                            mod_y=module_y,
                                            mod_h=module_h,
                                            mod_w=module_w
                                            )

        for i, gcell in enumerate(self.grid_occupied):
            self.grid_cells[i] = gcell / grid_area

        return self.grid_cells

    def get_density_cost(self) -> float:
        """
        compute average of top 10% of grid cell density and take half of it
        """
        if self.FLAG_UPDATE_DENSITY:
            self.get_grid_cells_density()
            self.FLAG_UPDATE_DENSITY=False

        occupied_cells = sorted([gc for gc in self.grid_cells if gc != 0.0], reverse=True)
        density_cost = 0.0

        # take top 10%
        density_cnt = math.floor(len(self.grid_cells) * 0.1)

        # if grid cell smaller than 10, take the average over occupied cells
        if len(self.grid_cells) < 10:
            density_cost = float(sum(occupied_cells) / len(occupied_cells))
            return 0.5 * density_cost

        idx = 0
        sum_density = 0
        # take top 10%
        while idx < density_cnt and idx < len(occupied_cells):
            sum_density += occupied_cells[idx]
            idx += 1

        return 0.5 * float(sum_density / density_cnt)

    def set_canvas_size(self, width:float, height:float) -> float:
        """
        Set canvas size
        """
        self.width = width
        self.height = height

        # Flag updates
        self.FLAG_UPDATE_CONGESTION = True
        self.FLAG_UPDATE_DENSITY = True
        self.FLAG_UPDATE_NODE_MASK = True
        self.__reset_node_mask()
        self.FLAG_UPDATE_MACRO_AND_CLUSTERED_PORT_ADJ = True

        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)
        return True

    def get_canvas_width_height(self) -> Tuple[float, float]:
        """
        Return canvas size
        """
        return self.width, self.height

    def set_placement_grid(self, grid_col:int, grid_row:int) -> bool:
        """
        Set grid col/row
        """
        print("#[PLACEMENT GRID] Col: %d, Row: %d" % (grid_col, grid_row))
        self.grid_col = grid_col
        self.grid_row = grid_row

        # Flag updates
        self.FLAG_UPDATE_CONGESTION = True
        self.FLAG_UPDATE_DENSITY = True
        self.FLAG_UPDATE_NODE_MASK = True
        self.__reset_node_mask()
        self.FLAG_UPDATE_MACRO_AND_CLUSTERED_PORT_ADJ = True

        self.V_routing_cong = [0] * self.grid_col * self.grid_row
        self.H_routing_cong = [0] * self.grid_col * self.grid_row
        self.V_macro_routing_cong = [0] * self.grid_col * self.grid_row
        self.H_macro_routing_cong = [0] * self.grid_col * self.grid_row

        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)
        return True

    def get_grid_num_columns_rows(self) -> Tuple[int, int]:
        """
        Return grid col/row
        """
        return self.grid_col, self.grid_row

    def get_macro_indices(self) -> list:
        """
        Return all macro indices
        """
        return sorted(self.hard_macro_indices + self.soft_macro_indices)

    def set_project_name(self, project_name):
        """
        Set Project name
        """
        self.project_name = project_name

    def get_project_name(self) -> str:
        """
        Return Project name
        """
        return self.project_name
    
    def set_block_name(self, block_name:str) -> None:
        """
        Return Block name
        """
        self.block_name = block_name

    def get_block_name(self) -> str:
        """
        Return Block name
        """
        return self.block_name

    def set_routes_per_micron(self, hroutes_per_micron:float, vroutes_per_micron:float) -> None:
        """
        Set Routes per Micron
        """
        print("#[ROUTES PER MICRON] Hor: %.2f, Ver: %.2f" % (hroutes_per_micron, vroutes_per_micron))
        # Flag updates
        self.FLAG_UPDATE_CONGESTION = True

        self.hroutes_per_micron = hroutes_per_micron
        self.vroutes_per_micron = vroutes_per_micron

    def get_routes_per_micron(self) -> Tuple[float, float]:
        """
        Return Routes per Micron
        """
        return self.hroutes_per_micron, self.vroutes_per_micron

    def set_congestion_smooth_range(self, smooth_range:float) -> None:
        """
        Set congestion smooth range
        """
        print("#[CONGESTION SMOOTH RANGE] Smooth Range: %d" % (smooth_range))
        # Flag updates
        self.FLAG_UPDATE_CONGESTION = True

        self.smooth_range = math.floor(smooth_range)

    def get_congestion_smooth_range(self) -> float:
        """
        Return congestion smooth range
        """
        return self.smooth_range

    def set_overlap_threshold(self, overlap_thres:float) -> None:
        """
        Set Overlap Threshold
        """
        print("#[OVERLAP THRESHOLD] Threshold: %.4f" % (overlap_thres))
        self.overlap_thres = overlap_thres

    def get_overlap_threshold(self) -> float:
        """
        Return Overlap Threshold
        """
        return self.overlap_thres

    def set_canvas_boundary_check(self, ifCheck:bool) -> None:
        """
        boundary_check: Do a boundary check during node placement.
        """
        self.canvas_boundary_check = ifCheck

    def get_canvas_boundary_check(self) -> bool:
        """
        return canvas_boundary_check
        """
        return self.canvas_boundary_check

    def set_macro_routing_allocation(self, hrouting_alloc:float, vrouting_alloc:float) -> None:
        """
        Set Vertical/Horizontal Macro Allocation
        """
        # Flag updates
        self.FLAG_UPDATE_CONGESTION = True

        self.hrouting_alloc = hrouting_alloc
        self.vrouting_alloc = vrouting_alloc

    def get_macro_routing_allocation(self) -> Tuple[float, float]:
        """
        Return Vertical/Horizontal Macro Allocation
        """
        return self.hrouting_alloc, self.vrouting_alloc

    def __two_pin_net_routing(self, source_gcell, node_gcells, weight):
        """
        private function: Routing between 2-pin nets
        """
        temp_gcell = list(node_gcells)
        if temp_gcell[0] == source_gcell:
            sink_gcell = temp_gcell[1]
        else:
            sink_gcell = temp_gcell[0]

        # y
        row_min = min(sink_gcell[0], source_gcell[0])
        row_max = max(sink_gcell[0], source_gcell[0])

        # x
        col_min = min(sink_gcell[1], source_gcell[1])
        col_max = max(sink_gcell[1], source_gcell[1])

        # H routing
        for col_idx in range(col_min, col_max, 1):
            col = col_idx
            row = source_gcell[0]
            self.H_routing_cong[row * self.grid_col + col] += weight

        # V routing
        for row_idx in range(row_min, row_max, 1):
            row = row_idx
            col = sink_gcell[1]
            self.V_routing_cong[row * self.grid_col + col] += weight

    def __l_routing(self, node_gcells, weight):
        """
        private function: L_shape routing in 3-pin nets
        """
        node_gcells.sort(key = lambda x: (x[1], x[0]))
        y1, x1 = node_gcells[0]
        y2, x2 = node_gcells[1]
        y3, x3 = node_gcells[2]
        # H routing (x1, y1) to (x2, y1)
        for col in range(x1, x2):
            row = y1
            self.H_routing_cong[row * self.grid_col + col] += weight
        
        # H routing (x2, y2) to (x2, y3)
        for col in range(x2,x3):
            row = y2
            self.H_routing_cong[row * self.grid_col + col] += weight
        
        # V routing (x2, min(y1, y2)) to (x2, max(y1, y2))
        for row in range(min(y1, y2), max(y1, y2)):
            col = x2
            self.V_routing_cong[row * self.grid_col + col] += weight
        
        # V routing (x3, min(y2, y3)) to (x3, max(y2, y3))
        for row in range(min(y2, y3), max(y2, y3)):
            col = x3
            self.V_routing_cong[row * self.grid_col + col] += weight

    def __t_routing(self, node_gcells, weight):
        """
        private function: T_shape routing in 3-pin nets
        """
        node_gcells.sort()
        y1, x1 = node_gcells[0]
        y2, x2 = node_gcells[1]
        y3, x3 = node_gcells[2]
        xmin = min(x1, x2, x3)
        xmax = max(x1, x2, x3)

        # H routing (xmin, y2) to (xmax, y2)
        for col in range(xmin, xmax):
            row = y2
            self.H_routing_cong[row * self.grid_col + col] += weight
        
        # V routing (x1, y1) to (x1, y2)
        for row in range(min(y1, y2), max(y1, y2)):
            col = x1
            self.V_routing_cong[row * self.grid_col + col] += weight
        
        # V routing (x3, y3) to (x3, y2)
        for row in range(min(y2, y3), max(y2, y3)):
            col = x3
            self.V_routing_cong[row * self.grid_col + col] += weight

    def __three_pin_net_routing(self, node_gcells, weight):
        """
        private_function: Routing Scheme for 3-pin nets
        """
        temp_gcell = list(node_gcells)
        ## Sorted based on X
        temp_gcell.sort(key = lambda x: (x[1], x[0]))
        y1, x1 = temp_gcell[0]
        y2, x2 = temp_gcell[1]
        y3, x3 = temp_gcell[2]

        if x1 < x2 and x2 < x3 and min(y1, y3) < y2 and max(y1, y3) > y2:
            self.__l_routing(temp_gcell, weight)
        elif x2 == x3 and x1 < x2 and y1 < min(y2, y3):
            for col_idx in range(x1,x2,1):
                row = y1
                col = col_idx
                self.H_routing_cong[row * self.grid_col + col] += weight
            
            for row_idx in range(y1, max(y2,y3)):
                col = x2
                row = row_idx
                self.V_routing_cong[row * self.grid_col + col] += weight
        elif y2 == y3:
            for col in range(x1, x2):
                row = y1
                self.H_routing_cong[row * self.grid_col + col] += weight
            
            for col in range(x2, x3):
                row = y2
                self.H_routing_cong[row * self.grid_col + col] += weight
            
            for row in range(min(y2, y1), max(y2, y1)):
                col = x2
                self.V_routing_cong[row * self.grid_col + col] += weight
        else: 
            self.__t_routing(temp_gcell, weight)

    def __macro_route_over_grid_cell(self, mod_x, mod_y, mod_w, mod_h):
        """
        private function for add module to grid cells
        """
        # Two corners
        ur = (mod_x + (mod_w/2), mod_y + (mod_h/2))
        bl = (mod_x - (mod_w/2), mod_y - (mod_h/2))

        # construct block based on current module
        module_block = Block(
                            x_max=mod_x + (mod_w/2),
                            y_max=mod_y + (mod_h/2),
                            x_min=mod_x - (mod_w/2),
                            y_min=mod_y - (mod_h/2)
                            )

        # Only need two corners of a grid cell
        ur_row, ur_col = self.__get_grid_cell_location(*ur)
        bl_row, bl_col = self.__get_grid_cell_location(*bl)

        # check if out of bound
        if ur_row >= 0 and ur_col >= 0:
            if bl_row < 0:
                bl_row = 0

            if bl_col < 0:
                bl_col = 0
        else:
            # OOB, skip module
            return

        if bl_row >= 0 and bl_col >= 0:
            if ur_row > self.grid_row - 1:
                ur_row = self.grid_row - 1

            if ur_col > self.grid_col - 1:
                ur_col = self.grid_col - 1
        else:
            # OOB, skip module
            return
        
        if_PARTIAL_OVERLAP_VERTICAL = False
        if_PARTIAL_OVERLAP_HORIZONTAL = False

        for r_i in range(bl_row, ur_row + 1):
            for c_i in range(bl_col, ur_col + 1):
                # construct block based on current cell row/col
                grid_cell_block = Block(
                                        x_max= (c_i + 1) * self.grid_width,
                                        y_max= (r_i + 1) * self.grid_height,
                                        x_min= c_i * self.grid_width,
                                        y_min= r_i * self.grid_height
                                        )

                x_dist, y_dist = self.__overlap_dist(module_block, grid_cell_block)

                if ur_row != bl_row:
                    if (r_i == bl_row and abs(y_dist - self.grid_height) > 1e-5) or (r_i == ur_row and abs(y_dist - self.grid_height) > 1e-5):
                        if_PARTIAL_OVERLAP_VERTICAL = True
                
                if ur_col != bl_col:
                    if (c_i == bl_col and abs(x_dist - self.grid_width) > 1e-5) or (c_i == ur_col and abs(x_dist - self.grid_width) > 1e-5):
                        if_PARTIAL_OVERLAP_HORIZONTAL = True


                self.V_macro_routing_cong[r_i * self.grid_col + c_i] += x_dist * self.vrouting_alloc
                self.H_macro_routing_cong[r_i * self.grid_col + c_i] += y_dist * self.hrouting_alloc

        if if_PARTIAL_OVERLAP_VERTICAL:
            for r_i in range(ur_row, ur_row + 1):
                for c_i in range(bl_col, ur_col + 1):
                    grid_cell_block = Block(
                                        x_max= (c_i + 1) * self.grid_width,
                                        y_max= (r_i + 1) * self.grid_height,
                                        x_min= c_i * self.grid_width,
                                        y_min= r_i * self.grid_height
                                        )

                    x_dist, y_dist = self.__overlap_dist(module_block, grid_cell_block)
                    self.V_macro_routing_cong[r_i * self.grid_col + c_i] -= x_dist * self.vrouting_alloc

        if if_PARTIAL_OVERLAP_HORIZONTAL:
            for r_i in range(bl_row, ur_row + 1):
                for c_i in range(ur_col, ur_col + 1):
                    grid_cell_block = Block(
                                        x_max= (c_i + 1) * self.grid_width,
                                        y_max= (r_i + 1) * self.grid_height,
                                        x_min= c_i * self.grid_width,
                                        y_min= r_i * self.grid_height
                                        )

                    x_dist, y_dist = self.__overlap_dist(module_block, grid_cell_block)
                    self.H_macro_routing_cong[r_i * self.grid_col + c_i] -= y_dist * self.hrouting_alloc

    def __split_net(self, source_gcell, node_gcells):
        """
        private function: Split >3 pin net into multiple two-pin nets
        """
        splitted_netlist = []
        for node_gcell in node_gcells:
            if node_gcell != source_gcell:
                splitted_netlist.append({source_gcell, node_gcell})
        return splitted_netlist

    def get_vertical_routing_congestion(self):
        """
        Return Vertical Routing Congestion
        """
        if self.FLAG_UPDATE_CONGESTION:
            self.get_routing()
        
        return self.V_routing_cong

    def get_horizontal_routing_congestion(self):
        """
        Return Horizontal Routing Congestion
        """
        if self.FLAG_UPDATE_CONGESTION:
            self.get_routing()
        
        return self.H_routing_cong
                

    def get_routing(self):
        """
        H/V Routing Before Computing Routing Congestions
        """
        if self.FLAG_UPDATE_CONGESTION:
            self.grid_width = float(self.width/self.grid_col)
            self.grid_height = float(self.height/self.grid_row)

            self.grid_v_routes = self.grid_width * self.vroutes_per_micron
            self.grid_h_routes = self.grid_height * self.hroutes_per_micron

            # reset grid
            self.H_routing_cong = [0] * self.grid_row * self.grid_col
            self.V_routing_cong = [0] * self.grid_row * self.grid_col

            self.H_macro_routing_cong = [0] * self.grid_row * self.grid_col
            self.V_macro_routing_cong = [0] * self.grid_row * self.grid_col

            self.FLAG_UPDATE_CONGESTION = False

        for mod in self.modules_w_pins:
            curr_type = mod.get_type()
            # bounding box data structure
            node_gcells = set()
            source_gcell = None
            weight = 1

            # NOTE: connection only defined on PORT, soft/hard macro pins
            if curr_type == "PORT" and mod.get_sink():
                # add source grid location
                source_gcell = self.__get_grid_cell_location(*(mod.get_pos()))
                node_gcells.add(self.__get_grid_cell_location(*(mod.get_pos())))

                for sink_name in mod.get_sink():
                    for sink_pin in mod.get_sink()[sink_name]:
                        # retrieve indx in modules_w_pins
                        sink_idx = self.mod_name_to_indices[sink_pin]
                        # retrieve sink object
                        sink = self.modules_w_pins[sink_idx]
                        # retrieve grid location
                        node_gcells.add(self.__get_grid_cell_location(*(self.__get_pin_position(sink_idx))))

            elif curr_type == "MACRO_PIN" and mod.get_sink():
                # add source position
                mod_idx = self.mod_name_to_indices[mod.get_name()]
                node_gcells.add(self.__get_grid_cell_location(*(self.__get_pin_position(mod_idx))))
                source_gcell = self.__get_grid_cell_location(*(self.__get_pin_position(mod_idx)))

                if mod.get_weight() > 1:
                    weight = mod.get_weight()

                for input_list in mod.get_sink().values():
                    for sink_name in input_list:
                        # retrieve indx in modules_w_pins
                        sink_idx = self.mod_name_to_indices[sink_name]
                        # retrieve sink object
                        sink = self.modules_w_pins[sink_idx]
                        # retrieve grid location                                                                                                                                                                                                                                                 
                        node_gcells.add(self.__get_grid_cell_location(*(self.__get_pin_position(sink_idx))))
            
            elif curr_type == "MACRO" and self.is_node_hard_macro(self.mod_name_to_indices[mod.get_name()]):
                module_h = mod.get_height()
                module_w = mod.get_width()
                module_x, module_y = mod.get_pos()
                # compute overlap
                self.__macro_route_over_grid_cell(module_x, module_y, module_w, module_h)
            
            if len(node_gcells) == 2:
                self.__two_pin_net_routing(source_gcell=source_gcell,node_gcells=node_gcells, weight=weight)
            elif len(node_gcells) == 3:
                self.__three_pin_net_routing(node_gcells=node_gcells, weight=weight)
            elif len(node_gcells) > 3:
                for curr_net in self.__split_net(source_gcell=source_gcell, node_gcells=node_gcells):
                    self.__two_pin_net_routing(source_gcell=source_gcell, node_gcells=curr_net, weight=weight)

        # normalize routing congestion
        for idx, v_gcell in enumerate(self.V_routing_cong):
            self.V_routing_cong[idx] = float(v_gcell / self.grid_v_routes)
        
        for idx, h_gcell in enumerate(self.H_routing_cong):
            self.H_routing_cong[idx] = float(h_gcell / self.grid_h_routes)
        
        for idx, v_gcell in enumerate(self.V_macro_routing_cong):
            self.V_macro_routing_cong[idx] = float(v_gcell / self.grid_v_routes)
        
        for idx, h_gcell in enumerate(self.H_macro_routing_cong):
            self.H_macro_routing_cong[idx] = float(h_gcell / self.grid_h_routes)
        
        self.__smooth_routing_cong()

        # sum up routing congestion with macro congestion
        self.V_routing_cong = [sum(x) for x in zip(self.V_routing_cong, self.V_macro_routing_cong)]
        self.H_routing_cong = [sum(x) for x in zip(self.H_routing_cong, self.H_macro_routing_cong)]
 
    def get_routing_fast(self):
        """
            Fast version of get routing, using numba JIT and simple array operations.  
        """
        
        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)

        self.grid_v_routes = self.grid_width * self.vroutes_per_micron
        self.grid_h_routes = self.grid_height * self.hroutes_per_micron

        self.grid_v_routes = self.grid_width * self.vroutes_per_micron
        self.grid_h_routes = self.grid_height * self.hroutes_per_micron

        # reset grid
        self.H_routing_cong_fast = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.V_routing_cong_fast = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.H_macro_routing_cong_fast = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        self.V_macro_routing_cong_fast = np.zeros(self.grid_row * self.grid_col, dtype=np.float32)
        
        pin_row = np.clip(np.floor_divide(self.y_pos, self.grid_height).astype(np.int32), 0, self.grid_row - 1)
        pin_col = np.clip(np.floor_divide(self.x_pos, self.grid_width).astype(np.int32), 0, self.grid_col - 1)

        
        # accumulate the routing congestion 
        accumulate_net_routing(pin_row = pin_row, 
                               pin_col = pin_col, 
                               driver_pin = self.net_driver_pin, 
                               sink_offsets = self.net_sink_offsets, 
                               sink_pinks = self.net_sink_pins, 
                               weights = self.net_weights, 
                               H = self.H_routing_cong_fast, 
                               V = self.V_routing_cong_fast,
                               grid_cols = self.grid_col )
        
        # now compute the macro blockage congestion
        accumulate_macro_blockage(macro_x = self.x_pos[self.hmacro_indices_np],
                                  macro_y = self.y_pos[self.hmacro_indices_np],
                                  macro_w = self.hmacro_widths,
                                  macro_h = self.hmacro_heights,
                                  grid_w = self.grid_width,
                                  grid_h = self.grid_height,
                                  grid_rows = self.grid_row,
                                  grid_cols = self.grid_col,
                                  hrouting_alloc= self.hrouting_alloc,
                                  vrouting_alloc= self.vrouting_alloc,
                                  H_macro = self.H_macro_routing_cong_fast,
                                  V_macro = self.V_macro_routing_cong_fast )
        
        # normalize routing congestion
        self.H_routing_cong_fast /= self.grid_h_routes
        self.V_routing_cong_fast /= self.grid_v_routes
        
        # macro routing cong nromalize
        self.V_macro_routing_cong_fast /= self.grid_v_routes
        self.H_macro_routing_cong_fast /= self.grid_h_routes
        
        # smooth routing congestion
        smooth_routing_cong_fast(grid_col=self.grid_col,
                                    grid_row=self.grid_row,
                                    smooth_range=self.smooth_range,
                                    V_routing_cong=self.V_routing_cong_fast,
                                    H_routing_cong=self.H_routing_cong_fast)
        
        
        # sum up routing congestion with macro congestion
        self.V_routing_cong_fast = self.V_routing_cong_fast + self.V_macro_routing_cong_fast
        self.H_routing_cong_fast = self.H_routing_cong_fast + self.H_macro_routing_cong_fast
        
        
 
    def __smooth_routing_cong(self):
        """
        Smoothing V/H Routing congestion
        """
        temp_V_routing_cong = [0] * self.grid_col * self.grid_row
        temp_H_routing_cong = [0] * self.grid_col * self.grid_row

        # v routing cong
        for row in range(self.grid_row):
            for col in range(self.grid_col):
                lp = col - self.smooth_range
                if lp < 0:
                    lp = 0

                rp = col + self.smooth_range
                if rp >= self.grid_col:
                    rp = self.grid_col - 1
                
                gcell_cnt = rp - lp + 1

                val = self.V_routing_cong[row * self.grid_col + col] / gcell_cnt

                for ptr in range(lp, rp + 1, 1):
                    temp_V_routing_cong[row * self.grid_col + ptr] += val
        
        self.V_routing_cong = temp_V_routing_cong

        # h routing cong
        for row in range(self.grid_row):
            for col in range(self.grid_col):
                lp = row - self.smooth_range
                if lp < 0:
                    lp = 0

                up = row + self.smooth_range
                if up >= self.grid_row:
                    up = self.grid_row - 1
                
                gcell_cnt = up - lp + 1

                val = self.H_routing_cong[row * self.grid_col + col] / gcell_cnt

                for ptr in range(lp, up + 1, 1):
                    temp_H_routing_cong[ptr * self.grid_col + col] += val
        
        self.H_routing_cong = temp_H_routing_cong


    def is_node_soft_macro(self, node_idx) -> bool:
        """
        Return if node is a soft macro
        """
        try:
            return node_idx in self.soft_macro_indices
        except IndexError:
            print("[ERROR INDEX OUT OF RANGE] Can not process index at {}".format(node_idx))
            exit(1)

    def is_node_hard_macro(self, node_idx) -> bool:
        """
        Return if node is a hard macro
        """
        try:
            return node_idx in self.hard_macro_indices
        except IndexError:
            print("[ERROR INDEX OUT OF RANGE] Can not process index at {}".format(node_idx))
            exit(1)

    def get_node_name(self, node_idx: int) -> str:
        """
        Return node name based on given node index
        """
        try:
            return self.indices_to_mod_name[node_idx]
        except Exception:
            print("[ERROR NODE INDEX] Node not found!")
            exit(1)
    
    def get_node_index(self, node_name: str) -> int:
        """
        Return node index based on given node name
        """
        try:
            return self.mod_name_to_indices[node_name]
        except Exception:
            print("[ERROR NODE NAME] Node not found!")
            exit(1)
    
    def get_node_mask(self, node_idx: int, node_name: str=None) -> list:
        """
        Return node mask based on given node
        All legal positions must satisfy:
            - No Out-of-Bound
            - No Overlapping with previously placed MACROs
        """
        mod = self.modules_w_pins[node_idx]

        canvas_block = Block(x_max=self.width,
                            y_max=self.height,
                            x_min=0,
                            y_min=0)
        if mod.get_type() == "PORT" or mod.get_type() == "MACRO_PIN":
            mod_w = 1e-3
            mod_h = 1e-3
        else:
            mod_w = mod.get_width()
            mod_h = mod.get_height()

        temp_node_mask = np.array([1] * (self.grid_col * self.grid_row))\
            .reshape(self.grid_row, self.grid_col)

        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)

        for i in range(self.grid_row):
            for j in range(self.grid_col):
                # try every location
                # construct block based on current module dimenstion
                temp_x = j * self.grid_width + (self.grid_width/2)
                temp_y = i * self.grid_height + (self.grid_height/2)

                mod_block = Block(
                                    x_max=temp_x + (mod_w/2),
                                    y_max=temp_y + (mod_h/2),
                                    x_min=temp_x - (mod_w/2),
                                    y_min=temp_y - (mod_h/2)
                                )
                # check OOB
                if abs(self.__overlap_area(
                    block_i=canvas_block, block_j=mod_block) - (mod_w*mod_h)) > 1e-8:
                    temp_node_mask[i][j] = 0
                else:
                    # check overlapping
                    for pmod_idx in self.placed_macro:
                        pmod = self.modules_w_pins[pmod_idx]
                        if not pmod.get_placed_flag():
                            continue

                        p_x, p_y = pmod.get_pos()
                        p_w = pmod.get_width()
                        p_h = pmod.get_height()
                        pmod_block = Block(
                                            x_max=p_x + (p_w/2),
                                            y_max=p_y + (p_h/2),
                                            x_min=p_x - (p_w/2),
                                            y_min=p_y - (p_h/2)
                                            )
                        # if overlap with placed module
                        if self.__overlap_area(block_i=pmod_block, block_j=mod_block) > 0:
                             temp_node_mask[i][j] = 0
            
        return temp_node_mask.flatten()


    def get_node_type(self, node_idx: int) -> str:
        """
        Return node type
        """
        try:
            return self.modules_w_pins[node_idx].get_type()
        except IndexError:
            # NOTE: Google's API return NONE if out of range
            print("[WARNING INDEX OUT OF RANGE] Can not process index at {}".format(node_idx))
            return None

    
    def get_node_width_height(self, node_idx: int):
        """
        Return node dimension
        """
        mod = None
        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR NODE FIXED] Found {}. Only 'MACRO', 'macro', 'STDCELL'".format(mod.get_type())
                    +"'PORT' are considered to be fixable nodes")
            exit(1)
        except Exception:
            print("[ERROR NODE FIXED] Could not find module by node index")
            exit(1)

        return mod.get_width(), mod.get_height()

    def make_soft_macros_square(self):
        """
        [IGNORE] THIS DOES NOT AFFECT DENSITY. SHOULD WE IMPLEMENT THIS AT ALL?
        make soft macros as squares
        """
        for mod_idx in self.soft_macro_indices:
            mod = self.modules_w_pins[mod_idx]
            mod_area = mod.get_width() * mod.get_height()
            mod.set_width(math.sqrt(mod_area))
            mod.set_height(math.sqrt(mod_area))

    def update_soft_macros_position(self, coord_dict):
        """
        For sync-up with Google's plc_client after FD placer
        """
        for mod_idx in coord_dict.keys():
            self.modules_w_pins[mod_idx].set_pos(coord_dict[mod_idx])

    def set_soft_macro_position(self, node_idx, x_pos, y_pos):
        """
        used for updating soft macro position
        """
        self.modules_w_pins[node_idx].set_pos(x_pos, y_pos)

    def set_use_incremental_cost(self, use_incremental_cost):
        """
        NOT IMPLEMENTED
        """
        self.use_incremental_cost = use_incremental_cost

    def get_use_incremental_cost(self):
        """
        NOT IMPLEMENTED
        """
        return self.use_incremental_cost

    def get_macro_adjacency(self) -> list:
        """
        Compute Adjacency Matrix
        """
        # NOTE: in pb.txt, netlist input count exceed certain threshold will be ommitted
        #[MACRO][macro]

        if self.FLAG_UPDATE_MACRO_ADJ:
            # do some update
            self.FLAG_UPDATE_MACRO_ADJ = False

        module_indices = self.hard_macro_indices + self.soft_macro_indices
        macro_adj = [0] * (self.hard_macro_cnt + self.soft_macros_cnt) * (self.hard_macro_cnt + self.soft_macros_cnt)
        assert len(macro_adj) == (self.hard_macro_cnt + self.soft_macros_cnt) * (self.hard_macro_cnt + self.soft_macros_cnt)

        for row_idx, module_idx in enumerate(sorted(module_indices)):
            # row index
            # store temp module
            curr_module = self.modules_w_pins[module_idx]
            # get module name
            curr_module_name = curr_module.get_name()

            for col_idx, h_module_idx in enumerate(sorted(module_indices)):
                # col index
                entry = 0
                # store connected module
                h_module = self.modules_w_pins[h_module_idx]
                # get connected module name
                h_module_name = h_module.get_name()

                if curr_module_name in h_module.get_connection():
                    entry += h_module.get_connection()[curr_module_name]

                if h_module_name in curr_module.get_connection():
                    entry += curr_module.get_connection()[h_module_name]

                macro_adj[row_idx * (self.hard_macro_cnt + self.soft_macros_cnt) + col_idx] = entry
                macro_adj[col_idx * (self.hard_macro_cnt + self.soft_macros_cnt) + row_idx] = entry

        return macro_adj

    def get_macro_and_clustered_port_adjacency(self):
        """
        Compute Adjacency Matrix (Unclustered PORTs)
        if module is a PORT, assign it to nearest cell location even if OOB
        """

        #[MACRO][macro]
        module_indices = self.hard_macro_indices + self.soft_macro_indices

        #[Grid Cell] => [PORT]
        clustered_ports = {}
        for port_idx in self.port_indices:
            port = self.modules_w_pins[port_idx]
            x_pos, y_pos = port.get_pos()

            row, col = self.__get_grid_cell_location(x_pos=x_pos, y_pos=y_pos)

            # prevent OOB
            if row >= self.grid_row:
                row = self.grid_row - 1
            
            if row < 0:
                row = 0

            if col >= self.grid_col:
                col = self.grid_col - 1
            
            if col < 0:
                col = 0

            if (row, col) in clustered_ports:
                clustered_ports[(row, col)].append(port)
            else:
                clustered_ports[(row, col)] = [port]
        
        # NOTE: in pb.txt, netlist input count exceed certain threshold will be ommitted
        macro_adj = [0] * (len(module_indices) + len(clustered_ports)) * (len(module_indices) + len(clustered_ports))
        cell_location = [0] * len(clustered_ports)

        # instantiate macros
        for row_idx, module_idx in enumerate(sorted(module_indices)):
            # store temp module
            curr_module = self.modules_w_pins[module_idx]
            # get module name
            curr_module_name = curr_module.get_name()

            for col_idx, h_module_idx in enumerate(sorted(module_indices)):
                # col index
                entry = 0
                # store connected module
                h_module = self.modules_w_pins[h_module_idx]
                # get connected module name
                h_module_name = h_module.get_name()

                if curr_module_name in h_module.get_connection():
                    entry += h_module.get_connection()[curr_module_name]

                if h_module_name in curr_module.get_connection():
                    entry += curr_module.get_connection()[h_module_name]

                macro_adj[row_idx * (len(module_indices) + len(clustered_ports)) + col_idx] = entry
                macro_adj[col_idx * (len(module_indices) + len(clustered_ports)) + row_idx] = entry
        
        # instantiate clustered ports
        for row_idx, cluster_cell in enumerate(sorted(clustered_ports, key=lambda tup: tup[1])):
            # add cell location
            cell_location[row_idx] = cluster_cell[0] * self.grid_col + cluster_cell[1]

            # relocate to after macros
            row_idx += len(module_indices)

            # for each port within a grid cell
            for curr_port in clustered_ports[cluster_cell]:
                # get module name
                curr_port_name = curr_port.get_name()
                # assuming ports only connects to macros
                for col_idx, h_module_idx in enumerate(module_indices):
                    # col index
                    entry = 0
                    # store connected module
                    h_module = self.modules_w_pins[h_module_idx]
                    # get connected module name
                    h_module_name = h_module.get_name()
                
                    if curr_port_name in h_module.get_connection():
                        entry += h_module.get_connection()[curr_port_name]

                    if h_module_name in curr_port.get_connection():
                        entry += curr_port.get_connection()[h_module_name]
            
                    macro_adj[row_idx * (len(module_indices) + len(clustered_ports)) + col_idx] += entry
                    macro_adj[col_idx * (len(module_indices) + len(clustered_ports)) + row_idx] += entry
                
        return macro_adj, sorted(cell_location)

    def is_node_fixed(self, node_idx: int):
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR NODE FIXED] Found {}. Only 'MACRO', 'STDCELL'".format(mod.get_type())
                    +"'PORT' are considered to be fixable nodes")
            exit(1)
        except Exception:
            print("[ERROR NODE FIXED] Could not find module by node index")
            exit(1)

        return mod.get_fix_flag()

    def update_node_coords(self, node_idx, x_pos, y_pos):
        """
        Update Node location if node is 'MACRO', 'STDCELL', 'PORT'
        """
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR NODE LOCATION] Found {}. Only 'MACRO', 'macro', 'STDCELL'".format(mod.get_type())
                    +"'PORT' are considered to be placable nodes")
            exit(1)
        except Exception:
            print("[ERROR NODE LOCATION] Could not find module by node index")
            exit(1)
        
        mod.set_pos(x_pos, y_pos)

    def update_macro_orientation(self, node_idx, orientation):
        """ 
        Update macro orientation if node is 'MACRO'
        """
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO']
        except AssertionError:
            print("[ERROR MACRO ORIENTATION] Found {}. Only 'MACRO'".format(mod.get_type())
                    +" are considered to be ORIENTED")
            exit(1)
        except Exception:
            print("[ERROR MACRO ORIENTATION] Could not find module by node index")
            exit(1)
        
        mod.set_orientation(orientation)

        macro = self.modules_w_pins[node_idx]
        macro_name = macro.get_name()
        hard_macro_pins = self.hard_macros_to_inpins[macro_name]
        
        orientation = macro.get_orientation()

        # update all pin offset
        for pin_name in hard_macro_pins:
            pin = self.modules_w_pins[self.mod_name_to_indices[pin_name]]

            x_offset, y_offset = pin.get_offset()
            x_offset_org = x_offset
            if orientation == "N":
                pass
            elif orientation == "FN":
                x_offset = -x_offset
                pin.set_offset(x_offset, y_offset)
            elif orientation == "S":
                x_offset = -x_offset
                y_offset = -y_offset
                pin.set_offset(x_offset, y_offset)
            elif orientation == "FS":
                y_offset = -y_offset
                pin.set_offset(x_offset, y_offset)
            elif orientation == "E":
                x_offset = y_offset
                y_offset = -x_offset_org
                pin.set_offset(x_offset, y_offset)
            elif orientation == "FE":
                x_offset = -y_offset
                y_offset = -x_offset_org
                pin.set_offset(x_offset, y_offset)
            elif orientation == "W":
                x_offset = -y_offset
                y_offset = x_offset_org
                pin.set_offset(x_offset, y_offset)
            elif orientation == "FW":
                x_offset = y_offset
                y_offset = x_offset_org
                pin.set_offset(x_offset, y_offset)

    def update_port_sides(self):
        """
        Define Port "Side" by its location on canvas
        """
        pass

    def snap_ports_to_edges(self):
        pass

    def get_node_location(self, node_idx):
        """ 
        Return Node location if node is 'MACRO', 'STDCELL', 'PORT'
        """
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR NODE LOCATION] Found {}. Only 'MACRO', 'STDCELL'".format(mod.get_type())
                    +"'PORT' are considered to be placable nodes")
            exit(1)
        except Exception:
            print("[ERROR NODE PLACED] Could not find module by node index")
            exit(1)
        
        return mod.get_pos()
                       
    def get_grid_cell_of_node(self, node_idx):
        """ if grid_cell at grid crossing, break-tie to upper right
        """
        mod = None
        
        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO']
        except AssertionError:
            print("[ERROR NODE LOCATION] Found {}. Only 'MACRO'".format(mod.get_type())
                    +" can be called")
            exit(1)
        except Exception:
            print("[ERROR NODE LOCATION] Could not find module by node index")
            exit(1)
        
        row, col = self.__get_grid_cell_location(*mod.get_pos())

        return row * self.grid_col + col

    def get_macro_orientation(self, node_idx):
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO']
        except AssertionError:
            print("[ERROR MACRO ORIENTATION] Found {}. Only 'MACRO'".format(mod.get_type())
                    +" are considered to be ORIENTED")
            exit(1)
        except Exception:
            print("[ERROR MACRO ORIENTATION] Could not find module by node index")
            exit(1)
        
        return mod.get_orientation()

    def unfix_node_coord(self, node_idx):
        """
        Unfix a module
        """
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR UNFIX NODE] Found {}. Only 'MACRO', 'STDCELL'".format(mod.get_type())
                    +"'PORT' are considered to be fixable nodes")
            exit(1)
        except Exception:
            print("[ERROR UNFIX NODE] Could not find module by node index")
            exit(1)

        self.modules_w_pins[node_idx].set_fix_flag(False)
    
    def fix_node_coord(self, node_idx):
        """
        Fix a module
        """
        mod = None
        
        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR FIX NODE] Found {}. Only 'MACRO', 'STDCELL'"\
                .format(mod.get_type())
                +"'PORT' are considered to be fixable nodes")
            exit(1)
        except Exception:
            print("[ERROR FIX NODE] Could not find module by node index")
            exit(1)

        self.modules_w_pins[node_idx].set_fix_flag(True)

    def __update_node_mask(self):
        """
        TODO: should we reload the placed node?
        NOTE: NOT USED
        """
        return

        self.node_mask = np.array([1] * (self.grid_col * self.grid_row)).\
                        reshape(self.grid_row, self.grid_col)
        self.FLAG_UPDATE_NODE_MASK = False

        for pmod_idx in self.placed_macro:
            pmod = self.modules_w_pins[pmod_idx]
            if not pmod.get_placed_flag():
                continue
            
            p_x, p_y = pmod.get_pos()
            prow, pcol = self.__get_grid_cell_location(p_x, p_y)
            c_idx = prow * self.grid_col + pcol
            self.__place_node_mask(c_idx, pmod.get_width(), pmod.get_height())
    
    def __reset_node_mask(self):
        """
        Internal function for reseting node mask
        * All four sides cannot be used for placement
        """
        self.node_mask = np.array([1] * (self.grid_col * self.grid_row)).\
                        reshape(self.grid_row, self.grid_col)


    def __node_pad_cell(self, mod_width, mod_height):
        """
        Internal function for computing how much cells we need for padding
        This is to avoid overlapping on placement
        """
        self.grid_width = float(self.width/self.grid_col)
        self.grid_height = float(self.height/self.grid_row)

        cell_hor = math.ceil(((mod_width/2) - (self.grid_width/2)) / self.grid_width)
        cell_ver = math.ceil(((mod_height/2) - (self.grid_height/2)) / self.grid_height)

        return cell_hor, cell_ver

    def place_node(self, node_idx, grid_cell_idx):
        """
        Place the node into the center of the given grid_cell
        """
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR PLACE NODE] Found {}. Only 'MACRO', 'STDCELL'"\
                .format(mod.get_type())
                +"'PORT' are considered to be placable nodes")
            exit(1)
        except Exception:
            print("[ERROR PLACE NODE] Could not find module by node index")

        try: 
            assert grid_cell_idx <= self.grid_col * self.grid_row - 1
        except AssertionError:
            print("[WARNING PLACE NODE] Invalid Location. No node placed.")

        # TODO: add check valid clause
        if not mod.get_fix_flag():
            mod.set_pos(*self.__get_grid_cell_position(grid_cell_idx))
            self.placed_macro.append(self.mod_name_to_indices[mod.get_name()])
            mod.set_placed_flag(True)

            # update flag
            self.FLAG_UPDATE_CONGESTION = True
            self.FLAG_UPDATE_DENSITY = True
            # self.FLAG_UPDATE_NODE_MASK = True
            self.FLAG_UPDATE_WIRELENGTH = True

            self.__place_node_mask(grid_cell_idx, mod_width=mod.get_width(), mod_height=mod.get_height())

    def can_place_node(self, node_idx, grid_cell_idx):
        return self.get_node_mask(node_idx=node_idx)[grid_cell_idx]

    def unplace_node(self, node_idx):
        """
        Set the node's ifPlaced flag to False if not fixed node
        """
        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR UNPLACE NODE] Found {}. Only 'MACRO', 'STDCELL'".format(mod.get_type())
                    +"'PORT' are considered to be placable nodes")
            exit(1)
        except Exception:
            print("[ERROR UNPLACE NODE] Could not find module by node index")
            exit(1)

        if not mod.get_fix_flag():
            if node_idx in self.hard_macro_indices:
                mod.set_placed_flag(False)
                self.placed_macro.remove(node_idx)
                # update flag
                self.FLAG_UPDATE_CONGESTION = True
                self.FLAG_UPDATE_DENSITY = True
                # self.FLAG_UPDATE_NODE_MASK = True # placeholder
                self.FLAG_UPDATE_WIRELENGTH = True
            elif node_idx in self.soft_macro_indices:
                mod.set_placed_flag(False)
                # update flag
                self.FLAG_UPDATE_CONGESTION = True
                self.FLAG_UPDATE_DENSITY = True
                # self.FLAG_UPDATE_NODE_MASK = True # placeholder
                self.FLAG_UPDATE_WIRELENGTH = True
        else:
            print("[WARNING UNPLACE NODE] Trying to unplace a fixed node")

    def unplace_all_nodes(self):
        """
        Set all ifPlaced flag to False except for fixed nodes
        """
        for mod_idx in sorted(self.port_indices + self.hard_macro_indices + self.soft_macro_indices):
            mod = self.modules_w_pins[mod_idx]
            if mod.get_fix_flag():
                continue

            if mod.get_placed_flag():
                mod.set_placed_flag(False)
        
        self.placed_macro = []
        # update flag
        self.FLAG_UPDATE_CONGESTION = True
        self.FLAG_UPDATE_DENSITY = True
        # self.FLAG_UPDATE_NODE_MASK = True
        self.FLAG_UPDATE_WIRELENGTH = True
        self.__reset_node_mask()

    def is_node_placed(self, node_idx):
        mod = None

        try:
            mod = self.modules_w_pins[node_idx]
            assert mod.get_type() in ['MACRO', 'macro', 'STDCELL', 'PORT']
        except AssertionError:
            print("[ERROR NODE PLACED] Found {}. Only 'MACRO', 'STDCELL',".format(mod.get_type())
                    +"'PORT' are considered to be placable nodes")
            exit(1)
        except Exception:
            print("[ERROR NODE PLACED] Could not find module by node index")
            exit(1)

        mod = self.modules_w_pins[node_idx]
        return mod.get_placed_flag()

    def disconnect_nets(self):
        pass

    def get_source_filename(self):
        """
        return netlist path
        """
        return self.netlist_file

    def get_blockages(self):
        return self.blockages

    def create_blockage(self, minx, miny, maxx, maxy, blockage_rate):
        self.blockages.append([minx, miny, maxx, maxy, blockage_rate])

    def get_ref_node_id(self, node_idx=-1):
        """
        ref_node_id is used for macro_pins. Refers to the macro it belongs to.
        if input PORT, return itself
        """
        if self.modules_w_pins[node_idx].get_type() == "PORT":
            return node_idx

        if node_idx != -1:
            if node_idx in self.soft_macro_pin_indices or node_idx in self.hard_macro_pin_indices:
                pin = self.modules_w_pins[node_idx]
                return self.mod_name_to_indices[pin.get_macro_name()]
        return -1

    def save_placement(self, filename, info=""):
        """
        When writing out info line-by-line, add a "#" at front
        """
        with open(filename, 'w+') as f:
            for line in info.split('\n'):
                f.write("# " + line + '\n')

            # if first, no \newline
            HEADER = True

            for mod_idx in sorted(self.hard_macro_indices + self.soft_macro_indices + self.port_indices):
                # [node_index] [x] [y] [orientation] [fixed]
                mod = self.modules_w_pins[mod_idx]

                if HEADER:
                    f.write("{} {:g} {:g} {} {}".format(mod_idx,
                        *mod.get_pos(),
                        mod.get_orientation() if mod.get_orientation() else "-",
                        "1" if mod.get_fix_flag() else "0"))
                    HEADER = False
                else:
                    f.write("\n{} {:g} {:g} {} {}".format(mod_idx,
                            *mod.get_pos(),
                            mod.get_orientation() if mod.get_orientation() else "-",
                            "1" if mod.get_fix_flag() else "0"))

    def display_canvas( self,
                        annotate=True, 
                        amplify=False,
                        saveName=None,
                        show=True):
        """
        Non-google function, For quick canvas view
        """
        #define Matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(8,8), dpi=50)

        if amplify:
            PORT_SIZE = 4
            FONT_SIZE = 10
            PIN_SIZE = 4
        else:
            PORT_SIZE = 2
            FONT_SIZE = 5
            PIN_SIZE = 2

        # Plt config
        ax.margins(x=0.05, y=0.05)
        ax.set_aspect('equal', adjustable='box')

        # Construct grid
        x, y = np.meshgrid(np.linspace(0, self.width, self.grid_col + 1),\
             np.linspace(0, self.height, self.grid_row + 1))

        ax.plot(x, y, c='b', alpha=0.1) # use plot, not scatter
        ax.plot(np.transpose(x), np.transpose(y), c='b', alpha=0.2) # add this here

        # Construct module blocks
        for mod in self.modules_w_pins:
            if mod.get_type() == 'PORT' and mod.get_placed_flag():
                plt.plot(*mod.get_pos(),'ro', markersize=PORT_SIZE)
            elif mod.get_type() == 'MACRO' and mod.get_placed_flag():
                if not self.is_node_soft_macro(self.mod_name_to_indices[mod.get_name()]):
                    # hard macro
                    ax.add_patch(Rectangle((mod.get_pos()[0] - mod.get_width()/2, mod.get_pos()[1] - mod.get_height()/2),\
                        mod.get_width(), mod.get_height(),\
                        alpha=0.5, zorder=1000, facecolor='b', edgecolor='darkblue'))
                    if annotate:
                        ax.annotate(mod.get_name(), mod.get_pos(),  wrap=True,color='r', weight='bold', fontsize=FONT_SIZE, ha='center', va='center')
                else:
                    # soft macro
                    ax.add_patch(Rectangle((mod.get_pos()[0] - mod.get_width()/2, mod.get_pos()[1] - mod.get_height()/2),\
                        mod.get_width(), mod.get_height(),\
                        alpha=0.5, zorder=1000, facecolor='y'))
                    if annotate:
                        ax.annotate(mod.get_name(), mod.get_pos(), wrap=True,color='r', weight='bold', fontsize=FONT_SIZE, ha='center', va='center')
            elif mod.get_type() == 'MACRO_PIN':
                pin_idx = self.mod_name_to_indices[mod.get_name()]
                macro_idx = self.get_ref_node_id(pin_idx)
                macro = self.modules_w_pins[macro_idx]
                if macro.get_placed_flag():
                    plt.plot(*self.__get_pin_position(pin_idx),'bo', markersize=PIN_SIZE)
            # elif mod.get_type() == 'macro' :
            #     ax.add_patch(Rectangle((mod.get_pos()[0] - mod.get_width()/2, mod.get_pos()[1] - mod.get_height()/2),\
            #         mod.get_width(), mod.get_height(),\
            #         alpha=0.5, zorder=1000, facecolor='y'))
            #     if annotate:
            #         ax.annotate(mod.get_name(), mod.get_pos(), wrap=True,color='r', weight='bold', fontsize=FONT_SIZE, ha='center', va='center')
        if saveName:
            plt.savefig(saveName)
        if show:
            plt.show()

        plt.close('all')
    
    '''
    FD Placement below shares the same functionality as the FDPlacement/fd_placement.py
    '''
    def __ifOverlap(self, u_i, v_i, ux=0, uy=0, vx=0, vy=0):
        '''
        Detect if the two modules are overlapping or not (w/o using block structure)
        '''
        # extract first macro
        u_side = self.modules_w_pins[u_i].get_height()
        u_x1 = self.modules_w_pins[u_i].get_pos()[0] + ux - u_side/2 # left
        u_x2 = self.modules_w_pins[u_i].get_pos()[0] + ux + u_side/2 # right
        u_y1 = self.modules_w_pins[u_i].get_pos()[1] + uy + u_side/2 # top
        u_y2 = self.modules_w_pins[u_i].get_pos()[1] + uy - u_side/2 # bottom

        # extract second macro
        v_side = self.modules_w_pins[v_i].get_height()
        v_x1 = self.modules_w_pins[v_i].get_pos()[0] + vx - v_side/2 # left
        v_x2 = self.modules_w_pins[v_i].get_pos()[0] + vx + v_side/2 # right
        v_y1 = self.modules_w_pins[v_i].get_pos()[1] + vy + v_side/2 # top
        v_y2 = self.modules_w_pins[v_i].get_pos()[1] + vy - v_side/2 # bottom

        return u_x1 < v_x2 and u_x2 > v_x1 and u_y1 > v_y2 and u_y2 < v_y1
    
    def __repulsive_force(self, repel_factor, mod_i_idx, mod_j_idx, with_initialization=False):
        '''
        Calculate repulsive force between two nodes node_i, node_j
        '''
        # Only exert force when modules are overlapping
        # TODO: effects on PORTs
        if not self.__ifOverlap(u_i=node_i, v_i=node_j):
            # node_i_x, node_i_y, node_j_x, node_j_y
            return 0.0, 0.0, 0.0, 0.0
        print("[INFO] REPEL FORCE detects overlapping, exerting repelling")

        if with_initialization:
            # TODO: exerting SPRING FORCE
            pass

        # retrieve module instance
        mod_i = self.modules_w_pins[mod_i_idx]
        mod_j = self.modules_w_pins[mod_j_idx]

        # retrieve module position
        x_i, y_i = mod_i.get_pos()
        x_j, y_j = mod_j.get_pos()
        
        # get dist between x and y
        x_dist = x_i - x_j
        y_dist = y_i - y_j

        # get directional vector for node i
        x_i2j = x_dist
        xd_i2j = x_i2j / abs(x_dist)
        y_i2j = y_dist
        yd_i2j = y_i2j / abs(y_dist)

        # get directional vector for node j
        x_j2i = -1.0 * x_dist
        xd_j2i = x_j2i / abs(x_dist)
        y_j2i = -1.0 * y_dist
        yd_j2i = y_j2i / abs(y_dist)

        # detect boundaries and if driver or sink is MACRO
        # TODO: consider PORT to be inmoveable as well
        if self.is_node_hard_macro(mod_i_idx):
            # then, mod_j is should move towards mod_i
            i_force = self.__i_force(i)
            # node_i_x, node_i_y, node_j_x, node_j_y
            return 0.0, 0.0, i_force * xd_j2i, i_force * yd_j2i
        if self.is_node_hard_macro(mod_j_idx):
            # then, mod_i is should move towards mod_j
            i_force = self.__i_force(i)
            # node_i_x, node_i_y, node_j_x, node_j_y
            return i_force * xd_i2j, i_force * yd_i2j, 0.0, 0.0
        else:
            # between macro and macro, attract each to each other
            # Can result in heavily overlapping
            return 

    def __repulsive_force_hard_macro(self, repel_factor, h_node_i, s_node_j):
        '''
        Calculate repulsive force between hard macro and soft macro
        '''
        if repel_factor == 0.0:
            return 0.0, 0.0
        
        # retrieve module instance
        h_mod_i = self.modules_w_pins[h_node_i]
        s_mod_j = self.modules_w_pins[s_node_j]

        # retrieve module position
        x_i, y_i = h_mod_i.get_pos()
        x_j, y_j = s_mod_j.get_pos()
        
        # get dist between x and y
        x_dist = x_i - x_j
        y_dist = y_i - y_j

        # get dist of hypotenuse
        hypo_dist = math.sqrt(x_dist**2 + y_dist**2)
        
        # compute force in x and y direction
        if hypo_dist <= 1e-10 or self.__ifOverlap(h_node_i, s_node_j):
            return x_dist/hypo_dist * (h_mod_i.get_height()/2 + s_mod_j.get_height()/2),\
                    y_dist/hypo_dist * (h_mod_i.get_height()/2 + s_mod_j.get_height()/2)
        else:
            return 0.0, 0.0

    def __attractive_force(self, io_factor, attract_factor, pin_i_idx, pin_j_idx, io_flag = True, attract_exponent = 1, i = 1):
        '''
        Calculate repulsive force between two pins pin_i, pin_j
        '''
        
        # retrieve module position
        x_i, y_i = self.__get_pin_position(pin_idx=pin_i_idx)
        x_j, y_j = self.__get_pin_position(pin_idx=pin_j_idx)

        # get distance
        x_dist = x_i - x_j
        y_dist = y_i - y_j
        # if pins are close enough, dont attract futher
        if abs(x_dist) <= 1e-3 and abs(y_dist) <= 1e-3:
            return 0.0, 0.0, 0.0, 0.0
        elif abs(x_dist) <= 1e-3:
            x_i2j = 0
            xd_i2j = 0
            x_j2i = 0
            xd_j2i = 0

            y_i2j = -1.0 * (y_dist)
            yd_i2j = y_i2j / abs(y_dist)
            y_j2i = y_dist
            yd_j2i = y_j2i / abs(y_dist)
        elif abs(y_dist) <= 1e-3:
            x_i2j = -1.0 * (x_dist)
            xd_i2j = x_i2j / abs(x_dist)
            x_j2i = x_dist
            xd_j2i = x_j2i / abs(x_dist)

            y_i2j = 0
            yd_i2j = 0
            y_j2i = 0
            yd_j2i = 0
        else:
            # get directional vector for pin i
            x_i2j = -1.0 * (x_dist)
            xd_i2j = x_i2j / abs(x_dist)
            y_i2j = -1.0 * (y_dist)
            yd_i2j = y_i2j / abs(y_dist)

            # get directional vector for pin j
            x_j2i = x_dist
            xd_j2i = x_j2i / abs(x_dist)
            y_j2i = y_dist
            yd_j2i = y_j2i / abs(y_dist)

        # TODO: consider PORT to be inmoveable as well
        i_force = self.__i_force(i)
        # pin_i_x, pin_i_y, pin_j_x, pin_j_y 
        return i_force * xd_i2j, i_force * yd_i2j, i_force * xd_j2i, i_force * yd_j2i

            
    def __centralize_soft_macro(self, mod_id):
        '''
        Pull the modules to the nearest center of the gridcell
        '''
        if self.is_node_soft_macro(mod_id):
            # put everyting at center, regardless the overlapping issue
            mod = self.modules_w_pins[mod_id]
            mod.set_pos(self.width/2, self.height/2)

    def __initialization(self):
        '''
        Initialize soft macros to the center
        '''
        for mod_idx in self.soft_macro_indices:
            self.__centralize_soft_macro(mod_idx)

    def __boundary_check(self, mod_id):
        '''
        Make sure all the clusters are placed within the canvas
        '''
        mod = self.modules_w_pins[mod_id]
        mod_x, mod_y = mod.get_pos()
        
        if mod_x < 0.0:
            mod_x = 0.0
        
        if mod_x > self.width:
            mod_x = self.width

        if mod_y < 0.0:
            mod_y = 0.0

        if mod_y > self.height:
            mod_y = self.height

        mod.set_pos(mod_x, mod_y)
        
    def __fd_placement(self, io_factor, num_steps, max_move_distance, attract_factor, repel_factor, use_current_loc, verbose=True):
        '''
        Force-directed Placement for standard-cell clusters
        ''' 
        # store x/y displacement for all soft macro disp
        soft_macro_disp = {}

        def check_OOB(mod_id, x_disp, y_disp):
            mod = self.modules_w_pins[mod_id]
            mod_x, mod_y = mod.get_pos()
            mod_height = mod.get_height()
            mod_width = mod.get_width()

            # print(x_disp, y_disp, mod_x, mod_y, mod_width, mod_height)
            # boundary after displacement
            x_max = mod_x + mod_width/2 + x_disp
            y_max = mod_y + mod_height/2 + y_disp
            x_min = mod_x - mod_width/2 + x_disp
            y_min = mod_y - mod_height/2 + y_disp

            # print(x_max, x_min, y_max, y_min)
            # determine if move
            if x_min <= 0.0 or x_max >= self.width:
                x_disp = 0.0
            if y_min <= 0.0 or y_max >= self.height:
                y_disp = 0.0

            return x_disp, y_disp

        def getBBox(mod_id):
            mod = self.modules_w_pins[mod_id]
            x, y = mod.get_pos()
            width = mod.get_width()
            height = mod.get_height()
            lx = x - width / 2.0
            ly = y - height / 2.0
            ux = x + width / 2.0
            uy = y + height / 2.0
            return lx, ly, ux, uy

        def _check_overlap(mod_u, mod_v):
            '''
            Zhiang's implmentation
            '''
            u_lx, u_ly, u_ux, u_uy = getBBox(mod_u)
            v_lx, v_ly, v_ux, v_uy = getBBox(mod_v)

            if (u_lx >= v_ux or u_ux <= v_lx or u_ly >= v_uy or u_uy <= v_ly):
                # no overlap
                return None, None
            else:
                u_cx = (u_lx + u_ux) / 2.0
                u_cy = (u_ly + u_uy) / 2.0
                v_cx = (v_lx + v_ux) / 2.0
                v_cy = (v_ly + v_uy) / 2.0

                if u_cx == v_cx and u_cy == v_cy:
                    # fully overlap
                    x_dir = -1.0 / math.sqrt(2.0)
                    y_dir = -1.0 / math.sqrt(2.0)
                    return x_dir, y_dir
                else:
                    x_dir = u_cx - v_cx
                    y_dir = u_cy - v_cy
                    dist = math.sqrt(x_dir * x_dir + y_dir * y_dir)
                    return x_dir / dist, y_dir / dist
        
        def check_overlap(mod_u, mod_v):
            u_lx, u_ly, u_ux, u_uy = getBBox(mod_u)
            v_lx, v_ly, v_ux, v_uy = getBBox(mod_v)

            if (u_lx >= v_ux or u_ux <= v_lx or u_ly >= v_uy or u_uy <= v_ly):
                return 1, 1
            else:
                u_cx = (u_lx + u_ux) / 2.0
                u_cy = (u_ly + u_uy) / 2.0
                v_cx = (v_lx + v_ux) / 2.0
                v_cy = (v_ly + v_uy) / 2.0
                x_d = u_cx - v_cx
                y_d = u_cy - v_cy

                # set the minimum val to 1e-12
                # min_dist = 1e-2
                # if abs(x_d) <= min_dist:
                #     x_d = -1.0 * min_dist
                # if abs(y_d) <= min_dist:
                #     y_d = -1.0 * min_dist
                if x_d == 0:
                    x_d = 1
                else:
                    x_d /= abs(x_d)
                
                if y_d == 0:
                    y_d = 1
                else:
                    y_d /= abs(y_d)
                
                return x_d, y_d
        
        def add_displace(mod_id, x_disp, y_disp):
            '''
            Add the displacement
            '''
            if mod_id in self.soft_macro_indices:
                soft_macro_disp[mod_id][0] += x_disp
                soft_macro_disp[mod_id][1] += y_disp

        def update_location(mod_id, x_disp, y_disp):
            '''
            Update the displacement to the coordiante
            '''
            x_pos, y_pos = self.modules_w_pins[mod_id].get_pos()

            x_disp, y_disp = check_OOB(mod_id, x_disp, y_disp)
            # for debug purpose
            with open('os_debug.txt', 'a+') as the_file:
                the_file.write("{} {} {} {} {}\n".format(
                    mod_id,
                    x_pos + x_disp, 
                    y_pos + y_disp, 
                    x_disp, y_disp
                    ))
            self.modules_w_pins[mod_id].set_pos(x_pos + x_disp, y_pos + y_disp)

        def move_soft_macros(attract_factor, repel_factor, io_factor, max_displacement):
            # map to soft macro index
            for mod_idx in self.soft_macro_indices:
                soft_macro_disp[mod_idx] = [0.0, 0.0]
            calcAttractiveForce(attract_factor, io_factor, max_displacement)
            calcRepulsiveForce(repel_factor, max_displacement)
            max_x_disp = 0.0
            max_y_disp = 0.0

            for mod_idx in self.soft_macro_indices:
                max_x_disp = max(max_x_disp, abs(soft_macro_disp[mod_idx][0]))
                max_y_disp = max(max_y_disp, abs(soft_macro_disp[mod_idx][1]))
            
            # normalization
            if max_x_disp > 0.0:
                for mod_idx in self.soft_macro_indices:
                    soft_macro_disp[mod_idx][0] = (soft_macro_disp[mod_idx][0] / max_x_disp) * max_displacement
            if max_y_disp > 0.0:
                for mod_idx in self.soft_macro_indices:
                    soft_macro_disp[mod_idx][1] = (soft_macro_disp[mod_idx][1] / max_y_disp) * max_displacement
            for mod_idx in self.soft_macro_indices:
                update_location(mod_idx, soft_macro_disp[mod_idx][0], soft_macro_disp[mod_idx][1])

        def checkPinRelativePos(pin_u, pin_v):
            ux, uy = self.__get_pin_position(pin_u)
            vx, vy = self.__get_pin_position(pin_v)
            return -1.0 * (ux - vx), -1.0 * (uy - vy)

        def check_if_pin_close(pin_u, pin_v):
            ux, uy = self.__get_pin_position(pin_u)
            vx, vy = self.__get_pin_position(pin_v)
            
            move_x = True
            move_y = True

            # needs to be determined
            dist_thre = 1

            if abs(ux - uy) <= dist_thre:
                move_x = False
            
            if abs(vx - vy) <= dist_thre:
                move_y = False

            return move_x, move_y

        def calcRepulsiveForce(repel_factor, max_displacement):
            macro_list = sorted(self.hard_macro_indices + self.soft_macro_indices)
            for i in range(len(macro_list)):
                mod_u_idx = macro_list[i]
                for j in range(i + 1, len(macro_list)):
                    mod_v_idx = macro_list[j]
                    # if not self.__ifOverlap(mod_u_idx, mod_v_idx):
                    #     # if not overlapping, dont exert force
                    #     continue
                    
                    x_d, y_d = _check_overlap(mod_u_idx, mod_v_idx)
                    x_disp = 0.0
                    y_disp = 0.0
                    # print("debugging overlap: ", x_d, y_d)
                    # No overlap
                    if x_d == None: 
                        x_disp = 0.0
                    else:
                        # x_disp = repel_factor * 1.0 / x_d
                        x_disp = repel_factor * 1.0 * max_displacement * x_d
                    
                    # No overlap
                    if y_d == None:
                        y_disp = 0.0
                    else:
                        # y_disp = repel_factor * 1.0 / y_d
                        y_disp = repel_factor * 1.0 * max_displacement * y_d

                    # print("debugging: ", x_disp, y_disp)
                    add_displace(mod_u_idx, x_disp, y_disp)
                    add_displace(mod_v_idx, -1.0 * x_disp, -1.0 * y_disp)
        
        def calcAttractiveForce(attract_factor, io_factor, max_displacement):
            for driver_pin_name in self.nets.keys():
                # extract driver pin
                driver_pin_idx = self.mod_name_to_indices[driver_pin_name]
                driver_pin = self.modules_w_pins[driver_pin_idx]
                # extract driver macro
                driver_macro_idx = self.get_ref_node_id(driver_pin_idx)
                # extract net weight
                weight_factor = driver_pin.get_weight()
                for sink_pin_name in self.nets[driver_pin_name]:
                    sink_pin_idx = self.mod_name_to_indices[sink_pin_name]
                    sink_macro_idx = self.get_ref_node_id(sink_pin_idx)
                    # compute directional vector
                    x_d, y_d = checkPinRelativePos(driver_pin_idx, sink_pin_idx)
                    # if connection has port
                    if sink_pin_idx in self.port_indices or driver_pin_idx in self.port_indices:
                        force = weight_factor * io_factor * attract_factor
                    else:
                        force = weight_factor * attract_factor
                    
                    # only move when pin are not too close
                    # move_x, move_y = check_if_pin_close(driver_pin_idx, sink_pin_idx)
                    # x_disp = 0.0 if not move_x else force * x_d
                    # y_disp = 0.0 if not move_y else force * y_d
                    x_disp = force * x_d
                    y_disp = force * y_d

                    # add displacement to driver/sink pin 
                    add_displace(driver_macro_idx, x_disp, y_disp)
                    add_displace(sink_macro_idx, -1.0 * x_disp, -1.0 * y_disp)
            
        if use_current_loc == False:
            self.__initialization()
        for i in range(len(num_steps)):
            if verbose:
                print("[OPTIMIZING STDCELs] at num_step {}".format(i))
            attractive_factor = attract_factor[i]
            repulsive_factor = repel_factor[i]
            num_step = num_steps[i]
            max_displacement = max_move_distance[i]

            for j in range(num_step):
                if verbose:
                    print("[INFO] number of step {}".format(j))
                move_soft_macros(attractive_factor, repulsive_factor, io_factor, max_displacement)
        
    def optimize_stdcells(self, use_current_loc, move_stdcells, move_macros,
                        log_scale_conns, use_sizes, io_factor, num_steps,
                        max_move_distance, attract_factor, repel_factor):
        
        self.__fd_placement(io_factor, num_steps, max_move_distance, attract_factor, repel_factor, use_current_loc, verbose=True)

    # Board Entity Definition
    class Port:
        def __init__(self, name, x = 0.0, y = 0.0, side = "BOTTOM"):
            self.name = name
            self.x = float(x)
            self.y = float(y)
            self.side = side # "BOTTOM", "TOP", "LEFT", "RIGHT"
            self.sink = {} # standard cells, macro pins, ports driven by this cell
            self.connection = {} # [module_name] => edge degree
            self.fix_flag = True
            self.placement = 0 # needs to be updated
            self.orientation = None
            self.ifPlaced = True

        def get_name(self):
            return self.name
        
        def get_orientation(self):
            return self.orientation
        
        def get_height(self):
            return 0

        def get_width(self):
            return 0

        def get_weight(self):
            return 1.0

        def add_connection(self, module_name):
            # NOTE: assume PORT names does not contain slash
            ifPORT = False
            module_name_splited = module_name.rsplit('/', 1)
            if len(module_name_splited) == 1:
                ifPORT = not ifPORT

            if ifPORT:
                # adding PORT
                self.connection[module_name] = 1
            else:
                # adding soft/hard macros
                if module_name_splited[0] in self.connection.keys():
                    self.connection[module_name_splited[0]] += 1
                else:
                    self.connection[module_name_splited[0]] = 1

        def add_connections(self, module_names):
            # NOTE: assume PORT names does not contain slash
            for module_name in module_names:
                self.add_connection(module_name)

        def set_pos(self, x, y):
            self.x = x
            self.y = y

        def get_pos(self):
            return self.x, self.y

        def set_side(self, side):
            self.side = side

        def add_sink(self, sink_name):
            # NOTE: assume PORT names does not contain slash
            ifPORT = False
            sink_name_splited = sink_name.rsplit('/', 1)
            if len(sink_name_splited) == 1:
                ifPORT = not(ifPORT)

            if ifPORT:
                # adding PORT
                self.sink[sink_name] = [sink_name]
            else:
                # adding soft/hard macros
                if sink_name_splited[0] in self.sink.keys():
                    self.sink[sink_name_splited[0]].append(sink_name)
                else:
                    self.sink[sink_name_splited[0]] = [sink_name]

        def add_sinks(self, sink_names):
            # NOTE: assume PORT names does not contain slash
            for sink_name in sink_names:
                self.add_sink(sink_name)

        def get_sink(self):
            return self.sink

        def get_connection(self):
            return self.connection

        def get_type(self):
            return "PORT"
        
        def set_fix_flag(self, fix_flag):
            self.fix_flag = fix_flag
        
        def get_fix_flag(self):
            return self.fix_flag

        def set_placed_flag(self, ifPlaced):
            self.ifPlaced = ifPlaced
        
        def get_placed_flag(self):
            return self.ifPlaced

    class SoftMacro:
        def __init__(self, name, width, height, x = 0.0, y = 0.0):
            self.name = name
            self.width = float(width)
            self.height = float(height)
            self.x = float(x)
            self.y = float(y)
            self.connection = {} # [module_name] => edge degree
            self.orientation = None
            self.fix_flag = False
            self.ifPlaced = True
            self.location = 0 # needs to be updated

        def get_name(self):
            return self.name

        def add_connection(self, module_name, weight):
            # NOTE: assume PORT names does not contain slash
            ifPORT = False
            module_name_splited = module_name.rsplit('/', 1)
            if len(module_name_splited) == 1:
                ifPORT = not(ifPORT)

            if ifPORT:
                # adding PORT
                self.connection[module_name] = 1 * weight
            else:
                # adding soft/hard macros
                if module_name_splited[0] in self.connection.keys():
                    self.connection[module_name_splited[0]] += 1 * weight
                else:
                    self.connection[module_name_splited[0]] = 1 * weight

        def add_connections(self, module_names, weight):
            # NOTE: assume PORT names does not contain slash
            # consider weight on soft macro pins
            for module_name in module_names:
                self.add_connection(module_name, weight)

        def set_pos(self, x, y):
            self.x = x
            self.y = y

        def get_pos(self):
            return self.x, self.y

        def get_type(self):
            return "MACRO"

        def get_connection(self):
            return self.connection
        
        def set_orientation(self, orientation):
            self.orientation = orientation
        
        def get_orientation(self):
            return self.orientation

        def get_area(self):
            return self.width * self.height

        def get_height(self):
            return self.height

        def get_width(self):
            return self.width
        
        def set_height(self, height):
            self.height = height

        def set_width(self, width):
            self.width = width
        
        def set_location(self, grid_cell_idx):
            self.location = grid_cell_idx
        
        def get_location(self):
            return self.location
        
        def set_fix_flag(self, fix_flag):
            self.fix_flag = fix_flag
        
        def get_fix_flag(self):
            return self.fix_flag

        def set_placed_flag(self, ifPlaced):
            self.ifPlaced = ifPlaced
        
        def get_placed_flag(self):
            return self.ifPlaced

    class SoftMacroPin:
        def __init__(self, name, ref_id,
                    x = 0.0, y = 0.0,
                    macro_name = "", weight = 1.0):
            self.name = name
            self.ref_id = ref_id
            self.x = float(x)
            self.y = float(y)
            self.x_offset = 0.0 # not used
            self.y_offset = 0.0 # not used
            self.macro_name = macro_name
            self.weight = weight
            self.sink = {}

        def set_weight(self, weight):
            self.weight = weight

        def set_ref_id(self, ref_id):
            self.ref_id = ref_id

        def get_ref_id(self):
            return self.ref_id

        def get_weight(self):
            return self.weight

        def get_name(self):
            return self.name

        def get_macro_name(self):
            return self.macro_name

        def set_pos(self, x, y):
            self.x = x
            self.y = y

        def get_pos(self):
            return self.x, self.y

        def get_offset(self):
            return self.x_offset, self.y_offset

        def add_sink(self, sink_name):
            # NOTE: assume PORT names does not contain slash
            ifPORT = False
            sink_name_splited = sink_name.rsplit('/', 1)
            if len(sink_name_splited) == 1:
                ifPORT = not(ifPORT)

            if ifPORT:
                # adding PORT
                self.sink[sink_name] = [sink_name]
            else:
                # adding soft/hard macros
                if sink_name_splited[0] in self.sink.keys():
                    self.sink[sink_name_splited[0]].append(sink_name)
                else:
                    self.sink[sink_name_splited[0]] = [sink_name]

        def add_sinks(self, sink_names):
            # NOTE: assume PORT names does not contain slash
            for sink_name in sink_names:
                self.add_sink(sink_name)

        def get_sink(self):
            return self.sink

        def get_type(self):
            return "MACRO_PIN"

    class HardMacro:
        def __init__(self, name, width, height,
                     x = 0.0, y = 0.0, orientation = "N"):
            self.name = name
            self.width = float(width)
            self.height = float(height)
            self.x = float(x)
            self.y = float(y)
            self.orientation = orientation
            self.connection = {} # [module_name] => edge degree
            self.fix_flag = False
            self.ifPlaced = True
            self.location = 0 # needs to be updated

        def get_name(self):
            return self.name

        def add_connection(self, module_name, weight):
            # NOTE: assume PORT names does not contain slash
            ifPORT = False
            module_name_splited = module_name.rsplit('/', 1)
            if len(module_name_splited) == 1:
                ifPORT = not(ifPORT)

            if ifPORT:
                # adding PORT
                self.connection[module_name] = 1 * weight
            else:
                # adding soft/hard macros
                if module_name_splited[0] in self.connection.keys():
                    self.connection[module_name_splited[0]] += 1 * weight
                else:
                    self.connection[module_name_splited[0]] = 1 * weight

        def add_connections(self, module_names, weight):
            # NOTE: assume PORT names does not contain slash
            # consider weight on soft macro pins
            for module_name in module_names:
                self.add_connection(module_name, weight)

        def get_connection(self):
            return self.connection

        def set_pos(self, x, y):
            self.x = x
            self.y = y

        def get_pos(self):
            return self.x, self.y

        def set_orientation(self, orientation):
            self.orientation = orientation
        
        def get_orientation(self):
            return self.orientation

        def get_type(self):
            return "MACRO"

        def get_area(self):
            return self.width * self.height

        def get_height(self):
            return self.height

        def get_width(self):
            return self.width
        
        def set_location(self, grid_cell_idx):
            self.location = grid_cell_idx
        
        def get_location(self):
            return self.location
        
        def set_fix_flag(self, fix_flag):
            self.fix_flag = fix_flag
        
        def get_fix_flag(self):
            return self.fix_flag
        
        def set_placed_flag(self, ifPlaced):
            self.ifPlaced = ifPlaced
        
        def get_placed_flag(self):
            return self.ifPlaced

    class HardMacroPin:
        def __init__(self, name, ref_id,
                        x = 0.0, y = 0.0,
                        x_offset = 0.0, y_offset = 0.0,
                        macro_name = "", weight = 1.0):
            self.name = name
            self.ref_id = ref_id
            self.x = float(x)
            self.y = float(y)
            self.x_offset = float(x_offset)
            self.y_offset = float(y_offset)
            self.macro_name = macro_name
            self.weight = weight
            self.sink = {}
            self.ifPlaced = True

        def set_ref_id(self, ref_id):
            self.ref_id = ref_id

        def get_ref_id(self):
            return self.ref_id

        def set_weight(self, weight):
            self.weight = weight

        def get_weight(self):
            return self.weight

        def set_pos(self, x, y):
            self.x = x
            self.y = y

        def get_pos(self):
            return self.x, self.y

        def get_offset(self):
            return self.x_offset, self.y_offset
        
        def set_offset(self, x_offset, y_offset):
            self.x_offset = x_offset
            self.y_offset = y_offset

        def get_name(self):
            return self.name

        def get_macro_name(self):
            return self.macro_name

        def add_sink(self, sink_name):
            # NOTE: assume PORT names does not contain slash
            ifPORT = False
            sink_name_splited = sink_name.rsplit('/', 1)
            if len(sink_name_splited) == 1:
                ifPORT = not(ifPORT)

            if ifPORT:
                # adding PORT
                self.sink[sink_name] = [sink_name]
            else:
                # adding soft/hard macros
                if sink_name_splited[0] in self.sink.keys():
                    self.sink[sink_name_splited[0]].append(sink_name)
                else:
                    self.sink[sink_name_splited[0]] = [sink_name]

        def add_sinks(self, sink_names):
            # NOTE: assume PORT names does not contain slash
            for sink_name in sink_names:
                self.add_sink(sink_name)

        def get_sink(self):
            return self.sink

        def get_type(self):
            return "MACRO_PIN"

def main():
    test_netlist_dir = './Plc_client/test/'+\
        'ariane_68_1.3'
    netlist_file = os.path.join(test_netlist_dir,
                                'netlist.pb.txt')
    plc = PlacementCostAccelerated(netlist_file)

    print(plc.get_block_name())
    print("Area: ", plc.get_area())
    print("Wirelength: ", plc.get_wirelength())
    print("# HARD_MACROs     :         %d"%(plc.get_hard_macros_count()))
    print("# HARD_MACRO_PINs :         %d"%(plc.get_hard_macro_pins_count()))
    print("# MACROs          :         %d"%(plc.get_hard_macros_count() + plc.get_soft_macros_count()))
    print("# MACRO_PINs      :         %d"%(plc.get_hard_macro_pins_count() + plc.get_soft_macro_pins_count()))
    print("# PORTs           :         %d"%(plc.get_ports_count()))
    print("# SOFT_MACROs     :         %d"%(plc.get_soft_macros_count()))
    print("# SOFT_MACRO_PINs :         %d"%(plc.get_soft_macro_pins_count()))
    print("# STDCELLs        :         0")

if __name__ == '__main__':
    main()
