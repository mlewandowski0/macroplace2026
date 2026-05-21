from dataclasses import dataclass
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import math

from macro_place.benchmark import Benchmark

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import validate_placement, compute_proxy_cost, compute_proxy_cost_incremental, _set_placement_fast, load_benchmark_accel, load_benchmark_accel_from_dir, _set_placement_fast_moved  # noqa: E402
from trajectory_logger import TrajectoryLogger  

import torch.nn.functional as F


def _ph_env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return default if v is None else int(v)


def _ph_env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return default if v is None else float(v)


def _ph_env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def gaussian_kernel(device, dtype, k=3, sigma=1.0):
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    g = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (g / g.sum()).view(1, 1, k, k)

def smooth_flat(grid_flat, num_rows, num_cols, kernel):
    pad = kernel.shape[-1] // 2
    
    x = grid_flat.view(1, 1, num_rows, num_cols)
    return F.conv2d(x, kernel, padding=pad).view(-1)



@dataclass
class ProblemRepresentation:
    pos_init : torch.Tensor 
    sizes : torch.Tensor 
    
    movable_idx : torch.Tensor 
    movable_macro_idx : torch.Tensor 
    fixed_macro_idx : torch.Tensor 
    hard_macro_idx : torch.Tensor 
    
    pin_parent : torch.Tensor 
    pin_offset : torch.Tensor 
    pin2net : torch.Tensor 
    num_nets : int 
    
    bin_boxes : torch.Tensor
    bin_area : torch.Tensor
    num_rows : int 
    num_cols : int 
    
    canvas_w : float 
    canvas_h : float 

    def to(self, device: torch.device) -> "ProblemRepresentation":
        """Move all tensor fields once so the optimization loop stays on-device."""
        return ProblemRepresentation(
            pos_init=self.pos_init.to(device),
            sizes=self.sizes.to(device),
            movable_idx=self.movable_idx.to(device),
            movable_macro_idx=self.movable_macro_idx.to(device),
            fixed_macro_idx=self.fixed_macro_idx.to(device),
            hard_macro_idx=self.hard_macro_idx.to(device),
            pin_parent=self.pin_parent.to(device),
            pin_offset=self.pin_offset.to(device),
            pin2net=self.pin2net.to(device),
            num_nets=self.num_nets,
            bin_boxes=self.bin_boxes.to(device),
            bin_area=self.bin_area.to(device),
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            canvas_w=self.canvas_w,
            canvas_h=self.canvas_h,
        )
        
        
def scatter_logsumexp(values, index, num_groups):
    max_per_group = torch.full(
        (num_groups,), -torch.inf, device=values.device, dtype=values.dtype
    )
    max_per_group = max_per_group.scatter_reduce(
        dim=0, index=index, src=values, reduce="amax", include_self=True,
    )

    shifted_values = values - max_per_group[index]
    exp_shifted = torch.exp(shifted_values)

    sum_exp = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    sum_exp = sum_exp.scatter_add(0, index, exp_shifted)

    return max_per_group + torch.log(sum_exp.clamp_min(1e-10))
    

def smooth_hpwl(full_pos, pin_parent, pin_offset, pin2net, num_nets, gamma,
                num_net, canvas_w, canvas_h):

    pin_pos = full_pos[pin_parent] + pin_offset    # [P, 2]
    x = pin_pos[:, 0]
    y = pin_pos[:, 1]

    x_max = gamma * scatter_logsumexp(x / gamma, pin2net, num_nets)
    x_min = -gamma * scatter_logsumexp(-x / gamma, pin2net, num_nets)
    y_max = gamma * scatter_logsumexp(y / gamma, pin2net, num_nets)
    y_min = -gamma * scatter_logsumexp(-y / gamma, pin2net, num_nets)

    wl_per_net = (x_max - x_min) + (y_max - y_min)
    
    denom = (canvas_w + canvas_h) * max(num_net, 1)
    return wl_per_net.sum() / denom


def gamma_schedule(t, canvas_size):
    gamma_start = canvas_size / 50
    gamma_end = canvas_size / 2000
    return gamma_start * (gamma_end / gamma_start) ** t


def rectangle_bin_occupancy(pos, sizes, macro_idx, bin_boxes, num_rows, num_cols):
    p = pos[macro_idx]
    s = sizes[macro_idx]

    # Macro bounding boxes in absolute coords.
    x0 = p[:, 0] - 0.5 * s[:, 0]
    x1 = p[:, 0] + 0.5 * s[:, 0]
    y0 = p[:, 1] - 0.5 * s[:, 1]
    y1 = p[:, 1] + 0.5 * s[:, 1]

    bx0 = bin_boxes[:, 0]
    by0 = bin_boxes[:, 1]
    bx1 = bin_boxes[:, 2]
    by1 = bin_boxes[:, 3]

    ox = torch.relu(
        torch.minimum(x1[:, None], bx1[None, :])
        - torch.maximum(x0[:, None], bx0[None, :])
    )
    oy = torch.relu(
        torch.minimum(y1[:, None], by1[None, :])
        - torch.maximum(y0[:, None], by0[None, :])
    )

    # 2D overlap area per (macro, bin)
    area = ox * oy       # [M, B]
    occ = area.sum(dim=0)

    return occ


def density_loss(
    full_pos,
    sizes,
    movable_macro_idx,
    fixed_occ,
    bin_boxes,
    bin_area,
    target_density,
    num_rows,
    num_cols,
    frac,
    kernel,
    hotspot_weight=0.1,
    hotspot_tau=0.05,
    use_tilos=False,
):
    movable_occ = rectangle_bin_occupancy(
        full_pos, sizes, movable_macro_idx, bin_boxes, num_rows, num_cols,
    )

    movable_occ = smooth_flat(movable_occ, num_rows=num_rows, num_cols=num_cols, kernel=kernel)

    if use_tilos:
        total_occ = movable_occ + fixed_occ
        density_ratio = total_occ / bin_area.clamp_min(1e-9)
        topk = soft_topk_mean(
            density_ratio, frac=frac, beta=0.02,
            mask_zeros=True, zero_eps=1e-6,
        )
        return 0.5 * topk, total_occ

    capacity = target_density * bin_area
    available = torch.clamp(capacity - fixed_occ, min=1e-9)
    overflow = torch.relu(movable_occ - available)
    relative = overflow / capacity.clamp_min(1e-9)

    topk = soft_topk_mean(relative, frac=frac)
    hotspot = softmax_hotspot(relative, tau=hotspot_tau)
    loss = topk + hotspot_weight * hotspot

    return loss, movable_occ + fixed_occ


def soft_topk_mean(values, frac=0.10, beta=0.02, mask_zeros=False, zero_eps=1e-6):
    flat = values.flatten()

    if mask_zeros:
        with torch.no_grad():
            nonzero = flat[flat > zero_eps]
            if nonzero.numel() < 2:
                return flat.sum() * 0.0
            n_total = flat.numel()

            top_k_total = max(1, int(frac * n_total))
            n_nz = nonzero.numel()
            adj_frac = min(1.0, top_k_total / n_nz)
            q = torch.quantile(nonzero, 1.0 - adj_frac)

        nz_mask = (flat > zero_eps).to(flat.dtype)
        weights = torch.sigmoid((flat - q) / beta) * nz_mask
        return (weights * flat).sum() / weights.sum().clamp_min(1e-9)

    with torch.no_grad():
        q = torch.quantile(flat.detach(), 1.0 - frac)

    weights = torch.sigmoid((flat - q) / beta)
    return (weights * flat).sum() / weights.sum().clamp_min(1e-9)


def hard_overlap_loss(full_pos, sizes, hard_macro_idx, gap=1e-3):

    p = full_pos[hard_macro_idx]
    s = sizes[hard_macro_idx]

    x0 = p[:, 0] - 0.5 * s[:, 0] - gap
    x1 = p[:, 0] + 0.5 * s[:, 0] + gap
    y0 = p[:, 1] - 0.5 * s[:, 1] - gap
    y1 = p[:, 1] + 0.5 * s[:, 1] + gap

    ox = torch.relu(
        torch.minimum(x1[:, None], x1[None, :])
        - torch.maximum(x0[:, None], x0[None, :])
    )
    oy = torch.relu(
        torch.minimum(y1[:, None], y1[None, :])
        - torch.maximum(y0[:, None], y0[None, :])
    )

    area = ox * oy

    n = len(hard_macro_idx)
    mask = torch.triu(
        torch.ones((n, n), device=full_pos.device, dtype=full_pos.dtype),
        diagonal=1,
    )

    return (area * mask).sum()

def soft_net_bbox(full_pos, pin_parent, pin_offset, pin2net, num_nets, gamma):

    pin_pos = full_pos[pin_parent] + pin_offset

    x = pin_pos[:, 0]
    y = pin_pos[:, 1]

    x_max = gamma * scatter_logsumexp(x / gamma, pin2net, num_nets)
    x_min = -gamma * scatter_logsumexp(-x / gamma, pin2net, num_nets)
    y_max = gamma * scatter_logsumexp(y / gamma, pin2net, num_nets)
    y_min = -gamma * scatter_logsumexp(-y / gamma, pin2net, num_nets)

    return x_min, y_min, x_max, y_max


def true_net_bbox(full_pos, pin_parent, pin_offset, pin2net, num_nets):

    pin_pos = full_pos[pin_parent] + pin_offset
    x = pin_pos[:, 0]
    y = pin_pos[:, 1]

    inf = float("inf")
    device, dtype = x.device, x.dtype

    x_max = torch.full((num_nets,), -inf, device=device, dtype=dtype)
    x_max = x_max.scatter_reduce(0, pin2net, x, reduce="amax", include_self=True)

    x_min = torch.full((num_nets,), inf, device=device, dtype=dtype)
    x_min = x_min.scatter_reduce(0, pin2net, x, reduce="amin", include_self=True)

    y_max = torch.full((num_nets,), -inf, device=device, dtype=dtype)
    y_max = y_max.scatter_reduce(0, pin2net, y, reduce="amax", include_self=True)

    y_min = torch.full((num_nets,), inf, device=device, dtype=dtype)
    y_min = y_min.scatter_reduce(0, pin2net, y, reduce="amin", include_self=True)

    return x_min, y_min, x_max, y_max


def softmax_hotspot(values, tau=0.05):
    return tau * torch.logsumexp(values.flatten() / tau, dim=0)


def axis_loss (demand, capacity, top_frac, 
               soft_topk_beta=0.02, hotspot_weight=0.1, hotspot_tau=0.05):
    scale = (demand.mean().detach() / capacity.mean().detach()).clamp_min(1e-6)
    capacity_scaled = capacity * scale 
    rel_over = torch.relu(demand - capacity_scaled) / capacity_scaled.clamp_min(1e-6)
   
    topk = soft_topk_mean(rel_over, frac=top_frac, beta=soft_topk_beta)
     
    hotspot = hotspot_weight * softmax_hotspot(rel_over, tau=hotspot_tau)
    return topk + hotspot_weight * hotspot

def bbox_congestion_loss(
    full_pos,
    pin_parent,
    pin_offset,
    pin2net,
    num_nets,
    bin_boxes,
    num_rows,
    num_cols,
    H_capacity,
    V_capacity,
    sizes=None,
    hard_macro_idx=None,
    hrouting_alloc=0.0,
    vrouting_alloc=0.0,
    smooth_kernel_h=None,
    smooth_kernel_v=None,
    net_weights=None,
    top_frac=0.05,
    soft_topk_beta=0.02,
    hotspot_weight=0.0,
    routing_model="lroute",
    row_y_centers=None,
    col_x_centers=None,
    bin_w=None,
    bin_h=None,
):

    x0, y0, x1, y1 = true_net_bbox(
        full_pos, pin_parent, pin_offset, pin2net, num_nets,
    )

    w = (x1 - x0).clamp_min(1e-6)
    h = (y1 - y0).clamp_min(1e-6)
    bbox_area = (w * h).clamp_min(1e-6)

    bx0 = bin_boxes[:, 0]
    by0 = bin_boxes[:, 1]
    bx1 = bin_boxes[:, 2]
    by1 = bin_boxes[:, 3]

    if routing_model == "lroute":
        assert row_y_centers is not None and col_x_centers is not None, \
            "lroute requires row_y_centers and col_x_centers"
        assert bin_w is not None and bin_h is not None, \
            "lroute requires bin_w, bin_h scalars"

        row_by0 = row_y_centers - 0.5 * bin_h
        row_by1 = row_y_centers + 0.5 * bin_h
        col_bx0 = col_x_centers - 0.5 * bin_w
        col_bx1 = col_x_centers + 0.5 * bin_w

        x_overlap = torch.relu(
            torch.minimum(x1[:, None], col_bx1[None, :])
            - torch.maximum(x0[:, None], col_bx0[None, :])
        ) 

        y_overlap = torch.relu(
            torch.minimum(y1[:, None], row_by1[None, :])
            - torch.maximum(y0[:, None], row_by0[None, :])
        )  # [N_nets, N_rows]

        y0_strip_lo = y0[:, None] - 0.5 * bin_h
        y0_strip_hi = y0[:, None] + 0.5 * bin_h
        y0_row = torch.relu(
            torch.minimum(row_by1[None, :], y0_strip_hi)
            - torch.maximum(row_by0[None, :], y0_strip_lo)
        ) / bin_h  # [N_nets, N_rows], values in [0, 1]

        y1_strip_lo = y1[:, None] - 0.5 * bin_h
        y1_strip_hi = y1[:, None] + 0.5 * bin_h
        y1_row = torch.relu(
            torch.minimum(row_by1[None, :], y1_strip_hi)
            - torch.maximum(row_by0[None, :], y1_strip_lo)
        ) / bin_h  # [N_nets, N_rows]

        # Col indicators (left/right strips for V demand).
        x0_strip_lo = x0[:, None] - 0.5 * bin_w
        x0_strip_hi = x0[:, None] + 0.5 * bin_w
        x0_col = torch.relu(
            torch.minimum(col_bx1[None, :], x0_strip_hi)
            - torch.maximum(col_bx0[None, :], x0_strip_lo)
        ) / bin_w

        x1_strip_lo = x1[:, None] - 0.5 * bin_w
        x1_strip_hi = x1[:, None] + 0.5 * bin_w
        x1_col = torch.relu(
            torch.minimum(col_bx1[None, :], x1_strip_hi)
            - torch.maximum(col_bx0[None, :], x1_strip_lo)
        ) / bin_w

        if net_weights is not None:
            H_scale = net_weights  # [N_nets]
            V_scale = net_weights
        else:
            H_scale = torch.ones_like(x0)
            V_scale = torch.ones_like(x0)


        weighted_x_overlap = x_overlap * H_scale[:, None]
        y_strip_sum = 0.5 * (y0_row + y1_row)
        H_demand_2d = torch.einsum("nc,nr->rc", weighted_x_overlap, y_strip_sum)
        H_demand = H_demand_2d.reshape(-1)

        weighted_y_overlap = y_overlap * V_scale[:, None]
        x_strip_sum = 0.5 * (x0_col + x1_col)
        V_demand_2d = torch.einsum("nr,nc->rc", weighted_y_overlap, x_strip_sum)
        V_demand = V_demand_2d.reshape(-1)

    else:
        # RUDY: uniform demand across bbox. Higher memory but kept for
        # ablation. routing_model="rudy".
        ox = torch.relu(
            torch.minimum(x1[:, None], bx1[None, :])
            - torch.maximum(x0[:, None], bx0[None, :])
        )
        oy = torch.relu(
            torch.minimum(y1[:, None], by1[None, :])
            - torch.maximum(y0[:, None], by0[None, :])
        )
        overlap_area = ox * oy

        H_demand_per_area = w / bbox_area
        V_demand_per_area = h / bbox_area

        if net_weights is not None:
            H_demand_per_area = H_demand_per_area * net_weights
            V_demand_per_area = V_demand_per_area * net_weights

        H_demand = (overlap_area * H_demand_per_area[:, None]).sum(dim=0)
        V_demand = (overlap_area * V_demand_per_area[:, None]).sum(dim=0)

    if smooth_kernel_h is not None:
        Hf = H_demand.view(1, 1, num_rows, num_cols)
        pad = smooth_kernel_h.shape[-2] // 2
        H_demand = F.conv2d(Hf, smooth_kernel_h, padding=(pad, 0)).view(-1)
    if smooth_kernel_v is not None:
        Vf = V_demand.view(1, 1, num_rows, num_cols)
        pad = smooth_kernel_v.shape[-1] // 2
        V_demand = F.conv2d(Vf, smooth_kernel_v, padding=(0, pad)).view(-1)

    if sizes is not None and hard_macro_idx is not None and hard_macro_idx.numel() > 0:
        macro_blockage_area = rectangle_bin_occupancy(
            full_pos, sizes, hard_macro_idx, bin_boxes, num_rows, num_cols,
        )
        H_blockage = macro_blockage_area * hrouting_alloc
        V_blockage = macro_blockage_area * vrouting_alloc
    else:
        H_blockage = torch.zeros_like(H_demand)
        V_blockage = torch.zeros_like(V_demand)

    H_util = (H_demand + H_blockage) / H_capacity.clamp_min(1e-9)
    V_util = (V_demand + V_blockage) / V_capacity.clamp_min(1e-9)

    util = torch.cat([H_util, V_util], dim=0)

    topk = soft_topk_mean(
        util, frac=top_frac, beta=soft_topk_beta,
        mask_zeros=False,
    )
    if hotspot_weight > 0:
        topk = topk + hotspot_weight * softmax_hotspot(util, tau=0.05)

    return topk, (H_demand, V_demand)


@torch.no_grad()
def project_to_canvas(param_pos, sizes, movable_idx, canvas_w, canvas_h, eps=1e-3):

    s = sizes[movable_idx]

    x_min = 0.5 * s[:, 0] + eps
    x_max = canvas_w - 0.5 * s[:, 0] - eps

    y_min = 0.5 * s[:, 1] + eps
    y_max = canvas_h - 0.5 * s[:, 1] - eps

    param_pos[:, 0].clamp_(x_min, x_max)
    param_pos[:, 1].clamp_(y_min, y_max)


def schedule(t, overlap_final=10.0, name="current", disp_final=0.0):

    wl_weight = 1.0

    displacement_weight = ramp(t, start=0.05, end=0.80, final=disp_final)

    if name == "current":
        density_weight = ramp(t, start=0.10, end=0.70, final=0.5)
        congestion_weight = ramp(t, start=0.05, end=0.50, final=1.5)
        overlap_weight = ramp(t, start=0.40, end=1.00, final=overlap_final)

    elif name == "soft_overlap":
        density_weight = ramp(t, start=0.00, end=0.50, final=1.0)
        congestion_weight = ramp(t, start=0.05, end=0.70, final=1.0)
        overlap_weight = ramp(t, start=0.10, end=0.80, final=1.0)

    elif name == "proxy_first":
        density_weight = ramp(t, start=0.00, end=0.45, final=2.0)
        congestion_weight = ramp(t, start=0.10, end=0.70, final=2.0)
        overlap_weight = ramp(t, start=0.25, end=1.00, final=0.5)

    elif name == "early_density":
        density_weight = 0.5 + ramp(t, start=0.00, end=0.50, final=2.0)
        congestion_weight = ramp(t, start=0.10, end=0.70, final=1.0)
        overlap_weight = ramp(t, start=0.10, end=0.80, final=1.0)

    elif name == "legal_heavy":
        density_weight = ramp(t, start=0.00, end=0.50, final=1.0)
        congestion_weight = ramp(t, start=0.10, end=0.80, final=1.0)
        overlap_weight = ramp(t, start=0.05, end=0.80, final=3.0)

    else:
        raise ValueError(f"unknown schedule name: {name!r}")

    return wl_weight, density_weight, congestion_weight, overlap_weight, displacement_weight


def ramp(t, start, end, final):
    if t <= start:
        return 0.0
    if t >= end:
        return final
    return final * (t - start) / (end - start)


def displacement_loss(full_pos, reference_pos, movable_idx):
    delta = full_pos[movable_idx] - reference_pos[movable_idx]
    return delta.pow(2).sum(dim=1).mean()


def assemble_full_positions(param_pos, pos_init, movable_idx):
    full_pos = pos_init.clone()
    full_pos[movable_idx] = param_pos
    return full_pos
        
class Dreamplace:


    def __init__(self, run_name: str = "") -> None:
        self._plc_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_iters = _ph_env_int("ITER", 800)
        self.lr = _ph_env_float("LR", 1e-2)
        self.target_density = _ph_env_float("TARGET_DENSITY", 0.7)
        
        self.optimizer_name = _ph_env_str("OPTIMIZER", "adam")

        self.SAVE_ROWS_EVERY = 25

        self.k = _ph_env_int("KERNEL_SIZE", 1)
        self.sigma = _ph_env_float("KERNEL_SIGMA", 1.0)
        self.density_frac = _ph_env_float("DENSITY_FRAC", 0.5)

        self.congestion_frac = _ph_env_float("CONGESTION_FRAC", 0.1)

        self.density_use_tilos = _ph_env_int("DENSITY_USE_TILOS", 0) != 0

        self.routing_model = _ph_env_str("CONG_MODEL", "rudy")

        self.disp_final = _ph_env_float("DISP_FINAL", 0.06)

        self.mass_scale_exp = _ph_env_float("MASS_SCALE", 0.0)

        self.net_weight_exponent = _ph_env_float("NET_WEIGHT_EXP", 0.5)

        self.soft_topk_beta=0.02
        self.hotspot_weight=0.1

        self.polish_budget_seconds = 0.0

        self.schedule_name = "current"

        self.hill_climb_budget_seconds = 600.0
        self.max_hc = 600.0
        
        self.hill_climb_rounds = 5
        self.hill_climb_k_neighbors = 10
        self.hotspot_tau=0.05



    def _get_plc(self, benchmark: Benchmark):
        name = benchmark.name
        if name in self._plc_cache:
            return self._plc_cache[name]

        ibm_dir = Path("external/MacroPlacement/Testcases/ICCAD04") / name

        if ibm_dir.exists():
            _, plc = load_benchmark_accel_from_dir(str(ibm_dir))
        else:
            base = Path("external/MacroPlacement/Flows/NanGate45") / name / "netlist" / "output_CT_Grouping"
            _, plc = load_benchmark_accel(
                str(base / "netlist.pb.txt"),
                str(base / "initial.plc"),
                name=name,
            )

        self._plc_cache[name] = plc
        return plc


    def _build_problem_representation(
        self, benchmark: Benchmark, plc
    ) -> ProblemRepresentation:
        """
        Flatten the Benchmark/plc state into the tensors the differentiable
        placer consumes.

        Object layout (length N = num_macros + num_ports):
          [0, num_hard)                    : hard macros
          [num_hard, num_macros)           : soft macros
          [num_macros, num_macros+num_ports): I/O ports (size = (0, 0))

        Pin table is built from plc.nets directly so each (net, pin) pair
        becomes its own row — benchmark.net_nodes collapses to unique
        parents and discards pin offsets, which smooth_hpwl needs.
        """
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        num_ports = int(benchmark.port_positions.shape[0])
        N = num_macros + num_ports

        # ---- positions / sizes ----
        pos_init = torch.zeros((N, 2), dtype=torch.float32)
        pos_init[:num_macros] = benchmark.macro_positions
        if num_ports > 0:
            pos_init[num_macros:] = benchmark.port_positions

        sizes = torch.zeros((N, 2), dtype=torch.float32)
        sizes[:num_macros] = benchmark.macro_sizes
        # ports keep (0, 0)

        # ---- index sets ----
        hard_macro_idx = torch.arange(num_hard, dtype=torch.long)

        macro_fixed = benchmark.macro_fixed  # [num_macros] bool
        fixed_macro_idx = torch.where(macro_fixed)[0]
        movable_macro_idx = torch.where(~macro_fixed)[0]
        # Ports are treated as fixed (no movable port objects).
        movable_idx = movable_macro_idx.clone()

        # ---- name -> bench_idx (mirror load_benchmark_accel) ----
        plc_idx_to_bench = {}
        for b_idx, p_idx in enumerate(benchmark.hard_macro_indices):
            plc_idx_to_bench[p_idx] = b_idx
        for off, p_idx in enumerate(benchmark.soft_macro_indices):
            plc_idx_to_bench[p_idx] = num_hard + off
        for off, p_idx in enumerate(plc.port_indices):
            plc_idx_to_bench[p_idx] = num_macros + off

        name_to_bench = {
            plc.modules_w_pins[p_idx].get_name(): b_idx
            for p_idx, b_idx in plc_idx_to_bench.items()
        }

        # pin_name (e.g. "MACRO/PIN") -> (parent_bench_idx, offset_x, offset_y)
        # Only hard-macro pins carry non-zero offsets in this codebase.
        hard_pin_info = {}
        for p_idx in plc.hard_macro_pin_indices:
            pin = plc.modules_w_pins[p_idx]
            parent_name = pin.get_macro_name() if hasattr(pin, "get_macro_name") else None
            if parent_name and parent_name in name_to_bench:
                hard_pin_info[pin.get_name()] = (
                    name_to_bench[parent_name],
                    float(pin.x_offset),
                    float(pin.y_offset),
                )

        # ---- pin tables ----
        pin_parent_list: list = []
        pin_offset_list: list = []
        pin2net_list: list = []

        net_id = 0
        for driver, sinks in plc.nets.items():
            net_pins = []
            for pin_name in [driver] + sinks:
                info = hard_pin_info.get(pin_name)
                if info is not None:
                    net_pins.append(info)
                    continue
                # Fall back to parent center (soft macros, ports).
                parent_name = pin_name.split("/")[0]
                parent_bench = name_to_bench.get(parent_name)
                if parent_bench is not None:
                    net_pins.append((parent_bench, 0.0, 0.0))

            if not net_pins:
                continue

            for parent_bench, ox, oy in net_pins:
                pin_parent_list.append(parent_bench)
                pin_offset_list.append((ox, oy))
                pin2net_list.append(net_id)
            net_id += 1

        num_nets = net_id
        pin_parent = torch.tensor(pin_parent_list, dtype=torch.long)
        pin_offset = torch.tensor(pin_offset_list, dtype=torch.float32) \
            if pin_offset_list else torch.zeros((0, 2), dtype=torch.float32)
        pin2net = torch.tensor(pin2net_list, dtype=torch.long)

        # ---- density / congestion grid ----
        num_rows = int(benchmark.grid_rows)
        num_cols = int(benchmark.grid_cols)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        bin_w = canvas_w / num_cols
        bin_h = canvas_h / num_rows

        rows = torch.arange(num_rows, dtype=torch.float32)
        cols = torch.arange(num_cols, dtype=torch.float32)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        rr = rr.reshape(-1)
        cc = cc.reshape(-1)

        bin_boxes = torch.stack(
            [cc * bin_w, rr * bin_h, (cc + 1) * bin_w, (rr + 1) * bin_h],
            dim=1,
        )
        bin_area = torch.full(
            (num_rows * num_cols,), bin_w * bin_h, dtype=torch.float32
        )

        return ProblemRepresentation(
            pos_init=pos_init,
            sizes=sizes,
            movable_idx=movable_idx,
            movable_macro_idx=movable_macro_idx,
            fixed_macro_idx=fixed_macro_idx,
            hard_macro_idx=hard_macro_idx,
            pin_parent=pin_parent,
            pin_offset=pin_offset,
            pin2net=pin2net,
            num_nets=num_nets,
            bin_boxes=bin_boxes,
            bin_area=bin_area,
            num_rows=num_rows,
            num_cols=num_cols,
            canvas_w=canvas_w,
            canvas_h=canvas_h,
        )
    
    

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Run differentiable global placement on a single benchmark.

        Pipeline:
          1. Load plc + flatten benchmark into tensor problem representation.
          2. Precompute fixed-macro occupancy (baseline for density loss).
          3. Derive per-bin routing capacity (threshold for congestion loss).
          4. Build Adam over only the movable centers.
          5. For each iteration:
               - rebuild full_pos by splicing param_pos into fixed positions
               - compute wl / density / congestion / overlap / displacement
               - apply scheduled weights, backward, step
               - project movables back into canvas
               - track best by total loss
          6. Return best_pos (truncated to macros, dropping ports).

        The output IS NOT GUARANTEED LEGAL — overlaps are tolerated. A
        legalizer + SA polish step (not yet implemented) is required before
        submitting to the proxy evaluator.
        """


        import random as _py_random
        base_seed = _ph_env_int("RANDOM_SEED", 42)
        name_hash = sum(ord(c) for c in benchmark.name) & 0xFFFF
        full_seed = (base_seed + name_hash) & 0x7FFFFFFF
        _py_random.seed(full_seed)
        import numpy as _np
        _np.random.seed(full_seed)
        torch.manual_seed(full_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(full_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if _ph_env_int("DETERMINISTIC", 1):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as e:
                print(f"[place] warning: use_deterministic_algorithms failed ({e})")

        plc = self._get_plc(benchmark)
        problem = self._build_problem_representation(benchmark, plc).to(self.device)
        print(f"Dreamplace torch device: {self.device}, seed={full_seed}")


        with torch.no_grad():
            fixed_occ = rectangle_bin_occupancy(
                problem.pos_init,
                problem.sizes,
                problem.fixed_macro_idx,
                problem.bin_boxes,
                problem.num_rows,
                problem.num_cols,
            )

        kernel = gaussian_kernel(device=self.device,
                                 dtype=torch.float32,
                                 k=self.k,
                                 sigma=self.sigma)

        fixed_occ = smooth_flat(fixed_occ, num_rows=problem.num_rows, num_cols=problem.num_cols, kernel=kernel )

        smooth_range = 2
        K = 2 * smooth_range + 1
        box = torch.full((K,), 1.0 / K, device=self.device, dtype=torch.float32)
        smooth_kernel_h = box.view(1, 1, K, 1)   # smooth along Y (rows)
        smooth_kernel_v = box.view(1, 1, 1, K)   # smooth along X (cols)

        hrouting_alloc = float(getattr(plc, "hrouting_alloc", 0.0))
        vrouting_alloc = float(getattr(plc, "vrouting_alloc", 0.0))
        print(f"[place] hrouting_alloc={hrouting_alloc:.3f}  "
              f"vrouting_alloc={vrouting_alloc:.3f}")

        bin_w = problem.canvas_w / problem.num_cols
        bin_h = problem.canvas_h / problem.num_rows

        row_y_centers = torch.arange(
            problem.num_rows, dtype=torch.float32, device=self.device,
        ) * bin_h + 0.5 * bin_h
        col_x_centers = torch.arange(
            problem.num_cols, dtype=torch.float32, device=self.device,
        ) * bin_w + 0.5 * bin_w
        cap_per_bin = float(
            benchmark.hroutes_per_micron * bin_h
            + benchmark.vroutes_per_micron * bin_w
        )
        
        H_capacity = torch.full(
            (problem.num_rows * problem.num_cols,),
            float(benchmark.hroutes_per_micron * bin_h),
            dtype=torch.float32,
            device=self.device,
        )
        V_capacity = torch.full(
            (problem.num_rows * problem.num_cols,),
            float(benchmark.vroutes_per_micron * bin_w),
            dtype=torch.float32, device=self.device,
        )

        pin_count_per_net = torch.zeros(
            problem.num_nets, device=self.device, dtype=torch.float32,
        )
        pin_count_per_net.scatter_add_(
            0, problem.pin2net,
            torch.ones_like(problem.pin2net, dtype=torch.float32),
        )
        net_weights = pin_count_per_net ** self.net_weight_exponent

        print(f"[place] net_weights: exp={self.net_weight_exponent}, "
              f"min={net_weights.min().item():.2f}, "
              f"max={net_weights.max().item():.2f}, "
              f"mean=1.00")

        param_pos = nn.Parameter(problem.pos_init[problem.movable_idx].clone())

        opt_name = self.optimizer_name.lower()
        if opt_name == "nesterov":
            optimizer = torch.optim.SGD(
                [param_pos], lr=self.lr, momentum=0.9, nesterov=True,
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam([param_pos], lr=self.lr)
        else:
            raise ValueError(f"unknown OPTIMIZER={self.optimizer_name!r}; "
                             f"expected 'adam' or 'nesterov'")
        print(f"[place] optimizer={opt_name} lr={self.lr}")

        num_hard = int(problem.hard_macro_idx.shape[0])
        num_movable = int(problem.movable_macro_idx.shape[0])
        overlap_base = _ph_env_float("OVERLAP_BASE", 10.0)
        overlap_final = overlap_base * max(1.0, num_hard / 246.0)

        reference_pos = problem.pos_init.clone()

        PROXY_EVAL_EVERY = 25
        best_proxy = float("inf")
        best_pos = None

        proxy_now = float("inf")
        overlaps_now = -1
        canvas_size = max(problem.canvas_w, problem.canvas_h)
        best_overlaps = -1
        
        rows = []
        
        # Expanded bucket thresholds: track more candidate snapshots so the
        # downstream legalize-all-buckets step has more chances to find a
        # better legal placement. The brief observed bucket k=5 sometimes
        # beats k=0 post-legalization — wider granularity captures this.
        bucket_thresholds = (0, 1, 2, 5, 10, 20, 50, 100, 10**9)
        buckets = {
            k: {"score": float("inf"), "pos": None, "overlaps_now": None}
            for k in bucket_thresholds
        }


        for it in range(self.num_iters):
            # Normalized time in [0, 1] drives both gamma and the weight schedule.
            t = it / max(1, self.num_iters - 1)

            optimizer.zero_grad()

            # Rebuild full position tensor — autograd connects param_pos
            # through the splice into all downstream losses.
            full_pos = assemble_full_positions(
                param_pos, problem.pos_init, problem.movable_idx
            )
            gamma = gamma_schedule(t, canvas_size)

            wl = smooth_hpwl(
                full_pos,
                problem.pin_parent,
                problem.pin_offset,
                problem.pin2net,
                problem.num_nets,
                gamma,
                problem.num_nets,
                problem.canvas_w,
                problem.canvas_h
            )

            d_loss, _density_map = density_loss(
                full_pos,
                problem.sizes,
                problem.movable_macro_idx,
                fixed_occ,
                problem.bin_boxes,
                problem.bin_area,
                self.target_density,
                problem.num_rows,
                problem.num_cols,
                kernel=kernel,
                frac=self.density_frac,
                use_tilos=self.density_use_tilos,
            )

            c_loss, _cong_map = bbox_congestion_loss(
                full_pos,
                problem.pin_parent,
                problem.pin_offset,
                problem.pin2net,
                problem.num_nets,
                problem.bin_boxes,
                problem.num_rows,
                problem.num_cols,
                H_capacity,
                V_capacity,
                sizes=problem.sizes,
                hard_macro_idx=problem.hard_macro_idx,
                hrouting_alloc=hrouting_alloc,
                vrouting_alloc=vrouting_alloc,
                smooth_kernel_h=smooth_kernel_h,
                smooth_kernel_v=smooth_kernel_v,
                net_weights=net_weights,
                top_frac=self.congestion_frac,
                soft_topk_beta=self.soft_topk_beta,
                # hotspot_weight=0 by default; matches TILOS (no separate
                # hotspot term, just top-5% mean over concat(H, V)).
                hotspot_weight=0.0,
                routing_model=self.routing_model,
                row_y_centers=row_y_centers,
                col_x_centers=col_x_centers,
                bin_w=bin_w,
                bin_h=bin_h,
            )

            ov = hard_overlap_loss(
                full_pos, problem.sizes, problem.hard_macro_idx, gap=1e-3
            )

            disp = displacement_loss(
                full_pos, reference_pos, problem.movable_idx
            )

            w_wl, w_d, w_c, w_ov, w_disp = schedule(
                t, overlap_final=overlap_final, name=self.schedule_name,
                disp_final=self.disp_final,
            )

            loss = (
                w_wl * wl
                + w_d * d_loss
                + w_c * c_loss
                + w_ov * ov
                + w_disp * disp
            )

            loss.backward()


            torch.nn.utils.clip_grad_norm_([param_pos], max_norm=10.0)
            optimizer.step()
            
            LR_FLOOR = 0.1
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * t))
            for g in optimizer.param_groups:
                g["lr"] = self.lr * max(LR_FLOOR, cos_factor)

            # Hard re-projection: optimizer.step() can push a macro outside
            # the canvas if its loss gradient pointed that way.
            project_to_canvas(
                param_pos,
                problem.sizes,
                problem.movable_idx,
                problem.canvas_w,
                problem.canvas_h,
                eps=1e-3,
            )

            if it % PROXY_EVAL_EVERY == 0 or it == self.num_iters - 1:
                with torch.no_grad():
                    snapshot = (
                        full_pos.detach()[: benchmark.num_macros]
                        .clone()
                        .cpu()
                    )
                    pc = compute_proxy_cost(
                        snapshot, benchmark, plc, with_comparison=False
                    )
                    proxy_now = float(pc["fast_proxy_cost"])
                    overlaps_now = pc["overlap_metrics_fast"]["overlap_count"]
                    
                    for k in buckets:
                        if overlaps_now <= k and proxy_now < buckets[k]["score"]:
                            buckets[k]["score"] = proxy_now
                            buckets[k]["pos"] = snapshot.clone()
                            buckets[k]["overlaps_now"] = overlaps_now
                    
                    
                if proxy_now < best_proxy:
                    best_proxy = proxy_now
                    best_pos = full_pos.detach().clone()
                    best_overlaps = overlaps_now
                    
                    
                rows.append({
                    "it" : it,
                    "loss" : float(loss.item()),
                    
                    # surrogates 
                    "wl_sur" : float(wl.item()),
                    "d_sur" : float(d_loss.item()),
                    "c_sur" : float(c_loss.item()),
                    "ov_sur" : float(ov.item()),
                    
                    # exact proxy components
                    "proxy" : float(pc["fast_proxy_cost"]),
                    "exact_wl": float(pc["fast_wirelength_cost"]),
                    "exact_den": float(pc["fast_density_cost"]),
                    "exact_cong": float(pc["fast_congestion_cost"]),
                    "overlaps": int(pc["overlap_metrics_fast"]["overlap_count"]),
                    
                    # schedule context 
                    "gamma" : float(gamma),
                    "w_d" : float(w_d),
                    "w_c" : float(w_c),
                    "w_ov" : float(w_ov)

                })

            if it % 50 == 0:
                # Rolling Pearson(surrogate, exact) on the snapshots
                # we've recorded so far. Target: all three > +0.7 (the
                # leg.txt baseline that produced 1.36 avg). Negative or
                # near-zero values mean the surrogate is decoupled from
                # the proxy — schedule/loss-formulation needs work.
                if len(rows) >= 5:
                    df_partial = pd.DataFrame(rows[-50:])  # recent window
                    try:
                        pcc_wl = float(df_partial["wl_sur"].corr(df_partial["exact_wl"]))
                        pcc_d = float(df_partial["d_sur"].corr(df_partial["exact_den"]))
                        pcc_c = float(df_partial["c_sur"].corr(df_partial["exact_cong"]))
                        pcc_str = f"  pcc[wl={pcc_wl:+.2f} d={pcc_d:+.2f} c={pcc_c:+.2f}]"
                    except Exception:
                        pcc_str = ""
                else:
                    pcc_str = ""
                print(
                    f"it={it:04d} "
                    f"loss={loss.item():.4f} "
                    f"wl={wl.item():.4f} "
                    f"d={d_loss.item():.4f} "
                    f"c={c_loss.item():.4f} "
                    f"ov={ov.item():.4f} "
                    f"proxy={proxy_now:.4f} (ovlp={overlaps_now}, best={best_proxy:.4f})"
                    f"{pcc_str}"
                )


        df = pd.DataFrame(data=rows)

        if not os.path.exists("logs_track"):
            os.makedirs("logs_track", exist_ok=True)
        df.to_csv(f"logs_track/dreamplace_{benchmark.name}.csv", index=False)

        # ---- surrogate-vs-exact diagnostics (Pearson correlation) -------
        # The doc's central instruction: change one thing per run, check
        # correlation each time. Print it inline so we don't have to grep
        # the CSV.
        if len(df) >= 3:
            try:
                pcc_cong = float(df["c_sur"].corr(df["exact_cong"]))
                pcc_den = float(df["d_sur"].corr(df["exact_den"]))
                pcc_wl = float(df["wl_sur"].corr(df["exact_wl"]))
            except Exception as e:
                pcc_cong = pcc_den = pcc_wl = float("nan")
                print(f"[place] correlation diag failed: {e}")
            print(f"[place] surrogate vs exact (Pearson):  "
                  f"wl={pcc_wl:+.3f}  den={pcc_den:+.3f}  cong={pcc_cong:+.3f}")

        for bucket in buckets.items():
            k, v = bucket

            print(f"k={k} : score={v['score']:.4f}, overlaps={v['overlaps_now']}")

        print(f"[place] returning best snapshot: proxy={best_proxy:.4f}, overlaps={best_overlaps}")
        assert best_pos is not None, "no snapshot was ever taken"

        # -----------------------------------------------------------------
        # LEGALIZE ALL BUCKETS, PICK BEST LEGAL PROXY
        # -----------------------------------------------------------------
        # Old behavior: pick the lowest-overlap-bucket non-None snapshot,
        # legalize once, return. This was conservative — sometimes a
        # higher-overlap bucket has a much lower raw proxy and legalizes
        # to a better legal proxy than the safe bucket.
        #
        # New behavior: legalize every non-None bucket, return the one
        # with the best EXACT post-legalization proxy. Branch-and-bound:
        # process buckets in order of ascending raw_score; skip any
        # bucket whose raw_score already exceeds the best legal score
        # found so far (legalization can only increase score).
        best = self._legalize_all_buckets_and_pick_best(buckets, benchmark, plc)

        if best is None:
            # Catastrophic: every bucket failed to legalize. Two-tier
            # fallback before giving up:
            #
            # (1) Jitter + retry. Often the legalizer chokes on 1-2
            #     interlocked macros (ibm10's "1 overlapping pair remains
            #     after all three stages"). A ±20nm random nudge on hard
            #     macro positions breaks the symmetry; the analytical
            #     structure is preserved so proxy stays close to raw.
            #
            # (2) Shelf-pack. Last resort. Loses the analytical signal
            #     entirely but guaranteed legal. VALID > INVALID for the
            #     evaluator.
            print("[place] no bucket legalized; trying jitter retry...")
            best = self._try_jitter_and_legalize(
                best_pos[:benchmark.num_macros].cpu(), benchmark, plc,
            )
            if best is not None:
                best["raw_score"] = best_proxy
                best["overlaps_raw"] = best_overlaps
                best["degradation"] = best["score"] - best_proxy
                print(f"[place] jitter rescued: legal_proxy={best['score']:.4f}")
            else:
                print("[place] jitter failed; shelf-pack fallback")
                shelf_pos = self._shelf_pack_fallback(
                    best_pos[:benchmark.num_macros].cpu(), benchmark,
                )
                pc_fb = compute_proxy_cost(
                    shelf_pos, benchmark, plc, with_comparison=False,
                )
                fb_proxy = float(pc_fb["fast_proxy_cost"])
                fb_overlaps = int(pc_fb["overlap_metrics_fast"]["overlap_count"])
                print(f"[place] shelf-pack fallback: "
                      f"proxy={fb_proxy:.4f}, overlaps={fb_overlaps}")
                best = {
                    "name": "shelf_pack_fallback",
                    "score": fb_proxy,
                    "pos": shelf_pos,
                    "raw_score": best_proxy,
                    "overlaps_raw": best_overlaps,
                    "degradation": fb_proxy - best_proxy,
                }

        print(f"[place] winner: {best['name']} -> "
              f"legal_proxy={best['score']:.4f} "
              f"(raw {best['raw_score']:.4f} with "
              f"{best['overlaps_raw']} overlaps, "
              f"degradation={best['degradation']:+.4f})")

        # -----------------------------------------------------------------
        # HANAN HILL CLIMB (deterministic local search, after legalize)
        # -----------------------------------------------------------------
        # Greedy: try Hanan-grid alternatives for each macro, accept
        # only if exact proxy improves. Bounded runtime, no randomness.
        # Off by default (hill_climb_budget_seconds=0); enable in __init__.
        hc_budget_base = float(getattr(self, "hill_climb_budget_seconds", 0.0))
        if hc_budget_base > 0:
            # Scale budget with macro count. Per roadmap Priority 4:
            # 9/17 benchmarks were hitting the fixed 120s cap on round 0,
            # leaving improvement on the table. Bigger benchmarks need
            # proportionally more time (per-macro evaluation cost is roughly
            # linear in num_hard).
            #
            # Formula: base budget + 0.4s per hard macro, capped at 300s.
            # For ibm01 (246 hard): 120 + 98 = 218s.
            # For ibm12 (651 hard): 120 + 260 = 380, capped to 300s.
            # For ibm17 (~900 hard): hits 300s cap.
            hc_budget = min(self.max_hc, hc_budget_base + 0.4 * num_hard)
            print(f"[place] hill_climb budget: {hc_budget:.0f}s "
                  f"(base={hc_budget_base:.0f}s, num_hard={num_hard})")

            from hanan_hill_climb import hill_climb
            climbed = hill_climb(
                best["pos"], benchmark, plc,
                max_rounds=getattr(self, "hill_climb_rounds", 5),
                time_budget_seconds=hc_budget,
                k_neighbors=getattr(self, "hill_climb_k_neighbors", 10),
                verbose=True,
            )
            climb_pc = compute_proxy_cost(climbed, benchmark, plc, with_comparison=False)
            climb_proxy = float(climb_pc["fast_proxy_cost"])
            climb_overlaps = climb_pc["overlap_metrics_fast"]["overlap_count"]
            print(f"[place] post-hill_climb: proxy={climb_proxy:.4f}, "
                  f"overlaps={climb_overlaps}, "
                  f"climb_delta={climb_proxy - best['score']:+.4f}")
            if climb_overlaps == 0 and climb_proxy < best["score"]:
                best = {**best, "pos": climbed, "score": climb_proxy,
                        "name": best["name"] + "+climb"}
            elif climb_overlaps != 0:
                print(f"[place] WARNING: hill_climb produced {climb_overlaps} "
                      f"overlaps! keeping pre-climb legal placement")

        polish_budget = float(getattr(self, "polish_budget_seconds", 0.0))
        if polish_budget > 0:
            print("POLISHING ENABLED:")
            from sa_polisher import polish
            print(f"[place] polishing for up to {polish_budget:.0f}s...")
            polished_pos = polish(
                best["pos"], benchmark, plc,
                time_budget_seconds=polish_budget, verbose=True,
            )
            polish_pc = compute_proxy_cost(
                polished_pos, benchmark, plc, with_comparison=False,
            )
            polish_proxy = float(polish_pc["fast_proxy_cost"])
            polish_overlaps = polish_pc["overlap_metrics_fast"]["overlap_count"]
            print(f"[place] post-polish: proxy={polish_proxy:.4f}, "
                  f"overlaps={polish_overlaps}, "
                  f"polish_delta={polish_proxy - best['score']:+.4f}")
            if polish_overlaps != 0:
                print(f"[place] WARNING: polish produced {polish_overlaps} "
                      f"overlaps! falling back to pre-polish legal placement")
                return best["pos"]
            return polished_pos

        return best["pos"]

    def _shelf_pack_fallback(self, pos, benchmark):

        legal = pos.clone()
        num_hard = benchmark.num_hard_macros
        sizes_np = benchmark.macro_sizes.cpu().numpy()

        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        GAP = 1e-3

        order = sorted(range(num_hard), key=lambda i: -sizes_np[i, 1])

        cur_x = 0.0
        cur_y = 0.0
        row_h = 0.0
        for i in order:
            w = float(sizes_np[i, 0])
            h = float(sizes_np[i, 1])
            # Wrap to next row if no horizontal room.
            if cur_x + w > canvas_w:
                cur_x = 0.0
                cur_y += row_h + GAP
                row_h = 0.0
            # Overflow into origin matches sa_placer's behavior — will
            # still leave a few overlaps but bounded, not catastrophic.
            if cur_y + h > canvas_h:
                legal[i, 0] = w / 2
                legal[i, 1] = h / 2
                continue
            legal[i, 0] = cur_x + w / 2
            legal[i, 1] = cur_y + h / 2
            cur_x += w + GAP
            row_h = max(row_h, h)

        return legal

    def _try_jitter_and_legalize(self, pos, benchmark, plc,
                                 max_tries=4, jitter_microns=0.02):
        """
        Sometimes the legalizer fails because 1-2 hard macros are
        interlocked in a way it can't unstick (the ibm10 case:
        "1 overlapping pair remains after all three stages"). A tiny
        random jitter on hard-macro positions usually breaks the symmetry
        and lets the legalizer succeed without losing the proxy structure.

        Tries `max_tries` different random seeds before giving up.
        Returns the best legal result, or None if all attempts fail.
        """
        from legalizer import legalize
        import numpy as np
        rng = np.random.default_rng(seed=1234)

        best = None
        for trial in range(max_tries):
            jitter = rng.uniform(
                -jitter_microns, jitter_microns,
                size=(benchmark.num_hard_macros, 2),
            ).astype(np.float32)
            candidate = pos.clone()
            candidate[:benchmark.num_hard_macros] += torch.from_numpy(jitter)
            try:
                legal = legalize(candidate, benchmark, verbose=False)
                pc = compute_proxy_cost(
                    legal, benchmark, plc, with_comparison=False,
                )
                score = float(pc["fast_proxy_cost"])
                ovl = int(pc["overlap_metrics_fast"]["overlap_count"])
                if ovl == 0 and (best is None or score < best["score"]):
                    best = {"name": f"jitter_t{trial}", "score": score,
                            "pos": legal, "trial": trial,
                            "jitter": jitter_microns}
                    print(f"  jitter trial {trial} (±{jitter_microns}µm): "
                          f"legal_score={score:.4f}")
            except Exception as e:
                print(f"  jitter trial {trial}: failed ({e})")
        return best

    def _legalize_all_buckets_and_pick_best(self, buckets, benchmark, plc):
        """
        Try legalizing every non-None bucket; return the candidate with
        the best post-legalization exact proxy (and zero overlaps).

        Returns: dict with keys {name, score, pos, raw_score, overlaps_raw,
        degradation}, or None if no bucket produced a legal placement.

        Branch-and-bound optimization:
          We process buckets in ascending order of raw_score. Since
          legalization can only worsen the score (or leave it unchanged
          on already-legal input), any bucket whose raw_score >= the
          best legal score found so far can be skipped — its legal
          output cannot beat the current best.

          This dramatically cuts wasted work on high-overlap buckets
          whose raw_score is low only because they're cheating on
          legality.
        """
        from legalizer import legalize

        # Collect non-None candidates and sort by raw_score asc.
        candidates = []
        for k, b in buckets.items():
            if b["pos"] is not None:
                candidates.append((k, b["pos"], b["score"], b["overlaps_now"]))
        candidates.sort(key=lambda c: c[2])

        if not candidates:
            return None

        best = None

        # Pretty-print the result table as we go — invaluable for debugging
        # bucket selection and per-benchmark legalization degradation.
        print(f"[place] legalize-all-buckets ({len(candidates)} candidates):")
        print(f"  {'bucket':<10} {'raw_score':>10} {'raw_ovlp':>9} "
              f"{'legal_score':>12} {'legal_ovlp':>11} {'degrad':>9}")

        for k, pos, raw_score, raw_overlaps in candidates:
            # Branch-and-bound prune: if even the raw score already loses,
            # legalizing won't help (legal_score >= raw_score in general).
            if best is not None and raw_score >= best["score"]:
                print(f"  bucket_{k:<3} {raw_score:>10.4f} {raw_overlaps:>9} "
                      f"{'SKIPPED':>12} (raw >= best legal {best['score']:.4f})")
                continue

            try:
                legal = legalize(pos, benchmark, verbose=False)
                pc = compute_proxy_cost(
                    legal, benchmark, plc, with_comparison=False,
                )
                legal_score = float(pc["fast_proxy_cost"])
                legal_overlaps = int(pc["overlap_metrics_fast"]["overlap_count"])
            except Exception as e:
                # Legalizer can raise on hopeless input. Keep going — other
                # buckets may still succeed.
                print(f"  bucket_{k:<3} {raw_score:>10.4f} {raw_overlaps:>9} "
                      f"FAILED: {e}")
                continue

            degradation = legal_score - raw_score
            mark = ""
            if legal_overlaps == 0 and (best is None or legal_score < best["score"]):
                mark = " *"
                best = {
                    "name": f"bucket_{k}",
                    "score": legal_score,
                    "pos": legal,
                    "raw_score": raw_score,
                    "overlaps_raw": raw_overlaps,
                    "degradation": degradation,
                }

            print(f"  bucket_{k:<3} {raw_score:>10.4f} {raw_overlaps:>9} "
                  f"{legal_score:>12.4f} {legal_overlaps:>11} "
                  f"{degradation:>+9.4f}{mark}")

        return best
