import numba
import numpy as np
import numba
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from macro_place.benchmark import Benchmark
from PlacementCost import PlacementCostAccelerated


import torch

"""
Benchmark loader - extracts data from PlacementCost into PyTorch tensors.

Leverages the existing MacroPlacement parser instead of reimplementing.
"""

import os
import torch
from typing import Optional, Tuple


def load_benchmark_accel(
    netlist_file: str, plc_file: Optional[str] = None, name: Optional[str] = None
) -> Tuple[Benchmark, PlacementCostAccelerated]:
    """
    Load benchmark from ICCAD04 format using PlacementCost parser.

    Args:
        netlist_file: Path to netlist.pb.txt
        plc_file: Optional path to initial.plc (if None, uses default placement)
        name: Optional benchmark name override (inferred from path if not given)

    Returns:
        Tuple of (Benchmark, PlacementCost) - Benchmark contains PyTorch tensors,
        PlacementCost object is needed for cost computation
    """
    # Initialize PlacementCost (parses netlist)
    plc = PlacementCostAccelerated(netlist_file)

    # Optionally restore placement from .plc file
    if plc_file:
        plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

    # Extract benchmark name from path if not provided.
    # IBM paths: .../ibm01/netlist.pb.txt  -> "ibm01"
    # NG45 paths: .../ariane133/netlist/output_CT_Grouping/netlist.pb.txt -> "ariane133"
    if name is None:
        name = os.path.basename(os.path.dirname(netlist_file))
        # NG45 designs have extra subdirectory levels; walk up to find the design name
        if name in ("output_CT_Grouping", "output_CodeElement"):
            name = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(netlist_file)))
            )

    # Extract canvas and grid info
    canvas_width, canvas_height = plc.get_canvas_width_height()
    grid_rows = plc.grid_row
    grid_cols = plc.grid_col
    hroutes_per_micron = plc.hroutes_per_micron
    vroutes_per_micron = plc.vroutes_per_micron

    # Extract hard macros
    hard_macro_plc_indices = plc.hard_macro_indices
    num_hard = len(hard_macro_plc_indices)

    macro_positions = []
    macro_sizes = []
    macro_fixed = []
    macro_names = []

    for idx in hard_macro_plc_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        w = node.get_width()
        h = node.get_height()
        fixed = node.get_fix_flag()
        macro_positions.append([x, y])
        macro_sizes.append([w, h])
        macro_fixed.append(fixed)
        macro_names.append(node.get_name())

    # Extract soft macros (standard cell clusters)
    soft_macro_plc_indices = plc.soft_macro_indices
    num_soft = len(soft_macro_plc_indices)

    for idx in soft_macro_plc_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        w = node.get_width()
        h = node.get_height()
        fixed = node.get_fix_flag()
        macro_positions.append([x, y])
        macro_sizes.append([w, h])
        macro_fixed.append(fixed)
        macro_names.append(node.get_name())

    num_macros = num_hard + num_soft

    # Extract hard macro pin offsets (relative to macro center)
    macro_pin_offsets = []
    pin_map = {}
    for idx in plc.hard_macro_pin_indices:
        pin = plc.modules_w_pins[idx]
        pin_macro = pin.get_macro_name() if hasattr(pin, "get_macro_name") else None
        if pin_macro:
            pin_map.setdefault(pin_macro, []).append(
                [pin.x_offset, pin.y_offset]
            )
    for macro_idx in hard_macro_plc_indices:
        macro_name = plc.modules_w_pins[macro_idx].get_name()
        offsets = pin_map.get(macro_name, [])
        macro_pin_offsets.append(
            torch.tensor(offsets, dtype=torch.float32) if offsets else torch.zeros(0, 2)
        )

    # Extract I/O port positions
    port_pos_list = []
    for idx in plc.port_indices:
        node = plc.modules_w_pins[idx]
        x, y = node.get_pos()
        port_pos_list.append([x, y])
    port_positions = (
        torch.tensor(port_pos_list, dtype=torch.float32)
        if port_pos_list
        else torch.zeros(0, 2)
    )

    # Convert to tensors
    macro_positions = torch.tensor(macro_positions, dtype=torch.float32)
    macro_sizes = torch.tensor(macro_sizes, dtype=torch.float32)
    macro_fixed = torch.tensor(macro_fixed, dtype=torch.bool)

    # Extract net connectivity
    # Build mapping from module/port names to benchmark tensor indices:
    #   hard macros -> [0, num_hard), soft macros -> [num_hard, num_hard+num_soft)
    #   ports -> num_macros + port_index
    plc_idx_to_bench = {}
    for bench_idx, plc_idx in enumerate(hard_macro_plc_indices):
        plc_idx_to_bench[plc_idx] = bench_idx
    for bench_idx_offset, plc_idx in enumerate(soft_macro_plc_indices):
        plc_idx_to_bench[plc_idx] = num_hard + bench_idx_offset
    for port_offset, plc_idx in enumerate(plc.port_indices):
        plc_idx_to_bench[plc_idx] = num_macros + port_offset

    # Map pin/module names to benchmark indices via their parent macro/port
    name_to_bench = {}
    for plc_idx, bench_idx in plc_idx_to_bench.items():
        mod = plc.modules_w_pins[plc_idx]
        name_to_bench[mod.get_name()] = bench_idx

    num_nets = int(plc.net_cnt)
    net_nodes = []
    net_weights_list = []
    for driver, sinks in plc.nets.items():
        nodes_in_net = set()
        for pin_name in [driver] + sinks:
            # Pin names are "MACRO/PIN" for macro pins or just "PORT" for ports
            parent = pin_name.split("/")[0]
            if parent in name_to_bench:
                nodes_in_net.add(name_to_bench[parent])
        if nodes_in_net:
            net_nodes.append(torch.tensor(sorted(nodes_in_net), dtype=torch.long))
            net_weights_list.append(1.0)

    num_nets = len(net_nodes)
    net_weights_tensor = torch.tensor(net_weights_list, dtype=torch.float32) if net_weights_list else torch.zeros(0, dtype=torch.float32)

    # Create Benchmark object
    benchmark = Benchmark(
        name=name,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        num_macros=num_macros,
        num_hard_macros=num_hard,
        num_soft_macros=num_soft,
        macro_positions=macro_positions,
        macro_sizes=macro_sizes,
        macro_fixed=macro_fixed,
        macro_names=macro_names,
        num_nets=num_nets,
        net_nodes=net_nodes,
        net_weights=net_weights_tensor,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        hroutes_per_micron=hroutes_per_micron,
        vroutes_per_micron=vroutes_per_micron,
        port_positions=port_positions,
        macro_pin_offsets=macro_pin_offsets,
        hard_macro_indices=hard_macro_plc_indices,
        soft_macro_indices=soft_macro_plc_indices,
    )

    return benchmark, plc

def load_benchmark_accel_from_dir(benchmark_dir: str) -> Tuple[Benchmark, PlacementCostAccelerated]:
    """
    Convenience wrapper to load from directory.

    Args:
        benchmark_dir: Path like "external/MacroPlacement/Testcases/ICCAD04/ibm01"

    Returns:
        Tuple of (Benchmark, PlacementCost)
    """
    netlist_file = os.path.join(benchmark_dir, "netlist.pb.txt")
    plc_file = os.path.join(benchmark_dir, "initial.plc")

    if not os.path.exists(netlist_file):
        raise FileNotFoundError(f"Netlist not found: {netlist_file}")

    if not os.path.exists(plc_file):
        print(f"Warning: No initial.plc found at {plc_file}, using default placement")
        plc_file = None

    return load_benchmark_accel(netlist_file, plc_file)

class BenchmarkNpy:
    """
    Placement benchmark in pure PyTorch tensors.

    All coordinates are in microns.
    All indices are 0-based.

    Tensors contain both hard macros (indices [0, num_hard_macros)) and
    soft macros (indices [num_hard_macros, num_macros)). Hard macros are
    the primary optimization targets; soft macros are standard cell clusters
    that should be co-optimized for best results.
    """
    
    # Core data
    name: str

    # Canvas
    canvas_width: float
    canvas_height: float

    # Macros (hard + soft, hard macros first)
    num_macros: int
    macro_positions: np.ndarray  # [num_macros, 2] - (x, y) centers
    macro_sizes: np.ndarray  # [num_macros, 2] - (width, height)
    macro_fixed: np.ndarray  # [num_macros] - bool, True if fixed
    macro_names: List[str]  # [num_macros] - names for debugging

    # Nets (hypergraph connectivity)
    num_nets: int
    net_nodes: List[np.ndarray]  # List of [nodes_in_net_i] - node indices
    net_weights: np.ndarray  # [num_nets] - net weights (default 1.0)

    # Grid (for metrics)
    grid_rows: int
    grid_cols: int

    # I/O ports (pins on the chip boundary)
    port_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # [num_ports, 2]

    # Hard macro pin offsets (relative to macro center)
    # List of [num_pins_i, 2] tensors, one per hard macro (indices [0, num_hard_macros))
    macro_pin_offsets: List[np.ndarray] = field(default_factory=list)

    # Routing parameters
    hroutes_per_micron: float = 11.285  # Horizontal routing tracks per micron
    vroutes_per_micron: float = 12.605  # Vertical routing tracks per micron

    # PlacementCost mapping (tensor index → PlacementCost module index)
    hard_macro_indices: List[int] = field(default_factory=list)
    soft_macro_indices: List[int] = field(default_factory=list)

    # Counts
    num_hard_macros: int = 0
    num_soft_macros: int = 0

    def __init__(self, regular_benchmark : Benchmark) -> None:
        self.name = regular_benchmark.name
        self.canvas_width = regular_benchmark.canvas_width
        self.canvas_height = regular_benchmark.canvas_height
        self.num_macros = regular_benchmark.num_macros
        self.macro_positions = regular_benchmark.macro_positions.cpu().numpy()
        self.macro_sizes = regular_benchmark.macro_sizes.cpu().numpy()
        self.macro_fixed = regular_benchmark.macro_fixed.cpu().numpy()
        self.macro_names = regular_benchmark.macro_names
        self.num_nets = regular_benchmark.num_nets
        self.net_nodes = [net.cpu().numpy() for net in regular_benchmark.net_nodes]
        self.net_weights = regular_benchmark.net_weights.cpu().numpy()
        self.grid_rows = regular_benchmark.grid_rows
        self.grid_cols = regular_benchmark.grid_cols
        self.port_positions = regular_benchmark.port_positions.cpu().numpy()
        self.macro_pin_offsets = [offsets.cpu().numpy() for offsets in regular_benchmark.macro_pin_offsets]
        self.hroutes_per_micron = regular_benchmark.hroutes_per_micron
        self.vroutes_per_micron = regular_benchmark.vroutes_per_micron
        self.hard_macro_indices = regular_benchmark.hard_macro_indices
        self.soft_macro_indices = regular_benchmark.soft_macro_indices
        self.num_hard_macros = regular_benchmark.num_hard_macros
        self.num_soft_macros = regular_benchmark.num_soft_macros


def validate_placement(
    placement: torch.Tensor, benchmark: Benchmark, check_overlaps: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate placement legality.

    Checks:
    - All macros within canvas bounds
    - No NaN/Inf values
    - Correct shape
    - Fixed macros at original positions
    - No macro overlaps (optional, can be slow for large designs)
    """
    violations = []

    if placement.shape != (benchmark.num_macros, 2):
        violations.append(
            f"Shape mismatch: expected {(benchmark.num_macros, 2)}, got {placement.shape}"
        )
        return False, violations

    if torch.isnan(placement).any():
        violations.append("Placement contains NaN values")
    if torch.isinf(placement).any():
        violations.append("Placement contains Inf values")

    x_coords = placement[:, 0]
    y_coords = placement[:, 1]
    widths = benchmark.macro_sizes[:, 0]
    heights = benchmark.macro_sizes[:, 1]

    x_min = x_coords - widths / 2
    x_max = x_coords + widths / 2
    y_min = y_coords - heights / 2
    y_max = y_coords + heights / 2

    if (x_min < 0).any() or (x_max > benchmark.canvas_width).any():
        violations.append("Macros outside horizontal canvas bounds")
    if (y_min < 0).any() or (y_max > benchmark.canvas_height).any():
        violations.append("Macros outside vertical canvas bounds")

    fixed_mask = benchmark.macro_fixed
    if fixed_mask.any():
        original_pos = benchmark.macro_positions[fixed_mask]
        new_pos = placement[fixed_mask]
        if not torch.allclose(original_pos, new_pos, atol=1e-3):
            violations.append("Fixed macros have been moved")

    if check_overlaps:
        overlap_count = 0
        num_hard = benchmark.num_hard_macros
        for i in range(num_hard):
            for j in range(i + 1, num_hard):
                lx_i, ux_i = x_min[i].item(), x_max[i].item()
                ly_i, uy_i = y_min[i].item(), y_max[i].item()
                lx_j, ux_j = x_min[j].item(), x_max[j].item()
                ly_j, uy_j = y_min[j].item(), y_max[j].item()

                if not (lx_i >= ux_j or ux_i <= lx_j or ly_i >= uy_j or uy_i <= ly_j):
                    overlap_count += 1
                    if overlap_count <= 5:
                        violations.append(f"Macros {i} and {j} overlap")

        if overlap_count > 5:
            violations.append(f"... and {overlap_count - 5} more overlaps")

    return len(violations) == 0, violations


def compute_overlap_metrics(
    placement: torch.Tensor, benchmark: Benchmark
) -> Dict[str, float]:
    """
    Compute overlap metrics for macro placement.

    Borrowed from intern_challenge placement.py and adapted for macro placement.

    Args:
        placement: [num_macros, 2] tensor of (x, y) center positions
        benchmark: Benchmark object with macro sizes

    Returns:
        Dictionary with:
            - overlap_count: Number of overlapping macro pairs
            - total_overlap_area: Total area of all overlaps (μm²)
            - max_overlap_area: Largest single overlap area (μm²)
            - num_macros_with_overlaps: Number of macros involved in at least one overlap
            - overlap_ratio: Fraction of macros with overlaps (0.0 = no overlaps, 1.0 = all overlap)
    """
    num_macros = placement.shape[0]

    if num_macros <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "num_macros_with_overlaps": 0,
            "overlap_ratio": 0.0,
        }

    # Extract positions and sizes
    positions = placement.cpu().detach().numpy()  # [N, 2]
    widths = benchmark.macro_sizes[:, 0].cpu().numpy()  # [N]
    heights = benchmark.macro_sizes[:, 1].cpu().numpy()  # [N]

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    macros_with_overlaps = set()

    # Check hard macro pairs only for overlap (soft macros naturally overlap)
    num_hard = getattr(benchmark, 'num_hard_macros', num_macros)
    for i in range(num_hard):
        for j in range(i + 1, num_hard):
            # Calculate center-to-center distances
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for non-overlap (sum of half-widths/heights)
            min_sep_x = (widths[i] + widths[j]) / 2.0
            min_sep_y = (heights[i] + heights[j]) / 2.0

            # Calculate overlap amounts in each dimension
            overlap_x = max(0.0, min_sep_x - dx)
            overlap_y = max(0.0, min_sep_y - dy)

            # Overlap occurs only if BOTH x and y overlap
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
                macros_with_overlaps.add(i)
                macros_with_overlaps.add(j)

    num_macros_with_overlaps = len(macros_with_overlaps)
    overlap_ratio = num_macros_with_overlaps / num_macros if num_macros > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "num_macros_with_overlaps": num_macros_with_overlaps,
        "overlap_ratio": overlap_ratio,
    }

def compute_overlap_metrics_fast(
    placement: torch.Tensor, benchmark: Benchmark
) -> Dict[str, float]:
    """
    Compute overlap metrics for macro placement.

    Borrowed from intern_challenge placement.py and adapted for macro placement.

    Args:
        placement: [num_macros, 2] tensor of (x, y) center positions
        benchmark: Benchmark object with macro sizes

    Returns:
        Dictionary with:
            - overlap_count: Number of overlapping macro pairs
            - total_overlap_area: Total area of all overlaps (μm²)
            - max_overlap_area: Largest single overlap area (μm²)
            - num_macros_with_overlaps: Number of macros involved in at least one overlap
            - overlap_ratio: Fraction of macros with overlaps (0.0 = no overlaps, 1.0 = all overlap)
    """
    num_macros = placement.shape[0]

    if num_macros <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "num_macros_with_overlaps": 0,
            "overlap_ratio": 0.0,
        }

    # Extract positions and sizes
    positions = placement.cpu().detach().numpy()  # [N, 2]
    widths = benchmark.macro_sizes[:, 0].cpu().numpy()  # [N]
    heights = benchmark.macro_sizes[:, 1].cpu().numpy()  # [N]

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    macros_with_overlaps = set()

    num_hard = benchmark.num_hard_macros
    i, j = np.triu_indices(num_hard, k=1)  # upper-triangle pair indices

    dx = np.abs(positions[i, 0] - positions[j, 0])
    dy = np.abs(positions[i, 1] - positions[j, 1])

    # minimum separation (sum of half-widths/heights)
    min_sep_x = (widths[i] + widths[j]) * 0.5
    min_sep_y = (heights[i] + heights[j]) * 0.5

    # intrusion per axis, clipped at 0 (non-overlapping pairs have ox=0 or oy=0)
    overlap_x = np.maximum(0.0, min_sep_x - dx)
    overlap_y = np.maximum(0.0, min_sep_y - dy)

    # Because both ox and oy are already non-negative, pair_area is 0 exactly
    # where the pair doesn't overlap — no np.where needed.
    pair_area = overlap_x * overlap_y
    overlap_mask = pair_area > 0.0

    overlap_count = int(overlap_mask.sum())
    total_overlap_area = float(pair_area.sum())
    max_overlap_area = float(pair_area.max(initial=0.0))

    # "involved" detection: mark each macro in the pair, no sort needed
    involved = np.zeros(num_hard, dtype=bool)
    involved[i[overlap_mask]] = True
    involved[j[overlap_mask]] = True
    num_macros_with_overlaps = int(involved.sum())
    overlap_ratio = num_macros_with_overlaps / num_macros if num_macros > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "num_macros_with_overlaps": num_macros_with_overlaps,
        "overlap_ratio": float(overlap_ratio),
    }

def __compute_overlap_metrics_fast(
    placement: torch.Tensor, benchmark: Benchmark
) -> Dict[str, float]:
    """
    Compute overlap metrics for hard macros (pairwise AABB intersections).

    Vectorized: builds an N×N upper-triangular matrix of pairwise overlap
    areas and reduces. For ibm01 (N≈246) this is <1 ms; scales as O(N²)
    memory but cache-friendly.
    """
    num_macros = placement.shape[0]
    num_hard = getattr(benchmark, "num_hard_macros", num_macros)

    if num_hard <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "num_macros_with_overlaps": 0,
            "overlap_ratio": 0.0,
        }

    positions = placement.detach().cpu().numpy()
    sizes = benchmark.macro_sizes.detach().cpu().numpy()

    x = positions[:num_hard, 0]
    y = positions[:num_hard, 1]
    w = sizes[:num_hard, 0]
    h = sizes[:num_hard, 1]

    # Pairwise distances and minimum non-overlap separations.
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    min_sep_x = (w[:, None] + w[None, :]) * 0.5
    min_sep_y = (h[:, None] + h[None, :]) * 0.5

    # Intrusion in each axis, clipped at 0.
    ox = np.maximum(0.0, min_sep_x - dx)
    oy = np.maximum(0.0, min_sep_y - dy)

    area = ox * oy                      # [N, N]; diag is macro's own area
    np.fill_diagonal(area, 0.0)

    # Upper triangle: each pair counted once.
    iu = np.triu_indices(num_hard, k=1)
    pair_area = area[iu]                # [N*(N-1)/2]

    has_overlap = pair_area > 0.0
    overlap_count = int(has_overlap.sum())
    total_overlap_area = float(pair_area.sum())
    max_overlap_area = float(pair_area.max(initial=0.0))

    # A macro is "involved" if any pair containing it has nonzero overlap.
    # Use the full symmetric matrix (row sum) to detect involvement.
    macros_involved = np.count_nonzero(area.sum(axis=1) > 0.0)
    overlap_ratio = macros_involved / num_macros if num_macros > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "num_macros_with_overlaps": int(macros_involved),
        "overlap_ratio": float(overlap_ratio),
    }

def _ensure_congestion_arrays(plc: PlacementCostAccelerated):
    """
    Ensure congestion arrays are properly sized for current grid.

    Args:
        plc: PlacementCost object
    """
    expected_size = plc.grid_col * plc.grid_row
    current_size = len(plc.H_routing_cong)

    if current_size != expected_size:
        # Reinitialize with correct size
        plc.V_routing_cong = [0] * expected_size
        plc.H_routing_cong = [0] * expected_size
        plc.V_macro_routing_cong = [0] * expected_size
        plc.H_macro_routing_cong = [0] * expected_size

def _set_placement_fast(plc: PlacementCostAccelerated, placement: torch.Tensor, benchmark: Benchmark):
    """
    Fast-path placement setter.

    Updates:
      * plc.x_pos / plc.y_pos for macros AND pins (flat numpy arrays read by
        the numba kernels — HPWL, congestion).
      * plc.modules_w_pins[macro_idx].set_pos(...) for HARD + SOFT macros
        (needed by plc.get_density_cost(), which iterates module objects).

    Does NOT update MACRO_PIN module objects (pins) — those are only read by
    the slow reference wirelength / congestion paths, which the SA loop never
    calls. If you need those, use _set_placement instead.

    ~200-500 µs per call for ibm01; ~1-2 ms for ibm18.

    Precondition: build_fast_representation has populated hmacro_indices_np,
    smacro_indices_np, hard_pin_indices, hard_pin_macro_slot,
    hard_pin_offset_x/y, soft_pin_indices, soft_pin_macro_slot.
    """
    placement_np = placement.detach().cpu().numpy()
    num_hard = benchmark.num_hard_macros
    hard_xy = placement_np[:num_hard]     # [num_hard, 2]
    soft_xy = placement_np[num_hard:]     # [num_soft, 2]

    # ---- flat numpy arrays: macro centers ----
    plc.x_pos[plc.hmacro_indices_np] = hard_xy[:, 0]
    plc.y_pos[plc.hmacro_indices_np] = hard_xy[:, 1]
    plc.x_pos[plc.smacro_indices_np] = soft_xy[:, 0]
    plc.y_pos[plc.smacro_indices_np] = soft_xy[:, 1]

    # ---- flat numpy arrays: hard macro pins (pin_pos = center + offset) ----
    plc.x_pos[plc.hard_pin_indices] = (
        hard_xy[plc.hard_pin_macro_slot, 0] + plc.hard_pin_offset_x
    )
    plc.y_pos[plc.hard_pin_indices] = (
        hard_xy[plc.hard_pin_macro_slot, 1] + plc.hard_pin_offset_y
    )

    # ---- flat numpy arrays: soft macro pins (offsets are 0) ----
    plc.x_pos[plc.soft_pin_indices] = soft_xy[plc.soft_pin_macro_slot, 0]
    plc.y_pos[plc.soft_pin_indices] = soft_xy[plc.soft_pin_macro_slot, 1]

    # ---- module objects: macro centers only ----
    # plc.get_density_cost() reads mod.get_pos() for every macro module.
    # Without this sync, density sees stale positions during SA and
    # appears constant — meaning SA never actually optimizes for density.
    modules = plc.modules_w_pins
    for i, macro_idx in enumerate(benchmark.hard_macro_indices):
        x, y = hard_xy[i]
        modules[macro_idx].set_pos(float(x), float(y))
    for i, macro_idx in enumerate(benchmark.soft_macro_indices):
        x, y = soft_xy[i]
        modules[macro_idx].set_pos(float(x), float(y))

    # ---- invalidate caches ----
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True

def _set_placement_fast_moved(plc, idxs, new_positions):
    
    for pos, s in zip(new_positions, idxs):
        x, y = pos[0], pos[1]
        
        mi = plc.hmacro_indices_np[s] 
        plc.x_pos[mi] = x 
        plc.y_pos[mi] = y
        
        a, b = plc.hard_slot_pin_offsets[s], plc.hard_slot_pin_offsets[s + 1]
        pins = plc.hard_slot_pin_indices[a:b]
        
        plc.x_pos[pins] = x + plc.hard_slot_pin_offset_x[a:b]
        plc.y_pos[pins] = y + plc.hard_slot_pin_offset_y[a:b]

    # ---- invalidate caches ----
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True
    
def compute_proxy_cost_incremental(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc: PlacementCostAccelerated,
    # Accepts either a list of ints (typical SA) or an int32 ndarray. Both
    # are coerced to ndarray by plc._as_int32_array inside the kernels.
    moved_slots=None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Incremental-HPWL proxy cost.

    Contract:
      * Caller must have ALREADY updated plc.x_pos / plc.y_pos via
        _set_placement_fast BEFORE calling this.
      * `moved_slots` is an int32 array of HARD-MACRO slot indices in
        [0, num_hard_macros). On the first call of a run it may be None or
        any value — the HPWL cache is built from scratch.
      * For subsequent calls, pass the slots of the macros that moved since
        the last call (1 for displace/teleport, 2 for swap).

    This version: HPWL is incremental, congestion + density use the existing
    fast full-compute paths. Overlap metrics use the vectorized version.
    """
    if weights is None:
        weights = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}

    # --- wirelength (incremental) ---
    if moved_slots is None or not plc._hpwl_incremental_ready:
        plc.init_hpwl_incremental()
        raw_hpwl = plc.total_hpwl
    else:
        raw_hpwl = plc.get_wirelength_incremental(moved_slots)

    denom = (plc.width + plc.height) * max(plc.net_cnt, 1)
    wl = raw_hpwl / denom

    # --- congestion (incremental) ---
    if moved_slots is None or not plc._congestion_incremental_ready:
        plc.init_congestion_incremental()
        cong = plc._finalize_abu()
    else:
        cong = plc.get_congestion_cost_incremental(moved_slots)

    # --- density (incremental) ---
    if moved_slots is None or not plc._density_incremental_ready:
        plc.init_density_incremental()
        den = plc._density_abu()
    else:
        den = plc.get_density_cost_incremental(moved_slots)

    proxy = weights["wirelength"] * wl + weights["density"] * den + weights["congestion"] * cong

    overlap_metrics = compute_overlap_metrics_fast(placement, benchmark)

    return {
        "proxy_cost":       float(proxy),
        "wirelength_cost":  float(wl),
        "density_cost":     float(den),
        "congestion_cost":  float(cong),
        "overlap_metrics":  overlap_metrics,
    }


def _set_placement(plc: PlacementCostAccelerated, placement: torch.Tensor, benchmark: Benchmark):
    """
    Set macro positions in PlacementCost object.

    Args:
        plc: PlacementCost object
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object with macro indices mapping
    """
    # Convert tensor to numpy for PlacementCost API
    placement_np = placement.cpu().numpy()

    # Build macro_name -> [pin_indices] lookup (cached on plc)
    if not hasattr(plc, '_macro_pin_map'):
        pin_map = {}
        for idx, mod in enumerate(plc.modules_w_pins):
            if mod.get_type() == 'MACRO_PIN' and hasattr(mod, 'get_macro_name'):
                name = mod.get_macro_name()
                if name not in pin_map:
                    pin_map[name] = []
                pin_map[name].append(idx)
        plc._macro_pin_map = pin_map

    # Set hard macro positions (indices [0, num_hard))
    for i, macro_idx in enumerate(benchmark.hard_macro_indices):
        x, y = placement_np[i]
        node = plc.modules_w_pins[macro_idx]
        node.set_pos(x, y)
        # Keep plc.x_pos / y_pos in sync at the macro's own index so kernels
        # reading positions for macro blockage (accumulate_macro_blockage)
        # see the updated placement.
        plc.x_pos[macro_idx] = x
        plc.y_pos[macro_idx] = y
        # Update pin positions (pin.get_pos() caches stale coordinates)
        for pin_idx in plc._macro_pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pin_idx]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)

            plc.x_pos[pin_idx] = x + pin.x_offset
            plc.y_pos[pin_idx] = y + pin.y_offset


    # Set soft macro positions (indices [num_hard, num_macros))
    num_hard = benchmark.num_hard_macros
    for i, macro_idx in enumerate(benchmark.soft_macro_indices):
        x, y = placement_np[num_hard + i]
        node = plc.modules_w_pins[macro_idx]
        node.set_pos(x, y)
        # Mirror the hard-macro update for consistency (soft macros aren't
        # routed through accumulate_macro_blockage today, but having the
        # array stay in-sync with set_pos() keeps everything coherent).
        plc.x_pos[macro_idx] = x
        plc.y_pos[macro_idx] = y
        for pin_idx in plc._macro_pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pin_idx]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)

            plc.x_pos[pin_idx] = x + pin.x_offset
            plc.y_pos[pin_idx] = y + pin.y_offset

    # Reinitialize congestion arrays with correct size
    # This is needed because the arrays may be incorrectly sized
    _ensure_congestion_arrays(plc)

    # Mark that costs need to be recomputed
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True

def compute_proxy_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc: PlacementCostAccelerated,
    weights: Optional[Dict[str, float]] = None,
    with_comparison = True
) -> Dict[str, float]:
    """
    Compute proxy cost using PlacementCost's ground truth evaluator.

    Args:
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object with circuit data
        plc: PlacementCost object (contains all netlist/placement data)
        weights: Optional cost weights {
            'wirelength': 1.0,
            'density': 0.5,
            'congestion': 0.5
        }

    Returns:
        {
            'proxy_cost': float,
            'wirelength_cost': float,
            'density_cost': float,
            'congestion_cost': float,
            'overlap_count': int,
            'total_overlap_area': float,
            'max_overlap_area': float,
            'num_macros_with_overlaps': int,
            'overlap_ratio': float,
        }
    """
    if weights is None:
        weights = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}

    # Set placement in PlacementCost object (if different from current)
    
    if with_comparison:
        _set_placement(plc, placement, benchmark)
    else:
        _set_placement_fast(plc=plc, placement=placement, benchmark=benchmark)

    slow_proxy = None
    slow_wirelength_cost = None
    slow_density_cost = None
    slow_congestion_cost = None


    fast_wirelength_cost = plc.get_cost_fast()
    fast_density_cost = plc.get_density_cost()
    fast_congestion_cost = plc.get_congestion_cost_fast()

    if with_comparison:
        # compute the slow and original option
        # Compute costs using PlacementCost methods
        slow_wirelength_cost = plc.get_cost()
        slow_density_cost = plc.get_density_cost()
        slow_congestion_cost = plc.get_congestion_cost()  # Fixed with monkey-patch above

        # Weighted sum (matching ISPD 2023 paper convention)
        slow_proxy = (
            weights["wirelength"] * slow_wirelength_cost
            + weights["density"] * slow_density_cost
            + weights["congestion"] * slow_congestion_cost
        )
        

    fast_proxy = (
        weights["wirelength"] * fast_wirelength_cost
        + weights["density"] * fast_density_cost
        + weights["congestion"] * fast_congestion_cost
    )


    # Compute overlap metrics
    
    overlap_metrics = None
    if with_comparison:
        overlap_metrics = compute_overlap_metrics(placement, benchmark)
    
    
    overlap_metrics_fast = compute_overlap_metrics_fast(placement, benchmark)

    return {
        "slow_proxy_cost": slow_proxy,
        "slow_wirelength_cost": slow_wirelength_cost,
        "slow_density_cost": slow_density_cost,
        "slow_congestion_cost": slow_congestion_cost,
        "fast_proxy_cost": fast_proxy,
        "fast_wirelength_cost": fast_wirelength_cost,
        "fast_density_cost": fast_density_cost,
        "fast_congestion_cost": fast_congestion_cost,
        "overlap_metrics" : overlap_metrics,  # Add all overlap metrics
        "overlap_metrics_fast" : overlap_metrics_fast
    }

