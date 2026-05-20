import pickle
from pathlib import Path
import torch

from macro_place.benchmark import Benchmark

class TrajectoryLogger:
    """Collect per-iteration snapshots of an SA run for offline visualization.

    Static data (sizes, canvas, fixed mask) is captured once. Per-snapshot data
    (positions, energy, cost, violations, temperature, accepted flag) is
    appended every `save_every` iterations.
    """

    def __init__(self, benchmark: Benchmark, save_every: int, output_path: Path):
        self.save_every = max(1, int(save_every))
        self.output_path = Path(output_path)
        self.static = {
            "name": benchmark.name,
            "canvas_width": float(benchmark.canvas_width),
            "canvas_height": float(benchmark.canvas_height),
            "macro_sizes": benchmark.macro_sizes.detach().cpu().numpy().copy(),
            "macro_fixed": benchmark.macro_fixed.detach().cpu().numpy().copy(),
            "num_hard_macros": int(benchmark.num_hard_macros),
            "num_macros": int(benchmark.num_macros),
            "macro_names": list(benchmark.macro_names),
        }
        self.snapshots: list[dict] = []

    def log(
        self,
        iteration: int,
        placement: torch.Tensor,
        energy: float,
        proxy_cost: float,
        violations: int,
        temperature: float,
        accepted: bool,
        delta_energy: float,
        moved_idx: int,
    ) -> None:
        if iteration % self.save_every != 0:
            return
        self.snapshots.append(
            {
                "iteration": int(iteration),
                "positions": placement.detach().cpu().numpy().copy(),
                "energy": float(energy),
                "proxy_cost": float(proxy_cost),
                "violations": int(violations),
                "temperature": float(temperature),
                "accepted": bool(accepted),
                "delta_energy": float(delta_energy),
                "moved_idx": int(moved_idx),
            }
        )

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump({"static": self.static, "snapshots": self.snapshots}, f)
        print(f"[trajectory] wrote {len(self.snapshots)} frames -> {self.output_path}")

