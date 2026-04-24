from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from optim_lab.optimizers import create_optimizer


class MatrixProblem(torch.nn.Module):
    def __init__(self, dim: int, rank: int, condition: float, seed: int) -> None:
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        u, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
        v, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g))
        spectrum = torch.logspace(0, -math.log10(condition), dim)
        target = u[:, :rank] @ torch.diag(torch.linspace(1.0, 0.1, rank)) @ v[:, :rank].T
        self.register_buffer("left", u @ torch.diag(spectrum) @ u.T)
        self.register_buffer("right", v @ torch.diag(spectrum.flip(0)) @ v.T)
        self.register_buffer("target", target)
        self.weight = torch.nn.Parameter(0.02 * torch.randn(dim, dim, generator=g))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, noise_std: float = 0.0) -> torch.Tensor:
        residual = self.left @ (self.weight - self.target) @ self.right
        loss = 0.5 * residual.square().mean() + 1e-4 * self.bias.square().mean()
        if noise_std:
            loss = loss + 0.0 * torch.randn((), device=loss.device) * noise_std
        return loss


def run_case(config: dict[str, Any], optimizer_name: str, seed: int, device: torch.device) -> dict[str, Any]:
    problem_cfg = config["problem"]
    model = MatrixProblem(
        dim=int(problem_cfg.get("dim", 64)),
        rank=int(problem_cfg.get("rank", 8)),
        condition=float(problem_cfg.get("condition", 1000.0)),
        seed=seed,
    ).to(device)
    optimizer_cfg = deepcopy(config.get("optimizer", {}))
    optimizer_cfg.pop("name", None)
    optimizer = create_optimizer(optimizer_name, model.named_parameters(), optimizer_cfg)
    steps = int(config.get("steps", 300))
    noise_std = float(problem_cfg.get("grad_noise_std", 0.0))
    start = time.perf_counter()
    status = "completed"
    final_loss = None
    for step in range(1, steps + 1):
        loss = model()
        if not torch.isfinite(loss):
            status = "failed_nonfinite_loss"
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if noise_std:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.add_(torch.randn_like(p.grad) * noise_std)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    return {
        "optimizer": optimizer_name,
        "seed": seed,
        "status": status,
        "steps_completed": step,
        "elapsed_sec": time.perf_counter() - start,
        "final_loss": final_loss,
        "diagnostics": getattr(optimizer, "last_stats", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matrix optimizer microbenchmarks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    device_name = config.get("device", "cpu")
    if device_name == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_root) / config.get("experiment_name", "microbench") / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    optimizers = config.get("optimizers", ["adamw", "muon", "cimuon"])
    seeds = config.get("seeds", [0, 1, 2])
    rows = []
    for optimizer_name in optimizers:
        for seed in seeds:
            row = run_case(config, optimizer_name, int(seed), device)
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    (out_dir / "microbench_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Wrote {out_dir / 'microbench_summary.json'}")


if __name__ == "__main__":
    main()

