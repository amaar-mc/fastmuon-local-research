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

from optim_lab.data import TokenStream
from optim_lab.models import GPTConfig, TinyGPT
from optim_lab.optimizers import create_optimizer


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(
    model: TinyGPT,
    stream: TokenStream,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
    generator: torch.Generator,
) -> dict[str, float]:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = stream.get_batch("val", batch_size, device, generator)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("Model returned no loss during eval")
        losses.append(float(loss.detach().cpu()))
    model.train()
    mean = sum(losses) / len(losses)
    var = sum((x - mean) ** 2 for x in losses) / max(len(losses), 1)
    return {"val_loss": mean, "val_loss_std": math.sqrt(var)}


def build_model(config: dict[str, Any]) -> TinyGPT:
    model_config = GPTConfig(**config["model"])
    return TinyGPT(model_config)


def run_training(
    config: dict[str, Any],
    *,
    optimizer_name: str | None = None,
    seed: int | None = None,
    output_root: str | Path | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    config = deepcopy(config)
    seed = int(config.get("seed", 0) if seed is None else seed)
    optimizer_name = optimizer_name or config["optimizer"]["name"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    optimizer_cfg = deepcopy(config.get("optimizer", {}))
    optimizer_cfg.pop("name", None)

    torch.manual_seed(seed)
    torch.set_float32_matmul_precision(config.get("matmul_precision", "high"))
    device = select_device(config.get("device", "auto"))
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(seed + 10_000)
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(seed + 20_000)

    stream = TokenStream.from_config(data_cfg, seed=seed)
    model = build_model(config).to(device)
    optimizer = create_optimizer(optimizer_name, model.named_parameters(), optimizer_cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.get("experiment_name", "local_optimizer_research")
    root = Path(output_root or config.get("output_root", "results"))
    label = run_label or f"{optimizer_name}_seed{seed}_{timestamp}"
    run_dir = root / experiment_name / label
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    max_steps = int(train_cfg.get("max_steps", 1000))
    batch_size = int(train_cfg.get("batch_size", 32))
    eval_interval = int(train_cfg.get("eval_interval", 100))
    eval_iters = int(train_cfg.get("eval_iters", 10))
    log_interval = int(train_cfg.get("log_interval", 10))
    grad_clip = train_cfg.get("grad_clip")
    target_val_loss = train_cfg.get("target_val_loss")

    start = time.perf_counter()
    last_time = start
    status = "completed"
    target_step = None
    final_val_loss = None
    step = 0

    with metrics_path.open("w") as metrics_file:
        initial_eval = evaluate(model, stream, batch_size, eval_iters, device, eval_generator)
        initial_record = {
            "step": 0,
            "elapsed_sec": 0.0,
            "optimizer": optimizer_name,
            "seed": seed,
            "train_loss": None,
            **initial_eval,
        }
        metrics_file.write(json.dumps(initial_record) + "\n")
        final_val_loss = initial_eval["val_loss"]

        for step in range(1, max_steps + 1):
            xb, yb = stream.get_batch("train", batch_size, device, batch_generator)
            _, loss = model(xb, yb)
            if loss is None or not torch.isfinite(loss):
                status = "failed_nonfinite_loss"
                break

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            if step % log_interval == 0 or step == 1 or step % eval_interval == 0:
                synchronize(device)
                now = time.perf_counter()
                record: dict[str, Any] = {
                    "step": step,
                    "elapsed_sec": now - start,
                    "step_sec": (now - last_time) / max(log_interval, 1),
                    "optimizer": optimizer_name,
                    "seed": seed,
                    "train_loss": float(loss.detach().cpu()),
                }
                last_time = now
                if hasattr(optimizer, "last_stats"):
                    record.update({f"opt_{k}": v for k, v in optimizer.last_stats.items()})
                if step % eval_interval == 0 or step == max_steps:
                    record.update(evaluate(model, stream, batch_size, eval_iters, device, eval_generator))
                    final_val_loss = record["val_loss"]
                    if target_val_loss is not None and target_step is None and final_val_loss <= target_val_loss:
                        target_step = step
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()

    elapsed = time.perf_counter() - start
    summary = {
        "experiment_name": experiment_name,
        "run_dir": str(run_dir),
        "optimizer": optimizer_name,
        "seed": seed,
        "status": status,
        "steps_completed": step,
        "elapsed_sec": elapsed,
        "tokens_seen": step * batch_size * int(data_cfg["block_size"]),
        "final_val_loss": final_val_loss,
        "target_val_loss": target_val_loss,
        "target_step": target_step,
        "device": str(device),
        "num_parameters": model.num_parameters(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyGPT with a selected optimizer.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--optimizer", default=None, help="Override optimizer name.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    args = parser.parse_args()

    summary = run_training(
        load_json(args.config),
        optimizer_name=args.optimizer,
        seed=args.seed,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

