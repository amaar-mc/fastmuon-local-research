from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from optim_lab.experiments.train_lm import deep_update, load_json, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequential local optimizer sweep.")
    parser.add_argument("--config", required=True, help="Path to JSON sweep config.")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    base_config = load_json(args.config)
    sweep = base_config.get("sweep", {})
    seeds = sweep.get("seeds", [base_config.get("seed", 0)])
    optimizers = sweep.get("optimizers", [{"name": base_config["optimizer"]["name"]}])
    summaries = []

    for opt_entry in optimizers:
        opt_name = opt_entry["name"]
        opt_overrides = deepcopy(opt_entry)
        opt_overrides.pop("name", None)
        for seed in seeds:
            config = deep_update(base_config, {"optimizer": {"name": opt_name, **opt_overrides}})
            label_parts = [opt_name, f"seed{seed}"]
            for key in ("fixed_rho", "pulse_interval", "pulse_rho_threshold", "ns_steps", "matrix_lr", "lr"):
                if key in opt_overrides:
                    value = str(opt_overrides[key]).replace(".", "p")
                    label_parts.append(f"{key}{value}")
            summary = run_training(
                config,
                optimizer_name=opt_name,
                seed=int(seed),
                output_root=args.output_root,
                run_label="_".join(label_parts),
            )
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True))

    output_root = Path(args.output_root or base_config.get("output_root", "results"))
    summary_path = output_root / base_config.get("experiment_name", "local_optimizer_research") / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
