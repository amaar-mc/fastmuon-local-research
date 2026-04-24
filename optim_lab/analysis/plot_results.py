from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def iter_metrics(root: Path):
    for path in root.rglob("metrics.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["_path"] = str(path)
                yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation curves and optimizer diagnostics.")
    parser.add_argument("root", nargs="?", default="results")
    parser.add_argument("--output", default="results/loss_curves.png")
    parser.add_argument("--diagnostic", default="opt_rho")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for row in iter_metrics(Path(args.root)):
        if row.get("val_loss") is not None:
            grouped[(row["optimizer"], row["seed"])].append(row)

    if not grouped:
        print("No validation metrics found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (optimizer, seed), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["step"])
        label = f"{optimizer}/s{seed}"
        axes[0].plot([r["step"] for r in rows], [r["val_loss"] for r in rows], label=label, alpha=0.85)
        diag_rows = [r for r in rows if args.diagnostic in r]
        if diag_rows:
            axes[1].plot([r["step"] for r in diag_rows], [r[args.diagnostic] for r in diag_rows], label=label, alpha=0.85)

    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_title(args.diagnostic)
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

