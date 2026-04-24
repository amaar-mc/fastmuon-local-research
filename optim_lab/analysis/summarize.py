from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_summaries(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("summary.json"):
        rows.append(json.loads(path.read_text()))
    return rows


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local optimizer runs.")
    parser.add_argument("root", nargs="?", default="results")
    args = parser.parse_args()

    rows = load_summaries(Path(args.root))
    if not rows:
        print("No summary.json files found.")
        return

    by_optimizer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_optimizer[row["optimizer"]].append(row)

    print("| optimizer | runs | failures | median final val | median target step | median sec |")
    print("|---|---:|---:|---:|---:|---:|")
    for optimizer, opt_rows in sorted(by_optimizer.items()):
        failures = sum(1 for row in opt_rows if row.get("status") != "completed")
        finals = [x for row in opt_rows if (x := maybe_float(row.get("final_val_loss"))) is not None]
        targets = [x for row in opt_rows if (x := maybe_float(row.get("target_step"))) is not None]
        secs = [x for row in opt_rows if (x := maybe_float(row.get("elapsed_sec"))) is not None]
        print(
            f"| {optimizer} | {len(opt_rows)} | {failures} | "
            f"{fmt(statistics.median(finals) if finals else None)} | "
            f"{fmt(statistics.median(targets) if targets else None)} | "
            f"{fmt(statistics.median(secs) if secs else None)} |"
        )


if __name__ == "__main__":
    main()

