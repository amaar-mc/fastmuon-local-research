from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.rglob("microbench_summary.json"):
        rows.extend(json.loads(path.read_text()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matrix microbenchmark outputs.")
    parser.add_argument("root", nargs="?", default="results")
    args = parser.parse_args()

    rows = load_rows(Path(args.root))
    if not rows:
        print("No microbench_summary.json files found.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["optimizer"]].append(row)

    print("| optimizer | runs | failures | median final loss | median sec | median rho |")
    print("|---|---:|---:|---:|---:|---:|")
    for optimizer, opt_rows in sorted(grouped.items()):
        failures = sum(1 for row in opt_rows if row.get("status") != "completed")
        losses = [float(row["final_loss"]) for row in opt_rows if row.get("final_loss") is not None]
        secs = [float(row["elapsed_sec"]) for row in opt_rows if row.get("elapsed_sec") is not None]
        rhos = [
            float(row.get("diagnostics", {}).get("rho"))
            for row in opt_rows
            if row.get("diagnostics", {}).get("rho") is not None
        ]
        loss = statistics.median(losses) if losses else None
        sec = statistics.median(secs) if secs else None
        rho = statistics.median(rhos) if rhos else None
        print(
            f"| {optimizer} | {len(opt_rows)} | {failures} | "
            f"{loss:.6g} | {sec:.6g} | {rho:.6g} |"
            if rho is not None
            else f"| {optimizer} | {len(opt_rows)} | {failures} | {loss:.6g} | {sec:.6g} | NA |"
        )


if __name__ == "__main__":
    main()

