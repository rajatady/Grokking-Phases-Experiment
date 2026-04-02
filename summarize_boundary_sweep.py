#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_runs(paths):
    runs = []
    for path in paths:
        with open(path) as f:
            runs.extend(json.load(f))
    return runs


def aggregate(runs):
    grouped = defaultdict(list)
    for run in runs:
        key = (run["checkpoint"], run["intervention"])
        grouped[key].append(run)

    rows = []
    for (checkpoint, intervention), group in sorted(grouped.items()):
        finals = [run["final_test_acc"] for run in group]
        maxima = [run["max_test_acc"] for run in group]
        rows.append({
            "checkpoint": checkpoint,
            "intervention": intervention,
            "num_seeds": len(group),
            "seed_list": ",".join(str(run.get("seed", "")) for run in group),
            "mean_final_test_acc": mean(finals),
            "mean_max_test_acc": mean(maxima),
            "min_final_test_acc": min(finals),
            "max_final_test_acc": max(finals),
        })
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    lines = [
        "| Checkpoint | Intervention | Seeds | Mean final | Mean max | Min final | Max final | Seed list |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['intervention']} | {row['num_seeds']} | "
            f"{row['mean_final_test_acc']:.3f} | {row['mean_max_test_acc']:.3f} | "
            f"{row['min_final_test_acc']:.3f} | {row['max_final_test_acc']:.3f} | "
            f"{row['seed_list']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed grokking ablation results.")
    parser.add_argument("inputs", nargs="+", help="Paths to per-seed ablation JSON files")
    parser.add_argument("--csv-output", default="boundary_sweep_summary.csv")
    parser.add_argument("--md-output", default="boundary_sweep_summary.md")
    args = parser.parse_args()

    runs = load_runs(args.inputs)
    rows = aggregate(runs)
    write_csv(rows, args.csv_output)
    write_markdown(rows, Path(args.md_output))
    print(f"Saved {args.csv_output} and {args.md_output}")


if __name__ == "__main__":
    main()
