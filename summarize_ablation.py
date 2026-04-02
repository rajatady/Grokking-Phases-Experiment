#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def load_results(path):
    with open(path) as f:
        return json.load(f)


def summarize_run(run):
    trajectory = run["trajectory"]
    values = [point["test_acc"] for point in trajectory]
    steps = [point["step"] for point in trajectory]
    last_delta = values[-1] - values[-2] if len(values) >= 2 else 0.0
    prev_delta = values[-2] - values[-3] if len(values) >= 3 else 0.0
    grokked = max(values) >= 0.95

    return {
        "intervention": run["intervention"],
        "checkpoint": run["checkpoint"],
        "last_step": steps[-1],
        "initial_test_acc": values[0],
        "final_test_acc": values[-1],
        "max_test_acc": max(values),
        "last_delta": last_delta,
        "prev_delta": prev_delta,
        "tail_values": ",".join(f"{v:.3f}" for v in values[-3:]),
        "grokked_95": grokked,
    }


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    lines = [
        "| Intervention | Checkpoint | Last step | Initial | Final | Max | Last delta | Prev delta | Tail | Grokked >=95% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['intervention']} | {row['checkpoint']} | {row['last_step']} | "
            f"{row['initial_test_acc']:.3f} | {row['final_test_acc']:.3f} | {row['max_test_acc']:.3f} | "
            f"{row['last_delta']:.3f} | {row['prev_delta']:.3f} | {row['tail_values']} | "
            f"{'yes' if row['grokked_95'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create shareable summaries from grokking ablation results.")
    parser.add_argument("input", help="Path to ablation JSON results.")
    parser.add_argument("--csv-output", default="ablation_summary.csv")
    parser.add_argument("--md-output", default="ablation_summary.md")
    args = parser.parse_args()

    rows = [summarize_run(run) for run in load_results(args.input)]
    rows.sort(key=lambda row: (row["checkpoint"], row["intervention"]))

    write_csv(rows, args.csv_output)
    write_markdown(rows, Path(args.md_output))

    print(f"Saved {args.csv_output} and {args.md_output}")


if __name__ == "__main__":
    main()
