"""
plot_results.py  –  Qiyao Ma (Week 3)

Generates comparative bar charts and a markdown summary table from experiment CSV outputs.

Requires: matplotlib

Usage (after running experiment.py --batch --out_csv results/):
    python -m scripts.plot_results --csv_dir results/ --out_dir figures/
    python -m scripts.plot_results --csv_dir results/ --table_only
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_METRICS_DISPLAY: List[Tuple[str, str, bool]] = [
    # (key, label, lower_is_better)
    ("completed_tasks",   "Completed Tasks",        False),
    ("avg_wait_time",     "Avg Wait Time (s)",      True),
    ("p95_wait_time",     "P95 Wait Time (s)",      True),
    ("avg_turnaround",    "Avg Turnaround (s)",     True),
    ("throughput",        "Throughput (tasks/s)",   False),
    ("utilization",       "GPU Utilization",        False),
    ("fairness_wait_std", "Fairness Wait Std (s)",  True),
    ("oom_events",        "OOM Events",             True),
]


def _load_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Load a two-row CSV (one per policy) into {policy: {metric: value}}."""
    data: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            policy = row["policy"]
            data[policy] = {k: float(v) for k, v in row.items() if k != "policy"}
    return data


def _find_csvs(csv_dir: str) -> Dict[str, str]:
    """Return {workload_label: csv_path} for all CSV files in the directory."""
    result: Dict[str, str] = {}
    for fname in sorted(os.listdir(csv_dir)):
        if fname.endswith(".csv"):
            label = fname.replace(".csv", "")
            result[label] = os.path.join(csv_dir, fname)
    return result


def print_markdown_table(
    workload_label: str,
    data: Dict[str, Dict[str, float]],
) -> None:
    """Print a markdown table comparing policies for one workload."""
    policies = list(data.keys())
    metric_keys = [m[0] for m in _METRICS_DISPLAY]
    metric_labels = [m[1] for m in _METRICS_DISPLAY]

    header = "| Metric | " + " | ".join(policies) + " |"
    sep = "|---|" + "---|" * len(policies)
    print(f"\n### {workload_label}\n")
    print(header)
    print(sep)
    for key, label in zip(metric_keys, metric_labels):
        row_vals = []
        for p in policies:
            val = data[p].get(key, float("nan"))
            row_vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        print(f"| {label} | " + " | ".join(row_vals) + " |")


def plot_workload(
    workload_label: str,
    data: Dict[str, Dict[str, float]],
    out_dir: str,
) -> None:
    """Generate and save a bar-chart figure comparing policies for one workload."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[plot_results] matplotlib not found – skipping plot generation.")
        return

    policies = list(data.keys())
    metric_keys = [m[0] for m in _METRICS_DISPLAY]
    metric_labels = [m[1] for m in _METRICS_DISPLAY]
    lower_flags = [m[2] for m in _METRICS_DISPLAY]

    n_metrics = len(metric_keys)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if n_metrics > 1 else [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bar_width = 0.6 / len(policies)

    for ax_idx, (key, label, lower) in enumerate(zip(metric_keys, metric_labels, lower_flags)):
        ax = axes_flat[ax_idx]
        xs = range(len(policies))
        vals = [data[p].get(key, 0.0) for p in policies]
        bars = ax.bar(
            [x + bar_width * i for i, x in enumerate(xs)],
            vals,
            width=bar_width,
            color=[colors[i % len(colors)] for i in range(len(policies))],
            label=policies,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xticks([x + bar_width * (len(policies) - 1) / 2 for x in xs])
        ax.set_xticklabels(policies, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        indicator = "(lower=better)" if lower else "(higher=better)"
        ax.set_xlabel(indicator, fontsize=7)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    # Hide unused axes
    for ax_idx in range(n_metrics, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(f"Scheduler Comparison – {workload_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = Path(out_dir) / f"{workload_label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_results] saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparative plots from experiment CSVs")
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="results",
        help="Directory containing per-workload CSV files from experiment.py --batch",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="figures",
        help="Output directory for PNG figures",
    )
    parser.add_argument(
        "--table_only",
        action="store_true",
        help="Print markdown tables only; skip figure generation",
    )
    args = parser.parse_args()

    csv_files = _find_csvs(args.csv_dir)
    if not csv_files:
        print(f"[plot_results] No CSV files found in '{args.csv_dir}'. "
              "Run experiment.py --batch first.")
        return

    for label, path in csv_files.items():
        data = _load_csv(path)
        print_markdown_table(label, data)
        if not args.table_only:
            plot_workload(label, data, args.out_dir)


if __name__ == "__main__":
    main()
