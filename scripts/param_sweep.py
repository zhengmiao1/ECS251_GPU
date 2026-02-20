"""
param_sweep.py  –  Zheng Miao (Week 3)

Grid sweep over short_threshold × aging_window for the memory-aware policy.
Prints a sensitivity table and optionally writes a CSV.

Usage:
    python -m scripts.param_sweep
    python -m scripts.param_sweep --workload llm_heavy --out_csv sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
from typing import Dict, List, Tuple

from .metrics import summarize_results
from .scheduler import MemoryAwareScheduler, SchedulerConfig, build_default_gpus
from .simulate import generate_tasks


_DEFAULT_SHORT_THRESHOLDS = [30.0, 60.0, 90.0, 120.0]
_DEFAULT_AGING_WINDOWS = [60.0, 120.0, 180.0, 240.0, 300.0]
_DEFAULT_SEEDS = [7, 11, 19, 23, 31]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sweep(
    short_thresholds: List[float],
    aging_windows: List[float],
    seeds: List[int],
    tasks: int,
    users: int,
    gpus: int,
    gpu_mem: float,
    workload: str,
    target_metrics: Tuple[str, ...] = ("avg_wait_time", "fairness_wait_std", "oom_events", "utilization"),
) -> List[Dict[str, object]]:
    """Run a full grid sweep and return one row per (short_threshold, aging_window) pair."""
    rows: List[Dict[str, object]] = []

    for st, aw in itertools.product(short_thresholds, aging_windows):
        seed_results: List[Dict[str, float]] = []
        for seed in seeds:
            task_list = generate_tasks(
                n=tasks,
                users=users,
                seed=seed,
                short_threshold=st,
                workload=workload,
            )
            scheduler = MemoryAwareScheduler(
                gpus=build_default_gpus(gpus, gpu_mem),
                config=SchedulerConfig(short_threshold=st, aging_window=aw),
            )
            seed_results.append(summarize_results(scheduler.schedule(task_list), num_gpus=gpus))

        row: Dict[str, object] = {
            "short_threshold": st,
            "aging_window": aw,
        }
        for metric in target_metrics:
            row[metric] = round(_mean([r[metric] for r in seed_results]), 4)
        rows.append(row)

    return rows


def _print_table(rows: List[Dict[str, object]], metrics: Tuple[str, ...]) -> None:
    col_keys = ["short_threshold", "aging_window", *metrics]
    header = "  ".join(f"{k:>18}" for k in col_keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(f"{row[k]:>18}" for k in col_keys)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sensitivity sweep for memory-aware scheduler")
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--gpu_mem", type=float, default=24.0)
    parser.add_argument("--workload", choices=["mixed", "llm_heavy", "vlm_heavy"], default="mixed")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in _DEFAULT_SEEDS))
    parser.add_argument(
        "--short_thresholds",
        type=str,
        default=",".join(str(v) for v in _DEFAULT_SHORT_THRESHOLDS),
        help="Comma-separated list of short_threshold values to sweep",
    )
    parser.add_argument(
        "--aging_windows",
        type=str,
        default=",".join(str(v) for v in _DEFAULT_AGING_WINDOWS),
        help="Comma-separated list of aging_window values to sweep",
    )
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    short_thresholds = [float(v.strip()) for v in args.short_thresholds.split(",") if v.strip()]
    aging_windows = [float(v.strip()) for v in args.aging_windows.split(",") if v.strip()]
    target_metrics: Tuple[str, ...] = ("avg_wait_time", "fairness_wait_std", "oom_events", "utilization")

    print(f"workload={args.workload}  seeds={seeds}")
    print(f"short_thresholds={short_thresholds}")
    print(f"aging_windows={aging_windows}\n")

    rows = sweep(
        short_thresholds=short_thresholds,
        aging_windows=aging_windows,
        seeds=seeds,
        tasks=args.tasks,
        users=args.users,
        gpus=args.gpus,
        gpu_mem=args.gpu_mem,
        workload=args.workload,
        target_metrics=target_metrics,
    )

    _print_table(rows, target_metrics)

    if args.out_csv:
        col_keys = ["short_threshold", "aging_window", *target_metrics]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=col_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved_csv={args.out_csv}")


if __name__ == "__main__":
    main()
