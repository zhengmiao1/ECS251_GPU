"""
param_sweep.py

Safety-buffer sensitivity sweep for the simulator setup used in the paper.
Outputs table-style rows for: Naive(0%), then MAS with configurable buffer levels.
"""
from __future__ import annotations

import argparse
import csv
from typing import Dict, List

from .metrics import summarize_results
from .scheduler import MemoryAwareScheduler, NaiveSharingScheduler, SchedulerConfig, build_default_gpus
from .simulate import generate_tasks


_DEFAULT_SEEDS = [7, 11, 19, 23, 31]
_DEFAULT_BUFFER_PCTS = [0, 5, 10, 15, 20]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sweep(
    buffer_pcts: List[float],
    seeds: List[int],
    tasks: int,
    users: int,
    gpus: int,
    gpu_mem: float,
    workload: str,
    short_threshold: float,
    aging_window: float,
    grace_secs: float,
    inter_arrival_mean: float,
) -> List[Dict[str, object]]:
    """Run sensitivity sweep and return one row per buffer percentage."""
    rows: List[Dict[str, object]] = []

    for pct in buffer_pcts:
        seed_results: List[Dict[str, float]] = []
        for seed in seeds:
            task_list = generate_tasks(
                n=tasks,
                users=users,
                seed=seed,
                short_threshold=short_threshold,
                workload=workload,
                inter_arrival_mean=inter_arrival_mean,
            )
            if pct == 0:
                scheduler = NaiveSharingScheduler(
                    gpus=build_default_gpus(gpus, gpu_mem),
                    config=SchedulerConfig(short_threshold=short_threshold, aging_window=aging_window),
                )
                label = "naive_0%"
            else:
                scheduler = MemoryAwareScheduler(
                    gpus=build_default_gpus(gpus, gpu_mem),
                    config=SchedulerConfig(
                        short_threshold=short_threshold,
                        aging_window=aging_window,
                        buffer_gb=gpu_mem * (pct / 100.0),
                        grace_secs=grace_secs,
                    ),
                )
                label = f"mas_{int(pct)}%"
            seed_results.append(summarize_results(scheduler.schedule(task_list), num_gpus=gpus, gpu_mem_gb=gpu_mem))

        row: Dict[str, object] = {
            "policy": label,
            "buffer_pct": pct,
        }
        for metric in ("avg_wait_time", "utilization_pct", "oom_rate_pct", "makespan"):
            row[metric] = round(_mean([r[metric] for r in seed_results]), 4)
        rows.append(row)

    return rows


def _print_table(rows: List[Dict[str, object]]) -> None:
    col_keys = ["policy", "buffer_pct", "avg_wait_time", "utilization_pct", "oom_rate_pct", "makespan"]
    header = "  ".join(f"{k:>18}" for k in col_keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = "  ".join(f"{row[k]:>18}" for k in col_keys)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety-buffer sensitivity sweep")
    parser.add_argument("--tasks", type=int, default=300)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--gpu_mem", type=float, default=40.0)
    parser.add_argument("--workload", choices=["mixed", "llm_heavy", "vlm_heavy"], default="mixed")
    parser.add_argument("--short_threshold", type=float, default=120.0)
    parser.add_argument("--aging_window", type=float, default=180.0)
    parser.add_argument("--grace_secs", type=float, default=20.0)
    parser.add_argument("--inter_arrival_mean", type=float, default=8.0)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in _DEFAULT_SEEDS))
    parser.add_argument(
        "--buffer_pcts",
        type=str,
        default=",".join(str(v) for v in _DEFAULT_BUFFER_PCTS),
        help="Comma-separated list of buffer percentages; include 0 for naive baseline",
    )
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    buffer_pcts = [float(v.strip()) for v in args.buffer_pcts.split(",") if v.strip()]

    print(f"workload={args.workload}  seeds={seeds}")
    print(f"buffer_pcts={buffer_pcts}\n")

    rows = sweep(
        buffer_pcts=buffer_pcts,
        seeds=seeds,
        tasks=args.tasks,
        users=args.users,
        gpus=args.gpus,
        gpu_mem=args.gpu_mem,
        workload=args.workload,
        short_threshold=args.short_threshold,
        aging_window=args.aging_window,
        grace_secs=args.grace_secs,
        inter_arrival_mean=args.inter_arrival_mean,
    )

    _print_table(rows)

    if args.out_csv:
        col_keys = ["policy", "buffer_pct", "avg_wait_time", "utilization_pct", "oom_rate_pct", "makespan"]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=col_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved_csv={args.out_csv}")


if __name__ == "__main__":
    main()
