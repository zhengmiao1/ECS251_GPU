from __future__ import annotations

import argparse
import csv
from typing import Dict, List

from .metrics import summarize_results
from .scheduler import FIFOScheduler, MemoryAwareScheduler, SchedulerConfig, build_default_gpus
from .simulate import generate_tasks


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_experiment(
    seeds: List[int],
    tasks: int,
    users: int,
    gpus: int,
    gpu_mem: float,
    short_threshold: float,
    aging_window: float,
    workload: str,
) -> Dict[str, Dict[str, float]]:
    reports: Dict[str, List[Dict[str, float]]] = {"memory": [], "fifo": []}

    for seed in seeds:
        task_list = generate_tasks(
            n=tasks,
            users=users,
            seed=seed,
            short_threshold=short_threshold,
            workload=workload,
        )

        memory_scheduler = MemoryAwareScheduler(
            gpus=build_default_gpus(gpus, gpu_mem),
            config=SchedulerConfig(
                short_threshold=short_threshold,
                aging_window=aging_window,
            ),
        )
        fifo_scheduler = FIFOScheduler(
            gpus=build_default_gpus(gpus, gpu_mem),
            config=SchedulerConfig(
                short_threshold=short_threshold,
                aging_window=aging_window,
            ),
        )

        reports["memory"].append(
            summarize_results(memory_scheduler.schedule(task_list), num_gpus=gpus)
        )
        reports["fifo"].append(
            summarize_results(fifo_scheduler.schedule(task_list), num_gpus=gpus)
        )

    summary: Dict[str, Dict[str, float]] = {}
    keys = reports["memory"][0].keys() if reports["memory"] else []
    for policy, policy_reports in reports.items():
        summary[policy] = {k: round(_mean([r[k] for r in policy_reports]), 4) for k in keys}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--gpu_mem", type=float, default=24.0)
    parser.add_argument("--short_threshold", type=float, default=60.0)
    parser.add_argument("--aging_window", type=float, default=180.0)
    parser.add_argument("--workload", choices=["mixed", "llm_heavy", "vlm_heavy"], default="mixed")
    parser.add_argument("--seeds", type=str, default="7,11,19,23,31")
    parser.add_argument("--out_csv", type=str, default="")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    summary = run_experiment(
        seeds=seeds,
        tasks=args.tasks,
        users=args.users,
        gpus=args.gpus,
        gpu_mem=args.gpu_mem,
        short_threshold=args.short_threshold,
        aging_window=args.aging_window,
        workload=args.workload,
    )

    print(f"workload={args.workload}, seeds={seeds}")
    print("policy, completed_tasks, avg_wait_time, p95_wait_time, avg_turnaround, throughput, utilization, fairness_wait_std, oom_events")
    for policy in ["fifo", "memory"]:
        r = summary[policy]
        print(
            f"{policy}, {r['completed_tasks']}, {r['avg_wait_time']}, {r['p95_wait_time']}, "
            f"{r['avg_turnaround']}, {r['throughput']}, {r['utilization']}, {r['fairness_wait_std']}, {r['oom_events']}"
        )

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["policy", *summary["memory"].keys()])
            for policy in ["fifo", "memory"]:
                writer.writerow([policy, *[summary[policy][k] for k in summary[policy].keys()]])
        print(f"saved_csv={args.out_csv}")


if __name__ == "__main__":
    main()
