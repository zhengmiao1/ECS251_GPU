from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

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
    out_db: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    reports: Dict[str, List[Dict[str, float]]] = {"memory": [], "fifo": []}

    store = None
    if out_db:
        from .db_store import ExperimentStore
        store = ExperimentStore(out_db)

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

        for policy_name, scheduler in [("memory", memory_scheduler), ("fifo", fifo_scheduler)]:
            result = scheduler.schedule(task_list)
            reports[policy_name].append(summarize_results(result, num_gpus=gpus))
            if store:
                run_id = store.insert_run(
                    policy=policy_name,
                    workload=workload,
                    seed=seed,
                    gpus=gpus,
                    gpu_mem=gpu_mem,
                    tasks=len(task_list),
                )
                store.insert_decisions(run_id, result.decisions)
                store.insert_results(run_id, result.results)

    if store:
        store.close()

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
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run all three workload modes and write per-workload CSVs into --out_csv as a directory",
    )
    parser.add_argument(
        "--out_db",
        type=str,
        default="",
        help="SQLite database path to persist all decisions and results",
    )
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    workload_modes = ["mixed", "llm_heavy", "vlm_heavy"] if args.batch else [args.workload]

    for workload in workload_modes:
        summary = run_experiment(
            seeds=seeds,
            tasks=args.tasks,
            users=args.users,
            gpus=args.gpus,
            gpu_mem=args.gpu_mem,
            short_threshold=args.short_threshold,
            aging_window=args.aging_window,
            workload=workload,
            out_db=args.out_db or None,
        )

        print(f"\nworkload={workload}, seeds={seeds}")
        print("policy, completed_tasks, avg_wait_time, p95_wait_time, avg_turnaround, throughput, utilization, fairness_wait_std, oom_events")
        for policy in ["fifo", "memory"]:
            r = summary[policy]
            print(
                f"{policy}, {r['completed_tasks']}, {r['avg_wait_time']}, {r['p95_wait_time']}, "
                f"{r['avg_turnaround']}, {r['throughput']}, {r['utilization']}, {r['fairness_wait_std']}, {r['oom_events']}"
            )

        if args.out_csv:
            if args.batch:
                csv_path = str(Path(args.out_csv) / f"{workload}.csv")
                Path(args.out_csv).mkdir(parents=True, exist_ok=True)
            else:
                csv_path = args.out_csv
            with open(csv_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["policy", *summary["memory"].keys()])
                for policy in ["fifo", "memory"]:
                    writer.writerow([policy, *[summary[policy][k] for k in summary[policy].keys()]])
            print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()
