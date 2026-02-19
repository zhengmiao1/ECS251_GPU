from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

from .event_logger import JsonlEventLogger
from .models import Task
from .scheduler import FIFOScheduler, MemoryAwareScheduler, SchedulerConfig, build_default_gpus
from .metrics import summarize_results


def _duration_class(duration: float, threshold: float) -> str:
    return "short" if duration <= threshold else "long"


def generate_tasks(
    n: int,
    users: int,
    seed: int,
    short_threshold: float,
    workload: str,
) -> List[Task]:
    random.seed(seed)
    tasks: List[Task] = []
    now = 0.0
    for i in range(n):
        inter_arrival = random.expovariate(1.0 / 8.0)
        now += inter_arrival

        if workload == "llm_heavy":
            long_prob = 0.2
        elif workload == "vlm_heavy":
            long_prob = 0.8
        else:
            long_prob = 0.4

        if random.random() < long_prob:
            duration = random.uniform(120, 900)
            mem_gb = random.choice([14, 16, 20, 24])
        else:
            duration = random.uniform(10, 80)
            mem_gb = random.choice([6, 8, 10, 12])

        task = Task(
            task_id=f"t{i}",
            user_id=f"u{random.randint(1, users)}",
            arrival_time=now,
            est_duration=duration,
            est_mem_gb=mem_gb,
            duration_class=_duration_class(duration, short_threshold),
        )
        tasks.append(task)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--gpu_mem", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--short_threshold", type=float, default=60.0)
    parser.add_argument("--aging_window", type=float, default=180.0)
    parser.add_argument("--policy", choices=["memory", "fifo", "both"], default="both")
    parser.add_argument("--workload", choices=["mixed", "llm_heavy", "vlm_heavy"], default="mixed")
    parser.add_argument("--log_dir", type=str, default="")
    args = parser.parse_args()

    tasks = generate_tasks(
        n=args.tasks,
        users=args.users,
        seed=args.seed,
        short_threshold=args.short_threshold,
        workload=args.workload,
    )

    policies = ["memory", "fifo"] if args.policy == "both" else [args.policy]

    for policy in policies:
        logger = None
        if args.log_dir:
            log_file = Path(args.log_dir) / f"{policy}_seed{args.seed}.jsonl"
            logger = JsonlEventLogger(str(log_file))

        gpus = build_default_gpus(args.gpus, args.gpu_mem)
        if policy == "memory":
            scheduler = MemoryAwareScheduler(
                gpus=gpus,
                config=SchedulerConfig(
                    short_threshold=args.short_threshold,
                    aging_window=args.aging_window,
                ),
                logger=logger,
            )
        else:
            scheduler = FIFOScheduler(
                gpus=gpus,
                config=SchedulerConfig(
                    short_threshold=args.short_threshold,
                    aging_window=args.aging_window,
                ),
                logger=logger,
            )

        result = scheduler.schedule(tasks)
        report = summarize_results(result, num_gpus=args.gpus)

        print(f"\npolicy={policy}, workload={args.workload}, seed={args.seed}")
        for k, v in report.items():
            print(f"{k}: {v}")

        if logger:
            logger.close()


if __name__ == "__main__":
    main()
