from __future__ import annotations

import argparse
import random
from typing import List

from .models import Task
from .scheduler import MemoryAwareScheduler, SchedulerConfig, build_default_gpus
from .metrics import summarize_results


def _duration_class(duration: float, threshold: float) -> str:
    return "short" if duration <= threshold else "long"


def generate_tasks(
    n: int,
    users: int,
    seed: int,
    short_threshold: float,
) -> List[Task]:
    random.seed(seed)
    tasks: List[Task] = []
    now = 0.0
    for i in range(n):
        inter_arrival = random.expovariate(1.0 / 10.0)
        now += inter_arrival
        duration = random.choice([random.uniform(10, 60), random.uniform(120, 600)])
        mem_gb = random.choice([6, 8, 10, 12, 16, 20])
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
    args = parser.parse_args()

    tasks = generate_tasks(
        n=args.tasks,
        users=args.users,
        seed=args.seed,
        short_threshold=args.short_threshold,
    )
    gpus = build_default_gpus(args.gpus, args.gpu_mem)
    scheduler = MemoryAwareScheduler(
        gpus=gpus,
        config=SchedulerConfig(short_threshold=args.short_threshold),
    )
    result = scheduler.schedule(tasks)
    report = summarize_results(result)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
