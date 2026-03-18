'''
This file is used to calculate the metrics for the scheduling algorithm.

The metrics are:
- Completed tasks
- Average wait time
- 95th percentile wait time
- Average turnaround time
- Throughput
- Utilization
- Fairness wait standard deviation
- OOM events
'''
from __future__ import annotations

import statistics
from typing import Dict

from .models import ScheduleResult


def summarize_results(
    result: ScheduleResult,
    num_gpus: int | None = None,
    gpu_mem_gb: float | None = None,
) -> Dict[str, float]:
    if not result.results:
        return {
            "completed_tasks": 0,
            "avg_wait_time": 0.0,
            "p95_wait_time": 0.0,
            "avg_turnaround": 0.0,
            "makespan": 0.0,
            "throughput": 0.0,
            "utilization": 0.0,
            "utilization_pct": 0.0,
            "fairness_wait_std": 0.0,
            "oom_events": 0,
            "oom_rate": 0.0,
            "oom_rate_pct": 0.0,
        }

    done_results = [r for r in result.results if r.status == "done"]
    completed = len(done_results)
    waits = sorted(r.wait_time for r in done_results) if done_results else [0.0]
    p95_idx = min(completed - 1, int(0.95 * (completed - 1)))
    earliest_arrival = min(d.task.arrival_time for d in result.results)
    makespan = max(r.end_time for r in result.results) - earliest_arrival
    avg_wait = (sum(r.wait_time for r in done_results) / completed) if completed else 0.0
    avg_turnaround = (
        sum((r.end_time - r.task.arrival_time) for r in done_results) / completed
        if completed
        else 0.0
    )
    throughput = completed / makespan if makespan > 0 else 0.0
    if num_gpus and num_gpus > 0 and gpu_mem_gb and gpu_mem_gb > 0 and makespan > 0:
        # Memory utilization over time proxy:
        #   sum(steady_mem_gb * runtime) / (num_gpus * gpu_mem_gb * makespan)
        mem_time = sum(
            r.task.est_mem_gb * max(0.0, r.end_time - r.start_time)
            for r in done_results
        )
        utilization = mem_time / (num_gpus * gpu_mem_gb * makespan)
    else:
        utilization = 0.0

    user_waits: dict[str, list[float]] = {}
    for task_res in done_results:
        user_waits.setdefault(task_res.task.user_id, []).append(task_res.wait_time)
    user_avg_waits = [sum(w) / len(w) for w in user_waits.values()]
    fairness_wait_std = statistics.pstdev(user_avg_waits) if len(user_avg_waits) > 1 else 0.0

    oom = sum(1 for r in result.results if r.status == "oom_killed")
    total_tasks = len({d.task.task_id for d in result.decisions}) if result.decisions else len(result.results)
    oom_rate = (oom / total_tasks) if total_tasks > 0 else 0.0
    return {
        "completed_tasks": completed,
        "avg_wait_time": round(avg_wait, 3),
        "p95_wait_time": round(waits[p95_idx], 3) if completed > 0 else 0.0,
        "avg_turnaround": round(avg_turnaround, 3),
        "makespan": round(makespan, 3),
        "throughput": round(throughput, 4),
        "utilization": round(utilization, 4),
        "utilization_pct": round(utilization * 100.0, 3),
        "fairness_wait_std": round(fairness_wait_std, 3),
        "oom_events": oom,
        "oom_rate": round(oom_rate, 4),
        "oom_rate_pct": round(oom_rate * 100.0, 3),
    }
