from __future__ import annotations

import statistics
from typing import Dict

from .models import ScheduleResult


def _merge_coverage(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    covered = 0.0
    cur_start, cur_end = sorted_intervals[0]
    for start, end in sorted_intervals[1:]:
        if start > cur_end:
            covered += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    covered += cur_end - cur_start
    return covered


def summarize_results(result: ScheduleResult, num_gpus: int | None = None) -> Dict[str, float]:
    if not result.results:
        return {
            "completed_tasks": 0,
            "avg_wait_time": 0.0,
            "p95_wait_time": 0.0,
            "avg_turnaround": 0.0,
            "throughput": 0.0,
            "utilization": 0.0,
            "fairness_wait_std": 0.0,
            "oom_events": 0,
        }

    completed = len(result.results)
    waits = sorted(r.wait_time for r in result.results)
    p95_idx = min(completed - 1, int(0.95 * (completed - 1)))
    makespan = max(r.end_time for r in result.results) - min(r.task.arrival_time for r in result.results)
    avg_wait = sum(r.wait_time for r in result.results) / completed
    avg_turnaround = sum((r.end_time - r.task.arrival_time) for r in result.results) / completed
    throughput = completed / makespan if makespan > 0 else 0.0
    if num_gpus and num_gpus > 0 and makespan > 0:
        per_gpu_intervals: dict[str, list[tuple[float, float]]] = {}
        for task_res in result.results:
            per_gpu_intervals.setdefault(task_res.gpu_id, []).append(
                (task_res.start_time, task_res.end_time)
            )
        total_busy = sum(_merge_coverage(intervals) for intervals in per_gpu_intervals.values())
        utilization = total_busy / (num_gpus * makespan)
    else:
        utilization = 0.0

    user_waits: dict[str, list[float]] = {}
    for task_res in result.results:
        user_waits.setdefault(task_res.task.user_id, []).append(task_res.wait_time)
    user_avg_waits = [sum(w) / len(w) for w in user_waits.values()]
    fairness_wait_std = statistics.pstdev(user_avg_waits) if len(user_avg_waits) > 1 else 0.0

    oom = sum(1 for d in result.decisions if not d.admitted)
    return {
        "completed_tasks": completed,
        "avg_wait_time": round(avg_wait, 3),
        "p95_wait_time": round(waits[p95_idx], 3),
        "avg_turnaround": round(avg_turnaround, 3),
        "throughput": round(throughput, 4),
        "utilization": round(utilization, 4),
        "fairness_wait_std": round(fairness_wait_std, 3),
        "oom_events": oom,
    }
