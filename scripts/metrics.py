from __future__ import annotations

from typing import Dict

from .models import ScheduleResult


def summarize_results(result: ScheduleResult) -> Dict[str, float]:
    if not result.results:
        return {
            "completed_tasks": 0,
            "avg_wait_time": 0.0,
            "avg_turnaround": 0.0,
            "oom_events": 0,
        }

    completed = len(result.results)
    avg_wait = sum(r.wait_time for r in result.results) / completed
    avg_turnaround = sum((r.end_time - r.task.arrival_time) for r in result.results) / completed
    oom = sum(1 for d in result.decisions if not d.admitted)
    return {
        "completed_tasks": completed,
        "avg_wait_time": round(avg_wait, 3),
        "avg_turnaround": round(avg_turnaround, 3),
        "oom_events": oom,
    }
