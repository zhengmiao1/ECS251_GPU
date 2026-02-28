"""
analysis.py  –  Zheng Miao (Week 4)

Priority-inversion analyzer for the memory-aware GPU scheduler.

A priority inversion occurs when a SHORT task (est_duration <= short_threshold) is
delayed in the pending queue while a LONG task that arrived *after* it was already
dispatched.  The anti-starvation aging rule in MemoryAwareScheduler resolves inversions
once the short task has waited >= aging_window seconds; this module detects and counts
how many such inversions occurred and whether they were eventually resolved.

Also provides print_policy_spec() which emits the formal pseudocode for the scheduling
policy used in the final report.

Usage:
    python -m scripts.analysis --tasks 100 --seed 7 --workload mixed
    python -m scripts.analysis --spec_only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from .models import ScheduleResult, TaskResult
from .scheduler import MemoryAwareScheduler, SchedulerConfig, build_default_gpus
from .simulate import generate_tasks


# ---------------------------------------------------------------------------
# Formal policy specification (printed for the report)
# ---------------------------------------------------------------------------

_POLICY_SPEC = """\
=======================================================================
 FORMAL POLICY SPECIFICATION  –  Memory-Aware GPU Scheduler  (v1.0)
=======================================================================

Parameters
----------
  short_threshold  : float   # tasks with est_duration <= threshold are "short"
  aging_window     : float   # seconds after which any deferred task is promoted
  max_gpu_mem      : float   # maximum single-GPU memory (GB); used for reject gate

Admission Control
-----------------
  ADMIT(task, gpus, now):
    IF task.est_mem_gb > max_gpu_mem:
        RETURN reject(task, reason="exceeds_gpu_capacity")

    gpu = argmax_{g in gpus : g.free_mem_gb >= task.est_mem_gb} g.free_mem_gb
    IF gpu is None:
        RETURN defer(task, reason="temporary_memory_pressure")

    RETURN dispatch(task, gpu)

Execution Ordering  (applied to the pending queue at each scheduling step)
-------------------
  ORDER(pending, now):
    FOR task IN pending:
        waited = now - task.arrival_time
        IF waited >= aging_window:
            priority(task) = (-1, -waited, task.arrival_time)   # top priority
        ELSE IF task.est_duration <= short_threshold:
            priority(task) = ( 0,  task.est_duration, task.arrival_time)
        ELSE:
            priority(task) = ( 1,  task.est_duration, task.arrival_time)
    RETURN sorted(pending, key=priority)   # ascending

Correctness Properties
----------------------
  P1 (No OOM):    A task is dispatched only when free_mem_gb >= est_mem_gb.
  P2 (No starvation): Any deferred task is promoted to top priority after
                  aging_window seconds, guaranteeing eventual dispatch
                  (provided est_mem_gb <= max_gpu_mem).
  P3 (Short-task preference): Short tasks precede long tasks in dispatch
                  order when both fit and neither has aged out.
=======================================================================
"""


def print_policy_spec() -> None:
    print(_POLICY_SPEC)


# ---------------------------------------------------------------------------
# Priority-inversion detection
# ---------------------------------------------------------------------------

@dataclass
class PriorityInversionEvent:
    """Records a single priority-inversion instance."""
    short_task_id: str
    short_arrival: float
    short_start: float          # when the short task was finally dispatched
    short_wait: float
    blocking_task_id: str       # a long task dispatched while the short task waited
    blocking_arrival: float
    blocking_start: float
    resolved_by_aging: bool     # True if short task waited >= aging_window before dispatch


def detect_priority_inversions(
    result: ScheduleResult,
    short_threshold: float = 60.0,
    aging_window: float = 180.0,
) -> List[PriorityInversionEvent]:
    """
    Scan a ScheduleResult and return all priority-inversion events.

    An inversion is recorded when:
      - A SHORT task T_s was deferred (wait_time > 0), AND
      - At least one LONG task T_l arrived after T_s but was dispatched before T_s.
    """
    short_results = [
        r for r in result.results
        if r.task.est_duration <= short_threshold and r.wait_time > 0.0
    ]
    long_results = [
        r for r in result.results
        if r.task.est_duration > short_threshold
    ]

    # Index long tasks by start_time for fast lookup
    long_by_start: dict[str, TaskResult] = {r.task.task_id: r for r in long_results}

    events: List[PriorityInversionEvent] = []

    for sr in short_results:
        # Find long tasks that: arrived after sr, started before sr
        blocking = [
            lr for lr in long_results
            if lr.task.arrival_time > sr.task.arrival_time
            and lr.start_time < sr.start_time
        ]
        for bl in blocking:
            events.append(
                PriorityInversionEvent(
                    short_task_id=sr.task.task_id,
                    short_arrival=sr.task.arrival_time,
                    short_start=sr.start_time,
                    short_wait=sr.wait_time,
                    blocking_task_id=bl.task.task_id,
                    blocking_arrival=bl.task.arrival_time,
                    blocking_start=bl.start_time,
                    resolved_by_aging=sr.wait_time >= aging_window,
                )
            )

    return events


def print_inversion_report(
    events: List[PriorityInversionEvent],
    policy_name: str,
) -> None:
    total = len(events)
    aged = sum(1 for e in events if e.resolved_by_aging)
    unresolved = total - aged

    print(f"\n[Priority-Inversion Report]  policy={policy_name}")
    print(f"  total inversions    : {total}")
    print(f"  resolved by aging   : {aged}")
    print(f"  unresolved (<aging) : {unresolved}")

    if total > 0:
        avg_short_wait = sum(e.short_wait for e in events) / total
        print(f"  avg short-task wait : {avg_short_wait:.1f} s")
        if unresolved > 0:
            print("\n  Unresolved inversion samples (short task, blocking long task):")
            shown = 0
            for e in events:
                if not e.resolved_by_aging:
                    print(
                        f"    {e.short_task_id} (arr={e.short_arrival:.1f}, wait={e.short_wait:.1f}s)"
                        f"  blocked by  {e.blocking_task_id} (arr={e.blocking_arrival:.1f},"
                        f" start={e.blocking_start:.1f})"
                    )
                    shown += 1
                    if shown >= 5:
                        print(f"    ... ({unresolved - shown} more)")
                        break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Priority-inversion analysis for GPU scheduler")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--gpu_mem", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--short_threshold", type=float, default=60.0)
    parser.add_argument("--aging_window", type=float, default=180.0)
    parser.add_argument("--workload", choices=["mixed", "llm_heavy", "vlm_heavy"], default="mixed")
    parser.add_argument(
        "--spec_only",
        action="store_true",
        help="Print the formal policy specification and exit",
    )
    args = parser.parse_args()

    print_policy_spec()
    if args.spec_only:
        return

    task_list = generate_tasks(
        n=args.tasks,
        users=args.users,
        seed=args.seed,
        short_threshold=args.short_threshold,
        workload=args.workload,
    )

    for policy_cls, label in [
        (MemoryAwareScheduler, "memory"),
        # FIFOScheduler has no aging, so all inversions are unresolved – useful contrast
        (__import__("scripts.scheduler", fromlist=["FIFOScheduler"]).FIFOScheduler, "fifo"),
    ]:
        scheduler = policy_cls(
            gpus=build_default_gpus(args.gpus, args.gpu_mem),
            config=SchedulerConfig(
                short_threshold=args.short_threshold,
                aging_window=args.aging_window,
            ),
        )
        result = scheduler.schedule(task_list)
        events = detect_priority_inversions(
            result,
            short_threshold=args.short_threshold,
            aging_window=args.aging_window,
        )
        print_inversion_report(events, policy_name=label)


if __name__ == "__main__":
    main()
