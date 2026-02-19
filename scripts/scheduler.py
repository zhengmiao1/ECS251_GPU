from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from .models import Task, GPUState, RunningTask, AdmissionDecision, TaskResult, ScheduleResult


@dataclass
class SchedulerConfig:
    prefer_short: bool = True
    short_threshold: float = 60.0
    aging_window: float = 180.0


class EventLogger(Protocol):
    def log(self, event_type: str, **payload: object) -> None:
        ...


class _NoopLogger:
    def log(self, event_type: str, **payload: object) -> None:
        del event_type, payload


class MemoryAwareScheduler:
    policy_name = "memory"

    def __init__(
        self,
        gpus: List[GPUState],
        config: SchedulerConfig | None = None,
        logger: EventLogger | None = None,
    ):
        self.gpus = gpus
        self.config = config or SchedulerConfig()
        self.logger = logger or _NoopLogger()
        self.max_gpu_mem = max((g.total_mem_gb for g in self.gpus), default=0.0)

    def _update_running(self, now: float) -> None:
        for gpu in self.gpus:
            gpu.running = [rt for rt in gpu.running if rt.end_time > now]

    def _pick_gpu(self, task: Task) -> GPUState | None:
        feasible = [g for g in self.gpus if g.free_mem_gb >= task.est_mem_gb]
        if not feasible:
            return None
        return max(feasible, key=lambda g: g.free_mem_gb)

    def _order_ready(self, ready: List[Task], now: float) -> List[Task]:
        if not self.config.prefer_short:
            return sorted(ready, key=lambda t: t.arrival_time)

        def priority(task: Task) -> tuple[int, float, float]:
            waited = max(0.0, now - task.arrival_time)
            if waited >= self.config.aging_window:
                return (-1, waited, task.arrival_time)
            short_or_long = 0 if task.est_duration <= self.config.short_threshold else 1
            return (short_or_long, task.est_duration, task.arrival_time)

        return sorted(ready, key=priority)

    def _log_task(self, event_type: str, task: Task, **extra: object) -> None:
        self.logger.log(
            event_type,
            policy=self.policy_name,
            task_id=task.task_id,
            user_id=task.user_id,
            arrival_time=task.arrival_time,
            est_duration=task.est_duration,
            est_mem_gb=task.est_mem_gb,
            **extra,
        )

    def schedule(self, tasks: List[Task]) -> ScheduleResult:
        tasks = sorted(tasks, key=lambda t: t.arrival_time)
        decisions: List[AdmissionDecision] = []
        results: List[TaskResult] = []
        pending: List[Task] = []
        now = 0.0

        i = 0
        while i < len(tasks) or pending:
            if i < len(tasks):
                now = max(now, tasks[i].arrival_time)
            self._update_running(now)

            while i < len(tasks) and tasks[i].arrival_time <= now:
                pending.append(tasks[i])
                i += 1

            ready = self._order_ready(pending, now=now)
            admitted_any = False
            remaining: List[Task] = []
            for task in ready:
                if task.est_mem_gb > self.max_gpu_mem:
                    decisions.append(
                        AdmissionDecision(
                            task=task,
                            admitted=False,
                            reason="exceeds_gpu_capacity",
                        )
                    )
                    self._log_task(
                        "reject",
                        task,
                        now=now,
                        reason="exceeds_gpu_capacity",
                    )
                    continue

                gpu = self._pick_gpu(task)
                if gpu is None:
                    remaining.append(task)
                    self._log_task(
                        "defer",
                        task,
                        now=now,
                        reason="temporary_memory_pressure",
                    )
                    continue

                start_time = now
                end_time = start_time + task.est_duration
                gpu.running.append(RunningTask(task=task, start_time=start_time, end_time=end_time))
                decisions.append(
                    AdmissionDecision(
                        task=task,
                        admitted=True,
                        reason="admitted",
                        gpu_id=gpu.gpu_id,
                    )
                )
                results.append(
                    TaskResult(
                        task=task,
                        gpu_id=gpu.gpu_id,
                        start_time=start_time,
                        end_time=end_time,
                        wait_time=start_time - task.arrival_time,
                    )
                )
                self._log_task(
                    "dispatch",
                    task,
                    now=now,
                    gpu_id=gpu.gpu_id,
                    wait_time=start_time - task.arrival_time,
                )
                admitted_any = True

            pending = remaining
            if not admitted_any:
                next_finish = min(
                    (rt.end_time for g in self.gpus for rt in g.running),
                    default=None,
                )
                if next_finish is None and i < len(tasks):
                    now = tasks[i].arrival_time
                elif next_finish is not None:
                    now = max(now, next_finish)
                else:
                    break

        return ScheduleResult(decisions=decisions, results=results)


class FIFOScheduler(MemoryAwareScheduler):
    policy_name = "fifo"

    def _pick_gpu(self, task: Task) -> GPUState | None:
        feasible = [g for g in self.gpus if g.free_mem_gb >= task.est_mem_gb]
        if not feasible:
            return None
        return sorted(feasible, key=lambda g: g.gpu_id)[0]

    def _order_ready(self, ready: List[Task], now: float) -> List[Task]:
        del now
        return sorted(
            ready,
            key=lambda t: (t.arrival_time, t.task_id),
        )


def build_default_gpus(count: int, mem_gb: float) -> List[GPUState]:
    return [GPUState(gpu_id=f"gpu{idx}", total_mem_gb=mem_gb) for idx in range(count)]
