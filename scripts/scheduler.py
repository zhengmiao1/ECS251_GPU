from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

from .models import Task, GPUState, RunningTask, AdmissionDecision, TaskResult, ScheduleResult


@dataclass
class SchedulerConfig:
    prefer_short: bool = True
    short_threshold: float = 120.0
    aging_window: float = 180.0
    buffer_gb: float = 4.0
    grace_secs: float = 20.0


class EventLogger(Protocol):
    def log(self, event_type: str, **payload: object) -> None:
        ...


class _NoopLogger:
    def log(self, event_type: str, **payload: object) -> None:
        del event_type, payload


def _task_priority_bucket(task: Task) -> int:
    order = {"small": 0, "medium": 1, "large": 2}
    return order.get(task.duration_class, 1)


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
        self._grace_reservations: Dict[str, List[Tuple[float, float]]] = {g.gpu_id: [] for g in self.gpus}

    def _update_running(self, now: float) -> None:
        for gpu in self.gpus:
            gpu.running = [rt for rt in gpu.running if rt.end_time > now]
            self._grace_reservations[gpu.gpu_id] = [
                (until, mem) for (until, mem) in self._grace_reservations[gpu.gpu_id] if until > now
            ]

    def _actual_used_mem(self, gpu: GPUState, now: float) -> float:
        used = 0.0
        for rt in gpu.running:
            mem = rt.task.est_mem_gb
            if now < rt.start_time + rt.task.spike_secs:
                mem += rt.task.startup_spike_gb
            used += mem
        return used

    def _grace_reserved_mem(self, gpu: GPUState) -> float:
        return sum(mem for _, mem in self._grace_reservations[gpu.gpu_id])

    def _effective_free_mem_for_admission(self, gpu: GPUState) -> float:
        # MAS subtracts both safety buffer and grace-period reservations.
        return gpu.free_mem_gb - self.config.buffer_gb - self._grace_reserved_mem(gpu)

    def _register_grace_reservation(self, gpu_id: str, now: float, task: Task) -> None:
        reserve = task.est_mem_gb + task.startup_spike_gb
        self._grace_reservations[gpu_id].append((now + self.config.grace_secs, reserve))

    def _runtime_oom(self, gpu: GPUState, now: float) -> bool:
        return self._actual_used_mem(gpu, now) > gpu.total_mem_gb + 1e-9

    def _pick_gpu(self, task: Task) -> GPUState | None:
        feasible = [
            g for g in self.gpus if self._effective_free_mem_for_admission(g) >= task.est_mem_gb
        ]
        if not feasible:
            return None
        return max(feasible, key=lambda g: self._effective_free_mem_for_admission(g))

    def _order_ready(self, ready: List[Task], now: float) -> List[Task]:
        if not self.config.prefer_short:
            return sorted(ready, key=lambda t: t.arrival_time)

        def priority(task: Task) -> tuple[int, float, float]:
            waited = max(0.0, now - task.arrival_time)
            if waited >= self.config.aging_window:
                return (-1, -waited, task.arrival_time)
            return (_task_priority_bucket(task), task.est_duration, task.arrival_time)

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
            startup_spike_gb=task.startup_spike_gb,
            **extra,
        )

    def _build_failed_result(self, task: Task, gpu_id: str, now: float, reason: str) -> TaskResult:
        self._log_task("task_oom", task, now=now, gpu_id=gpu_id, reason=reason)
        return TaskResult(
            task=task,
            gpu_id=gpu_id,
            start_time=now,
            end_time=now + 1e-3,
            wait_time=now - task.arrival_time,
            status="oom_killed",
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
                self._register_grace_reservation(gpu.gpu_id, now, task)
                decisions.append(
                    AdmissionDecision(
                        task=task,
                        admitted=True,
                        reason="admitted",
                        gpu_id=gpu.gpu_id,
                    )
                )
                self._log_task(
                    "dispatch",
                    task,
                    now=now,
                    gpu_id=gpu.gpu_id,
                    wait_time=start_time - task.arrival_time,
                )
                # MAS should avoid OOM by design, but keep runtime check for safety.
                if self._runtime_oom(gpu, now):
                    gpu.running.pop()
                    decisions.append(
                        AdmissionDecision(
                            task=task,
                            admitted=False,
                            reason="runtime_oom",
                            gpu_id=gpu.gpu_id,
                        )
                    )
                    results.append(self._build_failed_result(task, gpu.gpu_id, now, "runtime_oom"))
                else:
                    results.append(
                        TaskResult(
                            task=task,
                            gpu_id=gpu.gpu_id,
                            start_time=start_time,
                            end_time=end_time,
                            wait_time=start_time - task.arrival_time,
                            status="done",
                        )
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


class NaiveSharingScheduler(MemoryAwareScheduler):
    policy_name = "naive"

    def _effective_free_mem_for_admission(self, gpu: GPUState) -> float:
        # No safety buffer and no grace reservation.
        return gpu.free_mem_gb

    def _order_ready(self, ready: List[Task], now: float) -> List[Task]:
        del now
        return sorted(
            ready,
            key=lambda t: (t.arrival_time, t.task_id),
        )

    def _register_grace_reservation(self, gpu_id: str, now: float, task: Task) -> None:
        del gpu_id, now, task

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
                    decisions.append(AdmissionDecision(task=task, admitted=False, reason="exceeds_gpu_capacity"))
                    self._log_task("reject", task, now=now, reason="exceeds_gpu_capacity")
                    continue

                gpu = self._pick_gpu(task)
                if gpu is None:
                    remaining.append(task)
                    self._log_task("defer", task, now=now, reason="temporary_memory_pressure")
                    continue

                start_time = now
                end_time = now + task.est_duration
                gpu.running.append(RunningTask(task=task, start_time=start_time, end_time=end_time))
                decisions.append(AdmissionDecision(task=task, admitted=True, reason="admitted", gpu_id=gpu.gpu_id))
                self._log_task("dispatch", task, now=now, gpu_id=gpu.gpu_id, wait_time=start_time - task.arrival_time)

                if self._runtime_oom(gpu, now):
                    gpu.running.pop()
                    decisions.append(
                        AdmissionDecision(task=task, admitted=False, reason="runtime_oom", gpu_id=gpu.gpu_id)
                    )
                    results.append(self._build_failed_result(task, gpu.gpu_id, now, "runtime_oom"))
                    admitted_any = True
                else:
                    results.append(
                        TaskResult(
                            task=task,
                            gpu_id=gpu.gpu_id,
                            start_time=start_time,
                            end_time=end_time,
                            wait_time=start_time - task.arrival_time,
                            status="done",
                        )
                    )
                    admitted_any = True

            pending = remaining
            if not admitted_any:
                next_finish = min((rt.end_time for g in self.gpus for rt in g.running), default=None)
                if next_finish is None and i < len(tasks):
                    now = tasks[i].arrival_time
                elif next_finish is not None:
                    now = max(now, next_finish)
                else:
                    break

        return ScheduleResult(decisions=decisions, results=results)


class ExclusiveFIFOScheduler(MemoryAwareScheduler):
    policy_name = "exclusive"

    def _effective_free_mem_for_admission(self, gpu: GPUState) -> float:
        if gpu.running:
            return 0.0
        return gpu.total_mem_gb

    def _pick_gpu(self, task: Task) -> GPUState | None:
        feasible = [g for g in self.gpus if not g.running and g.total_mem_gb >= task.est_mem_gb]
        if not feasible:
            return None
        return sorted(feasible, key=lambda g: g.gpu_id)[0]

    def _order_ready(self, ready: List[Task], now: float) -> List[Task]:
        del now
        return sorted(ready, key=lambda t: (t.arrival_time, t.task_id))

    def _register_grace_reservation(self, gpu_id: str, now: float, task: Task) -> None:
        del gpu_id, now, task


# Backward compatibility with previous script names.
FIFOScheduler = NaiveSharingScheduler

def build_default_gpus(count: int, mem_gb: float) -> List[GPUState]:
    return [GPUState(gpu_id=f"gpu{idx}", total_mem_gb=mem_gb) for idx in range(count)]
