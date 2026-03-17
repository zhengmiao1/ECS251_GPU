from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Task:
    task_id: str
    user_id: str
    arrival_time: float
    est_duration: float
    est_mem_gb: float
    duration_class: str  # "short" or "long"


@dataclass
class RunningTask:
    task: Task
    start_time: float
    end_time: float


@dataclass
class GPUState:
    gpu_id: str
    total_mem_gb: float
    running: List[RunningTask] = field(default_factory=list)

    @property
    def used_mem_gb(self) -> float:
        return sum(t.task.est_mem_gb for t in self.running)

    @property
    def free_mem_gb(self) -> float:
        return max(0.0, self.total_mem_gb - self.used_mem_gb)


@dataclass
class AdmissionDecision:
    task: Task
    admitted: bool
    reason: str
    gpu_id: Optional[str] = None


@dataclass
class TaskResult:
    task: Task
    gpu_id: str
    start_time: float
    end_time: float
    wait_time: float


@dataclass
class ScheduleResult:
    decisions: List[AdmissionDecision]
    results: List[TaskResult]
