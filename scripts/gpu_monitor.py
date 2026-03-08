"""
gpu_monitor.py

Queries nvidia-smi for real-time GPU memory and process information.
Falls back to a mock mode when nvidia-smi is not available (for testing on CPU machines).
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GPUProcess:
    pid: int
    used_mem_mb: int
    process_name: str


@dataclass
class GPUInfo:
    gpu_id: int
    name: str
    total_mem_mb: int
    used_mem_mb: int
    free_mem_mb: int
    processes: List[GPUProcess] = field(default_factory=list)


def query_gpus() -> List[GPUInfo]:
    """
    Run nvidia-smi and return per-GPU memory statistics.

    Raises RuntimeError if nvidia-smi is unavailable or returns a non-zero exit code.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found. Is the NVIDIA driver installed?")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out.")

    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi exited with code {proc.returncode}: {proc.stderr.strip()}")

    gpus: List[GPUInfo] = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(GPUInfo(
            gpu_id=int(parts[0]),
            name=parts[1],
            total_mem_mb=int(parts[2]),
            used_mem_mb=int(parts[3]),
            free_mem_mb=int(parts[4]),
        ))

    return gpus


def query_gpu_processes() -> List[GPUProcess]:
    """
    Return a list of processes currently using GPU memory.
    """
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if proc.returncode != 0:
        return []

    processes: List[GPUProcess] = []
    for line in proc.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            processes.append(GPUProcess(
                pid=int(parts[0]),
                used_mem_mb=int(parts[1]),
                process_name=parts[2],
            ))
        except ValueError:
            continue

    return processes


def mock_gpus(count: int = 2, total_mem_mb: int = 24576) -> List[GPUInfo]:
    """
    Return a list of fake GPUs for testing on machines without NVIDIA GPUs.
    total_mem_mb default: 24576 = 24 GB
    """
    return [
        GPUInfo(
            gpu_id=i,
            name=f"Mock-GPU-{i}",
            total_mem_mb=total_mem_mb,
            used_mem_mb=0,
            free_mem_mb=total_mem_mb,
        )
        for i in range(count)
    ]
