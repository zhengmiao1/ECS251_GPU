# Detailed Explanation: Real GPU Scheduler (SLURM-like)

## Overview

The real GPU scheduler is a multi-component system that works like a lightweight
SLURM for a single machine.  It monitors live GPU memory via `nvidia-smi`, accepts
job submissions from users, and dispatches jobs to GPUs that have enough free
memory — including GPUs that are already partially occupied by other jobs.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    scheduler.db  (SQLite)                 │
│  jobs table: pending → running → done / failed / cancelled│
└──────────────────────────────────────────────────────────┘
          ▲  write                          ▲  write
          │                                 │
  submit.py / cancel.py            daemon.py (main loop)
  (user-facing CLI)                    │
                                        │ reads nvidia-smi
                                        ▼
                                  gpu_monitor.py
                                  (subprocess → nvidia-smi)
```

Five Python modules work together:

| File | Role |
|---|---|
| `gpu_monitor.py` | Query `nvidia-smi` for per-GPU memory stats |
| `job_store.py` | SQLite-backed persistent job queue |
| `daemon.py` | Main scheduling loop (runs continuously) |
| `submit.py` | CLI: users submit new jobs |
| `cancel.py` | CLI: cancel pending or kill running jobs |
| `status.py` | CLI: inspect GPU state and job queue |

---

## Component Deep-Dive

### 1. `gpu_monitor.py` — Live GPU Telemetry

Calls `nvidia-smi` with `--query-gpu` and `--format=csv,noheader,nounits` to get
structured output:

```
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
           --format=csv,noheader,nounits
```

Returns a list of `GPUInfo` dataclasses:
```
GPUInfo(gpu_id=0, name="NVIDIA A100", total_mem_mb=81920, used_mem_mb=12288, free_mem_mb=69632)
```

A `mock_gpus()` function creates fake GPU state for testing on machines without
NVIDIA hardware (e.g., MacBooks).

---

### 2. `job_store.py` — Persistent Job Queue

SQLite database (`scheduler.db`) with WAL mode for concurrent access.

**Schema (jobs table):**

| Column | Type | Description |
|---|---|---|
| `job_id` | INTEGER PK | Auto-increment |
| `user_id` | TEXT | Who submitted |
| `cmd` | TEXT | Shell command |
| `mem_mb` | INTEGER | Requested GPU memory |
| `est_secs` | REAL | Estimated runtime (scheduling hint) |
| `priority` | INTEGER | Higher = scheduled sooner |
| `status` | TEXT | `pending` / `running` / `done` / `failed` / `cancelled` |
| `gpu_id` | INTEGER | Assigned GPU index |
| `pid` | INTEGER | OS process ID |
| `submitted_at` | TEXT | ISO timestamp |
| `started_at` | TEXT | ISO timestamp |
| `ended_at` | TEXT | ISO timestamp |
| `exit_code` | INTEGER | Process exit code |

**Pending job ordering** (most important for scheduling fairness):
```sql
ORDER BY priority DESC, est_secs ASC, submitted_at ASC
```
- High-priority jobs first
- Among equal priority: shorter estimated runtime first (SRTF)
- FIFO tiebreak on submission time

---

### 3. `daemon.py` — The Scheduler Loop

The daemon runs an infinite poll loop:

```
while not stop:
    gpus = query_gpus()          # 1. get live GPU memory
    check_finished()             # 2. reap completed jobs
    effective_free = compute()   # 3. adjust for grace period
    schedule_pending(gpus)       # 4. dispatch pending jobs
    sleep(poll_interval)
```

#### Memory Accounting Problem

There is a race condition between when we start a job and when `nvidia-smi` reflects
its memory usage.  A newly launched PyTorch process takes 5–20 seconds to claim GPU
memory (CUDA context init).  Without correction, the daemon could dispatch two jobs
to the same GPU before either shows up in `nvidia-smi`, causing OOM.

**Solution: grace period tracking**

```python
_recent: Dict[job_id → (gpu_id, mem_mb, launch_epoch)]
```

When computing effective free memory for scheduling:
```
effective_free[gpu] = nvidia-smi free_mem
                    - sum(mem_mb for jobs in _recent if now - launch < grace_secs)
```

After `grace_secs` (default 30 s), the job's memory is visible in `nvidia-smi` and
we stop subtracting it from `_recent`.

#### GPU Selection: Most-Free-First

```python
candidates = {gpu: free for gpu, free in free.items() if free - buffer_mb >= needed_mb}
return max(candidates, key=lambda g: candidates[g])
```

Choosing the GPU with the *most* free memory (rather than least) tends to:
- Leave smaller GPUs free for large future jobs
- Reduce fragmentation across time

#### Safety Buffer

A fixed `buffer_mb` (default 512 MB) is subtracted from each GPU's free memory before
scheduling.  This absorbs:
- CUDA driver overhead (~100–200 MB)
- Memory estimation errors in user submissions
- Temporary fragmentation from CUDA allocator

#### Job Ordering in the Pending Queue

The `get_pending()` query returns jobs ordered by `priority DESC, est_secs ASC, submitted_at ASC`.

This implements **Shortest Remaining Time First (SRTF)** within each priority level.
Long-running jobs are not starved because users can submit them with `--priority 1` or higher.

---

### 4. `submit.py` — Job Submission

Users specify:
- `--user`: their identity (arbitrary string)
- `--mem_gb`: peak GPU memory needed (converted to MB internally)
- `--cmd`: the shell command to run
- `--est_hours` / `--est_mins`: estimated duration (used for ordering)
- `--priority`: optional urgency override

The job is written to the SQLite DB in `pending` state.  The daemon picks it up on
its next poll cycle (within `poll_interval` seconds).

---

### 5. Daemon: Job Launch

When a job is dispatched, the daemon calls `subprocess.Popen` with:
```python
env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)   # restricts the job to one GPU
env["SCHEDULER_JOB_ID"] = str(job_id)       # passthrough for logging
```

Using `shell=True` so the `cmd` string is interpreted by `/bin/sh`, supporting
pipelines, variable expansion, etc.

stdout/stderr are redirected to `/dev/null`.  For production use, redirect to
per-job log files using: `cmd = f"{user_cmd} > logs/job_{job_id}.log 2>&1"`.

---

### 6. GPU Sharing

Unlike exclusive-GPU schedulers (SLURM's default), this scheduler assigns jobs to
GPUs based purely on memory availability.  Multiple jobs can run on the same GPU
simultaneously, as long as the total committed memory fits.

Example:
```
GPU 0: 24 GB total
  - job 3: 8 GB   (running)
  - job 5: 10 GB  (running)
  Available: 24 - 8 - 10 - 0.5 (buffer) = 5.5 GB → can still accept a 4 GB job
```

This is the key advantage over SLURM's default exclusive mode and matches how
inference workloads (vLLM, TGI) actually run on shared GPU clusters.

---

## Data Flow: End-to-End

```
User runs:
  python -m scripts.submit --user alice --mem_gb 8 --cmd "python train.py"

  → JobStore.submit() inserts row into jobs (status=pending)
  → prints: "Submitted job_id=7"

Daemon (running in background) on next poll:
  → query_gpus() → GPU0: free=12000MB, GPU1: free=6000MB
  → get_pending() → [job7 (8192MB)]
  → pick_gpu(8192) → GPU0 (12000MB free, most free)
  → Popen("python train.py", env={CUDA_VISIBLE_DEVICES=0})
  → mark_running(job_id=7, gpu_id=0, pid=18432)
  → _recent[7] = (0, 8192, now)
  → log: "Dispatched job 7 user=alice mem=8192MB gpu=0 pid=18432"

Daemon on subsequent polls:
  → proc.poll() returns None → still running
  → nvidia-smi now shows GPU0: free=3500MB (job claimed memory)
  → grace period expires → _recent[7] removed
  → GPU0 effective free = 3500MB - 512MB buffer = 2988MB

When alice's job finishes:
  → proc.poll() returns 0 → mark_done(job_id=7, exit_code=0)
  → log: "Job 7 finished exit_code=0"
  → GPU0 memory freed in next nvidia-smi poll
```

---

## Mock Mode (No GPU Machine)

Run with `--mock` to simulate GPU state entirely in software:

```bash
python -m scripts.daemon --mock --mock_gpus 2 --mock_mem_gb 24 --poll 3
```

In mock mode, GPU free memory is computed from the daemon's own job tracking
rather than `nvidia-smi`.  This allows full end-to-end testing of scheduling
logic on any machine (including MacBooks without NVIDIA GPUs).

---

## Comparison with the Simulation (`scheduler.py`)

| Aspect | Simulation (`scheduler.py`) | Real Daemon (`daemon.py`) |
|---|---|---|
| GPU state | Synthetic `GPUState` objects | Live `nvidia-smi` |
| Job execution | Instant (simulated time) | Real OS processes |
| Memory tracking | Exact (est_mem_gb) | Estimated + grace period |
| Persistence | None (in-memory) | SQLite WAL |
| Multi-user | Simulated user IDs | Real OS users |
| Time | Discrete event simulation | Wall-clock |

The simulation is used for policy evaluation (comparing FIFO vs memory-aware);
the daemon is the actual deployable scheduler.

---

## Scheduling Policy Summary

```
ADMIT(job):
  for each gpu in GPUs:
    effective_free = nvidia_smi_free[gpu] - recently_committed[gpu]
    if effective_free - buffer_mb >= job.mem_mb:
      candidate_gpus.add(gpu)
  if candidate_gpus is empty: defer (stay pending)
  else: dispatch to argmax(effective_free) in candidate_gpus

ORDER(pending):
  sort by (priority DESC, est_secs ASC, submitted_at ASC)
```

Properties:
- **No OOM**: job is dispatched only when effective free ≥ mem_mb + buffer
- **GPU sharing**: multiple jobs per GPU allowed up to memory limit
- **Short-first**: minimises average wait time for quick jobs
- **No starvation**: users can set higher priority; priority accumulation can be
  added later (aging)
- **Crash safety**: SQLite WAL survives daemon restarts; orphaned jobs are reaped
