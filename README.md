# ECS251 GPU Scheduling Project 🚀

A lightweight GPU job scheduler for multi-user environments.

This project supports:
- per-job GPU count and memory requests
- queueing when resources are insufficient
- priority + FIFO scheduling
- optional per-job GPU cap
- real GPU mode (`nvidia-smi`) and mock mode (no GPU required)

---

## Repository

- Code repository: `https://github.com/zhengmiao1/ECS251_GPU`
- Branch used for submission: `main`

---

## Table of Contents

- [Overview](#overview)
- [Quick Start (3 minutes)](#quick-start-3-minutes)
- [Usage](#usage)
- [Scheduling Policy](#scheduling-policy)
- [Project Plan to Code Mapping](#project-plan-to-code-mapping)
- [Simulation Mode (No GPU)](#simulation-mode-no-gpu)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides a daemon-based scheduler that continuously monitors GPU availability and dispatches queued jobs as OS processes.

Two execution modes are supported:

1. **Real mode** 🖥️: reads live GPU state via `nvidia-smi`.
2. **Mock mode** 🧪: simulates GPUs in software for development/demo on CPU-only machines.

---

## Quick Start (3 minutes) ⚡

### 1) Prerequisites ✅

- Python 3.9+ (recommended)
- Linux/macOS shell
- For real mode: NVIDIA GPU + `nvidia-smi` available in PATH

### 2) Start daemon ▶️

#### Mock mode (recommended for first run)

```bash
python -m scripts.daemon --mock --mock_gpus 2 --mock_mem_gb 24 --poll 3 --db scheduler.db
```

#### Real mode

```bash
python -m scripts.daemon --db scheduler.db --poll 10 --buffer_mb 512
```

Optional: limit per-job GPU request

```bash
python -m scripts.daemon --db scheduler.db --max_gpus 2
```

### 3) Submit jobs 📥

```bash
python -m scripts.submit --user alice --mem_gb 8 --num_gpus 1 --est_hours 2 \
  --cmd "python train.py" --db scheduler.db

python -m scripts.submit --user bob --mem_gb 4 --est_mins 30 --priority 2 \
  --cmd "python eval.py" --db scheduler.db

python -m scripts.submit --user carol --mem_gb 16 --num_gpus 2 --est_hours 4 \
  --cmd "python pretrain.py" --db scheduler.db
```

### 4) Check status 📊

```bash
python -m scripts.status --db scheduler.db
python -m scripts.status --user alice --db scheduler.db
python -m scripts.status --mock --db scheduler.db
```

### 5) Cancel job 🛑

```bash
python -m scripts.cancel --job_id 3 --db scheduler.db
python -m scripts.cancel --job_id 2 --force --db scheduler.db
```

Stop daemon with `Ctrl+C`.

---

## Usage

### CLI commands (cheat sheet)

| Command | Purpose | Example |
| --- | --- | --- |
| `scripts.daemon` | Start scheduler daemon | `python -m scripts.daemon --db scheduler.db` |
| `scripts.submit` | Submit a new job | `python -m scripts.submit --user alice --mem_gb 8 --cmd "python train.py"` |
| `scripts.status` | Inspect jobs and GPU state | `python -m scripts.status --db scheduler.db` |
| `scripts.cancel` | Cancel pending or kill running job | `python -m scripts.cancel --job_id 2 --force` |
| `scripts.demo` | Run mock scenarios | `python -m scripts.demo --scenario 1` |

---

## Scheduling Policy 🧠

When dispatching pending jobs, the scheduler applies:

1. **Validation**
   - Reject immediately if `num_gpus > --max_gpus` (when configured)

2. **Queue ordering**
   - Higher `priority` first
   - Tie-breaker: FIFO (submission order)

3. **GPU placement**
   - Prefer fully idle GPUs first
   - If none can fit, use busy GPUs with enough remaining memory

4. **Backpressure**
   - If memory is insufficient on all GPUs, keep the job pending in queue

---

## Project Plan to Code Mapping

This section maps core project-plan concepts to concrete modules in the repository.

| Project-plan concept | Implementation location | Notes |
| --- | --- | --- |
| Job submission interface | `scripts/submit.py` | Validates user request fields and writes jobs to the queue store |
| Persistent job queue | `scripts/job_store.py` | Stores pending/running/completed jobs in SQLite |
| GPU monitoring | `scripts/gpu_monitor.py` | Reads GPU state from `nvidia-smi` (real mode) or mock state |
| Scheduling loop | `scripts/daemon.py` | Polls resources, selects dispatchable jobs, and launches processes |
| Priority + FIFO policy | `scripts/daemon.py` | Higher priority first; same priority in submission order |
| Per-job GPU limit enforcement | `scripts/daemon.py` | Rejects requests beyond `--max_gpus` |
| Status inspection | `scripts/status.py` | Prints queue state and GPU utilization summary |
| Job cancellation / force stop | `scripts/cancel.py` | Cancels pending jobs or force-stops running jobs |
| No-GPU scenario simulation | `scripts/demo.py` | Reproducible mock scenarios for validation and demonstration |

---

## Simulation Mode (No GPU) 🧪

Run predefined scenarios:

```bash
python -m scripts.demo
python -m scripts.demo --scenario 1
python -m scripts.demo --speed 3
```

| Scenario | Behavior shown |
| --- | --- |
| 1. No Conflict | Jobs dispatch immediately to idle GPUs |
| 2. GPU Sharing | Jobs share partially used GPUs when memory allows |
| 3. Conflict | Jobs stay queued when resources are insufficient |
| 4. Resolution | Queued jobs dispatch after resources are released |

---

## Project Structure 📁

```text
scripts/
  daemon.py          # scheduler daemon (main poll loop)
  submit.py          # CLI: submit a job
  status.py          # CLI: query jobs + GPU state
  cancel.py          # CLI: cancel/kill a job
  job_store.py       # SQLite job queue store
  gpu_monitor.py     # nvidia-smi interface + mock GPU support
  demo.py            # no-GPU scenario demo
docs/
  explain.md         # implementation walkthrough
  Functions.md       # function reference
```

---

## Configuration ⚙️

Common daemon options:

- `--db`: SQLite database path
- `--poll`: scheduler polling interval (seconds)
- `--max_gpus`: per-job GPU request cap
- `--buffer_mb`: memory safety buffer in real mode
- `--mock`, `--mock_gpus`, `--mock_mem_gb`: mock simulation options

---

## Troubleshooting 🩺

- `nvidia-smi: command not found`
  - Install NVIDIA driver/toolkit or use `--mock` mode
- Jobs remain pending
  - Expected if GPU memory is insufficient; verify with `scripts.status`
- Command exits unexpectedly
  - Check CLI arguments and DB path consistency (`--db` should match daemon)

---

## Contributing 🤝

For course work, keep PRs small and include:

- what changed
- why it changed
- how to reproduce/test (`daemon`, `submit`, `status`, `demo` commands)

---

## License 📄

If this is a course assignment, follow your course/repository policy.
Otherwise, add a standard OSS license (for example, MIT) and include a `LICENSE` file.
