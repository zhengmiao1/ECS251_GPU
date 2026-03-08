## ECS251 GPU Scheduling Project

This repository contains two related systems:

1. **Simulation** (`scheduler.py`, `simulate.py`, `experiment.py`) — discrete-event
   simulator for comparing memory-aware vs FIFO scheduling policies on synthetic LLM/VLM workloads.

2. **Real GPU Scheduler** (`daemon.py`, `submit.py`, `status.py`, `cancel.py`) —
   a deployable SLURM-like daemon that monitors live GPU memory via `nvidia-smi` and
   dispatches user jobs to GPUs with sufficient free memory, supporting GPU sharing.

---

### Project Structure

```
scripts/
  models.py          core data structures (simulation)
  scheduler.py       memory-aware policy + FIFO baseline (simulation)
  metrics.py         evaluation metrics (simulation)
  simulate.py        synthetic workload generator/runner
  experiment.py      multi-seed baseline experiments
  plot_results.py    bar charts + markdown tables from experiment CSVs
  analysis.py        priority-inversion detection + formal policy spec
  db_store.py        SQLite persistence for experiment runs
  event_logger.py    JSONL event logger for admission/dispatch traces
  gpu_monitor.py     nvidia-smi interface (real scheduler)
  job_store.py       SQLite job queue (real scheduler)
  daemon.py          scheduler daemon – main loop (real scheduler)
  submit.py          CLI: submit a job (real scheduler)
  status.py          CLI: view GPU state + queue (real scheduler)
  cancel.py          CLI: cancel/kill a job (real scheduler)
docs/
  PROPOSAL.md        project proposal
  explain.md         detailed implementation explanation
```

---

## Quickest Demo (self-contained, no GPU needed)

Run the interactive mock demo — simulates 5 users submitting 8 jobs with a live
dashboard showing queue/running/done state at each stage:

```bash
python -m scripts.demo                 # ~35 second demo
python -m scripts.demo --speed 4      # 4x faster (~9 seconds)
python -m scripts.demo --gpus 1       # single GPU — more contention, longer queues
```

---

## Real GPU Scheduler — Step-by-Step

### Step 1: Start the daemon

**With real NVIDIA GPUs:**
```bash
python -m scripts.daemon --db scheduler.db --poll 10 --buffer_mb 512
```

**Without a GPU (mock mode for testing):**
```bash
python -m scripts.daemon --mock --mock_gpus 2 --mock_mem_gb 24 --poll 3 --db scheduler.db
```

Leave this running in a terminal.  It will log GPU state and scheduling decisions
every `--poll` seconds.

---

### Step 2: Submit jobs

Open another terminal and submit jobs from different users:

```bash
# Alice submits a training job needing 8 GB, estimated 2 hours
python -m scripts.submit --user alice --mem_gb 8 --est_hours 2 \
    --cmd "python train.py --dataset imagenet" --db scheduler.db

# Bob submits a quick evaluation needing 4 GB, 30 minutes
python -m scripts.submit --user bob --mem_gb 4 --est_mins 30 \
    --cmd "python eval.py --checkpoint ckpt.pt" --db scheduler.db

# Carol submits a large job needing 20 GB
python -m scripts.submit --user carol --mem_gb 20 --est_hours 4 \
    --cmd "python pretrain.py --config large.yaml" --db scheduler.db

# Dave submits an urgent job with higher priority
python -m scripts.submit --user dave --mem_gb 6 --est_mins 15 \
    --cmd "python inference.py" --priority 1 --db scheduler.db
```

The daemon dispatches each job to a GPU with enough free memory (including GPUs
that already have other jobs running, as long as memory fits).

---

### Step 3: Check status

```bash
# View GPU state + all queued/running/completed jobs
python -m scripts.status --db scheduler.db

# Filter by user
python -m scripts.status --user alice --db scheduler.db

# Show all history
python -m scripts.status --all --db scheduler.db

# Mock mode status (no NVIDIA GPU)
python -m scripts.status --mock --db scheduler.db
```

Example output:
```
=== GPU State ===
  GPU 0  NVIDIA A100-SXM4         [########--------]  12288/24576 MB (50%)
  GPU 1  NVIDIA A100-SXM4         [####------------]   6144/24576 MB (25%)

=== Running (3) ===
  [   1] RUNNING    user=alice      mem= 8192MB  est=  7200s  pri=0  gpu=0  pid=18432  started=2026-03-07T10:00:05
           cmd: python train.py --dataset imagenet
  [   2] RUNNING    user=bob        mem= 4096MB  est=  1800s  pri=0  gpu=1  pid=18445  started=2026-03-07T10:00:05
           cmd: python eval.py --checkpoint ckpt.pt
  [   4] RUNNING    user=dave       mem= 6144MB  est=   900s  pri=1  gpu=0  pid=18460  started=2026-03-07T10:00:15
           cmd: python inference.py

=== Pending (1) ===
  [   3] PENDING    user=carol      mem=20480MB  est= 14400s  pri=0  gpu=?  submitted=2026-03-07T10:00:10
           cmd: python pretrain.py --config large.yaml
```

Carol's 20 GB job stays pending until a GPU frees up enough memory.

---

### Step 4: Cancel a job

```bash
# Cancel a pending job
python -m scripts.cancel --job_id 3 --db scheduler.db

# Force-kill a running job (sends SIGKILL)
python -m scripts.cancel --job_id 2 --force --db scheduler.db
```

---

### Step 5: Stop the daemon

Press `Ctrl+C` in the daemon terminal.  The daemon catches `SIGINT` and shuts down
cleanly after the current poll cycle.

---

## GPU Sharing Example

Unlike exclusive schedulers, this daemon allows multiple jobs on the same GPU:

```
GPU 0  24 GB total
  job 1  alice   8 GB  (running)
  job 4  dave    6 GB  (running)
  free:  24 - 8 - 6 - 0.5 buffer = 9.5 GB  →  can accept up to ~9 GB more
```

The daemon will keep assigning jobs to GPU 0 until memory is exhausted,
then fall back to GPU 1, then defer until memory becomes available.

---

## Simulation (Original)

Quick single run comparing memory-aware vs FIFO:
```bash
python -m scripts.simulate --tasks 120 --gpus 2 --gpu_mem 24 --policy both --workload mixed
```

Multi-seed batch experiment:
```bash
python -m scripts.experiment --tasks 200 --gpus 2 --gpu_mem 24 --batch --out_csv results/
```

Generate comparison charts:
```bash
python -m scripts.plot_results --csv_dir results/ --out_dir figures/
```

Priority-inversion analysis + formal policy spec:
```bash
python -m scripts.analysis --tasks 100 --seed 7 --workload mixed
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Memory source | `nvidia-smi` | Ground truth; handles any CUDA workload |
| GPU sharing | Allowed | Maximises utilisation for inference/mixed workloads |
| Job ordering | FIFO + priority | Fair ordering within priority tier; no starvation via explicit priority |
| Memory buffer | 512 MB default | Absorbs CUDA driver overhead + estimation errors |
| Grace period | 30 s default | Prevents double-booking during CUDA context init lag |
| Persistence | SQLite WAL | Survives daemon restart; concurrent read-access |

See `docs/explain.md` for the full implementation walkthrough.
