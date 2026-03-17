# ECS251 GPU Scheduling Project

A GPU job scheduler that manages multiple users sharing a cluster of GPUs. Users submit jobs specifying how much GPU memory and how many GPUs they need. The scheduler assigns GPUs, queues jobs when resources are full, enforces a per-job GPU limit, and respects user-defined priorities.

The project includes a live daemon that reads actual GPU state via `nvidia-smi` and dispatches jobs as real OS processes, and a mock demo mode that simulates GPUs in software so you can run everything without any GPU hardware.

---

## How the Scheduler Works

1. Users submit jobs specifying `--mem_gb`, `--num_gpus`, and optionally `--priority`.
2. The daemon polls GPU state every few seconds and tries to dispatch pending jobs.
3. **GPU assignment strategy:**
   - First, assign to fully idle GPUs (no other jobs running on them).
   - If no idle GPU has enough free memory, fall back to busy GPUs that still have room.
4. If a job requests more GPUs than the system's `--max_gpus` limit, it is **immediately rejected**.
5. If no GPU has enough free memory, the job **waits in queue**.
6. Queue order: higher `--priority` first; same priority is served in submission order (FIFO).

---

## Running the Real Scheduler (Step by Step)

### Step 1 — Start the daemon

Without a GPU (mock mode):

```bash
python -m scripts.daemon --mock --mock_gpus 2 --mock_mem_gb 24 --poll 3 --db scheduler.db
```

With real NVIDIA GPUs:

```bash
python -m scripts.daemon --db scheduler.db --poll 10 --buffer_mb 512
```

Add `--max_gpus 2` to cap how many GPUs any single job may request.

### Step 2 — Submit jobs

```bash
# Alice: 8 GB, 1 GPU, estimated 2 hours
python -m scripts.submit --user alice --mem_gb 8 --num_gpus 1 --est_hours 2 \
    --cmd "python train.py" --db scheduler.db

# Bob: 4 GB, high priority
python -m scripts.submit --user bob --mem_gb 4 --est_mins 30 --priority 2 \
    --cmd "python eval.py" --db scheduler.db

# Carol: 2 GPUs, 16 GB total
python -m scripts.submit --user carol --mem_gb 16 --num_gpus 2 --est_hours 4 \
    --cmd "python pretrain.py" --db scheduler.db
```

### Step 3 — Check status

```bash
python -m scripts.status --db scheduler.db          # all jobs + GPU state
python -m scripts.status --user alice --db scheduler.db   # filter by user
python -m scripts.status --mock --db scheduler.db   # mock GPU state
```

### Step 4 — Cancel a job

```bash
python -m scripts.cancel --job_id 3 --db scheduler.db          # cancel pending
python -m scripts.cancel --job_id 2 --force --db scheduler.db  # kill running
```

### Step 5 — Stop the daemon

Press `Ctrl+C`. The daemon finishes the current poll cycle and exits cleanly.

---

## Simulation (No GPU Needed)

Run the interactive scenario demo — no GPU or setup required:

```bash
python -m scripts.demo                   # run all four scenarios
python -m scripts.demo --scenario 1      # run one specific scenario
python -m scripts.demo --speed 3         # 3x faster
```

Each scenario is self-contained and starts from a clean state:

| Scenario | What it shows |
| --- | --- |
| 1 — No Conflict | Two idle GPUs take new jobs immediately, one job per GPU |
| 2 — GPU Sharing | New jobs share partially-used GPUs when enough memory remains |
| 3 — Conflict | GPUs are too full to admit new work, so jobs wait in the queue |
| 4 — Resolution | After running jobs finish, queued jobs are dispatched in priority order |

---

## Project Structure

```
scripts/
  daemon.py          scheduler daemon — main poll loop
  submit.py          CLI: submit a job
  status.py          CLI: view GPU state and queue
  cancel.py          CLI: cancel or kill a job
  job_store.py       SQLite job queue
  gpu_monitor.py     nvidia-smi interface + mock GPU support
  demo.py            scenario demo (no GPU needed)
docs/
  explain.md         full implementation walkthrough
  Functions.md       function reference by topic
```
