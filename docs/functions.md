# Function Summary

All functions grouped by what they do, regardless of which file they live in.

---

## 1. Job Submission & Cancellation

**Submit a job**
Users call `submit()` (in `job_store.py`) with a user ID, shell command, memory requirement, number of GPUs, estimated duration, and priority. It inserts a new row into the SQLite `jobs` table with status `pending` and returns the assigned `job_id`. The CLI wrapper in `submit.py` converts human-friendly units (GB, hours) before calling this.

**Cancel a pending job**
`cancel()` flips a `pending` job to `cancelled` in the database. It only works on jobs that have not yet started. Returns `True` if the cancellation succeeded.

**Kill a running job**
`kill_running()` marks a `running` job as `failed` in the database and returns the OS process ID so the caller can send `SIGKILL`. The `cancel.py` CLI calls `os.kill(pid, SIGKILL)` after receiving that PID.

**Reject a job immediately**
`reject()` marks a `pending` job as `failed` with a `"rejected: ..."` note. Called by the daemon when a job requests more GPUs than the system-wide limit allows — the job can never be satisfied so it is removed from the queue immediately rather than waiting forever.

---

## 2. GPU State Querying

**Query real GPUs**
`query_gpus()` runs `nvidia-smi --query-gpu` and parses the CSV output into a list of `GPUInfo` objects containing each GPU's index, name, total memory, used memory, and free memory.

**Query GPU processes**
`query_gpu_processes()` runs `nvidia-smi --query-compute-apps` to list all processes currently consuming GPU memory, returning their PID, memory usage, and process name.

**Simulate GPUs without hardware**
`mock_gpus(count, total_mem_mb)` creates a list of fake `GPUInfo` objects for testing on machines without NVIDIA GPUs. The daemon updates their `used_mem_mb` and `free_mem_mb` each poll cycle based on running-job accounting.

---

## 3. Memory Accounting (Real Scheduler)

**Track committed memory per GPU**
`_committed_per_gpu()` queries all running jobs from the database and sums how much memory each GPU has been promised. For multi-GPU jobs the total memory is split evenly across the assigned GPUs. This tells the scheduler which GPUs are idle versus busy.

**Compute effective free memory**
`_effective_free(gpus)` adjusts nvidia-smi's reported free memory by subtracting the memory of jobs launched within the last `grace_secs` seconds. This prevents double-booking during the lag between when a job starts and when its GPU memory consumption appears in nvidia-smi.

---

## 4. GPU Assignment

**Pick GPUs for a job (real scheduler)**
`_pick_gpus(needed_mb, num_gpus, free, committed)` uses a two-tier strategy:
1. First, look for fully idle GPUs (no committed memory) that each have enough room for their share of the job.
2. If there are not enough idle GPUs, fall back to busy GPUs that still have sufficient free memory.
Within each tier, GPUs are sorted by descending free memory so the most headroom is preserved. Returns a sorted list of GPU IDs or `None` if no valid assignment exists.

**Pick a GPU for a task (simulation — memory-aware policy)**
`_pick_gpu(task)` in `MemoryAwareScheduler` finds all GPUs with `free_mem_gb >= task.est_mem_gb` and returns the one with the most free memory (best-fit descending). Returns `None` if no GPU fits.

**Pick a GPU for a task (simulation — FIFO policy)**
`_pick_gpu(task)` in `FIFOScheduler` finds all eligible GPUs and returns the one with the lowest GPU ID, giving deterministic, round-robin-style placement.

---

## 5. Job Ordering & Priority

**Order the pending queue (real scheduler)**
`get_pending()` fetches all pending jobs from the database sorted by `priority DESC, submitted_at ASC`. Higher-priority jobs go first; within the same priority level, jobs are served in submission order (FIFO).

**Order the pending queue (simulation — memory-aware policy)**
`_order_ready(ready, now)` applies three ordering rules:
1. Any task that has waited longer than `aging_window` seconds is promoted to the front (anti-starvation).
2. Short tasks (estimated duration ≤ `short_threshold`) come before long tasks.
3. Ties are broken by arrival time.

**Order the pending queue (simulation — FIFO policy)**
`_order_ready(ready, now)` in `FIFOScheduler` sorts purely by `(arrival_time, task_id)` with no short-task preference or aging.

---

## 6. Job Dispatching & Lifecycle (Real Scheduler)

**Launch a job**
`_launch(job, gpu_ids)` forks a subprocess running the job's shell command with `CUDA_VISIBLE_DEVICES` set to the assigned GPU IDs. It also injects `SCHEDULER_JOB_ID` and `SCHEDULER_USER` as environment variables. Returns the process PID on success.

**Check for finished jobs**
`_check_finished()` calls `poll()` on every tracked subprocess. For any that have exited it calls `mark_done()` with the exit code and removes the job from internal tracking.

**Reap orphaned jobs**
`_reap_orphaned()` runs once at daemon startup. It looks for jobs marked `running` in the database that are not owned by the current daemon process, checks whether their PIDs still exist using `psutil`, and marks them `failed` if the process is gone. This handles the case where the daemon was restarted after a crash.

**Mark a job running**
`mark_running(job_id, gpu_id, pid)` updates the database row to status `running`, recording the assigned GPU(s), PID, and start timestamp.

**Mark a job done or failed**
`mark_done(job_id, exit_code, note)` sets status to `done` if exit code is 0, otherwise `failed`. Records the end timestamp and an optional note.

---

## 7. Main Scheduling Loop (Real Scheduler)

**One scheduling cycle**
`_schedule(gpus)` is called each poll interval. It processes every pending job in priority order:
- If the job requests more GPUs than the `max_gpus` limit, it is immediately rejected.
- Otherwise, `_pick_gpus` is called to find a valid GPU assignment using the idle-first strategy.
- If an assignment is found, the job is launched and the in-loop free/committed accounting is updated so later jobs in the same cycle see accurate state.
- If no GPUs are available, the job stays pending until the next cycle.

**Main daemon loop**
`run()` is the blocking entry point. It registers signal handlers, calls `_reap_orphaned` once, then loops forever: query GPUs → check finished jobs → log GPU state → run one scheduling cycle → sleep for `poll_interval` seconds.

---

## 8. Simulation Engine

**Generate a synthetic workload**
`generate_tasks(n, users, seed, short_threshold, workload)` produces `n` tasks with Poisson-distributed inter-arrivals. The `workload` mode controls the fraction of memory-heavy long tasks: `mixed` = 40%, `llm_heavy` = 20%, `vlm_heavy` = 80%. Users are assigned randomly.

**Run the discrete-event simulation**
`schedule(tasks)` in `MemoryAwareScheduler` / `FIFOScheduler` processes all tasks in arrival-time order. At each simulated time step it: removes finished tasks from GPUs, collects newly arrived tasks, sorts the pending queue, then admits or defers each task. When nothing can be admitted it fast-forwards simulation time to the next GPU free event or task arrival.

**Replay a stored run**
`_replay_run(db_path, run_id, ...)` reconstructs the task list from a saved SQLite experiment, re-runs the original scheduler policy, and compares replayed metrics against stored values to verify determinism.

---

## 9. Performance Metrics

**Merge intervals**
`_merge_coverage(intervals)` merges a list of possibly-overlapping `(start, end)` time intervals and returns the total covered duration. Used internally to compute GPU utilization without double-counting jobs that run concurrently.

**Compute all scheduling metrics**
`summarize_results(result, num_gpus)` takes a completed simulation result and returns a dict of eight metrics:
- `completed_tasks` — number of tasks that ran
- `avg_wait_time` — mean time from arrival to dispatch
- `p95_wait_time` — 95th-percentile wait time
- `avg_turnaround` — mean time from arrival to completion
- `throughput` — tasks per second over the makespan
- `utilization` — fraction of total GPU-time that was busy
- `fairness_wait_std` — standard deviation of per-user average wait times
- `oom_events` — number of tasks rejected for exceeding GPU memory capacity

---

## 10. Priority-Inversion Analysis

**Detect inversions**
`detect_priority_inversions(result, short_threshold, aging_window)` scans a simulation result for cases where a short task waited while a long task that arrived later was dispatched first. Returns a list of `PriorityInversionEvent` records, each noting whether the aging mechanism eventually resolved the inversion.

**Print the inversion report**
`print_inversion_report(events, policy_name)` summarizes inversion counts (total, resolved by aging, unresolved), average short-task wait time, and up to five sample unresolved inversions.

**Print the formal policy spec**
`print_policy_spec()` prints the pseudocode specification for the memory-aware scheduling policy, covering admission control rules, execution ordering, and correctness properties (no OOM, no starvation, short-task preference).

---

## 11. Experiment Running & Persistence

**Run a batch experiment**
`run_experiment(seeds, tasks, users, gpus, gpu_mem, ...)` runs both `MemoryAwareScheduler` and `FIFOScheduler` across multiple random seeds for a given workload, averages the metrics, and optionally persists all decisions and results to a SQLite database via `ExperimentStore`.

**Grid sweep over hyperparameters**
`sweep(short_thresholds, aging_windows, seeds, ...)` runs the memory-aware scheduler over every combination of `short_threshold` and `aging_window` values, averaged across multiple seeds. Returns one result row per parameter pair.

**Store experiment results**
`insert_run`, `insert_decisions`, `insert_results` in `ExperimentStore` write one experiment run, its admission decisions, and its task results into SQLite for offline querying.

**Query stored results**
`query_runs`, `per_user_wait_stats`, `per_gpu_utilization`, `wait_distribution`, `gpu_utilization_timeline`, `compare_runs` in `ExperimentStore` provide various views over stored experiment data — per-user fairness, GPU busy time, wait-time histograms, time-series utilization, and direct policy comparisons.

---

## 12. Reporting & Visualization

**Generate a markdown report**
`generate_evaluation_section(csv_dir)` reads per-workload CSV files produced by the experiment runner and assembles a complete markdown Evaluation section with a setup table, per-workload metric tables with percent-change columns, narrative findings paragraphs, aggregated overall findings, and a threats-to-validity discussion.

**Print a markdown comparison table**
`print_markdown_table(workload_label, data)` prints a single markdown table comparing all policy metrics for one workload, suitable for pasting into a report.

**Plot bar charts**
`plot_workload(workload_label, data, out_dir)` generates a multi-panel bar chart (one subplot per metric) comparing scheduler policies, annotates each bar with its value, highlights the winner per metric, and saves the figure as a PNG.

---

## 13. Logging & Display Utilities

**Log a scheduling event**
`JsonlEventLogger.log(event_type, **payload)` writes a JSON object to a `.jsonl` file with the event type and any keyword arguments. Used during simulation to record dispatch, defer, and reject decisions for offline analysis.

**Format memory as a bar**
`_mem_bar(used, total, width)` renders an ASCII progress bar like `[####----] 8192/24576 MB (33%)` for terminal display.

**Format duration as text**
`_fmt_secs(secs)` converts a number of seconds into a compact human-readable string such as `"1h30m"`, `"45m00s"`, or `"30s"`.

**Apply terminal colors**
`_color(status, text)` wraps text in ANSI escape codes matching the job status (yellow for pending, green for running, blue for done, red for failed, magenta for rejected).

**Print a live status snapshot**
`print_snapshot(store, gpus_cfg, title, t0)` (demo) and `print_gpu_state` / `print_jobs` (status CLI) render the current GPU memory bars and job table to the terminal, showing running, queued, and completed jobs at a glance.
