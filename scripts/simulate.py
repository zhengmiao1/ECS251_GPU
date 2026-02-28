from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional

from .event_logger import JsonlEventLogger
from .models import Task
from .scheduler import FIFOScheduler, MemoryAwareScheduler, SchedulerConfig, build_default_gpus
from .metrics import summarize_results


def _duration_class(duration: float, threshold: float) -> str:
    return "short" if duration <= threshold else "long"


def generate_tasks(
    n: int,
    users: int,
    seed: int,
    short_threshold: float,
    workload: str,
) -> List[Task]:
    random.seed(seed)
    tasks: List[Task] = []
    now = 0.0
    for i in range(n):
        inter_arrival = random.expovariate(1.0 / 8.0)
        now += inter_arrival

        if workload == "llm_heavy":
            long_prob = 0.2
        elif workload == "vlm_heavy":
            long_prob = 0.8
        else:
            long_prob = 0.4

        if random.random() < long_prob:
            duration = random.uniform(120, 900)
            mem_gb = random.choice([14, 16, 20, 24])
        else:
            duration = random.uniform(10, 80)
            mem_gb = random.choice([6, 8, 10, 12])

        task = Task(
            task_id=f"t{i}",
            user_id=f"u{random.randint(1, users)}",
            arrival_time=now,
            est_duration=duration,
            est_mem_gb=mem_gb,
            duration_class=_duration_class(duration, short_threshold),
        )
        tasks.append(task)
    return tasks


def _load_tasks_from_db(db_path: str, run_id: int) -> tuple[List[Task], str, int, int, float]:
    """
    Reconstruct the original task list from a stored run in SQLite.
    Returns (tasks, policy, gpus, seed, gpu_mem_gb).
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    run_row = conn.execute(
        "SELECT policy, gpus, seed, gpu_mem_gb FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if run_row is None:
        conn.close()
        raise ValueError(f"run_id={run_id} not found in {db_path}")

    policy = run_row["policy"]
    gpus = run_row["gpus"]
    seed = run_row["seed"]
    gpu_mem_gb = run_row["gpu_mem_gb"]

    decision_rows = conn.execute(
        "SELECT task_id, user_id, arrival_time, est_duration, est_mem_gb, duration_class "
        "FROM decisions WHERE run_id = ? ORDER BY arrival_time",
        (run_id,),
    ).fetchall()
    conn.close()

    seen: dict[str, Task] = {}
    tasks: List[Task] = []
    for row in decision_rows:
        tid = row["task_id"]
        if tid not in seen:
            t = Task(
                task_id=tid,
                user_id=row["user_id"],
                arrival_time=row["arrival_time"],
                est_duration=row["est_duration"],
                est_mem_gb=row["est_mem_gb"],
                duration_class=row["duration_class"],
            )
            seen[tid] = t
            tasks.append(t)

    return tasks, policy, gpus, seed, gpu_mem_gb


def _replay_run(
    db_path: str,
    run_id: int,
    short_threshold: float,
    aging_window: float,
) -> None:
    """
    Reconstruct and re-execute a stored run; compare replayed metrics to stored ones.
    """
    from .db_store import ExperimentStore
    from .metrics import summarize_results as _summarize

    tasks, policy, gpus, seed, gpu_mem_gb = _load_tasks_from_db(db_path, run_id)

    print(f"\n[replay] run_id={run_id}  policy={policy}  gpus={gpus}  "
          f"gpu_mem={gpu_mem_gb} GB  seed={seed}  tasks={len(tasks)}")

    gpu_list = build_default_gpus(gpus, gpu_mem_gb)
    config = SchedulerConfig(short_threshold=short_threshold, aging_window=aging_window)
    if policy == "memory":
        scheduler = MemoryAwareScheduler(gpus=gpu_list, config=config)
    else:
        scheduler = FIFOScheduler(gpus=gpu_list, config=config)

    replayed_result = scheduler.schedule(tasks)
    replayed_metrics = _summarize(replayed_result, num_gpus=gpus)

    # Load stored metrics from DB
    store = ExperimentStore(db_path)
    stored_rows = store._conn.execute(
        "SELECT wait_time, end_time - arrival_time AS turnaround "
        "FROM task_results WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    store.close()

    if stored_rows:
        n = len(stored_rows)
        stored_avg_wait = sum(r["wait_time"] for r in stored_rows) / n
        stored_avg_ta = sum(r["turnaround"] for r in stored_rows) / n
    else:
        stored_avg_wait = stored_avg_ta = 0.0

    # Query stored OOM count (store is already closed — open a fresh connection)
    _store2 = ExperimentStore(db_path)
    stored_oom = float(_store2._conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE run_id=? AND admitted=0", (run_id,)
    ).fetchone()[0])
    _store2.close()

    comparisons = [
        ("completed_tasks", float(len(stored_rows)),   float(replayed_metrics["completed_tasks"])),
        ("avg_wait_time",   round(stored_avg_wait, 3), replayed_metrics["avg_wait_time"]),
        ("avg_turnaround",  round(stored_avg_ta, 3),   replayed_metrics["avg_turnaround"]),
        ("oom_events",      stored_oom,                float(replayed_metrics["oom_events"])),
    ]

    print(f"\n{'Metric':<22} {'Stored':>12} {'Replayed':>12} {'Match':>8}")
    print("-" * 58)
    for metric, stored_val, replayed_val in comparisons:
        match = "OK" if abs(stored_val - replayed_val) < 1e-3 else "DIFF"
        print(f"{metric:<22} {stored_val:>12.3f} {replayed_val:>12.3f} {match:>8}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--users", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--gpu_mem", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--short_threshold", type=float, default=60.0)
    parser.add_argument("--aging_window", type=float, default=180.0)
    parser.add_argument("--policy", choices=["memory", "fifo", "both"], default="both")
    parser.add_argument("--workload", choices=["mixed", "llm_heavy", "vlm_heavy"], default="mixed")
    parser.add_argument("--log_dir", type=str, default="")
    # Replay mode (Zaishuo Xia, Week 4)
    parser.add_argument(
        "--replay",
        type=str,
        default="",
        metavar="DB_PATH",
        help="SQLite DB path; replays the run specified by --run_id",
    )
    parser.add_argument(
        "--run_id",
        type=int,
        default=1,
        help="run_id to replay from --replay DB",
    )
    args = parser.parse_args()

    # --replay mode: reconstruct + re-execute a stored run
    if args.replay:
        _replay_run(
            db_path=args.replay,
            run_id=args.run_id,
            short_threshold=args.short_threshold,
            aging_window=args.aging_window,
        )
        return

    tasks = generate_tasks(
        n=args.tasks,
        users=args.users,
        seed=args.seed,
        short_threshold=args.short_threshold,
        workload=args.workload,
    )

    policies = ["memory", "fifo"] if args.policy == "both" else [args.policy]

    for policy in policies:
        logger = None
        if args.log_dir:
            log_file = Path(args.log_dir) / f"{policy}_seed{args.seed}.jsonl"
            logger = JsonlEventLogger(str(log_file))

        gpus = build_default_gpus(args.gpus, args.gpu_mem)
        if policy == "memory":
            scheduler = MemoryAwareScheduler(
                gpus=gpus,
                config=SchedulerConfig(
                    short_threshold=args.short_threshold,
                    aging_window=args.aging_window,
                ),
                logger=logger,
            )
        else:
            scheduler = FIFOScheduler(
                gpus=gpus,
                config=SchedulerConfig(
                    short_threshold=args.short_threshold,
                    aging_window=args.aging_window,
                ),
                logger=logger,
            )

        result = scheduler.schedule(tasks)
        report = summarize_results(result, num_gpus=args.gpus)

        print(f"\npolicy={policy}, workload={args.workload}, seed={args.seed}")
        for k, v in report.items():
            print(f"{k}: {v}")

        if logger:
            logger.close()


if __name__ == "__main__":
    main()
