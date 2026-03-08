"""
submit.py  –  Submit a job to the GPU scheduler queue.

The scheduler daemon picks up pending jobs on its next poll cycle and
dispatches them to a GPU with sufficient free memory.

Usage examples
--------------
    # 8 GB, estimated 2 hours
    python -m scripts.submit --user alice --mem_gb 8 --est_hours 2 --cmd "python train.py"

    # 4 GB, estimated 30 minutes, higher priority
    python -m scripts.submit --user bob --mem_gb 4 --est_mins 30 --cmd "python eval.py" --priority 1

    # Minimal (no time estimate – job will still be scheduled, just ordered last among equals)
    python -m scripts.submit --user carol --mem_gb 12 --cmd "bash run.sh"
"""
from __future__ import annotations

import argparse

from .job_store import JobStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a job to the GPU scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default="scheduler.db",
                        help="Scheduler database path (default: scheduler.db)")
    parser.add_argument("--user", required=True,
                        help="User ID (e.g. alice)")
    parser.add_argument("--cmd", required=True,
                        help="Shell command to execute (quoted if it contains spaces)")
    parser.add_argument("--mem_gb", type=float, required=True,
                        help="Estimated peak GPU memory in GB")
    parser.add_argument("--est_hours", type=float, default=0.0,
                        help="Estimated duration in hours")
    parser.add_argument("--est_mins", type=float, default=0.0,
                        help="Estimated duration in minutes (additive with --est_hours)")
    parser.add_argument("--priority", type=int, default=0,
                        help="Scheduling priority: higher value = dispatched sooner (default: 0)")
    args = parser.parse_args()

    mem_mb = int(args.mem_gb * 1024)
    est_secs = args.est_hours * 3600.0 + args.est_mins * 60.0

    store = JobStore(args.db)
    job_id = store.submit(
        user_id=args.user,
        cmd=args.cmd,
        mem_mb=mem_mb,
        est_secs=est_secs,
        priority=args.priority,
    )
    store.close()

    print(f"Submitted  job_id={job_id}  user={args.user}  mem={mem_mb} MB  "
          f"est={est_secs:.0f}s  priority={args.priority}")
    print(f"  cmd: {args.cmd}")
    print(f"The daemon will dispatch this job on its next poll cycle.")


if __name__ == "__main__":
    main()
