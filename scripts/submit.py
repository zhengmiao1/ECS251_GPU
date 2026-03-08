"""
submit.py  –  Submit a job to the GPU scheduler queue.

The scheduler daemon picks up pending jobs on its next poll cycle and
dispatches them to GPU(s) with sufficient free memory.

Usage examples
--------------
    # Single GPU, 8 GB, estimated 2 hours
    python -m scripts.submit --user alice --mem_gb 8 --est_hours 2 --cmd "python train.py"

    # Explicitly request 2 GPUs with 16 GB total (8 GB per GPU)
    python -m scripts.submit --user bob --mem_gb 16 --num_gpus 2 --est_hours 4 --cmd "python pretrain.py"

    # High priority, 30-minute eval
    python -m scripts.submit --user carol --mem_gb 4 --est_mins 30 --cmd "python eval.py" --priority 1
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
                        help="Total GPU memory needed in GB (split evenly across --num_gpus GPUs)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to request (default: 1). "
                             "Scheduler finds N GPUs each with at least mem_gb/N free. "
                             "Jobs requesting more than the daemon's --max_gpus limit are "
                             "immediately rejected.")
    parser.add_argument("--est_hours", type=float, default=0.0,
                        help="Estimated duration in hours")
    parser.add_argument("--est_mins", type=float, default=0.0,
                        help="Estimated duration in minutes (additive with --est_hours)")
    parser.add_argument("--priority", type=int, default=0,
                        help="Scheduling priority: higher value = dispatched sooner (default: 0)")
    args = parser.parse_args()

    if args.num_gpus < 1:
        parser.error("--num_gpus must be at least 1")

    mem_mb = int(args.mem_gb * 1024)
    est_secs = args.est_hours * 3600.0 + args.est_mins * 60.0

    store = JobStore(args.db)
    job_id = store.submit(
        user_id=args.user,
        cmd=args.cmd,
        mem_mb=mem_mb,
        num_gpus=args.num_gpus,
        est_secs=est_secs,
        priority=args.priority,
    )
    store.close()

    gpu_str = f"{args.num_gpus} GPU{'s' if args.num_gpus > 1 else ''}"
    share_str = f" ({mem_mb // args.num_gpus} MB each)" if args.num_gpus > 1 else ""
    print(f"Submitted  job_id={job_id}  user={args.user}  "
          f"mem={mem_mb} MB total  {gpu_str}{share_str}  "
          f"est={est_secs:.0f}s  priority={args.priority}")
    print(f"  cmd: {args.cmd}")
    print(f"The daemon will dispatch this job on its next poll cycle.")


if __name__ == "__main__":
    main()
