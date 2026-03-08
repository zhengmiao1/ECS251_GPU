"""
cancel.py  –  Cancel a pending job or kill a running job.

Usage
-----
    python -m scripts.cancel --job_id 5               # cancel if pending
    python -m scripts.cancel --job_id 5 --force       # kill if running (sends SIGKILL)
"""
from __future__ import annotations

import argparse
import os
import signal

from .job_store import JobStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Cancel or kill a scheduled job")
    parser.add_argument("--job_id", type=int, required=True,
                        help="Job ID to cancel")
    parser.add_argument("--db", default="scheduler.db")
    parser.add_argument("--force", action="store_true",
                        help="Kill a running job (sends SIGKILL to its process)")
    args = parser.parse_args()

    store = JobStore(args.db)

    if args.force:
        pid = store.kill_running(args.job_id)
        if pid is None:
            # Try cancel as pending fallback
            ok = store.cancel(args.job_id)
            if ok:
                print(f"Job {args.job_id} was pending – cancelled.")
            else:
                print(f"Job {args.job_id} not found or already finished.")
        else:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"Sent SIGKILL to pid={pid}  (job {args.job_id}).")
            except ProcessLookupError:
                print(f"Process pid={pid} already gone. Job {args.job_id} marked failed.")
            except PermissionError:
                print(f"No permission to kill pid={pid}. Job {args.job_id} marked failed in DB.")
    else:
        ok = store.cancel(args.job_id)
        if ok:
            print(f"Job {args.job_id} cancelled.")
        else:
            job = store.get_job(args.job_id)
            if job is None:
                print(f"Job {args.job_id} not found.")
            else:
                print(f"Job {args.job_id} is in state '{job['status']}' – cannot cancel. "
                      "Use --force to kill running jobs.")

    store.close()


if __name__ == "__main__":
    main()
