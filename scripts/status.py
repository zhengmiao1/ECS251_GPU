"""
status.py  –  Show live GPU state and job queue.

Usage
-----
    python -m scripts.status                    # summary view
    python -m scripts.status --user alice       # jobs for a specific user
    python -m scripts.status --all              # show all historical jobs
    python -m scripts.status --db /path/to.db   # custom DB path
"""
from __future__ import annotations

import argparse
from typing import Dict, List

from .gpu_monitor import GPUInfo, mock_gpus, query_gpus
from .job_store import JobStore


_STATUS_COLORS = {
    "pending":   "\033[33m",   # yellow
    "running":   "\033[32m",   # green
    "done":      "\033[34m",   # blue
    "failed":    "\033[31m",   # red
    "cancelled": "\033[90m",   # dark grey
    "rejected":  "\033[35m",   # magenta
}
_RESET = "\033[0m"


def _color(status: str, text: str) -> str:
    return _STATUS_COLORS.get(status, "") + text + _RESET


def _mem_bar(used_mb: int, total_mb: int, width: int = 24) -> str:
    frac = min(1.0, used_mb / total_mb) if total_mb > 0 else 0.0
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    pct = frac * 100
    return f"[{bar}] {used_mb:>6}/{total_mb} MB ({pct:.0f}%)"


def _fmt_secs(secs: float) -> str:
    if secs <= 0:
        return "?"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def print_gpu_state(gpus: List[GPUInfo]) -> None:
    print("\n=== GPU State ===")
    if not gpus:
        print("  (no GPUs found)")
        return
    for g in gpus:
        bar = _mem_bar(g.used_mem_mb, g.total_mem_mb)
        print(f"  GPU {g.gpu_id}  {g.name:<28}  {bar}")


def print_jobs(jobs: List[Dict], title: str) -> None:
    if not jobs:
        return
    print(f"\n=== {title} ({len(jobs)}) ===")
    for j in jobs:
        status = j["status"]
        num_gpus = j.get("num_gpus") or 1
        gpu_raw = j.get("gpu_id")
        if gpu_raw is not None:
            gpu_str = f"gpu={gpu_raw}" + (" [x2]" if num_gpus > 1 else "")
        else:
            gpu_str = f"gpu=?  need={num_gpus}GPU{'s' if num_gpus > 1 else ' '}"
        est_str = _fmt_secs(j.get("est_secs", 0))
        label = f"{status.upper():<9}"
        base = (
            f"  [{j['job_id']:>4}] {_color(status, label)}  "
            f"user={j['user_id']:<10}  mem={j['mem_mb']:>5}MB  ngpu={num_gpus}  "
            f"est={est_str:>6}  pri={j['priority']}  {gpu_str}"
        )
        if status == "running":
            base += f"  pid={j['pid']}  started={j.get('started_at','?')}"
        elif status in ("done", "failed"):
            base += f"  exit={j.get('exit_code','?')}  ended={j.get('ended_at','?')}"
        elif status == "pending":
            base += f"  submitted={j.get('submitted_at','?')}"
        note = j.get("note") or ""
        if note:
            base += f"  [{note}]"
        print(base)
        print(f"           cmd: {j['cmd'][:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU scheduler status viewer")
    parser.add_argument("--db", default="scheduler.db")
    parser.add_argument("--user", default="",
                        help="Filter jobs by user ID")
    parser.add_argument("--all", action="store_true",
                        help="Show all historical jobs (not just recent 30)")
    parser.add_argument("--mock", action="store_true",
                        help="Show mock GPU state (for testing without NVIDIA GPU)")
    parser.add_argument("--mock_gpus", type=int, default=2)
    parser.add_argument("--mock_mem_gb", type=float, default=24.0)
    args = parser.parse_args()

    # GPU state
    try:
        if args.mock:
            gpus = mock_gpus(args.mock_gpus, int(args.mock_mem_gb * 1024))
        else:
            gpus = query_gpus()
        print_gpu_state(gpus)
    except RuntimeError as exc:
        print(f"\n=== GPU State ===\n  [unavailable: {exc}]")

    # Jobs
    store = JobStore(args.db)
    limit = 200 if args.all else 50

    if args.user:
        jobs = store.user_jobs(args.user, limit=limit)
    else:
        jobs = store.all_jobs(limit=limit)
    store.close()

    running  = [j for j in jobs if j["status"] == "running"]
    pending  = [j for j in jobs if j["status"] == "pending"]
    done     = [j for j in jobs if j["status"] in ("done", "cancelled")]
    failed   = [j for j in jobs if j["status"] == "failed"
                and not (j.get("note") or "").startswith("rejected:")]
    rejected = [j for j in jobs if j["status"] == "failed"
                and (j.get("note") or "").startswith("rejected:")]

    print_jobs(running,  "Running")
    print_jobs(pending,  "Pending (queued)")
    print_jobs(rejected, "Rejected")
    print_jobs(failed,   "Failed")
    print_jobs(done,     "Recent Completed")
    print()


if __name__ == "__main__":
    main()
