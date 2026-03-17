"""
demo.py  —  GPU Scheduler Demo (mock mode, no GPU needed)

Four self-contained scenarios, each starting from a clean state.

  Scenario 1  No Conflict   — jobs dispatched immediately to idle GPUs
  Scenario 2  GPU Sharing   — new jobs placed on busy GPUs with free memory
  Scenario 3  Conflict      — queue, priority ordering, and rejection
  Scenario 4  Resolution    — queue drains in priority order as memory frees

Usage:
    python -m scripts.demo                   # run all four scenarios
    python -m scripts.demo --scenario 1      # run one specific scenario
    python -m scripts.demo --speed 3         # 3x faster
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time

from .daemon import SchedulerDaemon
from .job_store import JobStore


# ── ANSI colours ───────────────────────────────────────────────────────────────
_C = {
    "reset":    "\033[0m",  "bold":     "\033[1m",
    "pending":  "\033[33m", "running":  "\033[32m",
    "done":     "\033[34m", "failed":   "\033[31m",
    "rejected": "\033[35m", "header":   "\033[36m",
    "dim":      "\033[90m",
}


def c(key: str, text: str) -> str:
    return _C.get(key, "") + text + _C["reset"]


# ── Display helpers ────────────────────────────────────────────────────────────

def _mem_bar(used: int, total: int, width: int = 20) -> str:
    frac = min(1.0, used / total) if total else 0.0
    bar = "#" * int(frac * width) + "-" * (width - int(frac * width))
    return f"[{bar}] {used:>6}/{total} MB ({frac*100:3.0f}%)"


def _fmt_dur(secs: float) -> str:
    if secs <= 0:
        return "?"
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h:   return f"{h}h{m:02d}m"
    if m:   return f"{m}m{s:02d}s"
    return f"{s}s"


def _status_tag(status: str, note: str = "") -> str:
    if status == "failed" and note.startswith("rejected:"):
        return c("rejected", "REJECTED")
    return {
        "pending": c("pending",  "QUEUED  "),
        "running": c("running",  "RUNNING "),
        "done":    c("done",     "DONE    "),
        "failed":  c("failed",   "FAILED  "),
    }.get(status, status)


def print_snapshot(store: JobStore, ngpus: int, mem_mb: int,
                   title: str, t0: float) -> None:
    elapsed = time.time() - t0

    # Build per-GPU usage from running jobs
    running_jobs = store.get_running()
    used_per_gpu: dict[int, int] = {}
    for j in running_jobs:
        gids = [int(g) for g in str(j.get("gpu_id") or "").split(",") if g]
        share = j["mem_mb"] // max(len(gids), 1)
        for gid in gids:
            used_per_gpu[gid] = used_per_gpu.get(gid, 0) + share

    sep = "─" * 70
    print(f"\n{sep}")
    print(c("bold", c("header", f"  T+{elapsed:4.1f}s   {title}")))
    print(sep)

    # GPU rows
    print(c("bold", "  GPUs"))
    for i in range(ngpus):
        used = used_per_gpu.get(i, 0)
        jobs_here = [j for j in running_jobs
                     if str(i) in str(j.get("gpu_id") or "").split(",")]
        if jobs_here:
            owners = ", ".join(
                f"{j['user_id']}({j['mem_mb'] // max(len(str(j.get('gpu_id','')).split(',')),1)}MB)"
                for j in jobs_here
            )
        else:
            owners = c("dim", "idle")
        print(f"    GPU {i}  {_mem_bar(used, mem_mb)}  {owners}")

    # Classify jobs
    all_jobs = store.all_jobs(limit=30)
    running_j  = [j for j in all_jobs if j["status"] == "running"]
    pending_j  = sorted(
        [j for j in all_jobs if j["status"] == "pending"],
        key=lambda j: (-j["priority"], j["submitted_at"], j["job_id"]),
    )
    rejected_j = [j for j in all_jobs
                  if j["status"] == "failed"
                  and (j.get("note") or "").startswith("rejected:")]
    done_j     = [j for j in all_jobs
                  if j["status"] == "done"
                  or (j["status"] == "failed"
                      and not (j.get("note") or "").startswith("rejected:"))]

    counts = (f"{c('running','■ running')} {len(running_j)}  "
              f"{c('pending','■ queued')} {len(pending_j)}  "
              f"{c('rejected','■ rejected')} {len(rejected_j)}  "
              f"{c('done','■ done')} {len(done_j)}")
    print(f"\n  {counts}")
    print(f"  {'ID':>3}  {'Status':8}  {'User':<7}  {'Mem':>7}  {'Pri':>3}  {'GPU':<9}  Info")
    print(f"  {'─'*3}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*3}  {'─'*9}  {'─'*28}")

    qpos = 0
    for section in (running_j, pending_j, rejected_j, done_j):
        for j in section:
            note = j.get("note") or ""
            is_pending  = j["status"] == "pending"
            is_rejected = j["status"] == "failed" and note.startswith("rejected:")

            gpu_raw = j.get("gpu_id")
            gpu_col = (f"GPU {','.join(str(gpu_raw).split(','))}"
                       if gpu_raw is not None else "  —      ")

            if is_pending:
                qpos += 1
                pri, mem = j["priority"], j["mem_mb"]
                info = f"{c('pending', f'Q{qpos}')} pri={pri}  needs {mem}MB"
            elif is_rejected:
                info = note.replace("rejected: ", c("rejected", "✗ "), 1)
            elif j["status"] == "running":
                info = f"started {j.get('started_at','?')[-8:]}"
            else:
                info = f"exit={j.get('exit_code','?')}  ended {j.get('ended_at','?')[-8:]}"

            print(f"  {j['job_id']:>3}  {_status_tag(j['status'], note)}  "
                  f"{j['user_id']:<7}  {j['mem_mb']:>5}MB  {j['priority']:>3}  "
                  f"{gpu_col:<9}  {info}")
    print(sep)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _start(ngpus: int, mem_mb: int, speed: float,
           max_gpus: int | None = None) -> tuple:
    """Create a fresh daemon + job store. Returns (daemon, store, db_path, t0)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    daemon = SchedulerDaemon(
        db_path=db_path,
        poll_interval=max(0.3, 1.0 / speed),
        buffer_mb=512,
        grace_secs=max(2.0, 5.0 / speed),
        max_gpus=max_gpus,
        mock=True,
        mock_gpus_count=ngpus,
        mock_gpu_mem_mb=mem_mb,
    )
    import logging
    logging.getLogger("gpu-scheduler").setLevel(logging.WARNING)
    threading.Thread(target=daemon.run, daemon=True).start()
    return daemon, JobStore(db_path), db_path, time.time()


def _cleanup(daemon: SchedulerDaemon, store: JobStore, db_path: str) -> None:
    daemon._stop = True
    time.sleep(0.5)
    store.close()
    try:
        os.unlink(db_path)
    except OSError:
        pass


def _sub(store: JobStore, user: str, mem_gb: float, num_gpus: int,
         real_dur: float, est_secs: float, speed: float,
         priority: int = 0) -> int:
    jid = store.submit(
        user_id=user,
        cmd=f"sleep {real_dur / speed:.1f}",
        mem_mb=int(mem_gb * 1024),
        num_gpus=num_gpus,
        est_secs=est_secs,
        priority=priority,
    )
    pri_note = f"  {c('running', f'priority={priority}')}" if priority > 0 else ""
    ngpu_str = f"{num_gpus}GPU{'s' if num_gpus > 1 else ' '}"
    print(c("dim", f"    submit #{jid:<2}  {user:<8}  "
            f"{mem_gb:.0f} GB  {ngpu_str}  est {_fmt_dur(est_secs)}{pri_note}"))
    return jid


def _w(secs: float, speed: float) -> None:
    time.sleep(secs / speed)


def _jobs_by_user(store: JobStore) -> dict[str, dict]:
    jobs = {}
    for job in store.all_jobs(limit=200):
        jobs[job["user_id"]] = job
    return jobs


def _wait_until(store: JobStore, speed: float, predicate, timeout: float = 8.0) -> dict[str, dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = _jobs_by_user(store)
        if predicate(jobs):
            return jobs
        time.sleep(max(0.03, 0.15 / max(speed, 0.1)))
    raise RuntimeError("demo state did not converge before timeout")


def _is_running_on(jobs: dict[str, dict], user: str, gpu_id: int) -> bool:
    job = jobs.get(user)
    return bool(job and job["status"] == "running" and str(job.get("gpu_id")) == str(gpu_id))


# ── Scenario 1: No Conflict ────────────────────────────────────────────────────

def scenario_no_conflict(ngpus: int, mem_mb: int, speed: float) -> None:
    """
    Both GPUs are completely idle.
    Alice (10 GB) and Bob (12 GB) are dispatched immediately,
    one to each GPU — no waiting, no sharing.
    """
    print(c("bold", "\n" + "=" * 70))
    print(c("bold", c("header", "  Scenario 1 — No Conflict")))
    print(c("dim",  "  Both GPUs idle.  Each job claims its own GPU immediately."))
    print(c("bold", "=" * 70))

    daemon, store, db_path, t0 = _start(ngpus, mem_mb, speed)

    print(c("dim", "\n  Submitting jobs:"))
    _sub(store, "alice", 10, 1, real_dur=30, est_secs=3600, speed=speed)
    _w(0.3, speed)
    _sub(store, "bob",  12, 1, real_dur=30, est_secs=2700, speed=speed)
    _wait_until(
        store,
        speed,
        lambda jobs: _is_running_on(jobs, "alice", 0) and _is_running_on(jobs, "bob", 1),
    )

    print_snapshot(store, ngpus, mem_mb,
                   "alice on GPU 0, bob on GPU 1 — each owns an idle GPU", t0)
    _cleanup(daemon, store, db_path)


# ── Scenario 2: GPU Sharing ────────────────────────────────────────────────────

def scenario_sharing(ngpus: int, mem_mb: int, speed: float) -> None:
    """
    GPUs are already in use but still have free memory.
    Carol and Dave are placed on the busy GPUs alongside existing jobs.

    GPU 0: alice (10 GB) + carol (6 GB) = 16 GB used
    GPU 1: bob  (12 GB) + dave  (4 GB) = 16 GB used
    """
    print(c("bold", "\n" + "=" * 70))
    print(c("bold", c("header", "  Scenario 2 — GPU Sharing")))
    print(c("dim",  "  New jobs join GPUs that are in use but have remaining memory."))
    print(c("bold", "=" * 70))

    daemon, store, db_path, t0 = _start(ngpus, mem_mb, speed)

    print(c("dim", "\n  Wave 1 — fill GPUs partially:"))
    _sub(store, "alice", 10, 1, real_dur=30, est_secs=3600, speed=speed)
    _w(0.3, speed)
    _sub(store, "bob",  12, 1, real_dur=30, est_secs=2700, speed=speed)
    _wait_until(
        store,
        speed,
        lambda jobs: _is_running_on(jobs, "alice", 0) and _is_running_on(jobs, "bob", 1),
    )

    print_snapshot(store, ngpus, mem_mb,
                   "After wave 1 — alice on GPU 0 (14 GB free), bob on GPU 1 (12 GB free)", t0)

    print(c("dim", "\n  Wave 2 — share the busy GPUs:"))
    _sub(store, "carol", 6, 1, real_dur=30, est_secs=300, speed=speed)
    _w(0.3, speed)
    _sub(store, "dave",  4, 1, real_dur=30, est_secs=600, speed=speed)
    _wait_until(
        store,
        speed,
        lambda jobs: (
            _is_running_on(jobs, "alice", 0)
            and _is_running_on(jobs, "carol", 0)
            and _is_running_on(jobs, "bob", 1)
            and _is_running_on(jobs, "dave", 1)
        ),
    )

    print_snapshot(store, ngpus, mem_mb,
                   "After wave 2 — carol shares GPU 0, dave shares GPU 1", t0)
    _cleanup(daemon, store, db_path)


# ── Scenario 3: Conflict ───────────────────────────────────────────────────────

def scenario_conflict(ngpus: int, mem_mb: int, speed: float) -> None:
    """
    GPUs are nearly full (20 GB each used, only 3.5 GB free per GPU).
    Three jobs cannot fit and are queued.
    Priority ordering: Grace (priority=2) goes to the front of the queue.

    Queue order shown here: Grace Q1 (pri=2) → Frank Q2 → Eve Q3
    """
    print(c("bold", "\n" + "=" * 70))
    print(c("bold", c("header", "  Scenario 3 — Conflict: Queue Forms Under Memory Pressure")))
    print(c("dim",  "  GPUs nearly full.  New jobs cannot fit and must wait in queue."))
    print(c("bold", "=" * 70))

    daemon, store, db_path, t0 = _start(ngpus, mem_mb, speed)

    print(c("dim", "\n  Fill GPUs (20 GB each, leaving only ~3.5 GB free per GPU):"))
    _sub(store, "alice", 20, 1, real_dur=30, est_secs=7200, speed=speed)
    _w(0.3, speed)
    _sub(store, "bob",   20, 1, real_dur=30, est_secs=7200, speed=speed)
    _wait_until(
        store,
        speed,
        lambda jobs: _is_running_on(jobs, "alice", 0) and _is_running_on(jobs, "bob", 1),
    )

    print(c("dim", "\n  Submit jobs that cannot fit:"))
    _sub(store, "frank",  4, 1, real_dur=5,  est_secs=300,  speed=speed, priority=0)
    _w(0.3, speed)
    _sub(store, "eve",   18, 1, real_dur=10, est_secs=7200, speed=speed, priority=0)
    _w(0.2, speed)
    _sub(store, "grace",  4, 1, real_dur=5,  est_secs=300,  speed=speed, priority=2)
    _wait_until(
        store,
        speed,
        lambda jobs: (
            _is_running_on(jobs, "alice", 0)
            and _is_running_on(jobs, "bob", 1)
            and jobs.get("grace", {}).get("status") == "pending"
            and jobs.get("frank", {}).get("status") == "pending"
            and jobs.get("eve", {}).get("status") == "pending"
        ),
    )

    print_snapshot(store, ngpus, mem_mb,
                   "Queue: grace Q1 (pri=2) → frank Q2 → eve Q3", t0)
    _cleanup(daemon, store, db_path)


# ── Scenario 4: Resolution ─────────────────────────────────────────────────────

def scenario_resolution(ngpus: int, mem_mb: int, speed: float) -> None:
    """
    Same conflict setup as Scenario 3, but Alice and Bob are short jobs.
    When they finish, the queue drains in priority order:

      Step 1  Alice and Bob running; grace/eve/frank all queued.
      Step 2  Alice and Bob finish.  Grace (pri=2) dispatched first.
              Eve (18 GB) goes to the now-idle GPU 1.
              Frank shares GPU 0 with Grace.
      Step 3  All jobs complete.
    """
    print(c("bold", "\n" + "=" * 70))
    print(c("bold", c("header", "  Scenario 4 — Resolution: Queue Drains in Priority Order")))
    print(c("dim",  "  Short jobs free memory; high-priority job dispatched first."))
    print(c("bold", "=" * 70))

    daemon, store, db_path, t0 = _start(ngpus, mem_mb, speed, max_gpus=2)

    print(c("dim", "\n  Fill GPUs with short jobs (8 s each):"))
    _sub(store, "alice", 20, 1, real_dur=8, est_secs=7200, speed=speed)
    _w(0.3, speed)
    _sub(store, "bob",   20, 1, real_dur=8, est_secs=7200, speed=speed)
    _wait_until(
        store,
        speed,
        lambda jobs: _is_running_on(jobs, "alice", 0) and _is_running_on(jobs, "bob", 1),
    )

    print(c("dim", "\n  Submit queued jobs while GPUs are full:"))
    _sub(store, "eve",   18, 1, real_dur=8, est_secs=7200, speed=speed, priority=0)
    _w(0.3, speed)
    _sub(store, "frank",  4, 1, real_dur=5, est_secs=300,  speed=speed, priority=0)
    _w(0.2, speed)
    _sub(store, "grace",  4, 1, real_dur=5, est_secs=300,  speed=speed, priority=2)
    _wait_until(
        store,
        speed,
        lambda jobs: (
            _is_running_on(jobs, "alice", 0)
            and _is_running_on(jobs, "bob", 1)
            and jobs.get("grace", {}).get("status") == "pending"
            and jobs.get("frank", {}).get("status") == "pending"
            and jobs.get("eve", {}).get("status") == "pending"
        ),
    )

    print_snapshot(store, ngpus, mem_mb,
                   "Step 1 — alice and bob running; grace, frank, and eve are queued", t0)

    _wait_until(
        store,
        speed,
        lambda jobs: (
            jobs.get("alice", {}).get("status") == "done"
            and jobs.get("bob", {}).get("status") == "done"
            and _is_running_on(jobs, "grace", 0)
            and _is_running_on(jobs, "frank", 0)
            and _is_running_on(jobs, "eve", 1)
        ),
        timeout=20.0,
    )

    print_snapshot(store, ngpus, mem_mb,
                   "Step 2 — grace dispatched first (pri=2); eve on GPU 1; frank shares GPU 0",
                   t0)

    # wait for remaining jobs (grace/frank ~5s, eve ~8s from dispatch at ~t=9)
    deadline = time.time() + 15.0 / speed
    while time.time() < deadline:
        if not [j for j in store.all_jobs() if j["status"] in ("pending", "running")]:
            break
        time.sleep(0.4 / speed)

    print_snapshot(store, ngpus, mem_mb, "Step 3 — all jobs completed", t0)

    # Summary
    all_jobs  = store.all_jobs()
    done_jobs = [j for j in all_jobs if j["status"] == "done"]
    waits = []
    for j in done_jobs:
        if j.get("started_at") and j.get("submitted_at"):
            from datetime import datetime
            try:
                wt = (datetime.fromisoformat(j["started_at"]) -
                      datetime.fromisoformat(j["submitted_at"])).total_seconds()
                waits.append((j["user_id"], wt))
            except ValueError:
                pass
    if waits:
        print(c("bold", "\n  Queue wait times:"))
        for user, wt in sorted(waits, key=lambda x: -x[1]):
            bar = "#" * min(int(wt), 30)
            print(f"    {user:<8}  {wt:5.1f}s  {c('pending', bar)}")

    _cleanup(daemon, store, db_path)


# ── Runner ─────────────────────────────────────────────────────────────────────

_SCENARIOS = {
    1: ("No Conflict",           scenario_no_conflict),
    2: ("GPU Sharing",           scenario_sharing),
    3: ("Conflict",              scenario_conflict),
    4: ("Resolution",            scenario_resolution),
}


def run_demo(ngpus: int = 2, mem_gb: float = 24.0,
             speed: float = 1.0, scenario: int = 0) -> None:
    mem_mb = int(mem_gb * 1024)

    header = (f"  GPU Scheduler Demo   "
              f"{ngpus} GPU × {mem_gb:.0f} GB   speed={speed}x   buffer=512 MB")
    print(c("bold", c("header", "\n" + header)))

    to_run = [_SCENARIOS[scenario]] if scenario in _SCENARIOS else list(_SCENARIOS.values())
    for name, fn in to_run:
        fn(ngpus, mem_mb, speed)
        time.sleep(0.5 / speed)

    print(c("bold", c("header", "\n  Demo complete.\n")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU scheduler demo (mock mode, no GPU needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scenarios:\n"
            "  1  No Conflict  — jobs dispatched to idle GPUs immediately\n"
            "  2  GPU Sharing  — new jobs placed on busy GPUs with free memory\n"
            "  3  Conflict     — queue, priority ordering, rejection\n"
            "  4  Resolution   — queue drains in priority order as memory frees\n"
        ),
    )
    parser.add_argument("--scenario", type=int, default=0,
                        choices=[1, 2, 3, 4],
                        help="Run one scenario (default: run all four)")
    parser.add_argument("--gpus", type=int, default=2,
                        help="Number of simulated GPUs (default: 2)")
    parser.add_argument("--mem_gb", type=float, default=24.0,
                        help="Memory per GPU in GB (default: 24)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier: 2 = 2x faster (default: 1)")
    args = parser.parse_args()
    run_demo(ngpus=args.gpus, mem_gb=args.mem_gb,
             speed=args.speed, scenario=args.scenario)


if __name__ == "__main__":
    main()
