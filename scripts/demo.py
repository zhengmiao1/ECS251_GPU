"""
demo.py  —  Realistic multi-user GPU scheduling demo (mock mode)

Simulates a real-world scenario:
  - A daemon monitors 2 "GPUs" (24 GB each)
  - Five users submit jobs at staggered intervals (as they would in practice)
  - Some jobs are immediately dispatched; others queue because GPUs are full
  - Short jobs finish and free memory; queued jobs are then dispatched
  - Status snapshots are printed at key moments showing queue / running / done

Usage:
    python -m scripts.demo               # standard demo (~35 seconds real time)
    python -m scripts.demo --speed 3     # 3x faster (~12 seconds)
    python -m scripts.demo --gpus 1      # single GPU — more contention
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from typing import List

from .daemon import SchedulerDaemon
from .job_store import JobStore
from .gpu_monitor import mock_gpus


# ── ANSI colours ──────────────────────────────────────────────────────────────
_C = {
    "reset":   "\033[0m",
    "bold":    "\033[1m",
    "pending": "\033[33m",   # yellow
    "running": "\033[32m",   # green
    "done":    "\033[34m",   # blue
    "failed":  "\033[31m",   # red
    "header":  "\033[36m",   # cyan
    "dim":     "\033[90m",   # grey
}


def c(key: str, text: str) -> str:
    return _C.get(key, "") + text + _C["reset"]


# ── Workload definition ────────────────────────────────────────────────────────

# Each entry: (user, label, mem_gb, real_duration_secs, est_duration_secs, priority, submit_delay)
# submit_delay: scenario-seconds after demo start (scaled by --speed at runtime)
#
# Multi-GPU trigger logic  (2 × 24 GB GPUs, 512 MB safety buffer):
#
#   Step 1 — alice, bob, frank all arrive before the first daemon poll (~0.3 s real).
#             Scheduler orders by est_secs ASC: bob(2700) → alice(3600) → frank(7200).
#
#   Step 2 — After bob and alice are dispatched:
#             GPU 0 free = 24 - 12 = 12 GB   (bob)
#             GPU 1 free = 24 - 14 = 10 GB   (alice)
#
#   Step 3 — frank needs 16 GB:
#             Single GPU 0: 12 GB < 16 GB  ✗
#             Single GPU 1: 10 GB < 16 GB  ✗
#             GPU 0 + GPU 1 combined: share = 16//2 = 8 GB each
#               GPU 0: 12 - 0.5 buffer = 11.5 ≥ 8  ✓
#               GPU 1: 10 - 0.5 buffer =  9.5 ≥ 8  ✓
#             → frank dispatched across BOTH GPUs  ★
#
#   Later — carol, eve, dave arrive and fill remaining GPU space.
SCENARIO: List[tuple] = [
    # user      label                    mem_gb  real_dur  est_dur  pri  submit_at
    # ── submitted close together → all pending in first daemon poll ──────────
    ("alice",  "LLM finetune   (14 GB)",   14,     14,      3600,   0,   0.0),
    ("bob",    "VLM pretrain   (12 GB)",   12,     12,      2700,   0,   0.2),  # poll<0.3s
    ("frank",  "LLM pretrain   (16 GB)",   16,     16,      7200,   0,   0.4),  # → MULTI-GPU ★
    # ── smaller jobs arrive after frank is already running ───────────────────
    ("dave",   "Quick test     ( 2 GB)",    2,      3,        60,   0,   4.5),  # fits on GPU0
    ("carol",  "Inference      ( 4 GB)",    4,      5,       300,   0,   5.0),  # queued until bob frees
    ("eve",    "Eval run       ( 6 GB)",    6,      6,       600,   0,   5.5),  # queued until bob frees
]


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _mem_bar(used: int, total: int, width: int = 20) -> str:
    frac = min(1.0, used / total) if total else 0.0
    filled = int(frac * width)
    pct = frac * 100
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {used:>6}/{total} MB ({pct:3.0f}%)"


def _fmt_secs(secs: float) -> str:
    if secs <= 0:
        return "  ?   "
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"  {s:2d}s "


def _status_tag(status: str) -> str:
    tags = {
        "pending": c("pending", "QUEUE  "),
        "running": c("running", "RUNNING"),
        "done":    c("done",    "DONE   "),
        "failed":  c("failed",  "FAILED "),
    }
    return tags.get(status, status)


# ── Status snapshot printer ────────────────────────────────────────────────────

def print_snapshot(store: JobStore, gpus_cfg: tuple, title: str, t0: float) -> None:
    elapsed = time.time() - t0
    ngpus, mem_mb = gpus_cfg

    # Rebuild GPU used memory from running jobs (gpu_id may be "0" or "0,1")
    running_jobs = store.get_running()
    used_per_gpu = {}
    for j in running_jobs:
        gpu_id_str = j.get("gpu_id")
        if gpu_id_str is None:
            continue
        gids = [int(g) for g in str(gpu_id_str).split(",")]
        share = j["mem_mb"] // len(gids)
        for gid in gids:
            used_per_gpu[gid] = used_per_gpu.get(gid, 0) + share

    sep = "─" * 72
    print(f"\n{sep}")
    print(c("bold", c("header", f"  ⏱  T+{elapsed:4.1f}s   {title}")))
    print(sep)

    # GPU rows
    print(c("bold", "  GPUs"))
    for i in range(ngpus):
        used = used_per_gpu.get(i, 0)
        bar  = _mem_bar(used, mem_mb)
        jobs_here = [j for j in running_jobs
                     if i in [int(g) for g in str(j.get("gpu_id") or "").split(",") if g]]
        def _owner_label(j: dict, gpu_idx: int) -> str:
            gids = [int(g) for g in str(j.get("gpu_id") or "").split(",") if g]
            tag = "[MULTI]" if len(gids) > 1 else ""
            share = j["mem_mb"] // len(gids)
            return f"{j['user_id']}({share}MB{tag})"
        owners = ", ".join(_owner_label(j, i) for j in jobs_here) or "idle"
        print(f"    GPU {i}  {bar}  ← {owners}")

    # Job table
    all_jobs = store.all_jobs(limit=30)
    pending  = [j for j in all_jobs if j["status"] == "pending"]
    running  = [j for j in all_jobs if j["status"] == "running"]
    done     = [j for j in all_jobs if j["status"] in ("done", "failed")]

    print()
    print(c("bold", f"  Jobs   {c('running','■ running')} {len(running)}   "
            f"{c('pending','■ pending')} {len(pending)}   "
            f"{c('done','■ done')} {len(done)}"))
    print(f"  {'ID':>3}  {'Status':7}  {'User':<8}  {'Mem':>7}  {'Est':>6}  "
          f"{'Pri':>3}  {'GPU':>4}  Command")
    print(f"  {'─'*3}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*3}  {'─'*4}  {'─'*28}")

    display = sorted(all_jobs, key=lambda j: (
        {"running": 0, "pending": 1, "done": 2, "failed": 2}.get(j["status"], 3),
        j["job_id"],
    ))
    for j in display:
        gpu_raw = j.get("gpu_id")
        if gpu_raw is None:
            gpu_str = "  —  "
        else:
            gids = str(gpu_raw).split(",")
            gpu_str = f"GPU {','.join(gids)}"
            if len(gids) > 1:
                gpu_str = c("header", gpu_str + " ★")   # highlight multi-GPU jobs
        cmd_short = j["cmd"][:28]
        print(f"  {j['job_id']:>3}  {_status_tag(j['status'])}  "
              f"{j['user_id']:<8}  {j['mem_mb']:>5}MB  "
              f"{_fmt_secs(j.get('est_secs',0)):>6}  {j['priority']:>3}  "
              f"{gpu_str}  {cmd_short}")

    print(sep)


# ── Demo runner ────────────────────────────────────────────────────────────────

def run_demo(ngpus: int = 2, mem_gb: float = 24.0, speed: float = 1.0) -> None:
    mem_mb = int(mem_gb * 1024)
    gpus_cfg = (ngpus, mem_mb)

    # Temp DB
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    print(c("bold", c("header", "\n  GPU Scheduler — Live Demo (mock mode)")))
    print(f"  {ngpus} GPUs × {mem_gb:.0f} GB each   speed={speed}x   DB={db_path}")

    # Start daemon
    daemon = SchedulerDaemon(
        db_path=db_path,
        poll_interval=max(0.3, 1.0 / speed),
        buffer_mb=512,
        grace_secs=max(2.0, 5.0 / speed),
        mock=True,
        mock_gpus_count=ngpus,
        mock_gpu_mem_mb=mem_mb,
    )
    # Suppress daemon logs during demo — we print our own snapshots
    import logging
    logging.getLogger("gpu-scheduler").setLevel(logging.WARNING)

    daemon_thread = threading.Thread(target=daemon.run, daemon=True)
    daemon_thread.start()

    store = JobStore(db_path)
    t0 = time.time()

    def wait(secs: float) -> None:
        time.sleep(secs / speed)

    def submit(user: str, label: str, mem_gb_j: float,
               real_dur: float, est_secs: float, priority: int) -> int:
        cmd = f"sleep {real_dur / speed:.1f}"  # real sleep scaled by speed
        jid = store.submit(
            user_id=user,
            cmd=cmd,
            mem_mb=int(mem_gb_j * 1024),
            est_secs=est_secs,
            priority=priority,
        )
        print(c("dim", f"\n  → {user} submitted job {jid}: {label}  "
                f"({mem_gb_j:.0f} GB, est {_fmt_secs(est_secs).strip()})"))
        return jid

    # ── Phase 1: initial burst ─────────────────────────────────────────────
    print(c("bold", "\n\n  Phase 1 — Users submit jobs (alice=14GB on GPU0, bob=12GB on GPU1)"))
    print(c("dim",  "            frank will submit 18GB — neither GPU alone has that much free"))

    prev_delay = 0.0
    submitted_so_far = []
    for (user, label, mem_gb_j, real_dur, est_secs, pri, submit_at) in SCENARIO:
        delta = submit_at - prev_delay
        if delta > 0:
            wait(delta)
        prev_delay = submit_at
        jid = submit(user, label, mem_gb_j, real_dur, est_secs, pri)
        submitted_so_far.append(jid)

    # Give daemon a moment to dispatch first wave
    wait(2.5)
    print_snapshot(store, gpus_cfg,
                   "After submissions — frank (18GB) dispatched via ★ MULTI-GPU", t0)

    # ── Phase 2: short jobs finish ─────────────────────────────────────────
    print(c("bold", "\n\n  Phase 2 — Short jobs complete; GPUs partially free"))
    wait(4.5)
    print_snapshot(store, gpus_cfg, "Mid-run: alice+bob finishing, frank still spanning both GPUs", t0)

    # ── Phase 3: all settling ──────────────────────────────────────────────
    wait(5.0)
    print_snapshot(store, gpus_cfg, "Late run: all jobs completing", t0)

    # ── Wait for everything to finish ─────────────────────────────────────
    deadline = time.time() + 30.0 / speed
    while time.time() < deadline:
        all_j = store.all_jobs()
        active = [j for j in all_j if j["status"] in ("pending", "running")]
        if not active:
            break
        wait(0.5)

    daemon._stop = True
    wait(0.8)

    print_snapshot(store, gpus_cfg, "Final state — all jobs completed", t0)

    # ── Summary stats ──────────────────────────────────────────────────────
    all_jobs = store.all_jobs()
    done_jobs = [j for j in all_jobs if j["status"] == "done"]
    if done_jobs:
        wait_times = []
        for j in done_jobs:
            if j.get("started_at") and j.get("submitted_at"):
                from datetime import datetime
                fmt = "%Y-%m-%dT%H:%M:%S"
                try:
                    wt = (datetime.fromisoformat(j["started_at"]) -
                          datetime.fromisoformat(j["submitted_at"])).total_seconds()
                    wait_times.append(wt)
                except ValueError:
                    pass
        print(c("bold", "\n  Summary"))
        print(f"    Total jobs submitted : {len(all_jobs)}")
        print(f"    Completed            : {len(done_jobs)}")
        print(f"    Failed               : {sum(1 for j in all_jobs if j['status'] == 'failed')}")
        if wait_times:
            print(f"    Avg queue wait       : {sum(wait_times)/len(wait_times):.1f}s  "
                  f"max={max(wait_times):.1f}s")
    print()

    store.close()
    try:
        os.unlink(db_path)
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU scheduler realistic demo (mock mode)")
    parser.add_argument("--gpus", type=int, default=2,
                        help="Number of simulated GPUs (default: 2)")
    parser.add_argument("--mem_gb", type=float, default=24.0,
                        help="Memory per GPU in GB (default: 24)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier: 2 = 2x faster (default: 1)")
    args = parser.parse_args()
    run_demo(ngpus=args.gpus, mem_gb=args.mem_gb, speed=args.speed)


if __name__ == "__main__":
    main()
