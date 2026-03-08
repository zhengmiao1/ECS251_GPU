"""
daemon.py  –  GPU Scheduler Daemon

Polls nvidia-smi every `poll_interval` seconds and dispatches pending jobs
to GPUs that have sufficient free memory.  Supports GPU sharing: multiple
jobs can run on the same GPU as long as the total committed memory fits.

Key design decisions
--------------------
1. GPU assignment priority:
   - FIRST try to assign to fully idle GPUs (no running jobs on them).
   - ONLY if there are not enough idle GPUs, fall back to partially-used
     GPUs that still have enough free memory.
   This keeps busy GPUs consolidated and leaves idle GPUs available for
   jobs that cannot share.

2. Rejection: jobs requesting more GPUs than `max_gpus` are immediately
   rejected (marked failed) rather than waiting in queue forever.

3. Memory accounting: effective_free[gpu] = nvidia-smi reported free
   MINUS memory committed to jobs we launched in the last `grace_secs`
   seconds (which may not yet show in nvidia-smi).  This prevents
   double-booking during the lag between job launch and GPU memory claim.

4. Job ordering: pending queue sorted by priority DESC, submitted_at ASC
   (FIFO within same priority).

5. Safety buffer: `buffer_mb` MB is kept free on every GPU to absorb
   estimation errors and CUDA/driver overhead.

Usage
-----
    python -m scripts.daemon [--db scheduler.db] [--poll 10] [--buffer_mb 512]
                             [--max_gpus 4] [--mock]
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

from .gpu_monitor import GPUInfo, mock_gpus, query_gpus
from .job_store import JobStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [daemon] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("gpu-scheduler")


class SchedulerDaemon:
    def __init__(
        self,
        db_path: str = "scheduler.db",
        poll_interval: float = 10.0,
        buffer_mb: int = 512,
        grace_secs: float = 30.0,
        max_gpus: Optional[int] = None,
        mock: bool = False,
        mock_gpus_count: int = 2,
        mock_gpu_mem_mb: int = 24576,
    ):
        self.store = JobStore(db_path)
        self.poll_interval = poll_interval
        self.buffer_mb = buffer_mb
        self.grace_secs = grace_secs
        self.max_gpus = max_gpus  # None means "no limit beyond available GPUs"
        self.mock = mock
        self._mock_gpus = mock_gpus(mock_gpus_count, mock_gpu_mem_mb) if mock else []

        # job_id -> subprocess.Popen  (for jobs launched by this daemon instance)
        self._procs: Dict[int, subprocess.Popen] = {}

        # job_id -> (gpu_ids: List[int], mem_mb, launch_epoch)
        # Tracks recent launches whose memory may not yet appear in nvidia-smi
        self._recent: Dict[int, tuple] = {}

        self._stop = False

    # ------------------------------------------------------------------
    # GPU querying
    # ------------------------------------------------------------------

    def _get_gpus(self) -> List[GPUInfo]:
        if self.mock:
            committed = self._committed_per_gpu()
            for g in self._mock_gpus:
                used = committed.get(g.gpu_id, 0)
                g.used_mem_mb = used
                g.free_mem_mb = max(0, g.total_mem_mb - used)
            return list(self._mock_gpus)
        return query_gpus()

    def _committed_per_gpu(self) -> Dict[int, int]:
        """Sum of mem_mb for all jobs currently running, split across their assigned GPUs."""
        committed: Dict[int, int] = {}
        for job in self.store.get_running():
            gpu_id_str = job.get("gpu_id")
            if gpu_id_str is None:
                continue
            gpu_ids = [int(g) for g in str(gpu_id_str).split(",")]
            share = job["mem_mb"] // len(gpu_ids)
            for gid in gpu_ids:
                committed[gid] = committed.get(gid, 0) + share
        return committed

    def _effective_free(self, gpus: List[GPUInfo]) -> Dict[int, int]:
        """
        Effective free memory per GPU = nvidia-smi free
        minus memory committed to recently launched jobs (grace period).
        """
        free: Dict[int, int] = {g.gpu_id: g.free_mem_mb for g in gpus}
        now = time.time()
        expired = []
        for job_id, (gpu_ids, mem_mb, epoch) in self._recent.items():
            if now - epoch < self.grace_secs:
                share = mem_mb // len(gpu_ids)
                for gid in gpu_ids:
                    free[gid] = max(0, free.get(gid, 0) - share)
            else:
                expired.append(job_id)
        for jid in expired:
            del self._recent[jid]
        return free

    def _pick_gpus(
        self,
        needed_mb: int,
        num_gpus: int,
        free: Dict[int, int],
        committed: Dict[int, int],
    ) -> Optional[List[int]]:
        """
        Find `num_gpus` GPUs for the job using a two-tier strategy:

        Tier 1 – Idle GPUs (no committed memory): assign here first.
                 Keeps idle GPUs available for large exclusive jobs and
                 avoids unnecessary contention.

        Tier 2 – Busy GPUs with remaining memory: use only if there are
                 not enough idle GPUs that each fit their share of the job.

        Within each tier, prefer GPUs with the most free memory (best-fit
        descending) to maximise remaining headroom.

        Returns a sorted list of GPU ids, or None if no valid assignment.
        """
        share = (needed_mb + num_gpus - 1) // num_gpus  # ceil(total / n) per GPU

        # Split eligible GPUs into idle vs busy
        idle: List[int] = []
        busy: List[int] = []
        for gid, fm in free.items():
            if fm - self.buffer_mb < share:
                continue  # not enough room for this job's share
            if committed.get(gid, 0) == 0:
                idle.append(gid)
            else:
                busy.append(gid)

        # Sort each tier by descending free memory
        idle.sort(key=lambda g: free[g], reverse=True)
        busy.sort(key=lambda g: free[g], reverse=True)

        # Prefer idle GPUs; fill remainder from busy if needed
        selected = (idle + busy)[:num_gpus]
        if len(selected) < num_gpus:
            return None
        return sorted(selected)

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def _launch(self, job: Dict, gpu_ids: List[int]) -> Optional[int]:
        """Fork a subprocess for the job with CUDA_VISIBLE_DEVICES set."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        env["SCHEDULER_JOB_ID"] = str(job["job_id"])
        env["SCHEDULER_USER"] = job["user_id"]

        try:
            proc = subprocess.Popen(
                job["cmd"],
                shell=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            log.error("Failed to launch job %d: %s", job["job_id"], exc)
            return None

        self._procs[job["job_id"]] = proc
        return proc.pid

    def _check_finished(self) -> None:
        """Poll running jobs; mark done/failed for those that have exited."""
        finished = []
        for job_id, proc in list(self._procs.items()):
            rc = proc.poll()
            if rc is not None:
                self.store.mark_done(job_id, rc)
                log.info("Job %d finished  exit_code=%d", job_id, rc)
                finished.append(job_id)
        for jid in finished:
            self._procs.pop(jid, None)
            self._recent.pop(jid, None)

    def _reap_orphaned(self) -> None:
        """
        Jobs marked 'running' in the DB but whose PID no longer exists
        (daemon was restarted).  Mark them failed so they don't block scheduling.
        """
        import psutil
        for job in self.store.get_running():
            jid = job["job_id"]
            if jid in self._procs:
                continue
            pid = job.get("pid")
            if pid is None:
                continue
            try:
                alive = psutil.pid_exists(pid)
            except Exception:
                alive = False
            if not alive:
                self.store.mark_done(jid, exit_code=-9, note="orphaned – pid gone on daemon restart")
                log.warning("Reaped orphaned job %d (pid=%d)", jid, pid)

    # ------------------------------------------------------------------
    # Main scheduling step
    # ------------------------------------------------------------------

    def _schedule(self, gpus: List[GPUInfo]) -> None:
        total_gpus = len(gpus)
        gpu_limit = self.max_gpus if self.max_gpus is not None else total_gpus

        free = self._effective_free(gpus)
        committed = self._committed_per_gpu()
        pending = self.store.get_pending()

        for job in pending:
            num_gpus = job.get("num_gpus") or 1

            # --- Rejection: job asks for more GPUs than the system allows ---
            if num_gpus > gpu_limit:
                reason = (
                    f"rejected: requested {num_gpus} GPU(s) exceeds "
                    f"max_gpus limit of {gpu_limit}"
                )
                self.store.reject(job["job_id"], reason)
                log.warning(
                    "Rejected job %d  user=%s  requested %d GPUs (limit=%d)",
                    job["job_id"], job["user_id"], num_gpus, gpu_limit,
                )
                continue

            # --- Try to assign GPUs (idle first, then partially-used) ---
            gpu_ids = self._pick_gpus(job["mem_mb"], num_gpus, free, committed)
            if gpu_ids is None:
                log.debug(
                    "Job %d deferred  user=%s  need=%dMB x %dGPU  free=%s",
                    job["job_id"], job["user_id"], job["mem_mb"], num_gpus,
                    {gid: f"{fm}MB" for gid, fm in free.items()},
                )
                continue

            pid = self._launch(job, gpu_ids)
            if pid is None:
                continue

            gpu_id_str = ",".join(str(g) for g in gpu_ids)
            self.store.mark_running(job["job_id"], gpu_id_str, pid)
            self._recent[job["job_id"]] = (gpu_ids, job["mem_mb"], time.time())

            # Update in-loop accounting so subsequent jobs in this cycle see updated free memory
            share = job["mem_mb"] // len(gpu_ids)
            for gid in gpu_ids:
                free[gid] = max(0, free[gid] - share)
                committed[gid] = committed.get(gid, 0) + share

            idle_flag = all(committed.get(gid, 0) == share for gid in gpu_ids)
            placement = "idle-GPU" if idle_flag else "shared-GPU"
            multi = " [MULTI-GPU]" if len(gpu_ids) > 1 else ""
            log.info(
                "Dispatched job %d  user=%-8s  mem=%4dMB  gpu=%s  pid=%d  %s%s",
                job["job_id"], job["user_id"], job["mem_mb"], gpu_id_str, pid,
                placement, multi,
            )

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def _log_gpu_state(self, gpus: List[GPUInfo], free: Dict[int, int]) -> None:
        parts = []
        for g in gpus:
            eff = free.get(g.gpu_id, g.free_mem_mb)
            parts.append(
                f"GPU{g.gpu_id}[used={g.used_mem_mb}MB eff_free={eff}MB/{g.total_mem_mb}MB]"
            )
        log.info("GPU state  %s", "  ".join(parts))

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        log.info(
            "Daemon started  poll=%.0fs  buffer=%dMB  grace=%.0fs  max_gpus=%s  mock=%s",
            self.poll_interval, self.buffer_mb, self.grace_secs,
            self.max_gpus if self.max_gpus is not None else "auto",
            self.mock,
        )
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            pass

        try:
            import psutil  # noqa: F401
            self._reap_orphaned()
        except ImportError:
            log.debug("psutil not installed – orphan reaping skipped")

        while not self._stop:
            try:
                gpus = self._get_gpus()
                self._check_finished()
                free = self._effective_free(gpus)
                self._log_gpu_state(gpus, free)
                self._schedule(gpus)
            except RuntimeError as exc:
                log.error("GPU query failed: %s", exc)
            except Exception as exc:
                log.exception("Unexpected error in scheduler loop: %s", exc)

            time.sleep(self.poll_interval)

        log.info("Daemon stopped.")
        self.store.close()

    def _handle_signal(self, signum: int, _frame: object) -> None:
        log.info("Received signal %d – shutting down after current cycle.", signum)
        self._stop = True


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU Scheduler Daemon")
    parser.add_argument("--db", default="scheduler.db",
                        help="Path to SQLite job database (default: scheduler.db)")
    parser.add_argument("--poll", type=float, default=10.0,
                        help="Polling interval in seconds (default: 10)")
    parser.add_argument("--buffer_mb", type=int, default=512,
                        help="Safety memory buffer per GPU in MB (default: 512)")
    parser.add_argument("--grace_secs", type=float, default=30.0,
                        help="Seconds to hold grace-period memory reservation (default: 30)")
    parser.add_argument("--max_gpus", type=int, default=None,
                        help="Maximum GPUs any single job may request. "
                             "Jobs exceeding this are immediately rejected. "
                             "(default: no limit beyond total available GPUs)")
    parser.add_argument("--mock", action="store_true",
                        help="Run without real GPUs – simulate GPU state in memory")
    parser.add_argument("--mock_gpus", type=int, default=2,
                        help="Number of mock GPUs (--mock only, default: 2)")
    parser.add_argument("--mock_mem_gb", type=float, default=24.0,
                        help="Memory per mock GPU in GB (--mock only, default: 24)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    daemon = SchedulerDaemon(
        db_path=args.db,
        poll_interval=args.poll,
        buffer_mb=args.buffer_mb,
        grace_secs=args.grace_secs,
        max_gpus=args.max_gpus,
        mock=args.mock,
        mock_gpus_count=args.mock_gpus,
        mock_gpu_mem_mb=int(args.mock_mem_gb * 1024),
    )
    daemon.run()


if __name__ == "__main__":
    main()
