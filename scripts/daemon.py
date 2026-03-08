"""
daemon.py  –  GPU Scheduler Daemon

Polls nvidia-smi every `poll_interval` seconds and dispatches pending jobs
to GPUs that have sufficient free memory.  Supports GPU sharing: multiple
jobs can run on the same GPU as long as the total committed memory fits.

Key design decisions
--------------------
1. Memory accounting: effective_free[gpu] = nvidia-smi reported free
   MINUS memory committed to jobs we launched in the last `grace_secs`
   seconds (which may not yet show in nvidia-smi).  This prevents
   double-booking during the lag between job launch and GPU memory claim.

2. GPU selection: "most-free-first" (best-fit descending) to pack jobs
   onto fewer GPUs and leave headroom for large jobs.

3. Job ordering: pending queue sorted by priority DESC, est_secs ASC
   (shortest estimated runtime first), then submission time ASC.
   This minimises average wait for short jobs while preventing starvation
   of long jobs via explicit priority escalation at submission time.

4. Safety buffer: `buffer_mb` MB is kept free on every GPU to absorb
   estimation errors and CUDA/driver overhead.

Usage
-----
    python -m scripts.daemon [--db scheduler.db] [--poll 10] [--buffer_mb 512] [--mock]
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
        mock: bool = False,
        mock_gpus_count: int = 2,
        mock_gpu_mem_mb: int = 24576,
    ):
        self.store = JobStore(db_path)
        self.poll_interval = poll_interval
        self.buffer_mb = buffer_mb
        self.grace_secs = grace_secs
        self.mock = mock
        self._mock_gpus = mock_gpus(mock_gpus_count, mock_gpu_mem_mb) if mock else []

        # job_id -> subprocess.Popen  (for jobs launched by this daemon instance)
        self._procs: Dict[int, subprocess.Popen] = {}

        # job_id -> (gpu_id, mem_mb, launch_epoch)
        # Tracks recent launches whose memory may not yet appear in nvidia-smi
        self._recent: Dict[int, tuple] = {}

        self._stop = False

    # ------------------------------------------------------------------
    # GPU querying
    # ------------------------------------------------------------------

    def _get_gpus(self) -> List[GPUInfo]:
        if self.mock:
            # In mock mode, keep a stable list but update used_mem_mb
            # based on our own running-job accounting so the demo is realistic.
            committed = self._committed_per_gpu()
            for g in self._mock_gpus:
                used = committed.get(g.gpu_id, 0)
                g.used_mem_mb = used
                g.free_mem_mb = max(0, g.total_mem_mb - used)
            return list(self._mock_gpus)
        return query_gpus()

    def _committed_per_gpu(self) -> Dict[int, int]:
        """Sum of mem_mb for all jobs currently running, split evenly across their assigned GPUs."""
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
        _recent stores: job_id -> (gpu_ids: List[int], mem_mb, epoch)
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

    def _pick_gpus(self, needed_mb: int, free: Dict[int, int]) -> Optional[List[int]]:
        """
        Try to find GPU(s) for a job needing `needed_mb` MB.

        Strategy:
          1. Single GPU: pick the GPU with most free memory >= needed_mb + buffer.
          2. Multi-GPU pair: if no single GPU fits, find the pair whose combined
             free memory fits, with each GPU contributing at least its share
             (needed_mb / 2).  Memory is assumed to be split evenly across GPUs.

        Returns a list of GPU ids (length 1 or 2), or None if no fit found.
        """
        # --- single GPU ---
        single_candidates = {
            gid: fm for gid, fm in free.items()
            if fm - self.buffer_mb >= needed_mb
        }
        if single_candidates:
            return [max(single_candidates, key=lambda g: single_candidates[g])]

        # --- multi-GPU pair ---
        share = (needed_mb + 1) // 2   # each GPU must hold at least this much
        gpu_ids = sorted(free.keys())
        best_pair: Optional[List[int]] = None
        best_total = 0
        for i in range(len(gpu_ids)):
            for j in range(i + 1, len(gpu_ids)):
                g1, g2 = gpu_ids[i], gpu_ids[j]
                if (free[g1] - self.buffer_mb >= share and
                        free[g2] - self.buffer_mb >= share):
                    total = free[g1] + free[g2]
                    if total > best_total:
                        best_total = total
                        best_pair = [g1, g2]
        return best_pair

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def _launch(self, job: Dict, gpu_ids: List[int]) -> Optional[int]:
        """
        Fork a subprocess for the job with CUDA_VISIBLE_DEVICES set.
        gpu_ids may be [0] for single-GPU or [0,1] for multi-GPU.
        Returns the PID on success, None on failure.
        """
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
        import psutil  # optional dependency
        for job in self.store.get_running():
            jid = job["job_id"]
            if jid in self._procs:
                continue  # we own this process
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
        free = self._effective_free(gpus)
        pending = self.store.get_pending()

        for job in pending:
            gpu_ids = self._pick_gpus(job["mem_mb"], free)
            if gpu_ids is None:
                log.debug(
                    "Job %d deferred  user=%s  need=%dMB  free=%s",
                    job["job_id"], job["user_id"], job["mem_mb"],
                    {gid: f"{fm}MB" for gid, fm in free.items()},
                )
                continue

            pid = self._launch(job, gpu_ids)
            if pid is None:
                continue

            gpu_id_str = ",".join(str(g) for g in gpu_ids)
            self.store.mark_running(job["job_id"], gpu_id_str, pid)
            self._recent[job["job_id"]] = (gpu_ids, job["mem_mb"], time.time())

            share = job["mem_mb"] // len(gpu_ids)
            for gid in gpu_ids:
                free[gid] = max(0, free[gid] - share)

            multi = " [MULTI-GPU]" if len(gpu_ids) > 1 else ""
            log.info(
                "Dispatched job %d  user=%-8s  mem=%4dMB  gpu=%s  pid=%d%s",
                job["job_id"], job["user_id"], job["mem_mb"], gpu_id_str, pid, multi,
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
            "Daemon started  poll=%.0fs  buffer=%dMB  grace=%.0fs  mock=%s",
            self.poll_interval, self.buffer_mb, self.grace_secs, self.mock,
        )
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            # signal() can only be called from the main thread; ignore in test/thread contexts
            pass

        # Try to reap jobs from a previous daemon run
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
                        help="Seconds to subtract recently-launched job memory from free (default: 30)")
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
        mock=args.mock,
        mock_gpus_count=args.mock_gpus,
        mock_gpu_mem_mb=int(args.mock_mem_gb * 1024),
    )
    daemon.run()


if __name__ == "__main__":
    main()
