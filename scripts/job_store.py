"""
job_store.py

SQLite-backed persistent job queue for the real GPU scheduler daemon.

Tables
------
jobs : one row per submitted job, tracks status lifecycle:
         pending → running → done | failed | cancelled
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional


_SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS jobs (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT    NOT NULL,
    cmd          TEXT    NOT NULL,
    mem_mb       INTEGER NOT NULL,          -- requested GPU memory in MB (total across all GPUs)
    num_gpus     INTEGER NOT NULL DEFAULT 1, -- number of GPUs requested
    est_secs     REAL    NOT NULL DEFAULT 0, -- estimated runtime in seconds (hint only)
    priority     INTEGER NOT NULL DEFAULT 0, -- higher value = higher priority
    status       TEXT    NOT NULL DEFAULT 'pending',  -- pending/running/done/failed/cancelled
    gpu_id       TEXT,                      -- assigned GPU index(es), e.g. "0" or "0,1"
    pid          INTEGER,                   -- OS process ID of the running job
    submitted_at TEXT    NOT NULL DEFAULT (datetime('now')),
    started_at   TEXT,
    ended_at     TEXT,
    exit_code    INTEGER,
    note         TEXT                       -- human-readable status note
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_user   ON jobs(user_id);
"""


class JobStore:
    """Thread-safe SQLite job store (WAL mode allows concurrent readers)."""

    def __init__(self, db_path: str = "scheduler.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        # Migrate older databases that predate the num_gpus column
        try:
            self._conn.execute("ALTER TABLE jobs ADD COLUMN num_gpus INTEGER NOT NULL DEFAULT 1")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def submit(
        self,
        user_id: str,
        cmd: str,
        mem_mb: int,
        num_gpus: int = 1,
        est_secs: float = 0.0,
        priority: int = 0,
    ) -> int:
        """Insert a new job in 'pending' state. Returns job_id."""
        cur = self._conn.execute(
            "INSERT INTO jobs (user_id, cmd, mem_mb, num_gpus, est_secs, priority, submitted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, cmd, mem_mb, num_gpus, est_secs, priority,
             datetime.now().isoformat(timespec="seconds")),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def mark_running(self, job_id: int, gpu_id, pid: int) -> None:
        """gpu_id may be an int (single GPU) or a string like '0,1' (multi-GPU)."""
        self._conn.execute(
            "UPDATE jobs SET status='running', gpu_id=?, pid=?, started_at=? WHERE job_id=?",
            (str(gpu_id), pid, datetime.now().isoformat(timespec="seconds"), job_id),
        )
        self._conn.commit()

    def mark_done(self, job_id: int, exit_code: int, note: str = "") -> None:
        status = "done" if exit_code == 0 else "failed"
        self._conn.execute(
            "UPDATE jobs SET status=?, ended_at=?, exit_code=?, note=? WHERE job_id=?",
            (status, datetime.now().isoformat(timespec="seconds"), exit_code, note, job_id),
        )
        self._conn.commit()

    def reject(self, job_id: int, reason: str) -> None:
        """Immediately reject a pending job (marks it failed with a reason note)."""
        self._conn.execute(
            "UPDATE jobs SET status='failed', ended_at=?, exit_code=-1, note=? "
            "WHERE job_id=? AND status='pending'",
            (datetime.now().isoformat(timespec="seconds"), reason, job_id),
        )
        self._conn.commit()

    def cancel(self, job_id: int) -> bool:
        """Cancel a pending job. Returns True if it was pending and is now cancelled."""
        cur = self._conn.execute(
            "UPDATE jobs SET status='cancelled', ended_at=?, note='user cancelled' "
            "WHERE job_id=? AND status='pending'",
            (datetime.now().isoformat(timespec="seconds"), job_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def kill_running(self, job_id: int, note: str = "killed by admin") -> Optional[int]:
        """
        Mark a running job as failed and return its PID so the caller can kill the process.
        Returns None if job is not in running state.
        """
        row = self._conn.execute(
            "SELECT pid FROM jobs WHERE job_id=? AND status='running'", (job_id,)
        ).fetchone()
        if row is None:
            return None
        pid = row["pid"]
        self._conn.execute(
            "UPDATE jobs SET status='failed', ended_at=?, exit_code=-1, note=? WHERE job_id=?",
            (datetime.now().isoformat(timespec="seconds"), note, job_id),
        )
        self._conn.commit()
        return pid

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_pending(self) -> List[Dict]:
        """
        Return pending jobs ordered by:
          1. priority DESC (higher priority first)
          2. submitted_at ASC  (FIFO within same priority)
        """
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status='pending' "
            "ORDER BY priority DESC, submitted_at ASC, job_id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_running(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status='running'"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_job(self, job_id: int) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None

    def all_jobs(self, limit: int = 50) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM jobs ORDER BY job_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def user_jobs(self, user_id: str, limit: int = 30) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE user_id=? ORDER BY job_id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
