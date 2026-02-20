"""
db_store.py  –  Zaishuo Xia (Week 3)

SQLite persistence layer for scheduler runs, admission decisions, and task results.
Enables offline querying and reproducible analysis without re-running simulations.

Usage:
    from scripts.db_store import ExperimentStore
    store = ExperimentStore("results.db")
    run_id = store.insert_run(policy="memory", workload="mixed", seed=7, gpus=2, gpu_mem=24.0)
    store.insert_decisions(run_id, result.decisions)
    store.insert_results(run_id, result.results)
    store.close()
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional

from .models import AdmissionDecision, TaskResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    policy      TEXT    NOT NULL,
    workload    TEXT    NOT NULL,
    seed        INTEGER NOT NULL,
    gpus        INTEGER NOT NULL,
    gpu_mem_gb  REAL    NOT NULL,
    tasks       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id),
    task_id     TEXT    NOT NULL,
    user_id     TEXT    NOT NULL,
    arrival_time REAL   NOT NULL,
    est_duration REAL   NOT NULL,
    est_mem_gb  REAL    NOT NULL,
    duration_class TEXT NOT NULL,
    admitted    INTEGER NOT NULL,   -- 1 = admitted, 0 = rejected
    reason      TEXT    NOT NULL,
    gpu_id      TEXT
);

CREATE TABLE IF NOT EXISTS task_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(run_id),
    task_id      TEXT    NOT NULL,
    user_id      TEXT    NOT NULL,
    gpu_id       TEXT    NOT NULL,
    arrival_time REAL    NOT NULL,
    start_time   REAL    NOT NULL,
    end_time     REAL    NOT NULL,
    wait_time    REAL    NOT NULL,
    turnaround   REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_results_run   ON task_results(run_id);
"""


class ExperimentStore:
    """SQLite-backed store for experiment runs and scheduler decisions."""

    def __init__(self, db_path: str = "experiments.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------

    def insert_run(
        self,
        policy: str,
        workload: str,
        seed: int,
        gpus: int,
        gpu_mem: float,
        tasks: int = 0,
    ) -> int:
        """Insert a new experiment run and return its run_id."""
        cur = self._conn.execute(
            "INSERT INTO runs (policy, workload, seed, gpus, gpu_mem_gb, tasks) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (policy, workload, seed, gpus, gpu_mem, tasks),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def insert_decisions(self, run_id: int, decisions: List[AdmissionDecision]) -> None:
        """Bulk-insert admission decisions for a run."""
        rows = [
            (
                run_id,
                d.task.task_id,
                d.task.user_id,
                d.task.arrival_time,
                d.task.est_duration,
                d.task.est_mem_gb,
                d.task.duration_class,
                1 if d.admitted else 0,
                d.reason,
                d.gpu_id,
            )
            for d in decisions
        ]
        self._conn.executemany(
            "INSERT INTO decisions "
            "(run_id, task_id, user_id, arrival_time, est_duration, est_mem_gb, "
            " duration_class, admitted, reason, gpu_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def insert_results(self, run_id: int, results: List[TaskResult]) -> None:
        """Bulk-insert task results for a run."""
        rows = [
            (
                run_id,
                r.task.task_id,
                r.task.user_id,
                r.gpu_id,
                r.task.arrival_time,
                r.start_time,
                r.end_time,
                r.wait_time,
                r.end_time - r.task.arrival_time,
            )
            for r in results
        ]
        self._conn.executemany(
            "INSERT INTO task_results "
            "(run_id, task_id, user_id, gpu_id, arrival_time, start_time, end_time, "
            " wait_time, turnaround) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def query_runs(
        self,
        policy: Optional[str] = None,
        workload: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Return all matching runs as a list of dicts."""
        clauses: List[str] = []
        params: List[object] = []
        if policy is not None:
            clauses.append("policy = ?")
            params.append(policy)
        if workload is not None:
            clauses.append("workload = ?")
            params.append(workload)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(f"SELECT * FROM runs {where}", params).fetchall()
        return [dict(r) for r in rows]

    def per_user_wait_stats(self, run_id: int) -> List[Dict[str, object]]:
        """Return per-user average and max wait time for a given run."""
        rows = self._conn.execute(
            "SELECT user_id, "
            "       AVG(wait_time) AS avg_wait, "
            "       MAX(wait_time) AS max_wait, "
            "       COUNT(*)       AS task_count "
            "FROM task_results "
            "WHERE run_id = ? "
            "GROUP BY user_id "
            "ORDER BY user_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def per_gpu_utilization(self, run_id: int) -> List[Dict[str, object]]:
        """Return total busy time per GPU for a given run."""
        rows = self._conn.execute(
            "SELECT gpu_id, "
            "       SUM(end_time - start_time) AS busy_time, "
            "       COUNT(*)                   AS task_count "
            "FROM task_results "
            "WHERE run_id = ? "
            "GROUP BY gpu_id "
            "ORDER BY gpu_id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
