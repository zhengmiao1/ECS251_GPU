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

    def wait_distribution(
        self,
        run_id: int,
        buckets: List[float] | None = None,
    ) -> List[Dict[str, object]]:
        """
        Return a histogram of task wait times for a given run.

        buckets: ascending list of right-edge values (seconds).
                 Default: [30, 60, 120, 300, 600, 1200, inf]
        Returns list of dicts: {bucket_label, count, pct}
        """
        if buckets is None:
            buckets = [30.0, 60.0, 120.0, 300.0, 600.0, 1200.0]

        rows = self._conn.execute(
            "SELECT wait_time FROM task_results WHERE run_id = ? ORDER BY wait_time",
            (run_id,),
        ).fetchall()
        wait_times = [r["wait_time"] for r in rows]
        total = len(wait_times)

        edges = list(buckets) + [float("inf")]
        labels = []
        for i, edge in enumerate(edges):
            lo = edges[i - 1] if i > 0 else 0.0
            if edge == float("inf"):
                labels.append(f">{lo:.0f}s")
            else:
                labels.append(f"{lo:.0f}–{edge:.0f}s")

        counts = [0] * len(edges)
        for wt in wait_times:
            for idx, edge in enumerate(edges):
                if wt <= edge:
                    counts[idx] += 1
                    break

        return [
            {
                "bucket": labels[i],
                "count": counts[i],
                "pct": round(100.0 * counts[i] / total, 2) if total > 0 else 0.0,
            }
            for i in range(len(edges))
        ]

    def gpu_utilization_timeline(
        self,
        run_id: int,
        bin_size: float = 60.0,
    ) -> List[Dict[str, object]]:
        """
        Return per-GPU busy fraction sampled at fixed time-bin intervals.

        Each row: {bin_start, gpu_id, busy_fraction}
        busy_fraction = (total overlap of task intervals with [bin_start, bin_start+bin_size])
                        / bin_size
        """
        rows = self._conn.execute(
            "SELECT gpu_id, start_time, end_time FROM task_results WHERE run_id = ?",
            (run_id,),
        ).fetchall()

        if not rows:
            return []

        t_min = min(r["start_time"] for r in rows)
        t_max = max(r["end_time"] for r in rows)

        # Snap t_min to bin boundary
        t_start = (t_min // bin_size) * bin_size
        timeline: List[Dict[str, object]] = []

        bin_start = t_start
        while bin_start < t_max:
            bin_end = bin_start + bin_size
            # Accumulate busy time per GPU for this bin
            gpu_busy: Dict[str, float] = {}
            for r in rows:
                overlap_start = max(r["start_time"], bin_start)
                overlap_end = min(r["end_time"], bin_end)
                if overlap_end > overlap_start:
                    gpu_id = r["gpu_id"]
                    gpu_busy[gpu_id] = gpu_busy.get(gpu_id, 0.0) + (overlap_end - overlap_start)
            for gpu_id, busy in sorted(gpu_busy.items()):
                timeline.append(
                    {
                        "bin_start": round(bin_start, 1),
                        "gpu_id": gpu_id,
                        "busy_fraction": round(min(busy / bin_size, 1.0), 4),
                    }
                )
            bin_start = bin_end

        return timeline

    def compare_runs(
        self,
        run_id_a: int,
        run_id_b: int,
    ) -> Dict[str, Dict[str, object]]:
        """
        Direct metric comparison between two stored runs.
        Returns {metric: {run_id_a: value, run_id_b: value, delta: b-a}}.
        """
        def _stats(run_id: int) -> Dict[str, float]:
            rows = self._conn.execute(
                "SELECT wait_time, end_time - arrival_time AS turnaround, "
                "       end_time - start_time AS duration "
                "FROM task_results WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            if not rows:
                return {}
            waits = sorted(r["wait_time"] for r in rows)
            n = len(waits)
            p95 = waits[min(n - 1, int(0.95 * (n - 1)))]
            avg_wait = sum(waits) / n
            avg_ta = sum(r["turnaround"] for r in rows) / n
            oom = self._conn.execute(
                "SELECT COUNT(*) AS c FROM decisions WHERE run_id = ? AND admitted = 0",
                (run_id,),
            ).fetchone()["c"]
            return {
                "completed_tasks": float(n),
                "avg_wait_time": round(avg_wait, 3),
                "p95_wait_time": round(p95, 3),
                "avg_turnaround": round(avg_ta, 3),
                "oom_events": float(oom),
            }

        stats_a = _stats(run_id_a)
        stats_b = _stats(run_id_b)
        result: Dict[str, Dict[str, object]] = {}
        for metric in stats_a:
            va = stats_a[metric]
            vb = stats_b.get(metric, 0.0)
            result[metric] = {
                f"run_{run_id_a}": va,
                f"run_{run_id_b}": vb,
                "delta": round(vb - va, 4),
            }
        return result

    def close(self) -> None:
        self._conn.close()
