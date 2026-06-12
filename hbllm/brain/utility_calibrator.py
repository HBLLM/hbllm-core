"""
Utility Calibration Framework.

Tracks predicted utility vs actual outcome to compute prediction error,
storing traces in a persistent SQLite database for feedback loop adjustment.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTrace:
    trace_id: str
    decision_point: str  # e.g. "planner_node:expand", "workspace:action"
    predicted_utility: float
    actual_outcome: float
    prediction_error: float
    timestamp: float
    metadata: dict[str, Any]


class UtilityCalibrator:
    """
    Tracks and records utility predictions vs actual outcomes.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._db_path = Path(data_dir) / "utility_calibration.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_traces (
                    trace_id TEXT PRIMARY KEY,
                    decision_point TEXT NOT NULL,
                    predicted_utility REAL NOT NULL,
                    actual_outcome REAL NOT NULL,
                    prediction_error REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_dp ON calibration_traces(decision_point)"
            )

    def record_trace(
        self,
        trace_id: str,
        decision_point: str,
        predicted_utility: float,
        actual_outcome: float,
        metadata: dict[str, Any] | None = None,
    ) -> CalibrationTrace:
        """
        Record a utility calibration trace.
        """
        import json

        prediction_error = predicted_utility - actual_outcome
        now = time.time()
        meta_str = json.dumps(metadata or {})

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO calibration_traces "
                "(trace_id, decision_point, predicted_utility, actual_outcome, prediction_error, timestamp, metadata_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    decision_point,
                    predicted_utility,
                    actual_outcome,
                    prediction_error,
                    now,
                    meta_str,
                ),
            )

        return CalibrationTrace(
            trace_id=trace_id,
            decision_point=decision_point,
            predicted_utility=predicted_utility,
            actual_outcome=actual_outcome,
            prediction_error=prediction_error,
            timestamp=now,
            metadata=metadata or {},
        )

    def get_traces(
        self, decision_point: str | None = None, limit: int = 100
    ) -> list[CalibrationTrace]:
        """
        Retrieve traces from the database.
        """
        import json

        query = "SELECT trace_id, decision_point, predicted_utility, actual_outcome, prediction_error, timestamp, metadata_json FROM calibration_traces"
        args = []
        if decision_point:
            query += " WHERE decision_point = ?"
            args.append(decision_point)
        query += " ORDER BY timestamp DESC LIMIT ?"
        args.append(limit)

        traces = []
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, args).fetchall()
            for r in rows:
                try:
                    meta = json.loads(r["metadata_json"] or "{}")
                except Exception:
                    meta = {}
                traces.append(
                    CalibrationTrace(
                        trace_id=r["trace_id"],
                        decision_point=r["decision_point"],
                        predicted_utility=r["predicted_utility"],
                        actual_outcome=r["actual_outcome"],
                        prediction_error=r["prediction_error"],
                        timestamp=r["timestamp"],
                        metadata=meta,
                    )
                )
        return traces

    def get_average_error(self, decision_point: str | None = None) -> float:
        """
        Get average absolute prediction error.
        """
        query = "SELECT AVG(ABS(prediction_error)) FROM calibration_traces"
        args = []
        if decision_point:
            query += " WHERE decision_point = ?"
            args.append(decision_point)

        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute(query, args).fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0

    def get_calibration_readiness(self) -> dict[str, Any]:
        """
        Return statistical parameters to determine bootstrap status.
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            rows = conn.execute(
                "SELECT predicted_utility, actual_outcome, prediction_error, decision_point FROM calibration_traces LIMIT 50"
            ).fetchall()

        sample_count = len(rows)
        if sample_count < 10:
            return {"bootstrap_active": True, "reason": "Insufficient samples"}

        errors = [r[2] for r in rows]
        mean_err = sum(errors) / sample_count
        variance = sum((x - mean_err) ** 2 for x in errors) / sample_count

        unique_domains = len(set(r[3] for r in rows))

        # Bootstrap is active if sample count is low, prediction error variance is high (noisy),
        # or domain coverage is extremely narrow.
        bootstrap_active = sample_count < 15 or variance > 0.05 or unique_domains < 3
        return {
            "bootstrap_active": bootstrap_active,
            "sample_count": sample_count,
            "variance": variance,
            "unique_domains": unique_domains,
            "reason": "Stable calibration achieved"
            if not bootstrap_active
            else "High variance or narrow domain",
        }

    def get_utility_percentiles(self) -> tuple[float, float, float]:
        """
        Get mixed short-term vs long-term utility percentiles mixed with external baseline anchor.

        Formula:
        percentile = 0.7 * live_percentile + 0.3 * anchor_percentile

        live_percentile is mixed 50/50 short (last 20) / long (last 100).
        anchor_percentile is frozen at (0.7, 0.3, 0.0).
        """
        with sqlite3.connect(str(self._db_path)) as conn:
            st_rows = conn.execute(
                "SELECT predicted_utility FROM calibration_traces ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()
            lt_rows = conn.execute(
                "SELECT predicted_utility FROM calibration_traces ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()

        if len(lt_rows) < 10:
            return 0.7, 0.3, 0.0

        st_scores = sorted([r[0] for r in st_rows])
        lt_scores = sorted([r[0] for r in lt_rows])

        def pct(scores, q):
            n = len(scores)
            return scores[int(n * q)]

        st_80, st_40, st_10 = pct(st_scores, 0.8), pct(st_scores, 0.4), pct(st_scores, 0.1)
        lt_80, lt_40, lt_10 = pct(lt_scores, 0.8), pct(lt_scores, 0.4), pct(lt_scores, 0.1)

        live_80 = 0.5 * st_80 + 0.5 * lt_80
        live_40 = 0.5 * st_40 + 0.5 * lt_40
        live_10 = 0.5 * st_10 + 0.5 * lt_10

        # Mix 70% live + 30% anchor baseline
        high = 0.7 * live_80 + 0.3 * 0.7
        med = 0.7 * live_40 + 0.3 * 0.3
        low = 0.7 * live_10 + 0.3 * 0.0
        return high, med, low

    def detect_drift(self) -> bool:
        """
        Detect utility score inflation or distribution drift using Z-score shift.
        Returns True if drift is detected (Z-score shift > 2.0).
        """
        import math

        with sqlite3.connect(str(self._db_path)) as conn:
            st_rows = conn.execute(
                "SELECT predicted_utility FROM calibration_traces ORDER BY timestamp DESC LIMIT 20"
            ).fetchall()
            lt_rows = conn.execute(
                "SELECT predicted_utility FROM calibration_traces ORDER BY timestamp DESC LIMIT 100"
            ).fetchall()

        if len(lt_rows) < 20:
            return False

        st_scores = [r[0] for r in st_rows]
        lt_scores = [r[0] for r in lt_rows]

        st_mean = sum(st_scores) / len(st_scores)
        lt_mean = sum(lt_scores) / len(lt_scores)

        lt_variance = sum((x - lt_mean) ** 2 for x in lt_scores) / len(lt_scores)
        lt_std_dev = math.sqrt(lt_variance)

        if lt_std_dev < 1e-5:
            # Avoid division by zero
            return False

        z_score = (st_mean - lt_mean) / lt_std_dev
        return abs(z_score) > 1.99
