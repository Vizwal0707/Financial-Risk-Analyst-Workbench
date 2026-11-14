"""DuckDB analytical engine manager for in-memory and embedded vectorized OLAP queries."""

import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class DuckDBManager:
    """Manages embedded DuckDB storage, Parquet view registration, and analytical queries."""

    def __init__(
        self, db_path: str = "data/fraud_analytics.duckdb", read_only: bool = False
    ) -> None:
        self.db_path = db_path
        self.read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Establishes or returns an active DuckDB connection."""
        if self._conn is None:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = duckdb.connect(database=self.db_path, read_only=self.read_only)
        return self._conn

    def register_parquet_view(self, view_name: str, parquet_path: str) -> None:
        """Registers a Parquet file as an analytical SQL view for zero-copy querying."""
        conn = self.connect()
        path_obj = Path(parquet_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Parquet source file not found at: {parquet_path}")
        escaped_path = str(path_obj.resolve()).replace("'", "''")
        conn.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped_path}')"
        )

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Executes a SQL query and returns the result as a pandas DataFrame."""
        conn = self.connect()
        if params:
            return conn.execute(sql, params).df()
        return conn.execute(sql).df()

    def execute(self, sql: str, params: list[Any] | None = None) -> None:
        """Executes a SQL statement (DDL/DML)."""
        conn = self.connect()
        if params:
            conn.execute(sql, params)
        else:
            conn.execute(sql)

    def profile_query_latency(self, sql: str, repetitions: int = 5) -> dict[str, float]:
        """Profiles execution latency of a SQL query in milliseconds."""
        conn = self.connect()
        latencies_ms: list[float] = []
        for _ in range(repetitions):
            start = time.perf_counter()
            conn.execute(sql).fetchall()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        latencies_ms.sort()
        mean_lat = sum(latencies_ms) / len(latencies_ms)
        p95_idx = int(0.95 * len(latencies_ms))
        p95_lat = latencies_ms[min(p95_idx, len(latencies_ms) - 1)]

        return {
            "min_ms": round(latencies_ms[0], 3),
            "mean_ms": round(mean_lat, 3),
            "p95_ms": round(p95_lat, 3),
            "repetitions": repetitions,
        }

    def close(self) -> None:
        """Closes the active database connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
