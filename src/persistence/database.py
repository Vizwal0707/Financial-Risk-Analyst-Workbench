"""Database connection manager with SQLite file persistence and PostgreSQL dual-mode compatibility."""

import asyncio
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SQLITE_PATH = "data/audit_scoring.db"

SQLITE_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS audit_scoring_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    step INTEGER NOT NULL,
    corridor_type TEXT NOT NULL,
    amount REAL NOT NULL,
    name_orig TEXT NOT NULL,
    oldbalance_orig REAL NOT NULL,
    name_dest TEXT NOT NULL,
    oldbalance_dest REAL NOT NULL,
    raw_probability REAL NOT NULL,
    calibrated_probability REAL NOT NULL,
    risk_score REAL NOT NULL,
    risk_tier TEXT NOT NULL,
    decision TEXT NOT NULL,
    action_code TEXT NOT NULL,
    decision_reasons TEXT,
    anomaly_score REAL,
    latency_ms REAL NOT NULL,
    model_version TEXT NOT NULL,
    cold_start_applied INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    explanation_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_request_id ON audit_scoring_logs (request_id);
CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_scoring_logs (decision);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_scoring_logs (created_at);
"""


def mask_account_id(account_id: str) -> str:
    """Masks customer/account identifiers for PCI-DSS-aligned audit log privacy best practice (non-certified prototype)."""
    acc = str(account_id).strip()
    if len(acc) <= 4:
        return "****"
    return f"{acc[:2]}****{acc[-4:]}"


def sanitize_db_url(url: str) -> str:
    """Masks database passwords in connection URLs for safe logging."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


class DatabaseManager:
    """Manages transactional audit persistence with SQLite or PostgreSQL backends.

    Architecture:
    - SQLite (Default): Local zero-dependency ACID storage with PRAGMA journal_mode=WAL.
    - PostgreSQL (Dual-mode): Configured via DATABASE_URL with SQLAlchemy connection pool (pool_pre_ping=True, pool_size=10).
    - Graceful Fallback: If PostgreSQL connection or initialization fails, falls back automatically to local SQLite with structured warnings.
    - Verification Status: PostgreSQL mode is functionally integrated and tested in dual-mode fallbacks, but marked as unverified in production until live testing on managed cloud infrastructure.
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url or os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")
        self.is_sqlite = self.db_url.startswith("sqlite")
        self.sqlite_path = self._resolve_sqlite_path()
        self.is_connected = False
        self._shared_memory_conn: Optional[sqlite3.Connection] = None
        self._pg_engine: Optional[Any] = None

    def _resolve_sqlite_path(self) -> str:
        if not self.is_sqlite or not self.db_url.startswith("sqlite"):
            return DEFAULT_SQLITE_PATH
        path_str = self.db_url.replace("sqlite:///", "").replace("sqlite://", "")
        if path_str == ":memory:" or not path_str:
            return ":memory:"
        Path(path_str).parent.mkdir(parents=True, exist_ok=True)
        return path_str

    def initialize_schema(self) -> None:
        """Runs DDL migrations to initialize audit tables."""
        if self.is_sqlite:
            if self.sqlite_path == ":memory:":
                # Retain open connection for shared in-memory database
                self._shared_memory_conn = sqlite3.connect("file:audit_mem?mode=memory&cache=shared", uri=True)
                self._shared_memory_conn.row_factory = sqlite3.Row
                self._shared_memory_conn.executescript(SQLITE_SCHEMA_DDL)
                self._shared_memory_conn.commit()
                self.is_connected = True
                logger.info("Initialized shared in-memory SQLite audit database.")
                return

            conn = sqlite3.connect(self.sqlite_path)
            try:
                # Enable WAL (Write-Ahead Logging) for high concurrent read/write throughput
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.executescript(SQLITE_SCHEMA_DDL)
                conn.commit()
                self.is_connected = True
                logger.info(f"Initialized SQLite audit database schema at: {self.sqlite_path} with WAL mode")
            finally:
                conn.close()
        else:
            # PostgreSQL connection attempt via SQLAlchemy with automatic fallback
            sanitized = sanitize_db_url(self.db_url)
            try:
                from sqlalchemy import create_engine, text
                self._pg_engine = create_engine(
                    self.db_url,
                    pool_pre_ping=True,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=1800,
                )
                with self._pg_engine.begin() as connection:
                    pg_ddl = SQLITE_SCHEMA_DDL.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                    connection.execute(text(pg_ddl))
                self.is_connected = True
                logger.info(f"Initialized PostgreSQL audit database schema at: {sanitized}")
            except Exception as e:
                logger.warning(
                    f"PostgreSQL connection to {sanitized} failed: {e}. "
                    f"Falling back to local SQLite at {DEFAULT_SQLITE_PATH}"
                )
                self.is_sqlite = True
                self.db_url = f"sqlite:///{DEFAULT_SQLITE_PATH}"
                self._pg_engine = None
                self.sqlite_path = self._resolve_sqlite_path()
                self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Returns standard SQLite connection."""
        if self.sqlite_path == ":memory:":
            conn = sqlite3.connect("file:audit_mem?mode=memory&cache=shared", uri=True, timeout=10.0)
        else:
            conn = sqlite3.connect(self.sqlite_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _execute_query_pg(self, query: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        from sqlalchemy import text
        parts = query.split("?")
        pg_query = parts[0]
        param_dict = {}
        for i in range(len(params)):
            param_dict[f"p{i}"] = params[i]
            pg_query += f":p{i}" + parts[i + 1]

        assert self._pg_engine is not None, "PostgreSQL engine is not initialized"
        with self._pg_engine.connect() as conn:
            result = conn.execute(text(pg_query), param_dict)
            return [dict(row._mapping) for row in result]

    def _execute_mutation_pg(self, statement: str, params: Tuple[Any, ...]) -> int:
        from sqlalchemy import text
        parts = statement.split("?")
        pg_stmt = parts[0]
        param_dict = {}
        for i in range(len(params)):
            param_dict[f"p{i}"] = params[i]
            pg_stmt += f":p{i}" + parts[i + 1]

        assert self._pg_engine is not None, "PostgreSQL engine is not initialized"
        with self._pg_engine.begin() as conn:
            result = conn.execute(text(pg_stmt), param_dict)
            return result.rowcount

    async def execute_query(self, query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        """Asynchronously executes SELECT query."""
        if not self.is_sqlite and self._pg_engine is not None:
            return await asyncio.to_thread(self._execute_query_pg, query, params)

        def _query() -> List[Dict[str, Any]]:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()

        return await asyncio.to_thread(_query)

    async def execute_mutation(self, statement: str, params: Tuple[Any, ...] = ()) -> int:
        """Asynchronously executes INSERT/UPDATE/DELETE statement with explicit rollback on error."""
        if not self.is_sqlite and self._pg_engine is not None:
            return await asyncio.to_thread(self._execute_mutation_pg, statement, params)

        def _mutate() -> int:
            conn = self.get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(statement, params)
                conn.commit()
                return cursor.rowcount
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return await asyncio.to_thread(_mutate)

    async def ping(self) -> bool:
        """Liveness check for audit database."""
        try:
            res = await self.execute_query("SELECT 1 AS alive;")
            return len(res) > 0 and res[0].get("alive") == 1
        except Exception as e:
            logger.error(f"Audit database ping failed: {e}")
            return False

    def close(self) -> None:
        """Gracefully closes open database connections and releases locks."""
        if self._shared_memory_conn is not None:
            try:
                self._shared_memory_conn.close()
            except Exception as e:
                logger.warning(f"Error closing shared memory connection: {e}")
            self._shared_memory_conn = None
        if self._pg_engine is not None:
            try:
                self._pg_engine.dispose()
            except Exception as e:
                logger.warning(f"Error disposing PostgreSQL engine: {e}")
            self._pg_engine = None
        self.is_connected = False
        logger.info("DatabaseManager connections closed.")
