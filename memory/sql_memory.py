"""
SQL memory backend using SQLite for the Reflexion memory study.

Provides structured retrieval by domain and error_type, enabling precise
credit assignment that is impossible with recency or semantic backends.
"""

# Database choice: SQLite (not Supabase or hosted PostgreSQL)
# Rationale for research reproducibility:
# 1. Zero external dependencies — runs entirely in-process, no network calls
# 2. No latency contamination — retrieval time is microseconds, not 10-100ms network round trips
#    that would confound our latency/cost metrics
# 3. Fully reproducible — experiment runs from `git clone` with no accounts or credentials
# 4. Sufficient scale — SQLite handles thousands of episodes with no performance issues
# If scaling to >100k episodes or multi-machine experiments, migrate to PostgreSQL via Docker
#    (a docker-compose.yml is provided in the repo root for this purpose).

import sqlite3
import logging
from memory.base import MemoryBackend

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         TEXT,
    domain          TEXT,
    attempt         INTEGER,
    success         INTEGER,
    reward          REAL,
    action_summary  TEXT,
    reflection      TEXT,
    error_type      TEXT,
    tokens_used     INTEGER,
    timestamp       TEXT
)
"""


class SQLMemory(MemoryBackend):
    """
    SQLite-backed episode memory with structured retrieval.

    Retrieval filters by domain first, then ranks by success DESC, reward DESC,
    timestamp DESC. The retrieve_by_error_type method enables structured credit
    assignment by error category — impossible with sliding window or vector backends.
    """

    def __init__(
        self,
        db_path: str = "./reflexion_episodes.db",
        retrieval_scope: str = "domain",
    ) -> None:
        """
        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for tests.
            retrieval_scope: 'domain' — filter by current domain before ranking (higher precision).
                             'global' — search all rows regardless of domain (higher recall).
        """
        self._db_path = db_path
        self._retrieval_scope = retrieval_scope
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def store(self, episode: dict) -> None:
        """Insert episode into the database. Converts success bool to int."""
        self._conn.execute(
            """
            INSERT INTO episodes
                (task_id, domain, attempt, success, reward, action_summary,
                 reflection, error_type, tokens_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode["task_id"],
                episode["domain"],
                episode["attempt"],
                int(episode["success"]),
                episode["reward"],
                episode["action_summary"],
                episode["reflection"],
                episode["error_type"],
                episode["tokens_used"],
                episode["timestamp"],
            ),
        )
        self._conn.commit()

    def retrieve(self, query: dict, k: int) -> list[dict]:
        """
        Retrieve up to k episodes, filtered by domain if retrieval_scope='domain'.

        Results are ordered by success ASC, timestamp DESC — failures first, most
        recent first within each group. This surfaces the agent's own failure
        reflections before its successes, which is the correct ordering for
        Reflexion: the agent already knows how to succeed when it succeeds; what it
        needs at retry time are the lessons extracted from past failures.

        (Prior ordering was success DESC which buried failures and caused SQL to
        underperform sliding window in the v1 experiment — see audit findings.)

        Logs a warning if fewer than k episodes are available.
        """
        if self._retrieval_scope == "domain":
            cursor = self._conn.execute(
                """
                SELECT * FROM episodes
                WHERE domain = ?
                ORDER BY success ASC, timestamp DESC
                LIMIT ?
                """,
                (query["domain"], k),
            )
        else:
            cursor = self._conn.execute(
                """
                SELECT * FROM episodes
                ORDER BY success ASC, timestamp DESC
                LIMIT ?
                """,
                (k,),
            )
        rows = [dict(row) for row in cursor.fetchall()]
        if len(rows) < k:
            logger.warning(
                "SQLMemory: requested k=%d but only %d episodes available "
                "(scope=%s, domain=%s). Warm-up phase likely.",
                k, len(rows), self._retrieval_scope, query.get("domain"),
            )
        # Convert success int back to bool
        for row in rows:
            row["success"] = bool(row["success"])
        return rows

    def retrieve_by_error_type(self, error_type: str, k: int) -> list[dict]:
        """
        Retrieve episodes matching a specific error type.

        This method enables structured credit assignment by error category —
        the key advantage of SQL over sliding window and vector DB.
        Useful for surfacing targeted lessons when the agent encounters a known failure mode.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM episodes
            WHERE error_type = ?
            ORDER BY reward DESC, timestamp DESC
            LIMIT ?
            """,
            (error_type, k),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            row["success"] = bool(row["success"])
        return rows

    def get_success_rate_by_domain(self, domain: str) -> float:
        """Return fraction of stored episodes for this domain where success=True."""
        cursor = self._conn.execute(
            "SELECT AVG(success) FROM episodes WHERE domain = ?",
            (domain,),
        )
        result = cursor.fetchone()[0]
        return float(result) if result is not None else 0.0

    def reset(self) -> None:
        """Drop and recreate the episodes table."""
        self._conn.execute("DROP TABLE IF EXISTS episodes")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()

    def count(self) -> int:
        """Return total number of stored episodes."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0]
