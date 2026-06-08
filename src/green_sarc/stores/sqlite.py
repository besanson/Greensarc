"""SQLite audit store — durable and queryable (audit P1-6).

JSONL is great for append-only replay; SQLite adds indexed, queryable runs
(filter by model/region/time, group by plan) without any third-party dependency.
Records are stored as a JSON blob plus a few indexed columns.

A single connection is held for the life of the store (``check_same_thread`` is
disabled and writes are guarded by a lock so the same store can serve a threaded
server), WAL is enabled to reduce writer/reader contention, and a ``meta`` table
records the schema version for future migrations (audit N-4 / 2.2).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterator, List

from green_sarc.auditor import AuditRecord

__all__ = ["SQLiteAuditStore"]

_SCHEMA_VERSION = 1


class SQLiteAuditStore:
    """Append audit records to a SQLite database (one held connection, WAL)."""

    def __init__(self, path: Any) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT,
                model TEXT,
                region TEXT,
                plan_id TEXT,
                timestamp REAL,
                data TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_model ON audit(model)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit(timestamp)")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_plan_ts ON audit(plan_id, timestamp)"
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        self._conn.commit()

    def append(self, record: AuditRecord) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit (action_id, model, region, plan_id, timestamp, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.action_id,
                    record.model,
                    record.region,
                    record.plan_id,
                    record.timestamp,
                    json.dumps(record.to_dict()),
                ),
            )
            self._conn.commit()

    def iter_records(self) -> Iterator[AuditRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM audit ORDER BY seq").fetchall()
        for (data,) in rows:
            yield AuditRecord.from_dict(json.loads(data))

    def list(self) -> List[AuditRecord]:
        return list(self.iter_records())

    def export_jsonl(self, path: Any) -> int:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with dest.open("w", encoding="utf-8") as out:
            for record in self.iter_records():
                out.write(json.dumps(record.to_dict()) + "\n")
                count += 1
        return count

    def close(self) -> None:
        with self._lock:
            self._conn.close()
