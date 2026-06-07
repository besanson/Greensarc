"""SQLite audit store — durable and queryable (audit P1-6).

JSONL is great for append-only replay; SQLite adds indexed, queryable runs
(filter by model/region/time) without any third-party dependency.  Records are
stored as a JSON blob plus a few indexed columns.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, List

from green_sarc.auditor import AuditRecord

__all__ = ["SQLiteAuditStore"]


class SQLiteAuditStore:
    """Append audit records to a SQLite database."""

    def __init__(self, path: Any) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT,
                model TEXT,
                region TEXT,
                timestamp REAL,
                data TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_model ON audit(model)")
        self._conn.commit()

    def append(self, record: AuditRecord) -> None:
        self._conn.execute(
            "INSERT INTO audit (action_id, model, region, timestamp, data) VALUES (?, ?, ?, ?, ?)",
            (
                record.action_id,
                record.model,
                record.region,
                record.timestamp,
                json.dumps(record.to_dict()),
            ),
        )
        self._conn.commit()

    def iter_records(self) -> Iterator[AuditRecord]:
        cursor = self._conn.execute("SELECT data FROM audit ORDER BY seq")
        for (data,) in cursor.fetchall():
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
        self._conn.close()
