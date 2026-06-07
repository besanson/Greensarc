"""JSON Lines audit store — durable, append-only, replayable.

Each audit record is one JSON object per line.  Because the audit log is also
the estimator's training data, an append-only JSONL file is the natural form: it
can be replayed to retrain an estimator from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List

from green_sarc.auditor import AuditRecord

__all__ = ["JSONLAuditStore"]


class JSONLAuditStore:
    """Append audit records to a JSON Lines file."""

    def __init__(self, path: Any) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict()) + "\n")

    def iter_records(self) -> Iterator[AuditRecord]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield AuditRecord.from_dict(json.loads(line))

    def list(self) -> List[AuditRecord]:
        return list(self.iter_records())

    def export_jsonl(self, path: Any) -> int:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with dest.open("w", encoding="utf-8") as out:
            for rec in self.iter_records():
                out.write(json.dumps(rec.to_dict()) + "\n")
                count += 1
        return count

    def close(self) -> None:  # noqa: D102 - file handles are not held open
        return None
