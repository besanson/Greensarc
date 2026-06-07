"""In-memory audit store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, List

from green_sarc.auditor import AuditRecord

__all__ = ["MemoryAuditStore"]


class MemoryAuditStore:
    """Keeps audit records in a list. Suitable for tests and short runs."""

    def __init__(self) -> None:
        self._records: List[AuditRecord] = []

    def append(self, record: AuditRecord) -> None:
        self._records.append(record)

    def list(self) -> List[AuditRecord]:
        return list(self._records)

    def iter_records(self) -> Iterator[AuditRecord]:
        return iter(list(self._records))

    def export_jsonl(self, path: Any) -> int:
        p = Path(path)
        with p.open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        return len(self._records)

    def close(self) -> None:  # noqa: D102 - nothing to release
        return None

    def __len__(self) -> int:
        return len(self._records)
