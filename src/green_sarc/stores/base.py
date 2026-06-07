"""The audit-store protocol.

An :class:`AuditStore` is where the Post-Action Auditor appends
:class:`~green_sarc.auditor.AuditRecord` objects — the ESG/audit log and the
estimator's training corpus.  Backends only need to append, list, and export.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Protocol, runtime_checkable

from green_sarc.auditor import AuditRecord

__all__ = ["AuditStore"]


@runtime_checkable
class AuditStore(Protocol):
    """Minimal durable store for audit records."""

    def append(self, record: AuditRecord) -> None:
        """Append one record."""

    def list(self) -> List[AuditRecord]:
        """Return all records in insertion order."""

    def iter_records(self) -> Iterator[AuditRecord]:
        """Stream records in insertion order."""

    def export_jsonl(self, path: Any) -> int:
        """Write every record to ``path`` as JSON Lines; return the count."""

    def close(self) -> None:
        """Release resources. Idempotent."""
