"""Command-line interface for Green SARC.

The ``green-sarc`` command inspects a JSON Lines audit log produced by the
Post-Action Auditor and prints predicted-vs-actual accuracy — the signal that
tells you whether the estimator is learning.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from green_sarc.stores.jsonl import JSONLAuditStore

__all__ = ["main"]


def _inspect(path: str) -> int:
    store = JSONLAuditStore(path)
    records = store.list()
    if not records:
        print(f"no audit records in {path}")
        return 0

    n = len(records)
    admitted = sum(1 for r in records if r.admitted)
    tripped = sum(1 for r in records if r.circuit_tripped)
    escalated = sum(1 for r in records if r.escalated)
    ran = [r for r in records if r.actual_cost > 0]

    print(f"audit log: {path}")
    print(f"  records           : {n}")
    print(f"  admitted          : {admitted}")
    print(f"  rejected/escalated: {n - admitted} ({escalated} escalated)")
    print(f"  circuit trips     : {tripped}")

    if ran:
        mae_cost = sum(abs(r.cost_error) for r in ran) / len(ran)
        mae_carbon = sum(abs(r.carbon_error) for r in ran) / len(ran)
        total_actual_cost = sum(r.actual_cost for r in ran)
        total_actual_carbon = sum(r.actual_carbon for r in ran)
        print(f"  actions executed  : {len(ran)}")
        print(f"  total tokens spent: {total_actual_cost:.0f}")
        print(f"  total carbon (g)  : {total_actual_carbon:.3f}")
        print(f"  MAE token cost    : {mae_cost:.1f}")
        print(f"  MAE carbon (g)    : {mae_carbon:.4f}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``green-sarc`` console script."""
    parser = argparse.ArgumentParser(prog="green-sarc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="summarise a JSONL audit log")
    inspect.add_argument("path", help="path to the audit .jsonl file")

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "inspect":
        return _inspect(args.path)
    parser.print_help()  # pragma: no cover - argparse enforces a subcommand
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
