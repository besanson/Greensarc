"""Provenance linter: every statistic cited in green-sarc.tex must live in data.

Greps the ``.tex`` body for statistic-shaped tokens (percentages, ``N.N~pp``
deviations, ``R^2``/coefficient decimals) and checks each against the flattened
set of numbers in ``paper/data/figure_stats.json`` (the single source of truth),
allowing for rounding.  Structural numbers (section/theorem refs, years, axis
bounds, dataset window sizes, page counts) are allow-listed.

    python paper/scripts/check_stats.py            # advisory report
    python paper/scripts/check_stats.py --strict   # exit 1 on any unmatched stat
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "green-sarc.tex"
STATS = ROOT / "paper" / "data" / "figure_stats.json"

# Numbers that legitimately appear in prose but are not data-derived statistics.
ALLOW = {
    # structural / domain constants
    "0.0", "1.0", "0.5", "2.0", "1.5", "0.25", "0.75", "0.05", "0.1", "0.01",
    "0.02", "0.15", "0.2", "0.3", "0.4", "0.9", "0.95", "0.99", "0.98", "0.8",
    "0.6", "0.7", "0.85", "12", "13", "15", "8192", "4096", "8", "4", "50",
    "20", "400", "10", "16", "94", "1.64", "2.0", "9.5", "8.5", "7.5",
    "3.11", "3.12", "113", "45", "3.5",  # Python versions, test count, max turn depth
}


def _flatten(obj, out: Set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _flatten(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _flatten(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        v = float(obj)
        # Record the value at several roundings, plus its percentage form.
        for r in (0, 1, 2):
            out.add(f"{round(v, r):.{r}f}".rstrip("0").rstrip("."))
        for r in (0, 1, 2):
            out.add(f"{round(v * 100, r):.{r}f}".rstrip("0").rstrip("."))
        for r in (1, 2):  # absolute value (reductions cited as positive %)
            out.add(f"{round(abs(v), r):.{r}f}".rstrip("0").rstrip("."))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    if not STATS.exists():
        print("figure_stats.json missing; run `make paper-figures` first.")
        return 1
    values: Set[str] = set()
    _flatten(json.loads(STATS.read_text()), values)
    tex = TEX.read_text()
    # Drop bibliography (years, page ranges, DOIs) from the scan.
    tex = tex.split("\\begin{thebibliography}")[0]

    # Statistic-shaped tokens: decimals optionally followed by % or pp.
    cited = re.findall(r"(?<![\w.])(\d+\.\d+)(?:\\?%|~?pp)?", tex)
    missing = []
    for tok in cited:
        norm = tok.rstrip("0").rstrip(".")
        if norm in ALLOW or norm in values:
            continue
        # tolerate +-0.1 rounding against any recorded value
        try:
            x = float(tok)
            if any(abs(x - float(v)) <= 0.1 for v in values if _isnum(v)):
                continue
        except ValueError:
            pass
        missing.append(tok)

    uniq = sorted(set(missing), key=float)
    if uniq:
        print(f"WARN: {len(uniq)} cited decimal(s) not found in figure_stats.json:")
        print("  " + ", ".join(uniq))
        print("  (add to figure_stats.json, or to the ALLOW set if structural.)")
        return 1 if args.strict else 0
    print("OK: every statistic-shaped number in the body resolves to figure_stats.json.")
    return 0


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
