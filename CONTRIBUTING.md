# Contributing

Thanks for your interest in Green SARC.

## Development setup

```bash
pip install -e ".[dev]"
# optional adapters: pip install -e ".[dev,mcp,otel,sarc,tiktoken]"
```

## Quality gate

Every change must pass the full gate (run before pushing):

```bash
make quality   # ruff (lint + format) · mypy · pytest
```

- **Lint/format:** ruff, line length 99 (`ruff check`, `ruff format`).
- **Types:** `mypy src/green_sarc`.
- **Tests:** `pytest -q`; async tests run under `asyncio_mode=auto`. Adapter tests
  for optional extras skip cleanly when the extra is not installed.

## Conventions

- Keep the **core framework-agnostic**: nothing under `src/green_sarc/` (outside
  `adapters/`) may import an orchestrator (KAOS) or SARC. Optional third-party
  imports in adapters are lazy.
- Match the surrounding style: `@dataclass(frozen=True)` value objects,
  `str, Enum` enums, `@runtime_checkable` Protocols, explicit `__all__`,
  Google-style docstrings, `from __future__ import annotations`.
- Add tests with every behavior change.

## Commits & PRs

- Write focused commits with a clear subject line and a body explaining the
  *why*. Reference the audit item (e.g. `P0-1`) when applicable.
- Open a PR against `main`; CI runs on Python 3.11 and 3.12.
