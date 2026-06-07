# Relationship to SARC

> **Source of truth:** the SARC framework lives at
> [github.com/besanson/sarc-governance](https://github.com/besanson/sarc-governance)
> (arXiv:2605.07728). Where this page describes SARC, defer to that repository.

Green SARC is an **application** of the SARC governance architecture to the
FinOps / GreenOps domain. The relationship is deliberately narrow and is stated
here minimally and factually.

## What Green SARC borrows

**Exactly one thing: the four-enforcement-site architecture.** SARC structures
runtime governance as four enforcement sites around a tool call. Green SARC keeps
that backbone and specialises each site for cost and carbon:

| SARC enforcement site | Green SARC site | Specialisation |
|---|---|---|
| Pre-Action Gate (PAG) | [`gate.py`](../src/green_sarc/gate.py) | Admit on a **learned cost/carbon forecast** at confidence `1 − delta`, not a static rule. |
| Action-Time Monitor (ATM) | [`monitor.py`](../src/green_sarc/monitor.py) | Circuit breaker on **loop count and marginal/total token cost**. |
| Post-Action Auditor (PAA) | [`auditor.py`](../src/green_sarc/auditor.py) | Log **predicted vs actual** cost/carbon; the log doubles as estimator training data. |
| Escalation Router (ER) | [`escalation.py`](../src/green_sarc/escalation.py) | Route to human review / deterministic fallback on **budget or carbon exhaustion**. |

That is the whole of the borrowing. Green SARC makes **no** claim about what SARC
governs beyond providing these four sites. (SARC's own enforcement model — its
constraint classes, responses, and class-to-point compatibility rules — is
documented in its repository, not asserted here.)

## What Green SARC does *not* take from SARC

- **No dependency.** Green SARC's core imports nothing from SARC. It is in a
  separate package (`green_sarc`) with no runtime dependencies at all. You can
  run it with no SARC, and no safety framework of any kind, present.
- **No safety / correctness / quality governance.** Green SARC governs **cost and
  carbon only**. It never tracks accuracy, safety, or output quality as a
  governed quantity. Those concerns belong to a safety layer (such as SARC), not
  here.

## How the two compose

Because both express governance as the *same four enforcement sites*, they can
share those sites in a single agent without conflict:

- A safety layer (e.g. SARC) governs correctness/safety at the gate and auditor.
- Green SARC governs cost/carbon at the *same* gate and auditor.

Composition is optional and never required. Green SARC is standalone first; a
safety layer is an independent, parallel concern. Nothing in Green SARC assumes a
safety layer is present, and nothing in it provides one.

### Composition in code: the SARC adapter

The composition above is realised concretely by
[`green_sarc/adapters/sarc.py`](../src/green_sarc/adapters/sarc.py). It expresses
Green SARC's predictive cost/carbon control as SARC
[`Constraint`](https://github.com/besanson/sarc-governance) objects and plugs
them into a SARC `GovernanceToolset`:

- the **Pre-Action Gate** becomes a HARD constraint at SARC's `PAG` site (it
  fires — and SARC blocks with `ConstraintViolation` — when the forecast does not
  fit the budget / carbon ceiling);
- the **Post-Action Auditor** becomes a SOFT constraint at SARC's `PAA` site (it
  reads actual token usage from the tool result, writes the predicted-vs-actual
  record, spends the budget, and retrains the estimator).

`wrap_toolset(toolset, governance, spec=safety_spec)` appends these to a caller's
existing safety `ConstraintSpec`, so **one** governed toolset enforces both
concerns at the shared four sites. This requires the optional extra
(`pip install 'green-sarc[sarc]'`); the Green SARC core still does not import
SARC. See [`examples/sarc_composition/run_demo.py`](../examples/sarc_composition/run_demo.py).

## Conventions mirrored from the SARC repo

So that a reader of one repository recognises the other, Green SARC follows
`sarc-governance`'s structure and style:

| Convention | SARC | Green SARC |
|---|---|---|
| Layout | `src/sarc_governance/`, `adapters/`, `stores/`, `tests/`, `examples/`, `docs/` | `src/green_sarc/`, `adapters/`, `stores/`, `tests/`, `examples/`, `docs/` |
| Packaging | setuptools, `pyproject.toml`, Python ≥ 3.11, MIT | identical |
| Tooling | ruff (line-length 99) + mypy + pytest-asyncio (`asyncio_mode=auto`); CI on 3.11/3.12 | identical |
| Value objects | `@dataclass(frozen=True)`, `str, Enum` enums, `@runtime_checkable` Protocols | identical |
| Records | `to_dict()` / `from_dict()` round-trips; explicit `__all__`; `from __future__ import annotations`; Google-style docstrings | identical |
| Escalation | async router; a broken handler is logged and suppressed so it can never break the agent loop | identical |
| KAOS/PAIS adapter | `sarc_governance/adapters/pais.py` | `green_sarc/adapters/mcp.py`, `pais_sidecar.py`, `otel.py` |

The intent is consistency, not coupling: the conventions are copied, the code is
not. See SARC's own [`docs/`](https://github.com/besanson/sarc-governance/tree/main/docs)
for the originals.

## Reference

Besanson, G. (2026). *Green SARC: Predictive FinOps as Governance-by-Architecture
for Agentic AI Systems.* Working paper. Borrows the four-enforcement-site
architecture from the SARC framework (arXiv:2605.07728,
[github.com/besanson/sarc-governance](https://github.com/besanson/sarc-governance)).
