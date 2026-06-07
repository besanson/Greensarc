# Green SARC documentation

Green SARC is a standalone, predictive **cost + carbon** governance layer for
agentic AI systems. It borrows its four-enforcement-site backbone from the SARC
framework and integrates with the KAOS orchestrator as an adapter.

This folder documents both relationships in depth. Start here:

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | The four enforcement sites and how `GreenGovernor` wires them; the `predict → act → log → retrain` loop; the Phase 1 / Phase 2 split. |
| [relationship-to-sarc.md](relationship-to-sarc.md) | Exactly what Green SARC borrows from the **SARC** framework, what it does **not**, the conventions it mirrors, and how the two compose. Links throughout to [`besanson/sarc-governance`](https://github.com/besanson/sarc-governance). |
| [kaos-integration.md](kaos-integration.md) | How **KAOS** calls Green SARC: the one-way dependency, the CRDs, and the three integration surfaces (MCP server, PAIS sidecar, OTel consumer) with deployment. Links throughout to [`axsaucedo/kaos`](https://github.com/axsaucedo/kaos) and [`axsaucedo/pydantic-ai-server`](https://github.com/axsaucedo/pydantic-ai-server). |
| [quickstart.md](quickstart.md) | Install, run the examples, govern your own loop, inspect the audit log. |

## Source-of-truth repositories

Green SARC deliberately depends on **neither** repo at runtime; it only aligns
with them. When this documentation describes SARC or KAOS, the authoritative
source is always the upstream repo — follow the links and verify against it.

- **SARC framework** — [github.com/besanson/sarc-governance](https://github.com/besanson/sarc-governance)
  (arXiv:2605.07728). The framework Green SARC borrows its four enforcement sites from.
- **KAOS** — [github.com/axsaucedo/kaos](https://github.com/axsaucedo/kaos).
  The Kubernetes-native agent orchestration framework that *calls* Green SARC.
- **PAIS** — [github.com/axsaucedo/pydantic-ai-server](https://github.com/axsaucedo/pydantic-ai-server).
  The agent runtime KAOS deploys; the sidecar adapter fronts its
  `/v1/chat/completions` endpoint.

## The one invariant

> The dependency runs **one way: KAOS → Green SARC**. Green SARC's core never
> imports SARC or KAOS. The orchestrator integrations live in
> [`src/green_sarc/adapters/`](../src/green_sarc/adapters/) as thin adapters on
> top of a framework-agnostic core that also runs standalone.
