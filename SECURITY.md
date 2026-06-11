# Security policy

Green SARC is an alpha research library. We still take security seriously.

## Reporting a vulnerability

Please report suspected vulnerabilities privately to **besanson@gmail.com**.
Do not open a public issue for security reports.

Include, where possible:

- a description of the issue and its impact,
- steps to reproduce (a minimal example or test),
- affected version / commit.

We aim to acknowledge reports within a few days and to coordinate a fix and
disclosure within a **90-day** window.

## Scope notes

- Green SARC governs **cost and carbon only**; it is not a safety/correctness
  control and should not be relied on as one.
- The MCP adapter's gating is **advisory** (an agent may decline to call it). For
  non-bypassable enforcement use the PAIS sidecar.
- The MCP server, if exposed, effectively reserves budget; deploy it behind
  authentication in multi-tenant environments.

## Sidecar authentication

The PAIS sidecar (`green_sarc.adapters.pais_sidecar.GreenSarcASGIMiddleware`)
supports an optional shared-secret check: set `GREEN_SARC_AUTH_TOKEN` and
governed paths require `Authorization: Bearer <token>`, returning `401`
otherwise. Health probes (`/healthz`, `/readyz`) are always exempt. When the
variable is unset the sidecar stays open (unchanged behaviour) and logs a
one-time warning.

Enabling it closes the **budget-drain** and **auditor-spoofing** surface
described in the working paper's threat model (§13) *for the sidecar transport*:
an unauthenticated caller can no longer burn another tenant's budget or inject
audit records through the gated endpoint. **MCP transport authentication remains
on the roadmap** — the MCP server should still be deployed behind a network
authentication layer in multi-tenant environments.

A rejected call returns `429` with a `Retry-After` header and a structured body
(`reason`, `predicted_tokens`, `budget_remaining`) so clients can back off
without scraping the error string.
