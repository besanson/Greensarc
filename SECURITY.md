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
