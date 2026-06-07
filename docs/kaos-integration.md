# KAOS integration

> **Source of truth:** KAOS lives at
> [github.com/axsaucedo/kaos](https://github.com/axsaucedo/kaos) and its agent
> runtime PAIS at
> [github.com/axsaucedo/pydantic-ai-server](https://github.com/axsaucedo/pydantic-ai-server).
> Field names, CRD schemas, and endpoints below should be verified against those
> repositories — they are the authority, not this page.

[KAOS](https://github.com/axsaucedo/kaos) is a Kubernetes-native framework for
deploying and orchestrating AI agents. In the Green SARC picture it is the
**caller**: KAOS orchestrates agents and Green SARC governs the cost and carbon
of each agent action.

## The one-way dependency

```
   KAOS (orchestrator)  ───────────────►  Green SARC (cost/carbon governance)
        the caller          one way            the callee

   Green SARC NEVER imports or depends on KAOS.
```

Green SARC's core
([`src/green_sarc/`](../src/green_sarc/), excluding `adapters/`) has no knowledge
of KAOS and no runtime dependencies. Every KAOS touchpoint is an adapter in
[`src/green_sarc/adapters/`](../src/green_sarc/adapters/). This is enforced
structurally: the adapters import the core, never the reverse, and the optional
`mcp` / `opentelemetry` packages are imported lazily so the core stays
dependency-free.

## Why integration is not a function call

KAOS is **Kubernetes-native, not an importable library**. From its
[README](https://github.com/axsaucedo/kaos):

- Agents are declared as **custom resources** — `kind: Agent`, `MCPServer`,
  `ModelAPI` under the API group `kaos.tools/v1alpha1`
  (see the operator's API types:
  [`agent_types.go`](https://github.com/axsaucedo/kaos/blob/main/operator/api/v1alpha1/agent_types.go),
  [`mcpserver_types.go`](https://github.com/axsaucedo/kaos/blob/main/operator/api/v1alpha1/mcpserver_types.go),
  [`modelapi_types.go`](https://github.com/axsaucedo/kaos/blob/main/operator/api/v1alpha1/modelapi_types.go)).
- Agents run on **PAIS** and expose an OpenAI-compatible
  `POST /v1/chat/completions` endpoint
  ([pydantic-ai-server](https://github.com/axsaucedo/pydantic-ai-server)).
- Tools integrate via **MCP** (Model Context Protocol): an agent lists MCP
  servers by name in `spec.mcpServers`, and the Agent controller injects each
  server's endpoint into the agent pod as environment variables
  (`MCP_SERVERS`, `MCP_SERVER_<NAME>_URL`); PAIS connects over MCP Streamable
  HTTP and discovers tools at runtime.
- Observability is via **OpenTelemetry** — PAIS emits per-request spans and
  metrics to an OTLP collector when telemetry is enabled.

So "KAOS calls Green SARC" is one of three concrete surfaces, each matching one
of these mechanisms.

## The three integration surfaces

Green SARC ships an adapter for each. They wrap the **same** framework-agnostic
core gate and auditor; choose by how much enforcement you need.

| Surface | KAOS mechanism | Adapter | Enforcement | KAOS change |
|---|---|---|---|---|
| **MCP server** | MCP tools (`spec.mcpServers`) | [`adapters/mcp.py`](../src/green_sarc/adapters/mcp.py) | Advisory (agent-invoked) | None — an `MCPServer` CR |
| **PAIS sidecar** | ASGI middleware on `/v1/chat/completions` | [`adapters/pais_sidecar.py`](../src/green_sarc/adapters/pais_sidecar.py) | **Hard** (HTTP 429, unbypassable) | Pod spec patch (sidecar) |
| **OTel consumer** | OpenTelemetry span stream | [`adapters/otel.py`](../src/green_sarc/adapters/otel.py) | Observe only | None |

### 1. MCP server (recommended default) — advisory

KAOS is MCP-native, so the lowest-touch surface is to run Green SARC as an
ordinary MCP server exposing two tools:

- `pre_action_gate(...)` → forecast + admit/reject verdict;
- `post_action_auditor(...)` → report actual token usage, write the audit record,
  retrain.

[`GreenSarcMCPService`](../src/green_sarc/adapters/mcp.py) holds the logic;
`build_mcp_server(...)` wraps it in a [FastMCP](https://github.com/modelcontextprotocol)
server (`pip install 'green-sarc[mcp]'`). It registers with KAOS as an `MCPServer`
custom resource — **no change to KAOS or PAIS** — and an agent opts in by listing
it under `spec.mcpServers`. See
[`examples/kaos_mcp_adapter/`](../examples/kaos_mcp_adapter/) for the manifests
and a runnable demo, and [Deployment](#deployment) below.

**Trade-off:** MCP tools are *agent-invoked*. The agent must be instructed to
consult the gate and honour its verdict; a non-cooperative agent can bypass it.
For guaranteed enforcement, use the sidecar.

### 2. PAIS sidecar — hard enforcement

PAIS exposes `POST /v1/chat/completions`
([pydantic-ai-server](https://github.com/axsaucedo/pydantic-ai-server)). The
sidecar puts the **same** core gate in ASGI middleware in front of that endpoint
so **every** model call is gated:

- on reject → return **HTTP 429**; the call never reaches the model;
- on admit → forward to PAIS, then read actual token usage from the response and
  write the audit record.

[`GreenSarcASGIMiddleware`](../src/green_sarc/adapters/pais_sidecar.py) wraps the
PAIS ASGI `app`; [`SidecarGate`](../src/green_sarc/adapters/pais_sidecar.py) is
the testable core. It is **dependency-free** (pure ASGI — no FastAPI/Starlette/
httpx import). Deploy it as a sidecar container in the agent pod via the Agent
CR's pod-spec override, or wrap the PAIS `app` directly.

See [`examples/pais_sidecar/run_demo.py`](../examples/pais_sidecar/run_demo.py)
for a runnable demo that gates a mock `/v1/chat/completions` and returns a 429.

### 3. OpenTelemetry consumer — observe actuals

PAIS emits per-request OpenTelemetry spans; real token usage lands there as
`gen_ai.usage.*` attributes (pydantic-ai's instrumentation). The Post-Action
Auditor's predicted-vs-actual log can be fed from that stream instead of an
explicit auditor call.
[`OTelActualsConsumer.ingest_span`](../src/green_sarc/adapters/otel.py) maps a
span to actuals (implemented and unit-tested); the live OTLP receiver that pushes
spans into it is a documented stub. Correlate a gated call with its span by
setting the `green_sarc.action_id` attribute as OpenTelemetry baggage at the gate.

## Known caveats (verify upstream)

- **PAIS reports `usage` as zero.** At the time of writing PAIS hardcodes the
  `usage` block of its chat-completion responses to `0`. The sidecar therefore
  falls back to a length-based token estimate, and the **OTel span is the best
  source of real actuals**. Track the upstream
  [pydantic-ai-server](https://github.com/axsaucedo/pydantic-ai-server) for when
  real usage is surfaced.
- **MCP gating is advisory.** See surface 1's trade-off.
- **CRD field names evolve.** The `kaos.tools/v1alpha1` schema is alpha; confirm
  field names and casing against the operator's
  [api/v1alpha1](https://github.com/axsaucedo/kaos/tree/main/operator/api/v1alpha1)
  before deploying.

## Deployment

A reference container and manifests are provided:

- [`deploy/Dockerfile`](../deploy/Dockerfile) builds an image that runs the Green
  SARC MCP server ([`examples/kaos_mcp_adapter/server.py`](../examples/kaos_mcp_adapter/server.py),
  configured from environment variables).
- [`examples/kaos_mcp_adapter/kaos_manifests.yaml`](../examples/kaos_mcp_adapter/kaos_manifests.yaml)
  registers it as an `MCPServer` and an `Agent` that consults it.

```bash
# Build and push the MCP server image (swap in your registry).
docker build -t ghcr.io/besanson/green-sarc-mcp:0.1.0 -f deploy/Dockerfile .
docker push ghcr.io/besanson/green-sarc-mcp:0.1.0

# Register with KAOS (the operator injects MCP_SERVER_GREEN_SARC_URL into agents
# that list `green-sarc` under spec.mcpServers).
kubectl apply -f examples/kaos_mcp_adapter/kaos_manifests.yaml
```

The server reads its budget and tables from environment variables
(`GREEN_SARC_TOKEN_BUDGET`, `GREEN_SARC_CARBON_CEILING`, `GREEN_SARC_DELTA`,
`GREEN_SARC_CARBON_INTENSITY`); see
[`server.py`](../examples/kaos_mcp_adapter/server.py).

## Standalone first

Every surface above is optional. Green SARC runs with **no orchestrator
present** — see [`examples/standalone_agent_loop/run_demo.py`](../examples/standalone_agent_loop/run_demo.py)
and [quickstart.md](quickstart.md). KAOS is one caller; the core does not require
it.
