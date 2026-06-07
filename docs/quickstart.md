# Quickstart

## Install

```bash
git clone https://github.com/besanson/greensarc
cd greensarc
pip install -e ".[dev]"            # core + test/lint tooling
# optional KAOS adapter extras:
pip install -e ".[dev,mcp,otel]"  # MCP server + OTel consumer deps
```

The core has **no runtime dependencies**. `mcp` and `opentelemetry-sdk` are
optional, pulled in only by their adapters. The PAIS sidecar needs nothing extra.

## Run the examples

```bash
# 1. Standalone: four sites on a mock agent — gate reject, breaker trip, audit log.
python examples/standalone_agent_loop/run_demo.py

# 2. KAOS MCP adapter (advisory): a mock agent driving the gate + auditor tools.
python examples/kaos_mcp_adapter/run_demo.py

# 3. PAIS sidecar (hard): gates a mock /v1/chat/completions, returns HTTP 429.
python examples/pais_sidecar/run_demo.py
```

## Govern your own loop

```python
import asyncio
from green_sarc import (
    Action, ActionOutcome, Budget, GreenGovernor, GateRejected,
    LearnedEstimator, TableCarbonModel, TableCostModel,
)

cost_model = TableCostModel()                                   # energy per token
carbon_model = TableCarbonModel(intensities={"eu-west": 230})   # gCO2e/kWh
budget = Budget(token_budget=10_000, carbon_ceiling=50.0, delta=0.05)

gov = GreenGovernor(
    budget=budget,
    estimator=LearnedEstimator(cost_model, carbon_model),
    cost_model=cost_model,
    carbon_model=carbon_model,
)

async def call_model(action: Action) -> ActionOutcome:
    # ... call the real model / tool, then report its actual token usage ...
    return ActionOutcome(result="...", actual_tokens=240)

async def main():
    action = Action(kind="chat.completion", model="gpt-x", region="eu-west",
                    prompt_tokens=120, max_tokens=180)
    try:
        result = await gov.run_action(action, call_model)
        print("ran:", result.actual_cost, "err:", result.audit.cost_error)
    except GateRejected as exc:
        print("blocked:", exc.decision.reason)

asyncio.run(main())
```

The estimator learns as it goes: the auditor logs predicted vs actual and feeds
each record back into the estimator (`predict → act → log → retrain`).

## Persist and inspect the audit log

```python
from green_sarc import GreenGovernor, JSONLAuditStore
gov = GreenGovernor(..., store=JSONLAuditStore("artifacts/audit.jsonl"))
```

```bash
green-sarc inspect artifacts/audit.jsonl   # predicted-vs-actual accuracy summary
```

## Develop

```bash
make quality   # ruff (lint + format) · mypy · pytest
```

## Where next

- [architecture.md](architecture.md) — the four sites and the learning loop.
- [kaos-integration.md](kaos-integration.md) — wire Green SARC into KAOS.
- [relationship-to-sarc.md](relationship-to-sarc.md) — what is borrowed from SARC.
