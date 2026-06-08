# Use it — govern your own agent in 5 minutes

Green SARC wraps the call your agent already makes. You supply one thing: an
`execute` coroutine that runs the action and reports its **actual token usage**.
Everything else (forecast, budget, breaker, audit) is wired for you.

## Install

```bash
pip install -e .            # from a clone (PyPI release: pip install green-sarc)
```

The core has zero runtime dependencies. Optional extras: `mcp`, `otel`, `sarc`,
`tiktoken`.

## The 3-line setup

```python
from green_sarc import GreenGovernor

gov = GreenGovernor.with_defaults(token_budget=200_000, usd_budget=5.00)
```

`with_defaults` wires a learning estimator plus the reference pricing/carbon
tables ([`green_sarc.data`](../src/green_sarc/data.py)) — approximate list prices
for common models (gpt-4o, claude-sonnet, …) and approximate grid intensity for
common cloud regions. **Override them for your real prices/grid:**

```python
from green_sarc import TableCostModel, ModelProfile, TableCarbonModel

gov = GreenGovernor.with_defaults(
    token_budget=200_000,
    usd_budget=5.00,
    cost_model=TableCostModel(profiles={
        "my-model": ModelProfile(energy_per_token_kwh=3e-7,
                                 usd_per_prompt_token=2e-6,
                                 usd_per_completion_token=8e-6),
    }),
    carbon_model=TableCarbonModel(intensities={"us-east-1": 370.0}),
)
```

## Wrap your call

```python
from green_sarc import Action, ActionOutcome, GateRejected

async def execute(action: Action) -> ActionOutcome:
    resp = await my_llm.chat(model=action.model, messages=msgs, max_tokens=action.max_tokens)
    return ActionOutcome(result=resp.text, actual_tokens=resp.usage.total_tokens)

action = Action(kind="chat.completion", model="gpt-4o", region="us-east-1",
                prompt_tokens=count_prompt_tokens(msgs), max_tokens=400)

try:
    result = await gov.run_action(action, execute)        # forecast → admit → run → audit
    use(result.result)
except GateRejected as exc:
    fallback(exc.decision.reason)                          # too expensive: down-route / cheaper model
```

That's it. The gate forecasts each call and admits it only if it fits the
remaining token / USD / carbon budget; the breaker kills runaway loops; the
auditor logs predicted-vs-actual and the estimator learns from it.

A complete, runnable version against any OpenAI-compatible endpoint is in
[`examples/openai_governed/run_demo.py`](../examples/openai_governed/run_demo.py).

## Persist what it learns

```python
from green_sarc import JSONLAuditStore

gov = GreenGovernor.with_defaults(token_budget=200_000, store=JSONLAuditStore("audit.jsonl"))
# ... later / after restart:
gov.estimator.bootstrap_from_jsonl("audit.jsonl")   # rehydrate the forecaster
```

Inspect accuracy any time:

```bash
green-sarc inspect audit.jsonl     # predicted-vs-actual MAE for tokens, USD, carbon
```

## Deploy under KAOS

You don't call Green SARC from KAOS in Python — KAOS is the caller, via an
adapter. See [kaos-integration.md](kaos-integration.md): the MCP server
(advisory), the PAIS sidecar (hard 429 enforcement), or the OTel consumer.
