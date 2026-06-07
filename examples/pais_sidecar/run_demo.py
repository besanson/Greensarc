"""KAOS adapter demo: Green SARC as a PAIS sidecar (hard enforcement).

Run it::

    python examples/pais_sidecar/run_demo.py

Unlike the MCP adapter (where the agent *chooses* to consult the gate), the
sidecar gates **every** call to the PAIS ``/v1/chat/completions`` endpoint. A
rejected call is answered with HTTP 429 and never reaches the model.

This demo stands up a tiny in-process mock of PAIS, wraps it with
``GreenSarcASGIMiddleware``, and drives a few chat-completion requests through
it — showing an admitted call (200, audited) and then a rejected call (429) once
the budget is too small for the forecast. No web server or HTTP client needed:
we call the ASGI app directly.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from green_sarc import Budget, LearnedEstimator, TableCarbonModel, TableCostModel
from green_sarc.adapters.pais_sidecar import GreenSarcASGIMiddleware, SidecarGate


def mock_pais(usage_total: int):
    """A minimal ASGI app standing in for the PAIS chat-completions endpoint."""

    async def app(scope, receive, send):
        more = True
        while more:
            msg = await receive()
            more = msg.get("more_body", False) if msg["type"] == "http.request" else False
        payload = json.dumps(
            {
                "id": "chatcmpl-demo",
                "object": "chat.completion",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Here you go."}}
                ],
                "usage": {"total_tokens": usage_total},
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": payload, "more_body": False})

    return app


async def call(app, request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Drive the ASGI app once with a POST to /v1/chat/completions."""
    body = json.dumps(request).encode()
    scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}
    sent = {"done": False}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    out: List[Dict[str, Any]] = []

    async def send(message):
        out.append(message)

    await app(scope, receive, send)
    return out


async def main() -> None:
    cost_model = TableCostModel()
    carbon_model = TableCarbonModel(intensities={"eu-west": 230.0})
    budget = Budget(token_budget=900.0, carbon_ceiling=5.0, delta=0.05)

    sidecar = SidecarGate(
        budget=budget,
        estimator=LearnedEstimator(cost_model, carbon_model, min_samples=2),
        cost_model=cost_model,
        carbon_model=carbon_model,
        region="eu-west",
    )
    # The sidecar wraps PAIS; the agent's model traffic flows through it.
    app = GreenSarcASGIMiddleware(mock_pais(usage_total=250), sidecar)

    requests = [
        {
            "model": "gpt-x",
            "messages": [{"role": "user", "content": "Summarise this."}],
            "max_tokens": 180,
        },
        {
            "model": "gpt-x",
            "messages": [{"role": "user", "content": "And again."}],
            "max_tokens": 180,
        },
        # By here the budget is nearly drained; a long-context call is rejected.
        {
            "model": "gpt-x-128k",
            "messages": [{"role": "user", "content": "x" * 400}],
            "max_tokens": 4000,
        },
    ]

    print("Driving /v1/chat/completions through the Green SARC sidecar:\n")
    for i, req in enumerate(requests):
        messages = await call(app, req)
        status = messages[0]["status"]
        if status == 429:
            err = json.loads(messages[1]["body"])["error"]
            print(f"  request {i}: HTTP 429 BLOCKED — {err['message']}")
        else:
            usage = json.loads(messages[1]["body"]).get("usage", {})
            print(
                f"  request {i}: HTTP {status} OK — model ran, "
                f"actual={usage.get('total_tokens')} tokens; "
                f"budget left={budget.remaining_tokens():.0f}"
            )

    print("\nAudit log (only admitted calls reach the model):")
    for rec in sidecar.store.list():
        print(
            f"  {rec.action_kind:<16} pred={rec.predicted_cost:7.1f} "
            f"actual={rec.actual_cost:7.1f} err={rec.cost_error:+7.1f} "
            f"carbon={rec.actual_carbon:.4f}g"
        )


if __name__ == "__main__":
    asyncio.run(main())
