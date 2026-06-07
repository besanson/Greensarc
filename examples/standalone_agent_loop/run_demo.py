"""Standalone demo: a small agent loop wrapped by the four enforcement sites.

Run it::

    python examples/standalone_agent_loop/run_demo.py

It shows, with no orchestrator and no external framework present:

1. an admitted action that runs and is audited (predicted vs actual),
2. a **gate rejection** when the forecast does not fit the remaining budget,
3. a **circuit-breaker trip** that kills a runaway retry loop, and
4. the resulting **auditor log**.

The "agent" here is a stand-in: ``fake_execute`` just reports a token count.
In a real system this coroutine would call the model / tool and return its
actual usage.
"""

from __future__ import annotations

import asyncio

from green_sarc import (
    Action,
    ActionOutcome,
    ActionTimeMonitor,
    Budget,
    CircuitTripped,
    EscalationEvent,
    EscalationRouter,
    GateRejected,
    GreenGovernor,
    LearnedEstimator,
    TableCarbonModel,
    TableCostModel,
)


def make_executor(tokens: float):
    """Return an async executor that reports a fixed token usage."""

    async def fake_execute(action: Action) -> ActionOutcome:
        # Pretend to call the model; report what it actually cost.
        return ActionOutcome(result=f"answer for {action.kind}", actual_tokens=tokens)

    return fake_execute


async def main() -> None:
    cost_model = TableCostModel()  # default energy profile
    carbon_model = TableCarbonModel(intensities={"eu-west": 230.0})  # gCO2e/kWh

    # A deliberately tight budget so we can watch the gate bite.
    budget = Budget(token_budget=1_500.0, carbon_ceiling=5.0, delta=0.05)

    async def on_escalation(event: EscalationEvent) -> None:
        print(f"  [ER] escalation: {event.reason.value} — {event.detail}")

    gov = GreenGovernor(
        budget=budget,
        estimator=LearnedEstimator(cost_model, carbon_model, min_samples=2),
        cost_model=cost_model,
        carbon_model=carbon_model,
        monitor=ActionTimeMonitor(max_loops=3),
        router=EscalationRouter(on_escalation),
    )

    def status() -> str:
        return (
            f"budget left: {budget.remaining_tokens():.0f} tokens, "
            f"{budget.remaining_carbon():.3f} gCO2e"
        )

    print("=== 1. Admitted actions (predict -> act -> log -> retrain) ===")
    for i in range(2):
        action = Action(
            kind="chat.completion",
            model="gpt-x",
            region="eu-west",
            prompt_tokens=120,
            max_tokens=180,
        )
        res = await gov.run_action(action, make_executor(250.0))
        print(
            f"  action {i}: admitted, predicted={res.decision.forecast.cost_hat:.0f} "
            f"actual={res.actual_cost:.0f} tokens "
            f"(source={res.decision.forecast.source}); {status()}"
        )

    print("\n=== 2. Gate rejection (forecast exceeds remaining budget) ===")
    # A never-seen, long-context action: the estimator has no history for this
    # key, so it cold-starts to the conservative worst case (prompt + max_tokens)
    # which does not fit the ~1000 tokens left.
    big = Action(
        kind="chat.completion",
        model="gpt-x-128k",
        region="eu-west",
        prompt_tokens=2_000,
        max_tokens=4_000,
    )
    try:
        await gov.run_action(big, make_executor(6_000.0))
    except GateRejected as exc:
        print(f"  rejected: {exc.decision.reason}")
        print(f"  (the action never ran; {status()})")

    print("\n=== 3. Circuit-breaker trip (runaway retry loop) ===")
    # max_loops=3 and we have already used 2 loops; a retry storm trips it.
    retry = Action(
        kind="retry.replan", model="gpt-x", region="eu-west", prompt_tokens=10, max_tokens=10
    )
    try:
        for _ in range(5):
            await gov.run_action(retry, make_executor(20.0))
    except CircuitTripped as exc:
        print(f"  breaker tripped: {exc.reason}")

    print("\n=== 4. Auditor log (ESG record + estimator training data) ===")
    for rec in gov.store.list():
        print(
            f"  {rec.action_kind:<16} admitted={rec.admitted!s:<5} "
            f"pred={rec.predicted_cost:7.1f} actual={rec.actual_cost:7.1f} "
            f"err={rec.cost_error:+7.1f} carbon={rec.actual_carbon:.4f}g "
            f"tripped={rec.circuit_tripped}"
        )


if __name__ == "__main__":
    asyncio.run(main())
