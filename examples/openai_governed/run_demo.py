"""Govern a *real* OpenAI-compatible agent loop with Green SARC.

This is the copy-paste path for using Green SARC on real traffic (OpenAI, Azure
OpenAI, or any OpenAI-compatible server — including a local one via
``OPENAI_BASE_URL``). The whole integration is the ``execute`` coroutine plus the
one-line ``GreenGovernor.with_defaults(...)``.

Run it::

    pip install openai
    export OPENAI_API_KEY=sk-...        # and optionally OPENAI_BASE_URL=...
    python examples/openai_governed/run_demo.py

With no key/SDK it prints how to run instead of failing — the point here is the
~10 lines of integration, not the network call.
"""

from __future__ import annotations

import asyncio
import os

from green_sarc import (
    DEFAULT_REGION,
    Action,
    ActionOutcome,
    GateRejected,
    GreenGovernor,
)

PROMPTS = [
    "Name three primary colors.",
    "Summarize the plot of Hamlet in one sentence.",
    "List two benefits of unit tests.",
    "What is the capital of Japan?",
]


async def main() -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("This live example needs the OpenAI SDK:  pip install openai")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) to run this live.")
        return

    client = AsyncOpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL from the env
    model = os.environ.get("GREEN_SARC_MODEL", "gpt-4o-mini")

    # One line wires the four sites with sensible defaults + the reference
    # pricing/carbon tables. Budgets are yours to set.
    gov = GreenGovernor.with_defaults(token_budget=20_000, usd_budget=0.25)

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        action = Action(
            kind="chat.completion",
            model=model,
            region=DEFAULT_REGION,
            prompt_tokens=max(1, len(prompt) // 4),  # rough; the estimator learns the rest
            max_tokens=200,
        )

        async def execute(act: Action, _messages=messages) -> ActionOutcome:
            resp = await client.chat.completions.create(
                model=act.model, messages=_messages, max_tokens=act.max_tokens
            )
            usage = resp.usage
            total = float(usage.total_tokens) if usage else 0.0
            return ActionOutcome(result=resp.choices[0].message.content, actual_tokens=total)

        try:
            result = await gov.run_action(action, execute)
            print(
                f"[ok]    {prompt[:40]:<40} "
                f"actual={result.actual_cost:.0f} tok  ${result.audit.actual_usd:.5f}  "
                f"left: {gov.budget.remaining_tokens():.0f} tok / ${gov.budget.remaining_usd():.4f}"
            )
        except GateRejected as exc:
            print(f"[block] {prompt[:40]:<40} {exc.decision.reason}")

    print(
        f"\nSpent: ${gov.budget.usd_spent:.4f}, {gov.budget.carbon_spent:.4f} gCO2e "
        f"across {len(gov.store.list())} audited calls; "
        f"{gov.budget.remaining_tokens():.0f} tokens left"
    )
    print("\nAudit log (the ESG record + the estimator's training data):")
    for rec in gov.store.list():
        print(
            f"  pred={rec.predicted_cost:7.1f}  actual={rec.actual_cost:7.1f}  "
            f"err={rec.cost_error:+7.1f}  ${rec.actual_usd:.5f}  {rec.actual_carbon:.4f} gCO2e"
        )


if __name__ == "__main__":
    asyncio.run(main())
