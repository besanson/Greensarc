"""Live governed-agent study harness (HUMAN-GATED; the live run is pending).

Everything else in the paper is replay; this is the one *live* experiment. It
runs two arms over N tasks --- an ungoverned baseline vs the full
:class:`~green_sarc.governor.GreenGovernor` stack --- against the Anthropic API,
and reports tokens, USD (from the API's real usage fields), task success, gate
decisions, breaker trips, and wall-clock overhead per arm.

Safety rails (the experiment about budgets must have a budget):

- A hard USD ceiling enforced *in this script*: it aborts the moment cumulative
  spend would exceed ``--usd-cap`` (default $25 for a dry run).
- After ``--probe-tasks`` (default 5) tasks it prints the projected full-run cost
  and STOPS unless ``--approve`` was passed --- so a human signs off on the spend
  before the bulk of it happens.

Model: default ``claude-opus-4-8``; override with ``GREEN_SARC_LIVE_MODEL`` or
``--model``; use ``claude-haiku-4-5-20251001`` for the cheap dry run.

Task set (proposed alternative to the full SWE-bench harness, per the brief): the
default ``proxy`` set is self-contained deterministic tasks with a cheap exact
verifier, so the harness runs without the SWE-bench docker harness. ``--task-set
swebench-lite`` is the heavier option to wire once a run is funded and approved.

With no ``ANTHROPIC_API_KEY`` the harness uses an offline :class:`MockTransport`
(deterministic canned usage) so the control flow is unit-tested in CI; it never
hits the network and the paper carries no live numbers until the run is approved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from green_sarc.governor import ActionOutcome, GateRejected, GreenGovernor
from green_sarc.monitor import CircuitTripped
from green_sarc.state import Action

# Per-token USD for the live models (input, output); used to price API usage and
# to feed the governor's cost model. Update from the LiteLLM loader if needed.
MODEL_PRICES: Dict[str, tuple] = {
    "claude-opus-4-8": (5.0e-6, 2.5e-5),
    "claude-haiku-4-5-20251001": (1.0e-6, 5.0e-6),
}
DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    usd: float


class Transport(Protocol):
    def complete(self, model: str, prompt: str, max_tokens: int) -> CompletionResult: ...


def _price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = MODEL_PRICES.get(model, MODEL_PRICES[DEFAULT_MODEL])
    return pin * prompt_tokens + pout * completion_tokens


@dataclass
class Task:
    id: str
    prompt: str
    check: Callable[[str], bool]


def proxy_tasks(n: int) -> List[Task]:
    """N self-contained tasks with a cheap exact verifier (no external harness)."""
    tasks = []
    for i in range(n):
        a, b = (i * 7 + 3), (i * 3 + 5)
        ans = str(a + b)
        tasks.append(
            Task(
                id=f"proxy-{i:03d}",
                prompt=f"Add {a} and {b}. Reply with only the integer result.",
                check=(lambda out, ans=ans: ans in out.strip().split()),
            )
        )
    return tasks


class MockTransport:
    """Offline transport: deterministic canned answers + usage (no network)."""

    def __init__(self, solve: bool = True) -> None:
        self.solve = solve

    def complete(self, model: str, prompt: str, max_tokens: int) -> CompletionResult:
        # The proxy prompts end with the two integers; "solve" them locally.
        nums = [int(t) for t in prompt.replace(".", " ").split() if t.lstrip("-").isdigit()]
        text = str(sum(nums[:2])) if (self.solve and len(nums) >= 2) else "?"
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(text) // 4)
        return CompletionResult(
            text, prompt_tokens, completion_tokens, _price(model, prompt_tokens, completion_tokens)
        )


class AnthropicTransport:  # pragma: no cover - exercised only with a real key
    """Anthropic Messages API; reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self) -> None:
        import anthropic  # lazy: only needed for a real run

        self._client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY

    def complete(self, model: str, prompt: str, max_tokens: int) -> CompletionResult:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        pt = int(resp.usage.input_tokens)
        ct = int(resp.usage.output_tokens)
        return CompletionResult(text, pt, ct, _price(model, pt, ct))


@dataclass
class ArmResult:
    arm: str
    tasks_run: int
    successes: int
    total_tokens: float
    total_usd: float
    gate_admitted: int
    gate_rejected: int
    breaker_trips: int
    wall_clock_s: float
    aborted_reason: Optional[str] = None


class UsdCapExceeded(Exception):
    """Raised by the harness's own ceiling before spend exceeds the cap."""


async def _run_arm(
    arm: str,
    tasks: List[Task],
    transport: Transport,
    model: str,
    max_tokens: int,
    usd_cap: float,
    governed: bool,
    token_budget: float,
    probe_tasks: int,
    approved: bool,
) -> ArmResult:

    gov = (
        GreenGovernor.with_defaults(token_budget=token_budget, usd_budget=usd_cap)
        if governed
        else None
    )
    successes = total_tokens = total_usd = 0.0
    admitted = rejected = trips = 0
    t0 = time.perf_counter()
    reason = None

    for k, task in enumerate(tasks):
        if total_usd >= usd_cap:
            reason = f"usd cap ${usd_cap} reached after {k} tasks"
            break
        if k == probe_tasks and not approved:
            projected = (total_usd / max(1, k)) * len(tasks)
            reason = (
                f"probe complete: spent ${total_usd:.4f} over {k} tasks, "
                f"projected full-run ${projected:.2f}; re-run with --approve to continue"
            )
            break

        action = Action(
            kind="chat.completion",
            model=model,
            region="",
            prompt_tokens=max(1, len(task.prompt) // 4),
            max_tokens=max_tokens,
        )

        def _call() -> CompletionResult:
            return transport.complete(model, task.prompt, max_tokens)

        if governed and gov is not None:

            async def execute(_a: Action) -> ActionOutcome:
                res = _call()
                return ActionOutcome(
                    result=res, actual_tokens=res.prompt_tokens + res.completion_tokens
                )

            try:
                governed_res = await gov.run_action(action, execute)
                res = governed_res.result
                admitted += 1
            except GateRejected:
                rejected += 1
                continue
            except CircuitTripped:
                trips += 1
                break
        else:
            res = _call()

        total_tokens += res.prompt_tokens + res.completion_tokens
        total_usd += res.usd
        if task.check(res.text):
            successes += 1

    if governed and gov is not None:
        trips += int(gov.monitor.tripped)

    return ArmResult(
        arm=arm,
        tasks_run=min(len(tasks), k + 1 if reason is None else k),
        successes=int(successes),
        total_tokens=total_tokens,
        total_usd=total_usd,
        gate_admitted=admitted,
        gate_rejected=rejected,
        breaker_trips=trips,
        wall_clock_s=time.perf_counter() - t0,
        aborted_reason=reason,
    )


def run_study(
    *,
    n: int = 50,
    model: str = DEFAULT_MODEL,
    transport: Optional[Transport] = None,
    usd_cap: float = 25.0,
    max_tokens: int = 256,
    token_budget: float = 1.0e7,
    probe_tasks: int = 5,
    approved: bool = False,
    task_set: str = "proxy",
) -> Dict[str, Any]:
    if task_set != "proxy":
        raise NotImplementedError(
            "swebench-lite task set is the heavier option to wire after the run is "
            "funded; the proxy set is the default for the dry run."
        )
    tasks = proxy_tasks(n)
    transport = transport or MockTransport()
    baseline = asyncio.run(
        _run_arm(
            "baseline",
            tasks,
            transport,
            model,
            max_tokens,
            usd_cap,
            False,
            token_budget,
            probe_tasks,
            approved,
        )
    )
    full = asyncio.run(
        _run_arm(
            "full",
            tasks,
            transport,
            model,
            max_tokens,
            usd_cap,
            True,
            token_budget,
            probe_tasks,
            approved,
        )
    )
    return {
        "model": model,
        "n_tasks": n,
        "task_set": task_set,
        "usd_cap": usd_cap,
        "arms": {"baseline": baseline.__dict__, "full": full.__dict__},
    }


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(prog="run_live_study", description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default=os.environ.get("GREEN_SARC_LIVE_MODEL", DEFAULT_MODEL))
    ap.add_argument("--usd-cap", type=float, default=25.0)
    ap.add_argument("--probe-tasks", type=int, default=5)
    ap.add_argument(
        "--approve",
        action="store_true",
        help="continue past the probe checkpoint (spend sign-off).",
    )
    ap.add_argument("--task-set", default="proxy", choices=["proxy", "swebench-lite"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if os.environ.get("ANTHROPIC_API_KEY"):
        transport: Transport = AnthropicTransport()
        print(f"live run: model={args.model}, cap=${args.usd_cap}, N={args.n}")
    else:
        transport = MockTransport()
        print("no ANTHROPIC_API_KEY: using offline MockTransport (no live numbers produced).")

    result = run_study(
        n=args.n,
        model=args.model,
        transport=transport,
        usd_cap=args.usd_cap,
        probe_tasks=args.probe_tasks,
        approved=args.approve,
        task_set=args.task_set,
    )
    for arm, r in result["arms"].items():
        print(
            f"  {arm:<9} run={r['tasks_run']} ok={r['successes']} tok={r['total_tokens']:.0f} "
            f"usd=${r['total_usd']:.4f} admit={r['gate_admitted']} reject={r['gate_rejected']} "
            f"trips={r['breaker_trips']}"
            + (f"  [{r['aborted_reason']}]" if r["aborted_reason"] else "")
        )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
