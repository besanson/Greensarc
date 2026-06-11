"""Live-study harness control flow, exercised offline via MockTransport.

No network and no API key: these tests cover the two-arm loop, the USD ceiling,
and the probe-checkpoint STOP — the live run itself is a pending human action.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper.scripts.run_live_study import (  # noqa: E402
    MockTransport,
    proxy_tasks,
    run_study,
)


def test_proxy_tasks_are_solvable_by_mock():
    tasks = proxy_tasks(5)
    tp = MockTransport(solve=True)
    assert len(tasks) == 5
    for t in tasks:
        res = tp.complete("claude-haiku-4-5-20251001", t.prompt, 64)
        assert t.check(res.text)  # the mock "solves" the arithmetic proxy
        assert res.prompt_tokens > 0 and res.completion_tokens > 0
        assert res.usd > 0.0


def test_both_arms_run_and_report_metrics():
    out = run_study(
        n=8,
        transport=MockTransport(solve=True),
        probe_tasks=8,
        approved=True,
        model="claude-haiku-4-5-20251001",
    )
    base, full = out["arms"]["baseline"], out["arms"]["full"]
    assert base["successes"] == 8  # mock solves every proxy task
    assert full["successes"] == 8
    # The governed arm admits through the gate; the ungoverned arm does not gate.
    assert full["gate_admitted"] == 8
    assert base["gate_admitted"] == 0
    assert base["total_usd"] > 0.0 and full["total_usd"] > 0.0


def test_probe_checkpoint_stops_without_approval():
    out = run_study(
        n=50,
        transport=MockTransport(),
        probe_tasks=5,
        approved=False,
        model="claude-haiku-4-5-20251001",
    )
    base = out["arms"]["baseline"]
    # Without --approve the run halts at the probe checkpoint, not at task 50.
    assert base["aborted_reason"] is not None
    assert "projected" in base["aborted_reason"]
    assert base["tasks_run"] <= 5


def test_usd_cap_aborts_before_overspend():
    # A cap far below the per-task price forces an abort almost immediately.
    out = run_study(
        n=50,
        transport=MockTransport(),
        usd_cap=1e-9,
        probe_tasks=50,
        approved=True,
        model="claude-opus-4-8",
    )
    base = out["arms"]["baseline"]
    assert base["aborted_reason"] is not None
    assert "usd cap" in base["aborted_reason"]
    # The ungoverned arm's spend is only known post-call, so at most one task
    # completes before the ceiling trips; it never runs the full N.
    assert base["tasks_run"] <= 1


def test_swebench_lite_is_not_yet_wired():
    with pytest.raises(NotImplementedError, match="swebench-lite"):
        run_study(n=5, task_set="swebench-lite")
