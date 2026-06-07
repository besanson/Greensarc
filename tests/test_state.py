"""Tests for the augmented core state."""

from __future__ import annotations

from green_sarc.state import Action, Budget, GovernanceContext


def test_budget_remaining_and_spend():
    b = Budget(token_budget=1000.0, carbon_ceiling=10.0)
    assert b.remaining_tokens() == 1000.0
    assert b.remaining_carbon() == 10.0

    b.spend(tokens=400.0, carbon=3.0)
    assert b.remaining_tokens() == 600.0
    assert b.remaining_carbon() == 7.0
    assert not b.is_token_exhausted()
    assert not b.is_carbon_exhausted()


def test_budget_exhaustion_flags():
    b = Budget(token_budget=100.0, carbon_ceiling=1.0)
    b.spend(tokens=100.0, carbon=0.5)
    assert b.is_token_exhausted()
    assert not b.is_carbon_exhausted()

    b.spend(tokens=0.0, carbon=0.6)
    assert b.is_carbon_exhausted()
    assert b.remaining_carbon() < 0.0


def test_action_defaults_and_context():
    a = Action(kind="chat.completion")
    assert a.model == ""
    assert a.prompt_tokens is None
    assert a.payload == {}

    ctx = GovernanceContext(budget=Budget(10.0, 1.0), principal_id="p", session_id="s")
    d = ctx.to_dict()
    assert d["principal_id"] == "p"
    assert d["session_id"] == "s"
    assert "budget" not in d
