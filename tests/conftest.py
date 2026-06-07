"""Shared fixtures and helpers for the Green SARC test suite."""

from __future__ import annotations

from typing import List

import pytest

from green_sarc.escalation import EscalationEvent, EscalationRouter
from green_sarc.pricing import ModelProfile, TableCarbonModel, TableCostModel
from green_sarc.state import Action, Budget


@pytest.fixture
def cost_model() -> TableCostModel:
    """A cost model with a known per-token energy for deterministic carbon."""
    return TableCostModel(
        profiles={"test-model": ModelProfile(energy_per_token_kwh=1.0e-6)},
        default_profile=ModelProfile(energy_per_token_kwh=1.0e-6),
    )


@pytest.fixture
def carbon_model() -> TableCarbonModel:
    """A carbon model with a known intensity (500 gCO2e/kWh)."""
    return TableCarbonModel(intensities={"eu-west": 500.0}, default_intensity=500.0)


@pytest.fixture
def budget() -> Budget:
    """A generous budget so individual sites can be tested in isolation."""
    return Budget(token_budget=100_000.0, carbon_ceiling=1_000.0, delta=0.05)


def make_action(prompt: int = 100, max_tokens: int = 200) -> Action:
    """A standard LLM-call action against the test model/region."""
    return Action(
        kind="chat.completion",
        model="test-model",
        region="eu-west",
        prompt_tokens=prompt,
        max_tokens=max_tokens,
    )


class RecordingRouter(EscalationRouter):
    """Escalation router that records every event it receives."""

    def __init__(self) -> None:
        self.events: List[EscalationEvent] = []

        async def handler(event: EscalationEvent) -> None:
            self.events.append(event)

        super().__init__(handler)
