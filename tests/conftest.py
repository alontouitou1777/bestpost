"""Shared fixtures.

The suite never touches the network. ``fake_llm`` is a MagicMock whose return
values are already valid domain objects, so tests only override the one stage
whose behaviour they care about.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schemas import (  # noqa: E402
    AngleOption,
    ContentDrafts,
    CreativeAngles,
    DraftOption,
    QACheck,
    SafetyCheck,
    StrategicBrief,
)
from statestore import FileStateStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """A state store rooted in a directory unique to this test."""
    return FileStateStore(storage_dir=str(tmp_path / "states"))


@pytest.fixture
def sample_brief() -> StrategicBrief:
    return StrategicBrief(
        product="ARCHIE AI",
        target_audience="Growth marketers at seed-stage startups",
        content_goal="Lead generation",
        key_message="Ship a campaign in minutes, not weeks",
    )


@pytest.fixture
def sample_angles() -> CreativeAngles:
    return CreativeAngles(
        options=[
            AngleOption(
                option_id="1",
                style_name="Speed",
                content="Emphasise time saved.",
                pros=["Concrete"],
                cons=["Commoditised"],
                score_percentage=72,
            ),
            AngleOption(
                option_id="2",
                style_name="Craft",
                content="Emphasise quality of output.",
                pros=["Differentiated"],
                cons=["Harder to prove"],
                score_percentage=91,
            ),
        ]
    )


@pytest.fixture
def sample_drafts() -> ContentDrafts:
    return ContentDrafts(
        options=[
            DraftOption(
                option_id="1",
                headline="Your next campaign, drafted before lunch",
                body="Brief in. Campaign out.",
                call_to_action="Start free",
            )
        ],
        best_option_id="1",
        selection_reasoning="Leads with the concrete benefit.",
    )


@pytest.fixture
def fake_llm(sample_brief, sample_angles, sample_drafts) -> MagicMock:
    """An LLM service stand-in whose every stage succeeds by default."""
    llm = MagicMock()
    llm.extract_brief.return_value = sample_brief
    llm.generate_angles.return_value = sample_angles
    llm.generate_drafts.return_value = sample_drafts
    llm.check_safety.return_value = SafetyCheck(is_safe=True, risk_score=0.05)
    llm.evaluate_qa.return_value = QACheck(
        is_approved=True, score=8.8, reason="Clear and on brief."
    )
    llm.generate_final_package.return_value = "# Campaign package\n\nApproved."
    return llm
