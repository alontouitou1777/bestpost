"""Typed data models for the ARCHIE workflow.

Every artefact produced by the language model is validated against one of these
models before it is stored, so a malformed response fails fast and loudly
instead of propagating through the pipeline as an untyped dictionary.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowStatus(str, Enum):
    """Terminal or in-flight state of a single workflow run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FLAGGED_SAFETY_RISK = "FLAGGED_SAFETY_RISK"
    FLAGGED_FOR_HUMAN_REVIEW = "FLAGGED_FOR_HUMAN_REVIEW"
    FAILED = "FAILED"


class StepName(str, Enum):
    """The six stages of the pipeline, in execution order."""

    BRIEF = "Step 1: Brief extraction"
    ANGLES = "Step 2: Creative angles"
    DRAFTS = "Step 3: Content drafts"
    SAFETY = "Step 4: Safety check"
    QA = "Step 5: Quality assurance"
    PACKAGE = "Step 6: Final package"


# ---------------------------------------------------------------------------
# Step artefacts
# ---------------------------------------------------------------------------
class StrategicBrief(BaseModel):
    """Structured interpretation of the user's free-text request."""

    product: str
    target_audience: str
    content_goal: str
    key_message: str


class AngleOption(BaseModel):
    """One candidate creative direction, scored against the others."""

    option_id: str
    style_name: str
    content: str
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    score_percentage: int = Field(ge=0, le=100)

    @field_validator("score_percentage", mode="before")
    @classmethod
    def _coerce_score(cls, value: object) -> object:
        """Accept '87%' or '87' as well as a plain integer."""
        if isinstance(value, str):
            digits = re.sub(r"[^0-9]", "", value)
            return int(digits) if digits else 0
        if isinstance(value, float):
            return round(value)
        return value


class CreativeAngles(BaseModel):
    """The full set of angles considered for a brief."""

    options: list[AngleOption] = Field(default_factory=list)

    def best(self) -> AngleOption | None:
        """Return the highest scoring angle, or None when there are none."""
        return max(self.options, key=lambda o: o.score_percentage, default=None)


class DraftOption(BaseModel):
    """A single piece of ad copy written against the winning angle."""

    option_id: str
    headline: str
    body: str
    call_to_action: str


class ContentDrafts(BaseModel):
    """Several drafts plus the model's own pick and its reasoning."""

    options: list[DraftOption] = Field(default_factory=list)
    best_option_id: str = ""
    selection_reasoning: str = ""

    def best(self) -> DraftOption | None:
        """Return the chosen draft, falling back to the first available one."""
        for option in self.options:
            if option.option_id == self.best_option_id:
                return option
        return self.options[0] if self.options else None


class SafetyCheck(BaseModel):
    """Outcome of screening a draft for policy and brand risk."""

    is_safe: bool
    risk_score: float = Field(ge=0.0, le=1.0)
    flagged_issues: list[str] = Field(default_factory=list)


class QACheck(BaseModel):
    """Editorial verdict on a draft, scored out of ten."""

    is_approved: bool
    score: float = Field(ge=0.0, le=10.0)
    reason: str


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------
class WorkflowState(BaseModel):
    """Everything known about one run, persisted after every completed step.

    This object is the unit of resumability: if the process dies, reloading the
    state from disk is enough to continue exactly where it stopped.
    """

    workflow_id: str = Field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    original_prompt: str
    status: WorkflowStatus = WorkflowStatus.PENDING

    brief: StrategicBrief | None = None
    angles: CreativeAngles | None = None
    selected_angle: AngleOption | None = None
    content_drafts: ContentDrafts | None = None
    safety_check: SafetyCheck | None = None

    drafts_history: list[ContentDrafts] = Field(default_factory=list)
    qa_history: list[QACheck] = Field(default_factory=list)
    qa_rejection_count: int = 0
    max_qa_retries: int = 3

    final_content: str | None = None
    error_message: str | None = None

    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def touch(self) -> None:
        """Refresh the modification timestamp."""
        self.updated_at = _utcnow()

    def last_qa_feedback(self) -> str | None:
        """Return the most recent rejection reason, used to steer a rewrite."""
        if self.qa_history and not self.qa_history[-1].is_approved:
            return self.qa_history[-1].reason
        return None

    def is_step_completed(self, step: StepName) -> bool:
        """Report whether a step already has a valid result on record.

        The orchestrator calls this before each stage so that resuming a run
        never repeats work that succeeded, and never pays for the same tokens
        twice.
        """
        if step is StepName.BRIEF:
            return self.brief is not None
        if step is StepName.ANGLES:
            return self.angles is not None
        if step is StepName.DRAFTS:
            return self.content_drafts is not None
        if step is StepName.SAFETY:
            return self.safety_check is not None
        if step is StepName.QA:
            return bool(self.qa_history) and self.qa_history[-1].is_approved
        if step is StepName.PACKAGE:
            return self.final_content is not None
        return False

    def completed_steps(self) -> list[StepName]:
        """List every step that currently has a result, for progress display."""
        return [step for step in StepName if self.is_step_completed(step)]
