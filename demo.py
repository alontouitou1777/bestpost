"""Instrumentation and fault injection for the LLM service.

This wrapper exists so the pipeline's resilience can be demonstrated and
measured without touching the orchestrator. It does two things:

*Counting* — records how many model calls each stage made, which is how you
verify that resuming a run really did skip the completed stages instead of
quietly recomputing them.

*Fault injection* — makes a chosen stage raise on demand, so a crash can be
reproduced deliberately rather than waited for.

It implements the same six methods as LLMService and delegates everything to
the wrapped instance, so the orchestrator cannot tell the difference.
"""

from __future__ import annotations

import logging

from llm_service import LLMService
from schemas import (
    AngleOption,
    ContentDrafts,
    CreativeAngles,
    DraftOption,
    QACheck,
    SafetyCheck,
    StepName,
    StrategicBrief,
)

logger = logging.getLogger("ArchieAgent.demo")

# Which underlying method each stage calls.
STEP_METHODS: dict[StepName, str] = {
    StepName.BRIEF: "extract_brief",
    StepName.ANGLES: "generate_angles",
    StepName.DRAFTS: "generate_drafts",
    StepName.SAFETY: "check_safety",
    StepName.QA: "evaluate_qa",
    StepName.PACKAGE: "generate_final_package",
}


class InjectedFailure(RuntimeError):
    """A deliberate failure, raised to demonstrate crash recovery."""


class InstrumentedLLMService:
    """Wraps an LLMService to count calls and optionally fail on cue."""

    def __init__(self, inner: LLMService, fail_at: StepName | None = None) -> None:
        """Create a wrapper.

        Args:
            inner: The real service that does the work.
            fail_at: Stage that should raise instead of running. None disables
                injection entirely, which is the normal case.
        """
        self.inner = inner
        self.fail_at = fail_at
        self.call_counts: dict[str, int] = {}

    @property
    def total_calls(self) -> int:
        """How many model calls this wrapper has made since it was created."""
        return sum(self.call_counts.values())

    def _guard(self, step: StepName) -> None:
        """Count the call, and raise first if this stage is the injected one."""
        if self.fail_at is step:
            logger.warning("Injecting a failure at %s", step.value)
            raise InjectedFailure(
                f"Simulated outage during {step.value}. "
                "Re-run with the same workflow ID to resume from here."
            )
        name = STEP_METHODS[step]
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

    # -- Delegated stages ---------------------------------------------------
    def extract_brief(self, prompt: str) -> StrategicBrief:
        self._guard(StepName.BRIEF)
        return self.inner.extract_brief(prompt)

    def generate_angles(self, brief: StrategicBrief) -> CreativeAngles:
        self._guard(StepName.ANGLES)
        return self.inner.generate_angles(brief)

    def generate_drafts(
        self,
        brief: StrategicBrief,
        angle: AngleOption | None,
        feedback_from_previous_qa: str | None = None,
    ) -> ContentDrafts:
        self._guard(StepName.DRAFTS)
        return self.inner.generate_drafts(
            brief, angle, feedback_from_previous_qa=feedback_from_previous_qa
        )

    def check_safety(self, draft: DraftOption) -> SafetyCheck:
        self._guard(StepName.SAFETY)
        return self.inner.check_safety(draft)

    def evaluate_qa(self, brief: StrategicBrief, draft: DraftOption) -> QACheck:
        self._guard(StepName.QA)
        return self.inner.evaluate_qa(brief, draft)

    def generate_final_package(
        self, brief: StrategicBrief, angle: AngleOption | None, draft: DraftOption
    ) -> str:
        self._guard(StepName.PACKAGE)
        return self.inner.generate_final_package(brief, angle, draft)
