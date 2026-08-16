"""Tests for the fault injection wrapper.

The point of these is that the resilience demo is honest: when a crash is
injected at stage three and the run is resumed, stages one and two must make
zero model calls the second time.
"""

from __future__ import annotations

import pytest

from Demo import InjectedFailure, InstrumentedLLMService
from orchestrator import Orchestrator
from schemas import StepName, WorkflowStatus


def test_the_wrapper_is_transparent_when_nothing_is_injected(store, fake_llm):
    """With no fault configured, the workflow behaves exactly as before."""
    llm = InstrumentedLLMService(fake_llm)

    state = Orchestrator(llm, store).run_workflow(prompt="Normal run")

    assert state.status is WorkflowStatus.COMPLETED
    assert llm.total_calls == 6, "one call per stage"


def test_injecting_a_fault_stops_the_run_at_that_stage(store, fake_llm):
    llm = InstrumentedLLMService(fake_llm, fail_at=StepName.DRAFTS)

    state = Orchestrator(llm, store).run_workflow(prompt="Crash at three")

    assert state.status is WorkflowStatus.FAILED
    assert state.is_step_completed(StepName.BRIEF)
    assert state.is_step_completed(StepName.ANGLES)
    assert not state.is_step_completed(StepName.DRAFTS)
    assert llm.call_counts == {"extract_brief": 1, "generate_angles": 1}


def test_resuming_after_an_injected_fault_recomputes_nothing(store, fake_llm):
    """The demo's central claim, asserted directly.

    Stage three fails, then a fresh run with the same id completes the
    workflow while making no calls at all for stages one and two.
    """
    crashing = InstrumentedLLMService(fake_llm, fail_at=StepName.DRAFTS)
    failed = Orchestrator(crashing, store).run_workflow(prompt="Resume me")
    assert failed.status is WorkflowStatus.FAILED

    resumed_llm = InstrumentedLLMService(fake_llm)
    resumed = Orchestrator(resumed_llm, store).run_workflow(
        prompt="Resume me", workflow_id=failed.workflow_id
    )

    assert resumed.status is WorkflowStatus.COMPLETED
    assert "extract_brief" not in resumed_llm.call_counts
    assert "generate_angles" not in resumed_llm.call_counts
    assert resumed_llm.total_calls == 4, "only stages three to six should run"

    # The reloaded artefacts are the originals, not regenerated copies.
    assert resumed.brief == failed.brief
    assert resumed.selected_angle == failed.selected_angle


@pytest.mark.parametrize(
    "stage,expected_prior_calls",
    [
        (StepName.BRIEF, 0),
        (StepName.ANGLES, 1),
        (StepName.DRAFTS, 2),
        (StepName.SAFETY, 3),
    ],
)
def test_a_crash_can_be_injected_at_any_stage(store, fake_llm, stage, expected_prior_calls):
    """Every stage is a valid injection point, and earlier ones still run."""
    llm = InstrumentedLLMService(fake_llm, fail_at=stage)

    state = Orchestrator(llm, store).run_workflow(prompt=f"Crash at {stage.name}")

    assert state.status is WorkflowStatus.FAILED
    assert llm.total_calls == expected_prior_calls


def test_the_injected_error_names_the_stage_and_the_fix(store, fake_llm):
    """The message shown to a viewer explains what happened and what to do."""
    llm = InstrumentedLLMService(fake_llm, fail_at=StepName.QA)

    state = Orchestrator(llm, store).run_workflow(prompt="Explain yourself")

    assert "Step 5" in state.error_message
    assert "resume" in state.error_message.lower()


def test_the_injected_failure_is_a_distinct_type(fake_llm):
    """Injected faults are distinguishable from real outages."""
    llm = InstrumentedLLMService(fake_llm, fail_at=StepName.BRIEF)

    with pytest.raises(InjectedFailure):
        llm.extract_brief("anything")
