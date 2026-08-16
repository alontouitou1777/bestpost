"""Tests for the six-stage workflow.

These cover the properties that make the pipeline safe to run in production:
it does not repeat paid work, it survives a crash, it improves rejected drafts
rather than retrying blindly, and it escalates instead of looping forever.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from orchestrator import Orchestrator
from schemas import CreativeAngles, QACheck, SafetyCheck, StepName, WorkflowStatus


def test_happy_path_runs_every_stage(store, fake_llm):
    """A clean run reaches COMPLETED and calls each stage exactly once."""
    state = Orchestrator(fake_llm, store).run_workflow(prompt="Launch ARCHIE")

    assert state.status is WorkflowStatus.COMPLETED
    assert state.final_content
    assert set(state.completed_steps()) == set(StepName)

    for stage in (
        fake_llm.extract_brief,
        fake_llm.generate_angles,
        fake_llm.generate_drafts,
        fake_llm.check_safety,
        fake_llm.evaluate_qa,
        fake_llm.generate_final_package,
    ):
        assert stage.call_count == 1


def test_highest_scoring_angle_is_selected(store, fake_llm):
    """The angle with the best score wins, not the first one returned."""
    state = Orchestrator(fake_llm, store).run_workflow(prompt="Launch ARCHIE")

    assert state.selected_angle is not None
    assert state.selected_angle.style_name == "Craft"
    assert state.selected_angle.score_percentage == 91


def test_rerunning_a_completed_workflow_costs_nothing(store, fake_llm):
    """A duplicate request returns the stored result without calling the model."""
    orchestrator = Orchestrator(fake_llm, store)
    first = orchestrator.run_workflow(prompt="Launch ARCHIE")

    second = orchestrator.run_workflow(prompt="Launch ARCHIE", workflow_id=first.workflow_id)

    assert second.status is WorkflowStatus.COMPLETED
    assert fake_llm.extract_brief.call_count == 1
    assert fake_llm.generate_final_package.call_count == 1


def test_failure_is_recorded_and_the_run_resumes_from_that_stage(store, fake_llm):
    """A stage failure leaves earlier work intact and is picked up on resume."""
    fake_llm.generate_angles.side_effect = RuntimeError("API outage")

    orchestrator = Orchestrator(fake_llm, store)
    failed = orchestrator.run_workflow(prompt="Resume me")

    assert failed.status is WorkflowStatus.FAILED
    assert failed.error_message is not None
    assert failed.brief is not None, "stage 1 result should survive the failure"
    assert failed.angles is None, "stage 2 should have no result"

    fake_llm.generate_angles.side_effect = None
    resumed = orchestrator.run_workflow(prompt="Resume me", workflow_id=failed.workflow_id)

    assert resumed.status is WorkflowStatus.COMPLETED
    assert fake_llm.extract_brief.call_count == 1, "stage 1 must not be recomputed"


def test_crash_recovery_across_a_process_restart(store, fake_llm, sample_brief):
    """State written to disk survives a fresh orchestrator and LLM client."""
    workflow_id = "crash_recovery_123"
    prompt = "Campaign that crashes midway"

    crashing_llm = MagicMock()
    crashing_llm.extract_brief.return_value = sample_brief
    crashing_llm.generate_angles.side_effect = RuntimeError("Process crashed")

    crashed = Orchestrator(crashing_llm, store).run_workflow(
        prompt=prompt, workflow_id=workflow_id
    )
    assert crashed.status is WorkflowStatus.FAILED
    assert crashing_llm.extract_brief.call_count == 1

    # A brand new orchestrator, as if the process had been restarted.
    recovered = Orchestrator(fake_llm, store).run_workflow(
        prompt=prompt, workflow_id=workflow_id
    )

    assert recovered.status is WorkflowStatus.COMPLETED
    assert recovered.brief == sample_brief
    assert fake_llm.extract_brief.call_count == 0, "the stored brief should be reused"

    # And a duplicate request after completion does no further work.
    Orchestrator(fake_llm, store).run_workflow(prompt=prompt, workflow_id=workflow_id)
    assert fake_llm.generate_angles.call_count == 1


def test_qa_rejection_feeds_the_reason_into_the_rewrite(store, fake_llm):
    """The second draft attempt receives the reviewer's specific complaint."""
    fake_llm.evaluate_qa.side_effect = [
        QACheck(is_approved=False, score=4.0, reason="Tone is too casual."),
        QACheck(is_approved=True, score=9.0, reason="Much better."),
    ]

    state = Orchestrator(fake_llm, store).run_workflow(prompt="Needs a rewrite")

    assert state.status is WorkflowStatus.COMPLETED
    assert state.qa_rejection_count == 1
    assert fake_llm.generate_drafts.call_count == 2

    first_call, second_call = fake_llm.generate_drafts.call_args_list
    assert first_call.kwargs["feedback_from_previous_qa"] is None
    assert second_call.kwargs["feedback_from_previous_qa"] == "Tone is too casual."


def test_repeated_rejection_escalates_to_a_human(store, fake_llm):
    """The loop stops at the retry budget instead of running forever."""
    fake_llm.evaluate_qa.return_value = QACheck(
        is_approved=False, score=2.0, reason="Always off brand."
    )

    state = Orchestrator(fake_llm, store).run_workflow(prompt="Hopeless brief")

    assert state.status is WorkflowStatus.FLAGGED_FOR_HUMAN_REVIEW
    assert state.qa_rejection_count == state.max_qa_retries == 3
    assert fake_llm.generate_drafts.call_count == 3
    assert fake_llm.generate_final_package.call_count == 0


def test_unsafe_copy_stops_before_review(store, fake_llm):
    """A safety failure halts the run and never reaches QA or packaging."""
    fake_llm.check_safety.return_value = SafetyCheck(
        is_safe=False,
        risk_score=0.92,
        flagged_issues=["Unverifiable medical claim"],
    )

    state = Orchestrator(fake_llm, store).run_workflow(prompt="Risky brief")

    assert state.status is WorkflowStatus.FLAGGED_SAFETY_RISK
    assert state.safety_check.flagged_issues == ["Unverifiable medical claim"]
    assert fake_llm.evaluate_qa.call_count == 0
    assert fake_llm.generate_final_package.call_count == 0


def test_every_stage_is_persisted_as_it_completes(store, fake_llm):
    """State on disk is current, not written only at the end."""
    fake_llm.generate_drafts.side_effect = RuntimeError("Stopped at stage 3")

    state = Orchestrator(fake_llm, store).run_workflow(prompt="Partial run")

    reloaded = store.load(state.workflow_id)
    assert reloaded is not None
    assert reloaded.status is WorkflowStatus.FAILED
    assert reloaded.is_step_completed(StepName.BRIEF)
    assert reloaded.is_step_completed(StepName.ANGLES)
    assert not reloaded.is_step_completed(StepName.DRAFTS)


def test_workflow_without_angles_still_produces_copy(store, fake_llm):
    """An empty angle set degrades gracefully rather than crashing."""
    fake_llm.generate_angles.return_value = CreativeAngles(options=[])

    state = Orchestrator(fake_llm, store).run_workflow(prompt="No angles")

    assert state.status is WorkflowStatus.COMPLETED
    assert state.selected_angle is None
    assert fake_llm.generate_drafts.call_args.args[1] is None
