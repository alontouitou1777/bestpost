"""The six-stage campaign workflow.

Design notes
------------
Two properties drive the structure of this module:

*Idempotency* — every stage is guarded by ``state.is_step_completed``. Calling
``run_workflow`` twice with the same id does no model work the second time, so
a duplicate request is free rather than expensive.

*Resumability* — state is written to the store after every stage. If the
process dies at stage four, the next invocation reloads the state and starts at
stage four; the first three stages are never recomputed.

Together they mean a crashed or retried run costs only the tokens it has not
already spent.
"""

from __future__ import annotations

import logging

from llm_service import LLMService
from schemas import (
    ContentDrafts,
    DraftOption,
    StepName,
    WorkflowState,
    WorkflowStatus,
)
from statestore import StateStore, validate_workflow_id

logger = logging.getLogger("ArchieAgent.orchestrator")


class Orchestrator:
    """Runs a brief through strategy, copywriting, safety and review."""

    def __init__(self, llm_service: LLMService, state_store: StateStore) -> None:
        self.llm = llm_service
        self.store = state_store

    # -- Entry point --------------------------------------------------------
    def run_workflow(self, prompt: str, workflow_id: str | None = None) -> WorkflowState:
        """Execute the pipeline, resuming any earlier run with the same id.

        Args:
            prompt: The campaign brief in free text.
            workflow_id: Identifier to resume. A new one is generated if absent.

        Returns:
            The final state, whatever its status. Failures are recorded on the
            state rather than raised, so the caller can inspect and resume.
        """
        state = self._load_or_create(prompt, workflow_id)

        if state.status is WorkflowStatus.COMPLETED:
            logger.info("Workflow '%s' already completed; nothing to do.", state.workflow_id)
            return state

        state.status = WorkflowStatus.RUNNING
        state.error_message = None
        self._save(state)

        try:
            self._step_brief(state)
            self._step_angles(state)
            terminated = self._review_loop(state)
            if terminated:
                return state
            self._step_package(state)
        except Exception as exc:  # noqa: BLE001 - recorded on the state for resume
            logger.error("Workflow '%s' failed: %s", state.workflow_id, exc)
            state.status = WorkflowStatus.FAILED
            state.error_message = str(exc)
            self._save(state)
            return state

        state.status = WorkflowStatus.COMPLETED
        self._save(state)
        logger.info("Workflow '%s' completed.", state.workflow_id)
        return state

    # -- Stage 1 ------------------------------------------------------------
    def _step_brief(self, state: WorkflowState) -> None:
        """Extract the strategic brief, unless it is already on record."""
        if state.is_step_completed(StepName.BRIEF):
            logger.info("Skipping %s (already done).", StepName.BRIEF.value)
            return

        logger.info("Running %s", StepName.BRIEF.value)
        state.brief = self.llm.extract_brief(state.original_prompt)
        self._save(state)

    # -- Stage 2 ------------------------------------------------------------
    def _step_angles(self, state: WorkflowState) -> None:
        """Generate candidate angles and select the highest scoring one."""
        if state.is_step_completed(StepName.ANGLES):
            logger.info("Skipping %s (already done).", StepName.ANGLES.value)
            return

        logger.info("Running %s", StepName.ANGLES.value)
        state.angles = self.llm.generate_angles(state.brief)
        state.selected_angle = state.angles.best()
        if state.selected_angle:
            logger.info("Selected angle: %s", state.selected_angle.style_name)
        self._save(state)

    # -- Stages 3 to 5 ------------------------------------------------------
    def _review_loop(self, state: WorkflowState) -> bool:
        """Draft, screen and review copy until it passes or the budget runs out.

        A rejected draft is discarded and rewritten with the reviewer's reason
        fed back into the prompt, so each attempt targets the stated problem.

        Returns:
            True if the workflow reached a terminal state inside the loop
            (safety risk or exhausted retries) and should stop.
        """
        while True:
            if state.is_step_completed(StepName.QA):
                logger.info("Skipping review loop (draft already approved).")
                return False

            draft = self._step_drafts(state)
            if draft is None:
                # Nothing to review; let the packaging stage handle the gap.
                return False

            if self._step_safety(state, draft):
                return True

            if self._step_qa(state, draft):
                return False

            if state.qa_rejection_count >= state.max_qa_retries:
                logger.warning(
                    "Workflow '%s' hit the QA retry limit; escalating to a human.",
                    state.workflow_id,
                )
                state.status = WorkflowStatus.FLAGGED_FOR_HUMAN_REVIEW
                self._save(state)
                return True

            # Discard the rejected draft so the next pass regenerates it.
            state.content_drafts = None
            state.safety_check = None
            self._save(state)

    def _step_drafts(self, state: WorkflowState) -> DraftOption | None:
        """Write copy for the selected angle, honouring any QA feedback."""
        if not state.is_step_completed(StepName.DRAFTS):
            feedback = state.last_qa_feedback()
            logger.info(
                "Running %s%s",
                StepName.DRAFTS.value,
                " (revision)" if feedback else "",
            )
            drafts: ContentDrafts = self.llm.generate_drafts(
                state.brief,
                state.selected_angle,
                feedback_from_previous_qa=feedback,
            )
            state.content_drafts = drafts
            state.drafts_history.append(drafts)
            self._save(state)

        return state.content_drafts.best() if state.content_drafts else None

    def _step_safety(self, state: WorkflowState, draft: DraftOption) -> bool:
        """Screen the draft. Returns True when the run must stop."""
        if not state.is_step_completed(StepName.SAFETY):
            logger.info("Running %s", StepName.SAFETY.value)
            state.safety_check = self.llm.check_safety(draft)
            self._save(state)

        if state.safety_check and not state.safety_check.is_safe:
            logger.warning(
                "Workflow '%s' flagged as unsafe: %s",
                state.workflow_id,
                ", ".join(state.safety_check.flagged_issues) or "no detail given",
            )
            state.status = WorkflowStatus.FLAGGED_SAFETY_RISK
            self._save(state)
            return True
        return False

    def _step_qa(self, state: WorkflowState, draft: DraftOption) -> bool:
        """Review the draft. Returns True when it is approved."""
        logger.info("Running %s", StepName.QA.value)
        verdict = self.llm.evaluate_qa(state.brief, draft)
        state.qa_history.append(verdict)

        if verdict.is_approved:
            logger.info("QA approved the draft with score %.1f", verdict.score)
            self._save(state)
            return True

        state.qa_rejection_count += 1
        logger.info(
            "QA rejected draft %d/%d: %s",
            state.qa_rejection_count,
            state.max_qa_retries,
            verdict.reason,
        )
        self._save(state)
        return False

    # -- Stage 6 ------------------------------------------------------------
    def _step_package(self, state: WorkflowState) -> None:
        """Assemble the approved copy into the final deliverable."""
        if state.is_step_completed(StepName.PACKAGE):
            logger.info("Skipping %s (already done).", StepName.PACKAGE.value)
            return

        draft = state.content_drafts.best() if state.content_drafts else None
        if draft is None:
            raise RuntimeError("Cannot build the final package without an approved draft.")

        logger.info("Running %s", StepName.PACKAGE.value)
        state.final_content = self.llm.generate_final_package(
            state.brief, state.selected_angle, draft
        )
        self._save(state)

    # -- Helpers ------------------------------------------------------------
    def _load_or_create(self, prompt: str, workflow_id: str | None) -> WorkflowState:
        """Return the stored state for this id, or a fresh one."""
        if workflow_id is None:
            state = WorkflowState(original_prompt=prompt)
            logger.info("Starting new workflow '%s'", state.workflow_id)
            return state

        validate_workflow_id(workflow_id)
        existing = self.store.load(workflow_id)
        if existing is not None:
            logger.info(
                "Resuming workflow '%s' (%d/%d steps already done)",
                workflow_id,
                len(existing.completed_steps()),
                len(StepName),
            )
            return existing

        logger.info("Starting new workflow '%s'", workflow_id)
        return WorkflowState(workflow_id=workflow_id, original_prompt=prompt)

    def _save(self, state: WorkflowState) -> None:
        """Persist the state so the run can survive a process restart."""
        state.touch()
        self.store.save(state)
