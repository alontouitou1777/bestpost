"""Command line entry point for running a workflow without the web UI."""

from __future__ import annotations

import argparse
import sys

from config import get_settings
from llm_service import LLMError, LLMService
from logging_config import setup_logging
from orchestrator import Orchestrator
from schemas import StepName, WorkflowState, WorkflowStatus
from statestore import InvalidWorkflowId, StateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="archie",
        description="Generate a marketing campaign from a short brief.",
        epilog='Example: python main.py "Launch a budgeting app for students" --id demo',
    )
    parser.add_argument("prompt", help="The campaign brief in plain language.")
    parser.add_argument(
        "--id",
        dest="workflow_id",
        default=None,
        help="Workflow id. Reuse one to resume an interrupted or failed run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full state as JSON instead of a readable summary.",
    )
    return parser


def print_summary(state: WorkflowState) -> None:
    """Render a run in a form that is readable in a terminal."""
    print(f"\nWorkflow : {state.workflow_id}")
    print(f"Status   : {state.status.value}")

    for step in StepName:
        mark = "x" if state.is_step_completed(step) else " "
        print(f"  [{mark}] {step.value}")

    if state.selected_angle:
        print(f"\nAngle    : {state.selected_angle.style_name} "
              f"({state.selected_angle.score_percentage}%)")

    if state.qa_history:
        last = state.qa_history[-1]
        verdict = "approved" if last.is_approved else "rejected"
        print(f"QA       : {verdict}, score {last.score}, after "
              f"{state.qa_rejection_count} rejection(s)")

    if state.status is WorkflowStatus.COMPLETED and state.final_content:
        print("\n" + "-" * 60)
        print(state.final_content)
    elif state.error_message:
        print(f"\nError    : {state.error_message}")
        print("Resume with: --id " + state.workflow_id)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_to_file)

    try:
        orchestrator = Orchestrator(
            llm_service=LLMService(settings=settings),
            state_store=StateStore(settings.state_dir),
        )
        state = orchestrator.run_workflow(prompt=args.prompt, workflow_id=args.workflow_id)
    except (LLMError, InvalidWorkflowId) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(state.model_dump_json(indent=2))
    else:
        print_summary(state)

    return 0 if state.status is WorkflowStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
