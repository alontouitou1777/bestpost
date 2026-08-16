"""Streamlit front end for the ARCHIE workflow."""

from __future__ import annotations

import streamlit as st

from config import get_settings
from demo import STEP_METHODS, InstrumentedLLMService
from llm_service import LLMError, LLMService
from logging_config import setup_logging
from orchestrator import Orchestrator
from schemas import StepName, WorkflowStatus
from statestore import InvalidWorkflowId, StateStore, validate_workflow_id

STATUS_STYLE = {
    WorkflowStatus.COMPLETED: ("✅", "success"),
    WorkflowStatus.FLAGGED_FOR_HUMAN_REVIEW: ("🙋", "warning"),
    WorkflowStatus.FLAGGED_SAFETY_RISK: ("🛑", "error"),
    WorkflowStatus.FAILED: ("⚠️", "error"),
    WorkflowStatus.RUNNING: ("⏳", "info"),
    WorkflowStatus.PENDING: ("•", "info"),
}

settings = get_settings()
setup_logging(settings.log_level, log_to_file=False)

st.set_page_config(page_title="ARCHIE - Agentic Campaign Workflow", layout="wide", page_icon="🚀")
st.title("🚀 ARCHIE")
st.caption(
    "A six-stage agentic workflow with self-correction, "
    "safety screening and resumable state."
)

store = StateStore(settings.state_dir)

# --- Sidebar ----------------------------------------------------------------
st.sidebar.header("🔑 Configuration")
settings.groq_api_key = st.sidebar.text_input(
    "Groq API key",
    value=settings.groq_api_key,
    type="password",
    help="Read from .env when present. Free keys at console.groq.com",
)
st.sidebar.caption(f"Model: `{settings.groq_model}`")

if not settings.is_configured:
    st.sidebar.warning("An API key is required to run a workflow.")

st.sidebar.divider()
st.sidebar.header("📂 Saved runs")

saved = store.list_ids()
selected = None
if saved:
    choice = st.sidebar.selectbox("Load a run", ["-- none --", *saved])
    if choice != "-- none --":
        selected = choice
        if st.sidebar.button("🗑️ Delete this run", use_container_width=True):
            store.delete(selected)
            st.rerun()
else:
    st.sidebar.caption("No saved runs yet.")

st.sidebar.divider()

# --- Sidebar: fault injection for demonstrating recovery --------------------
with st.sidebar.expander("🧪 Resilience demo"):
    st.caption(
        "Force a stage to fail, then re-run with the same workflow ID. "
        "The completed stages are reloaded from disk instead of recomputed."
    )
    crash_choice = st.selectbox(
        "Simulate a crash at",
        ["No crash", *[s.value for s in StepName]],
        help="Injects a failure at the chosen stage. Earlier stages still persist.",
    )
crash_stage = next((s for s in StepName if s.value == crash_choice), None)

# --- Input ------------------------------------------------------------------
loaded = store.load(selected) if selected else None

with st.form("workflow_form"):
    col_id, col_retries = st.columns([3, 1])
    workflow_id = col_id.text_input("Workflow ID", value=selected or "demo_run")
    max_retries = col_retries.number_input("QA retries", min_value=1, max_value=5, value=3)
    prompt = st.text_area(
        "Campaign brief",
        value=loaded.original_prompt if loaded else
        "A budgeting app that helps university students avoid overdraft fees",
        height=90,
    )
    submitted = st.form_submit_button("▶️ Run workflow", type="primary", use_container_width=True)

def execute(wid: str, brief: str, retries: int, fail_at: StepName | None) -> None:
    """Run or resume a workflow, recording how many model calls it took."""
    validate_workflow_id(wid)
    llm = InstrumentedLLMService(LLMService(settings=settings), fail_at=fail_at)

    with st.spinner("Running the workflow..."):
        state = Orchestrator(llm, store).run_workflow(prompt=brief, workflow_id=wid)
        if state.max_qa_retries != retries:
            state.max_qa_retries = int(retries)
            store.save(state)

    st.session_state["last_run"] = {
        "calls": llm.total_calls,
        "stages": dict(llm.call_counts),
        "resumed": bool(st.session_state.get("has_run_before", {}).get(wid)),
    }
    st.session_state.setdefault("has_run_before", {})[wid] = True


if submitted:
    if not settings.is_configured:
        st.error("Enter a Groq API key in the sidebar first.")
    elif not prompt.strip():
        st.error("Enter a campaign brief.")
    else:
        try:
            existing = store.load(workflow_id)
            if existing and existing.status is WorkflowStatus.COMPLETED:
                st.info("This run already completed. Change the ID to start a fresh one.")
            else:
                execute(workflow_id, prompt, max_retries, crash_stage)
                st.rerun()
        except InvalidWorkflowId as exc:
            st.error(str(exc))
        except LLMError as exc:
            st.error(f"The model could not complete the request: {exc}")

# --- Results ----------------------------------------------------------------
state = store.load(workflow_id)

if state is None:
    st.info("Run a workflow, or pick a saved one from the sidebar.")
    st.stop()

icon, level = STATUS_STYLE.get(state.status, ("•", "info"))
getattr(st, level)(f"{icon} **{state.status.value}** — workflow `{state.workflow_id}`")

if state.error_message:
    done = len(state.completed_steps())
    st.error(
        f"{state.error_message}\n\n"
        f"{done} of {len(StepName)} stages are saved to disk. "
        "Resuming continues from the stage that failed."
    )
    if st.button("⏭️ Resume from where it stopped", type="primary", use_container_width=True):
        try:
            execute(state.workflow_id, state.original_prompt, state.max_qa_retries, crash_stage)
            st.rerun()
        except LLMError as exc:
            st.error(f"The model could not complete the request: {exc}")

# Evidence that resuming skipped the finished work
last = st.session_state.get("last_run")
if last:
    skipped = [
        s.value.split(": ", 1)[-1]
        for s in StepName
        if STEP_METHODS[s] not in last["stages"] and state.is_step_completed(s)
    ]
    col_a, col_b = st.columns(2)
    col_a.metric("Model calls in the last run", last["calls"])
    col_b.metric("Stages reloaded from disk", len(skipped))
    if skipped:
        st.caption("Reused without recomputing: " + ", ".join(skipped))

# Stage progress
st.markdown("#### Pipeline")
for col, step in zip(st.columns(len(StepName)), StepName, strict=True):
    done = state.is_step_completed(step)
    label = step.value.split(": ", 1)[-1]
    col.markdown(f"{'✅' if done else '⬜'} **{label}**")

st.divider()

tab_brief, tab_angles, tab_review, tab_output, tab_raw = st.tabs(
    ["📋 Brief", "🎯 Angles", "🔁 Review loop", "📦 Final package", "💾 Raw state"]
)

with tab_brief:
    if state.brief:
        left, right = st.columns(2)
        left.metric("Product", state.brief.product)
        right.metric("Goal", state.brief.content_goal)
        st.markdown(f"**Audience** — {state.brief.target_audience}")
        st.markdown(f"**Key message** — {state.brief.key_message}")
    else:
        st.caption("Not generated yet.")

with tab_angles:
    if state.angles and state.angles.options:
        winner = state.selected_angle
        if winner:
            st.success(f"Winning angle: **{winner.style_name}** ({winner.score_percentage}%)")
        for col, option in zip(
            st.columns(len(state.angles.options)), state.angles.options, strict=True
        ):
            with col, st.container(border=True):
                st.markdown(f"#### {option.style_name}")
                st.progress(option.score_percentage / 100, text=f"{option.score_percentage}%")
                st.caption(option.content)
                st.markdown("**Pros**")
                for pro in option.pros:
                    st.markdown(f"- {pro}")
                st.markdown("**Cons**")
                for con in option.cons:
                    st.markdown(f"- {con}")
    else:
        st.caption("Not generated yet.")

with tab_review:
    if state.safety_check:
        check = state.safety_check
        st.markdown("##### Safety screening")
        (st.success if check.is_safe else st.error)(
            f"{'Passed' if check.is_safe else 'Flagged'} — risk score {check.risk_score:.2f}"
        )
        for issue in check.flagged_issues:
            st.markdown(f"- {issue}")

    st.markdown("##### Quality review")
    if state.qa_history:
        st.caption(
            f"{state.qa_rejection_count} rejection(s) out of a budget of {state.max_qa_retries}."
        )
        for index, verdict in enumerate(state.qa_history, start=1):
            with st.container(border=True):
                head, score = st.columns([4, 1])
                head.markdown(
                    f"**Attempt {index}** — {'approved' if verdict.is_approved else 'rejected'}"
                )
                score.metric("Score", f"{verdict.score:.1f}")
                st.caption(verdict.reason)
    else:
        st.caption("No review yet.")

    if len(state.drafts_history) > 1:
        st.markdown("##### Draft revisions")
        for index, draft_set in enumerate(state.drafts_history, start=1):
            best = draft_set.best()
            if best:
                with st.expander(f"Revision {index}: {best.headline}"):
                    st.write(best.body)

with tab_output:
    draft = state.content_drafts.best() if state.content_drafts else None
    if draft:
        with st.container(border=True):
            st.caption("🌐 Sponsored")
            st.markdown(f"### {draft.headline}")
            st.write(draft.body)
            st.button(draft.call_to_action, type="primary", disabled=True)
    if state.final_content:
        st.divider()
        st.markdown(state.final_content)
        st.download_button(
            "⬇️ Download package",
            data=state.final_content,
            file_name=f"{state.workflow_id}.md",
            mime="text/markdown",
        )
    elif not draft:
        st.caption("Nothing produced yet.")

with tab_raw:
    st.json(state.model_dump(mode="json"))
