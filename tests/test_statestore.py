"""Tests for persistence and identifier validation."""

from __future__ import annotations

import pytest

from schemas import WorkflowState, WorkflowStatus
from statestore import InvalidWorkflowId, validate_workflow_id


def test_state_survives_a_round_trip(store, sample_brief):
    """A saved state reloads as an equal object, nested models included."""
    original = WorkflowState(
        workflow_id="round_trip",
        original_prompt="Test",
        status=WorkflowStatus.RUNNING,
        brief=sample_brief,
    )
    store.save(original)

    reloaded = store.load("round_trip")

    assert reloaded is not None
    assert reloaded.brief == sample_brief
    assert reloaded.status is WorkflowStatus.RUNNING


def test_loading_an_unknown_id_returns_none(store):
    assert store.load("never_saved") is None


def test_corrupt_files_are_reported_as_missing(store):
    """Unparseable state does not crash the caller."""
    (store.storage_dir / "broken.json").write_text("{not json", encoding="utf-8")
    assert store.load("broken") is None


@pytest.mark.parametrize(
    "bad_id",
    ["../../etc/passwd", "a/b", "..", "", "with space", "quote'd", "x" * 65],
)
def test_unsafe_identifiers_are_rejected(bad_id):
    """Ids that could escape the state directory never reach the filesystem."""
    with pytest.raises(InvalidWorkflowId):
        validate_workflow_id(bad_id)


def test_traversal_attempt_writes_nothing(store, tmp_path):
    """A malicious id fails loudly instead of creating a file outside the store."""
    state = WorkflowState(workflow_id="ok_id", original_prompt="x")
    state.workflow_id = "../escaped"

    with pytest.raises(InvalidWorkflowId):
        store.save(state)

    assert not (tmp_path / "escaped.json").exists()


def test_listing_and_deleting(store):
    for wid in ("alpha", "beta"):
        store.save(WorkflowState(workflow_id=wid, original_prompt="x"))

    assert sorted(store.list_ids()) == ["alpha", "beta"]
    assert store.delete("alpha") is True
    assert store.delete("alpha") is False
    assert store.list_ids() == ["beta"]
    assert store.clear_all() == 1
