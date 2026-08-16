"""Filesystem-backed persistence for workflow state.

The interface is deliberately narrow (save / load / list / delete) so it can be
replaced by a SQLite or Postgres implementation without the orchestrator
noticing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from schemas import WorkflowState

logger = logging.getLogger("ArchieAgent.store")

_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidWorkflowId(ValueError):
    """Raised when an identifier is unsafe to use as a filename."""


def validate_workflow_id(workflow_id: str) -> str:
    """Return the id unchanged, or raise if it could escape the state directory.

    Only letters, digits, underscore and hyphen are allowed, which blocks path
    separators and parent-directory references such as '../../etc/passwd'.
    """
    if not isinstance(workflow_id, str) or not _VALID_ID.match(workflow_id):
        raise InvalidWorkflowId(
            "Workflow ID may contain only letters, digits, '_' and '-' (1-64 characters)."
        )
    return workflow_id


class StateStore:
    """Reads and writes one JSON document per workflow run."""

    def __init__(self, storage_dir: str = "states") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, workflow_id: str) -> Path:
        return self.storage_dir / f"{validate_workflow_id(workflow_id)}.json"

    def save(self, state: WorkflowState) -> None:
        """Persist a workflow state, overwriting any previous version."""
        path = self._path_for(state.workflow_id)
        path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("Saved workflow '%s' to %s", state.workflow_id, path)

    def load(self, workflow_id: str) -> WorkflowState | None:
        """Return a stored run, or None when it is absent or unreadable."""
        try:
            path = self._path_for(workflow_id)
        except InvalidWorkflowId:
            return None

        if not path.exists():
            return None

        try:
            return WorkflowState.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read workflow '%s': %s", workflow_id, exc)
            return None

    def list_ids(self) -> list[str]:
        """Return every workflow id in the state directory, newest first."""
        paths = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [p.stem for p in paths]

    def delete(self, workflow_id: str) -> bool:
        """Remove a run. Returns True if a file was actually deleted."""
        path = self._path_for(workflow_id)
        if path.exists():
            path.unlink()
            logger.info("Deleted workflow '%s'", workflow_id)
            return True
        return False

    def clear_all(self) -> int:
        """Delete every stored run and return how many were removed."""
        count = 0
        for path in self.storage_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count


# Kept so existing imports continue to resolve.
FileStateStore = StateStore
