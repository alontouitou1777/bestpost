"""UI-agnostic wrapper around the Groq chat completions API.

Each pipeline stage gets its own method that returns a validated Pydantic
model. Callers never see raw dictionaries, and a response that does not match
the expected shape is retried before it is allowed to fail the run.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

from groq import Groq
from pydantic import BaseModel

from config import Settings, get_settings
from schemas import (
    AngleOption,
    ContentDrafts,
    CreativeAngles,
    DraftOption,
    QACheck,
    SafetyCheck,
    StrategicBrief,
)

logger = logging.getLogger("ArchieAgent.llm")

TModel = TypeVar("TModel", bound=BaseModel)

_JSON_INSTRUCTION = " You MUST respond with a single valid JSON object and nothing else."


class LLMError(RuntimeError):
    """Raised when the model cannot be reached or returns unusable output."""


class LLMService:
    """Issues JSON-mode completions and returns validated domain objects."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        """Create a service.

        Args:
            settings: Configuration object. Defaults to reading the environment.
            client: Optional pre-built client, used to inject a fake in tests.
        """
        self.settings = settings or get_settings()

        if client is not None:
            self.client = client
        else:
            if not self.settings.is_configured:
                raise LLMError(
                    "GROQ_API_KEY is not set. Add it to your .env file or environment."
                )
            self.client = Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.settings.llm_timeout_seconds,
            )

        self.model = self.settings.groq_model

    # -- Stage 1 ------------------------------------------------------------
    def extract_brief(self, prompt: str) -> StrategicBrief:
        """Turn a free-text request into a structured strategic brief."""
        return self._generate(
            system="You are a senior brand strategist.",
            prompt=f"""Read this campaign request and extract its strategic core.

Request: "{prompt}"

Return JSON with exactly these keys:
{{
  "product": "what is being marketed",
  "target_audience": "who it is for, one specific segment",
  "content_goal": "the single measurable objective",
  "key_message": "the one idea the audience must remember"
}}""",
            model_cls=StrategicBrief,
        )

    # -- Stage 2 ------------------------------------------------------------
    def generate_angles(self, brief: StrategicBrief) -> CreativeAngles:
        """Propose and score three distinct creative directions."""
        return self._generate(
            system="You are a chief marketing officer.",
            prompt=f"""Propose three genuinely different creative angles for this brief.

Product: {brief.product}
Audience: {brief.target_audience}
Goal: {brief.content_goal}
Key message: {brief.key_message}

Score each angle from 0 to 100 on how well it serves the goal. Scores must
differ from one another. Return JSON with exactly these keys:
{{
  "options": [
    {{
      "option_id": "1",
      "style_name": "short name for the angle",
      "content": "two sentences describing the angle",
      "pros": ["strength", "strength"],
      "cons": ["weakness"],
      "score_percentage": 87
    }}
  ]
}}
The options array must contain exactly three entries with ids "1", "2", "3".""",
            model_cls=CreativeAngles,
        )

    # -- Stage 3 ------------------------------------------------------------
    def generate_drafts(
        self,
        brief: StrategicBrief,
        angle: AngleOption | None,
        feedback_from_previous_qa: str | None = None,
    ) -> ContentDrafts:
        """Write two ad copy drafts and pick the stronger one.

        Args:
            feedback_from_previous_qa: Rejection reason from the last QA pass.
                When present it is injected into the prompt so the rewrite
                addresses the specific problem rather than trying at random.
        """
        angle_text = (
            f"{angle.style_name}: {angle.content}" if angle else "No specific angle selected."
        )
        revision_note = ""
        if feedback_from_previous_qa:
            revision_note = f"""

A previous attempt was REJECTED by the quality reviewer for this reason:
"{feedback_from_previous_qa}"
Your new drafts must fix this specific problem."""

        return self._generate(
            system="You are an elite direct response copywriter.",
            prompt=f"""Write ad copy for this campaign.

Product: {brief.product}
Audience: {brief.target_audience}
Goal: {brief.content_goal}
Key message: {brief.key_message}
Chosen angle: {angle_text}{revision_note}

Return JSON with exactly these keys:
{{
  "options": [
    {{
      "option_id": "1",
      "headline": "attention-grabbing headline",
      "body": "two or three short paragraphs",
      "call_to_action": "short button label"
    }}
  ],
  "best_option_id": "1",
  "selection_reasoning": "why that draft is the strongest"
}}
The options array must contain exactly two entries with ids "1" and "2".""",
            model_cls=ContentDrafts,
        )

    # -- Stage 4 ------------------------------------------------------------
    def check_safety(self, draft: DraftOption) -> SafetyCheck:
        """Screen a draft for misleading, unsafe or non-compliant claims."""
        return self._generate(
            system="You are a compliance reviewer for advertising standards.",
            prompt=f"""Review this ad copy for policy risk: unverifiable claims,
medical or financial guarantees, discriminatory targeting, or deceptive urgency.

Headline: {draft.headline}
Body: {draft.body}
Call to action: {draft.call_to_action}

Return JSON with exactly these keys:
{{
  "is_safe": true,
  "risk_score": 0.1,
  "flagged_issues": ["describe each problem, empty list if none"]
}}
risk_score is a float between 0.0 and 1.0.""",
            model_cls=SafetyCheck,
        )

    # -- Stage 5 ------------------------------------------------------------
    def evaluate_qa(self, brief: StrategicBrief, draft: DraftOption) -> QACheck:
        """Judge whether a draft is good enough to ship."""
        return self._generate(
            system="You are a demanding editorial reviewer. You reject mediocre work.",
            prompt=f"""Judge this ad copy against its brief.

Key message it must land: {brief.key_message}
Audience: {brief.target_audience}
Goal: {brief.content_goal}

Headline: {draft.headline}
Body: {draft.body}
Call to action: {draft.call_to_action}

Score from 0 to 10. Approve only if the score is 7 or higher. If you reject it,
the reason must be specific and actionable so a writer can fix it.

Return JSON with exactly these keys:
{{
  "is_approved": true,
  "score": 8.5,
  "reason": "specific justification"
}}""",
            model_cls=QACheck,
        )

    # -- Stage 6 ------------------------------------------------------------
    def generate_final_package(
        self, brief: StrategicBrief, angle: AngleOption | None, draft: DraftOption
    ) -> str:
        """Assemble the approved work into a deliverable summary.

        This is the one stage that asks for prose rather than structured data,
        so it skips JSON mode entirely. Wrapping a multi-paragraph Markdown
        document inside a JSON string field is needlessly fragile: any stray
        quote or newline from the model breaks the parse.
        """
        angle_name = angle.style_name if angle else "n/a"
        package = self._complete_text(
            system="You are a campaign producer preparing a handover document.",
            prompt=f"""Write a short campaign handover document in Markdown.

Product: {brief.product}
Audience: {brief.target_audience}
Angle: {angle_name}
Headline: {draft.headline}
Body: {draft.body}
Call to action: {draft.call_to_action}

Include a one-line summary, the approved copy, and three suggested visual
directions for the creative team. Reply with the document only, no preamble.""",
        )
        if not package.strip():
            raise LLMError("Final package response was empty.")
        return package

    def _complete_text(self, system: str, prompt: str) -> str:
        """Request free-form text, retrying on transport errors."""
        attempts = max(1, self.settings.llm_max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.settings.llm_temperature,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                raise LLMError("Model returned an empty response.")
            except Exception as exc:  # noqa: BLE001 - re-raised as LLMError below
                last_error = exc
                logger.warning("Text attempt %d/%d failed: %s", attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(2 ** (attempt - 1))

        raise LLMError(f"Could not generate the final package: {last_error}") from last_error

    # -- Plumbing -----------------------------------------------------------
    def generate_json(self, prompt: str, system_prompt: str = "You are an AI assistant.") -> dict:
        """Send a prompt and return the parsed JSON response as a dictionary."""
        return self._with_retries(prompt, system_prompt, None)

    def _generate(self, system: str, prompt: str, model_cls: type[TModel]) -> TModel:
        """Send a prompt and return the response validated as model_cls."""
        return self._with_retries(prompt, system, model_cls)

    def _with_retries(
        self, prompt: str, system_prompt: str, model_cls: type[TModel] | None
    ) -> Any:
        """Attempt a completion repeatedly, backing off between failures.

        Retries cover transport errors, unparseable JSON, and responses that do
        not satisfy the target schema, since all three are transient in
        practice and a second attempt usually succeeds.
        """
        attempts = max(1, self.settings.llm_max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                raw = self._call_model(prompt, system_prompt)
                data = self._parse_json(raw)
                return model_cls.model_validate(data) if model_cls else data
            except Exception as exc:  # noqa: BLE001 - re-raised as LLMError below
                last_error = exc
                logger.warning("LLM attempt %d/%d failed: %s", attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(2 ** (attempt - 1))

        target = model_cls.__name__ if model_cls else "JSON"
        raise LLMError(
            f"Could not obtain a valid {target} response after {attempts} attempts: {last_error}"
        ) from last_error

    def _call_model(self, prompt: str, system_prompt: str) -> str:
        """Perform a single completion request and return the raw text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt + _JSON_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.settings.llm_temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError("Model returned an empty response.")
        return content

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Parse model output into a dict, raising LLMError on malformed JSON."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            preview = raw[:200].replace("\n", " ")
            raise LLMError(f"Model did not return valid JSON. Got: {preview!r}") from exc

        if not isinstance(parsed, dict):
            raise LLMError(f"Expected a JSON object, got {type(parsed).__name__}.")
        return parsed
