"""Tests for the model client: parsing, schema validation and retries.

A fake client stands in for the Groq SDK so nothing here touches the network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from config import Settings
from llm_service import LLMError, LLMService
from schemas import StrategicBrief


class FakeClient:
    """Returns queued responses in order and counts how often it was called."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_kwargs):
        self.calls += 1
        payload = self.responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(groq_api_key="test-key", llm_max_retries=3)


def service(settings: Settings, responses: list[str]) -> tuple[LLMService, FakeClient]:
    client = FakeClient(responses)
    return LLMService(settings=settings, client=client), client


BRIEF_JSON = json.dumps(
    {
        "product": "ARCHIE",
        "target_audience": "Marketers",
        "content_goal": "Leads",
        "key_message": "Fast campaigns",
    }
)


def test_a_valid_response_is_returned_as_a_typed_model(settings):
    llm, client = service(settings, [BRIEF_JSON])

    brief = llm.extract_brief("Launch ARCHIE")

    assert brief.product == "ARCHIE"
    assert client.calls == 1


def test_malformed_json_is_retried_then_raises(settings):
    llm, client = service(settings, ["not json", "still not json", "nope"])

    with pytest.raises(LLMError, match="StrategicBrief"):
        llm.extract_brief("x")

    assert client.calls == 3, "should exhaust the retry budget"


def test_a_transient_failure_recovers_on_the_next_attempt(settings):
    llm, client = service(settings, ["garbage", BRIEF_JSON])

    brief = llm.extract_brief("x")

    assert brief.product == "ARCHIE"
    assert client.calls == 2


def test_a_response_missing_required_fields_is_rejected(settings):
    incomplete = json.dumps({"product": "ARCHIE"})
    llm, client = service(settings, [incomplete] * 3)

    with pytest.raises(LLMError):
        llm.extract_brief("x")

    assert client.calls == 3


def test_percentage_strings_are_coerced_to_integers(settings):
    payload = json.dumps(
        {
            "options": [
                {
                    "option_id": "1",
                    "style_name": "Speed",
                    "content": "Fast.",
                    "pros": [],
                    "cons": [],
                    "score_percentage": "87%",
                }
            ]
        }
    )
    llm, _ = service(settings, [payload])

    angles = llm.generate_angles(
        StrategicBrief(
            product="p", target_audience="a", content_goal="g", key_message="k"
        )
    )

    assert angles.options[0].score_percentage == 87


def test_a_missing_api_key_is_reported_clearly():
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLMService(settings=Settings(groq_api_key=""))
