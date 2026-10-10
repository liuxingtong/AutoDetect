"""Tests for the LLM-backed novelty classifier role."""

from dataclasses import dataclass
from typing import Any, Sequence

from autodetect_repro.novelty import NoveltyLabel
from autodetect_repro.roles import LLMNoveltyClassifier


@dataclass
class _FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _FakeChat:
    content: str
    request_count: int = 1
    usage: _FakeUsage = _FakeUsage()


class _FakeClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 0,
    ) -> _FakeChat:
        self.calls.append(list(messages))
        return _FakeChat(self.responses.pop(0))


def test_llm_novelty_classifier_parses_each_label():
    client = _FakeClient(
        [
            '{"label": "Duplicate", "reason": "same scenario"}',
            '{"label": "Related", "reason": "shares trigger"}',
            '{"label": "Distinct", "reason": "new domain"}',
            '{"label": "Uncertain", "reason": "insufficient signal"}',
        ]
    )
    classifier = LLMNoveltyClassifier(client, max_tokens=64)

    assert classifier("a", []) is NoveltyLabel.DUPLICATE
    assert classifier("b", []) is NoveltyLabel.RELATED
    assert classifier("c", []) is NoveltyLabel.DISTINCT
    assert classifier("d", []) is NoveltyLabel.UNCERTAIN
    assert len(client.calls) == 4


def test_llm_novelty_classifier_falls_back_to_uncertain_on_invalid_json():
    client = _FakeClient(["not-json", "still not json"])
    classifier = LLMNoveltyClassifier(client, max_tokens=64)

    assert classifier("q", ["n1"]) is NoveltyLabel.UNCERTAIN


def test_llm_novelty_classifier_falls_back_on_invalid_label():
    client = _FakeClient(
        [
            '{"label": "mystery", "reason": "?"}',
            '{"label": "mystery", "reason": "?"}',  # repair attempt
        ]
    )
    classifier = LLMNoveltyClassifier(client, max_tokens=64)

    assert classifier("q", []) is NoveltyLabel.UNCERTAIN


def test_llm_novelty_classifier_includes_neighbors_and_kp_in_prompt():
    client = _FakeClient(['{"label": "Related", "reason": "shared"}'])
    classifier = LLMNoveltyClassifier(
        client, max_tokens=64, knowledge_point="word-count"
    )
    classifier("new candidate", ["old-1", "old-2"])

    assert len(client.calls) == 1
    prompt = client.calls[0][0]["content"]
    assert "new candidate" in prompt
    assert "old-1" in prompt
    assert "old-2" in prompt
    assert "word-count" in prompt
