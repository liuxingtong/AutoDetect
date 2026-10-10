"""Unit tests for the LLM-as-judge classes in
``autodetect_repro.calibration``.

The judge classes (SemanticJudge, BehaviorJudge) all delegate to
``roles._structured_call`` for the actual LLM call + JSON repair
loop.  The previous commit removed the duplicate ``_structured_call``
from ``calibration.py`` and made it the single source of truth.

These tests pin the *judge-side* contract:

- Happy path: a single well-formed JSON response yields a parsed
  JudgeVerdict with the right decision / confidence / reason.
- Repair path: the first response is invalid, the second one is
  valid -> the judge returns the valid one (no exception).
- Failure path: both responses are invalid -> the judge raises
  the package-level RoleError rather than silently returning a
  bogus verdict.

A ``FakeJudgeClient`` records every ``complete()`` call so the
tests can assert on prompt structure and on retry behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

import pytest


from autodetect_repro.calibration import (
    BEHAVIOR_LABELS,
    SEMANTIC_LABELS,
    BehaviorJudge,
    JudgeVerdict,
    SemanticJudge,
)
from autodetect_repro.client import ChatResult, TokenUsage
from autodetect_repro.roles import RoleError


# --- client stub ------------------------------------------------------------


class FakeJudgeClient:
    """A scripted replacement for ``SiliconFlowClient`` that returns
    a sequence of pre-canned responses and records every call so
    the tests can assert on the prompts."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.calls: List[Tuple[List[dict], float, int]] = []
        self._call_count = 0

    def complete(self, messages, *, temperature, max_tokens):
        self.calls.append((list(messages), temperature, max_tokens))
        idx = min(self._call_count, len(self.responses) - 1)
        self._call_count += 1
        return ChatResult(
            content=self.responses[idx],
            usage=TokenUsage(
                prompt_tokens=20,
                completion_tokens=10,
                total_tokens=30,
            ),
            request_count=1,
        )


# --- SemanticJudge ----------------------------------------------------------


def test_semantic_judge_happy_path():
    client = FakeJudgeClient(
        ['{"decision": "duplicate", "confidence": 0.9, "reason": "same text"}']
    )
    judge = SemanticJudge(client)
    verdict = judge.judge("Q1: write a list", "Q1: write a list")
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.decision == "duplicate"
    assert verdict.confidence == pytest.approx(0.9)
    assert verdict.reason == "same text"
    # And the validator only accepts the three documented labels.
    assert verdict.decision in SEMANTIC_LABELS
    # Exactly one call: no repair needed.
    assert len(client.calls) == 1


def test_semantic_judge_recovers_after_repair():
    """First response is bad JSON, second is good -> returns the
    good verdict without raising."""
    client = FakeJudgeClient(
        [
            "this is not JSON at all",  # 1st attempt: parse fails
            '{"decision": "related", "confidence": 0.6, "reason": "overlap"}',
        ]
    )
    judge = SemanticJudge(client)
    verdict = judge.judge("Q1", "Q2")
    assert verdict.decision == "related"
    assert verdict.confidence == pytest.approx(0.6)
    # Two calls: original + repair.
    assert len(client.calls) == 2
    # The second call's temperature is reset to 0.0 by _structured_call.
    _, temp2, _ = client.calls[1]
    assert temp2 == 0.0


def test_semantic_judge_rejects_invalid_label():
    client = FakeJudgeClient(
        [
            '{"decision": "banana", "confidence": 0.5, "reason": "r"}',
            '{"decision": "banana", "confidence": 0.5, "reason": "r"}',
        ]
    )
    judge = SemanticJudge(client)
    with pytest.raises(RoleError):
        judge.judge("Q1", "Q2")
    # Two attempts were made before giving up.
    assert len(client.calls) == 2


def test_semantic_judge_normalizes_label_case():
    """The validator lower-cases the decision; ensure that
    'Distinct' is accepted and surfaced as 'distinct'."""
    client = FakeJudgeClient(
        ['{"decision": "Distinct", "confidence": 0.7, "reason": "r"}']
    )
    judge = SemanticJudge(client)
    verdict = judge.judge("Q1", "Q2")
    assert verdict.decision == "distinct"


def test_semantic_judge_clamps_confidence():
    """Confidence > 1.0 is clamped to 1.0 by JudgeVerdict.__post_init__."""
    client = FakeJudgeClient(
        ['{"decision": "related", "confidence": 1.5, "reason": "r"}']
    )
    judge = SemanticJudge(client)
    verdict = judge.judge("Q1", "Q2")
    assert verdict.confidence == pytest.approx(1.0)


# --- BehaviorJudge ----------------------------------------------------------


def test_behavior_judge_happy_path():
    client = FakeJudgeClient(
        ['{"decision": "same", "confidence": 0.85, "reason": "same root cause"}']
    )
    judge = BehaviorJudge(client)
    verdict = judge.judge(
        {"violation": "v", "evidence": "e1"},
        {"violation": "v", "evidence": "e2"},
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.decision == "same"
    assert verdict.confidence == pytest.approx(0.85)
    assert verdict.reason == "same root cause"
    assert verdict.decision in BEHAVIOR_LABELS
    assert len(client.calls) == 1


def test_behavior_judge_recovers_after_repair():
    client = FakeJudgeClient(
        [
            "Random chatter, no JSON",
            '{"decision": "different", "confidence": 0.4, "reason": "different"}',
        ]
    )
    judge = BehaviorJudge(client)
    verdict = judge.judge({"violation": "v"}, {"violation": "w"})
    assert verdict.decision == "different"
    assert len(client.calls) == 2


def test_behavior_judge_rejects_invalid_label():
    client = FakeJudgeClient(
        [
            '{"decision": "yes", "confidence": 0.5, "reason": "r"}',
            '{"decision": "yes", "confidence": 0.5, "reason": "r"}',
        ]
    )
    judge = BehaviorJudge(client)
    with pytest.raises(RoleError):
        judge.judge({"v": 1}, {"v": 2})


def test_behavior_judge_accepts_uncertain():
    """The 'uncertain' label is a valid behavior decision that the
    calibration script needs to count separately."""
    client = FakeJudgeClient(
        ['{"decision": "uncertain", "confidence": 0.3, "reason": "ambiguous"}']
    )
    judge = BehaviorJudge(client)
    verdict = judge.judge({"v": 1}, {"v": 2})
    assert verdict.decision == "uncertain"
    assert verdict.confidence == pytest.approx(0.3)


# --- cross-cutting ----------------------------------------------------------


def test_judge_uses_zero_temperature_for_repair():
    """Both judges use temperature=0.0 on the first call, and
    _structured_call resets it to 0.0 on the repair attempt
    (not the caller's original temperature)."""
    client = FakeJudgeClient(
        [
            "garbage",
            '{"decision": "duplicate", "confidence": 0.5, "reason": "r"}',
        ]
    )
    judge = SemanticJudge(client)
    judge.judge("Q1", "Q2")
    # Both calls should be at 0.0 (the judge is deterministic).
    assert all(temp == 0.0 for (_, temp, _) in client.calls)


def test_judge_reason_truncated_to_240_chars():
    client = FakeJudgeClient(
        [
            '{"decision": "duplicate", "confidence": 0.5, "reason": "'
            + ("x" * 500) + '"}'
        ]
    )
    judge = SemanticJudge(client)
    verdict = judge.judge("Q1", "Q2")
    assert len(verdict.reason) <= 240


def test_judge_prompt_includes_both_inputs():
    """The judge should send the original prompts to the model;
    we verify this by inspecting what _structured_call forwarded."""
    client = FakeJudgeClient(
        ['{"decision": "duplicate", "confidence": 0.5, "reason": "r"}']
    )
    judge = SemanticJudge(client)
    judge.judge("alpha question", "beta question")
    assert len(client.calls) == 1
    messages, _, _ = client.calls[0]
    # The user message (the prompt) contains both inputs.
    prompt = messages[0]["content"]
    assert "alpha question" in prompt
    assert "beta question" in prompt
