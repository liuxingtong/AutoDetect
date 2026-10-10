"""LLM-as-judge calibration protocol for the adaptive framework.

Provides typed judges that, given a pair of inputs, classify them using
the same labels the runtime would emit.  Used in ``scripts/calibrate.py``
to measure agreement between the LLM judges and human ground truth
before running cross-mode comparison.

The judge contract is intentionally narrow:

- :class:`SemanticJudge`    classifies a *question pair* as Duplicate,
  Related, or Distinct.
- :class:`BehaviorJudge`    classifies a *signature pair* as Same,
  Different, or Uncertain.
- :class:`PromotionJudge`   was already implemented in ``roles.py`` and
  is not re-declared here.

All three judges return a frozen dataclass with ``decision``,
``confidence`` (0.0 - 1.0), and ``reason`` (free text ≤ 240 chars).
Confidence is the LLM's own self-rating; it lets the calibration
report plot agreement-vs-confidence curves without forcing every
downstream caller to be calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol

from .client import SiliconFlowClient
from .prompts import behavioral_clustering_prompt
from .roles import _structured_call
from .types import LimitsConfig


# ---------------------------------------------------------------------------
# Label enums.
# ---------------------------------------------------------------------------


SEMANTIC_LABELS = ("duplicate", "related", "distinct")
BEHAVIOR_LABELS = ("same", "different", "uncertain")


@dataclass(frozen=True)
class JudgeVerdict:
    """A single judge decision.

    Attributes
    ----------
    decision:
        One of the label strings declared above (lowercase).  For
        semantic judges: "duplicate" | "related" | "distinct".
        For behavior judges: "same" | "different" | "uncertain".
    confidence:
        LLM's own self-rated confidence in [0.0, 1.0].  A well-calibrated
        judge should have higher accuracy at higher confidence.
    reason:
        A short justification.  Capped to 240 chars by the parser to
        keep reports readable.
    raw:
        The parsed JSON payload as returned by the LLM.  Kept for
        debugging and for re-running the validation logic offline.
    """

    decision: str
    confidence: float
    reason: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # pragma: no cover - input hygiene
        # Normalize confidence into [0, 1] regardless of the LLM scale.
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(
                self, "confidence", max(0.0, min(1.0, self.confidence))
            )
        object.__setattr__(self, "reason", (self.reason or "")[:240])


# ---------------------------------------------------------------------------
# Prompt templates.
# ---------------------------------------------------------------------------


def _semantic_judge_prompt(question_a: str, question_b: str) -> str:
    """Wrap the existing novelty_prompt with strict judge instructions.

    We deliberately reuse the prompt the runtime LLMNoveltyClassifier
    uses, so that the judge and the runtime see the same input format.
    The wrapper adds an explicit confidence request and forbids
    explanation outside JSON.
    """
    return (
        "You are an AutoDetect semantic-novelty judge. Compare two candidate "
        "questions and decide whether they test substantively the same thing. "
        "Return strict JSON with fields: decision, confidence, reason. "
        "decision is one of duplicate, related, or distinct. "
        "confidence is a number in [0, 1] representing your certainty. "
        "reason is a short justification of <= 240 characters.\n"
        "Duplicate means changing only entities / surface names while the "
        "tested condition and expected answer pattern are the same. "
        "Related means the questions share some structure but differ in at "
        "least one material test variable. "
        "Distinct means the questions target different tested conditions.\n"
        "question_a={a}\nquestion_b={b}"
    ).format(a=question_a, b=question_b)


def _behavior_judge_prompt(signature_a: Dict[str, Any], signature_b: Dict[str, Any]) -> str:
    """Wrap the existing behavioral_clustering_prompt with judge framing.

    The runtime analyzer uses the same prompt, so calibration is
    measured on the same model contract the runtime depends on.
    """
    return (
        "You are an AutoDetect behavior-clustering judge. Compare two "
        "Failure Signatures and decide whether they describe the same "
        "underlying failure mechanism. Return strict JSON with fields: "
        "decision, confidence, reason. decision is one of same, different, "
        "or uncertain. confidence is a number in [0, 1] representing your "
        "certainty. reason is a short justification of <= 240 characters.\n"
        "Same means the observed behavior and trigger condition are "
        "materially the same. Different means a different mechanism. "
        "Uncertain means the evidence does not let you decide.\n"
        "signature_a={a}\nsignature_b={b}"
    ).format(a=signature_a, b=signature_b)


# ---------------------------------------------------------------------------
# Validation callbacks for _structured_call.
# ---------------------------------------------------------------------------


def _validate_semantic(payload: Dict[str, Any]) -> JudgeVerdict:
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in SEMANTIC_LABELS:
        raise ValueError("invalid semantic decision: {}".format(decision))
    confidence = float(payload.get("confidence", 0.0))
    reason = str(payload.get("reason", "")).strip()
    return JudgeVerdict(decision=decision, confidence=confidence,
                        reason=reason, raw=dict(payload))


def _validate_behavior(payload: Dict[str, Any]) -> JudgeVerdict:
    decision = str(payload.get("decision", "")).strip().lower()
    if decision not in BEHAVIOR_LABELS:
        raise ValueError("invalid behavior decision: {}".format(decision))
    confidence = float(payload.get("confidence", 0.0))
    reason = str(payload.get("reason", "")).strip()
    return JudgeVerdict(decision=decision, confidence=confidence,
                        reason=reason, raw=dict(payload))


# ---------------------------------------------------------------------------
# Public judge classes.
# ---------------------------------------------------------------------------


class _BaseJudge(Protocol):
    def judge(self, *args: Any, **kwargs: Any) -> JudgeVerdict: ...


class SemanticJudge:
    """LLM-as-judge for question-pair semantic novelty."""

    def __init__(
        self,
        client: SiliconFlowClient,
        limits: Optional[LimitsConfig] = None,
    ) -> None:
        self.client = client
        self.max_tokens = (limits.max_output_tokens if limits else 1024)

    def judge(self, question_a: str, question_b: str) -> JudgeVerdict:
        return _structured_call(
            self.client,
            _semantic_judge_prompt(question_a, question_b),
            max_tokens=self.max_tokens,
            temperature=0.0,
            validate=_validate_semantic,
        )


class BehaviorJudge:
    """LLM-as-judge for behavior-signature clustering."""

    def __init__(
        self,
        client: SiliconFlowClient,
        limits: Optional[LimitsConfig] = None,
    ) -> None:
        self.client = client
        self.max_tokens = (limits.max_output_tokens if limits else 1024)

    def judge(
        self, signature_a: Dict[str, Any], signature_b: Dict[str, Any]
    ) -> JudgeVerdict:
        return _structured_call(
            self.client,
            _behavior_judge_prompt(signature_a, signature_b),
            max_tokens=self.max_tokens,
            temperature=0.0,
            validate=_validate_behavior,
        )
