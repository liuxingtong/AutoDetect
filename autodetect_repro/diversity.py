"""Deterministic semantic and behavioral diversity measurements.

The module deliberately contains no model or network calls.  Embeddings and
LLM decisions are supplied by callers, which keeps the metrics reproducible
and makes them usable for offline calibration.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence, Set


@dataclass(frozen=True)
class SemanticMetrics:
    mean_novelty: float
    unique_question_gain: int


@dataclass(frozen=True)
class BehavioralMetrics:
    richness: int
    gain: int
    discovery_rate: float


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have the same non-zero dimension")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_novelty(similarities: Sequence[float]) -> float:
    """Return ``1 - mean cosine similarity`` for a candidate's kNN set."""
    if not similarities:
        return 1.0
    return 1.0 - sum(similarities) / len(similarities)


def semantic_metrics(
    current_questions: Sequence[str],
    history_questions: Sequence[str],
    *,
    embeddings: Mapping[str, Sequence[float]],
    duplicate_ids: Set[str],
    k: int = 1,
) -> SemanticMetrics:
    """Measure a round against a frozen history snapshot.

    ``duplicate_ids`` is the result of the semantic filter (not inferred from
    the embedding distance).  Current-round candidates are compared only to
    ``history_questions`` so candidate ordering cannot affect the score.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    novelties = []
    available_history = [item for item in history_questions if item in embeddings]
    for question in current_questions:
        vector = embeddings.get(question)
        if vector is None:
            continue
        similarities = sorted(
            (_cosine(vector, embeddings[item]) for item in available_history),
            reverse=True,
        )[:k]
        novelties.append(semantic_novelty(similarities))
    mean_novelty = sum(novelties) / len(novelties) if novelties else 0.0
    unique_gain = sum(1 for question in current_questions if question not in duplicate_ids)
    return SemanticMetrics(mean_novelty, unique_gain)


def behavioral_metrics(
    previous_count: int,
    current_labels: Sequence[str],
    *,
    target_calls: int,
) -> BehavioralMetrics:
    """Update confirmed behavior-cluster counts from a round's labels.

    ``None``/empty labels represent Same or Uncertain outcomes and are not
    counted as newly confirmed clusters.  Callers should pass only labels that
    are confirmed Different clusters, with prior labels de-duplicated.
    """
    if previous_count < 0 or target_calls < 0:
        raise ValueError("counts cannot be negative")
    new_labels = {label for label in current_labels if label}
    richness = previous_count + len(new_labels)
    gain = len(new_labels)
    rate = gain / target_calls if target_calls else 0.0
    return BehavioralMetrics(richness, gain, rate)
