"""Semantic candidate filtering with pluggable embeddings and LLM labeling."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Sequence, Tuple

from .diversity import _cosine, semantic_novelty


class NoveltyLabel(str, Enum):
    DUPLICATE = "duplicate"
    RELATED = "related"
    DISTINCT = "distinct"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class CandidateDecision:
    candidate: str
    label: NoveltyLabel
    semantic_novelty: float
    neighbors: Tuple[str, ...]


@dataclass(frozen=True)
class FilterResult:
    selected: Tuple[CandidateDecision, ...]
    rejected: Tuple[CandidateDecision, ...]
    history_snapshot: Tuple[str, ...]


class NoveltyFilter:
    def __init__(self, *, embedding: Callable[[str], Sequence[float]], top_k: int = 3):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.embedding = embedding
        self.top_k = top_k

    def _neighbors(self, candidate: str, history: Sequence[str]) -> Tuple[Tuple[str, ...], float]:
        vector = self.embedding(candidate)
        ranked = sorted(
            ((other, _cosine(vector, self.embedding(other))) for other in history),
            key=lambda pair: pair[1],
            reverse=True,
        )[: self.top_k]
        return tuple(item[0] for item in ranked), semantic_novelty([item[1] for item in ranked])

    def filter(
        self,
        candidates: Sequence[str],
        *,
        history: Sequence[str],
        classify: Callable[[str, Sequence[str]], NoveltyLabel],
        budget: int,
    ) -> FilterResult:
        if budget < 0:
            raise ValueError("budget cannot be negative")
        snapshot = tuple(history)
        seen = set(snapshot)
        selected: List[CandidateDecision] = []
        rejected: List[CandidateDecision] = []
        for candidate in candidates:
            text = str(candidate).strip()
            if not text or text in seen:
                decision = CandidateDecision(text, NoveltyLabel.DUPLICATE, 0.0, ())
                rejected.append(decision)
                continue
            neighbors, novelty = self._neighbors(text, snapshot)
            label = NoveltyLabel(classify(text, neighbors))
            decision = CandidateDecision(text, label, novelty, neighbors)
            if label is NoveltyLabel.DUPLICATE:
                rejected.append(decision)
            else:
                selected.append(decision)
                seen.add(text)
        selected.sort(key=lambda item: item.semantic_novelty, reverse=True)
        overflow = selected[budget:]
        rejected.extend(overflow)
        return FilterResult(tuple(selected[:budget]), tuple(rejected), snapshot)
