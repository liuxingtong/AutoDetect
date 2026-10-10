"""Failure signatures and offline behavioral novelty clustering."""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Callable, Dict, List, Sequence

from .diversity import BehavioralMetrics, behavioral_metrics


class BehaviorLabel(str, Enum):
    SAME = "same"
    DIFFERENT = "different"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class FailureSignature:
    violation: str
    observed_behavior: str
    trigger_context: str
    evidence: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorCluster:
    cluster_id: str
    representative: FailureSignature


class BehavioralAnalyzer:
    def __init__(self) -> None:
        self._clusters: List[BehaviorCluster] = []
        self._round_new: List[str] = []

    @property
    def clusters(self) -> Sequence[BehaviorCluster]:
        return tuple(self._clusters)

    def add(
        self,
        signature: FailureSignature,
        classify: Callable[[FailureSignature, Sequence[FailureSignature]], BehaviorLabel],
    ) -> BehaviorLabel:
        representatives = tuple(cluster.representative for cluster in self._clusters)
        label = BehaviorLabel(classify(signature, representatives))
        if label is BehaviorLabel.DIFFERENT:
            cluster_id = "behavior-{:04d}".format(len(self._clusters) + 1)
            self._clusters.append(BehaviorCluster(cluster_id, signature))
            self._round_new.append(cluster_id)
        return label

    def metrics(self, *, target_calls: int) -> BehavioralMetrics:
        return behavioral_metrics(
            previous_count=len(self._clusters) - len(self._round_new),
            current_labels=self._round_new,
            target_calls=target_calls,
        )

    def finish_round(self) -> None:
        self._round_new.clear()
