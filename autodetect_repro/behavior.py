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
        self._last_category_id: str = ""

    @property
    def clusters(self) -> Sequence[BehaviorCluster]:
        return tuple(self._clusters)

    def add(
        self,
        signature: FailureSignature,
        classify: Callable[[FailureSignature, Sequence[FailureSignature]], BehaviorLabel],
    ) -> BehaviorLabel:
        # A classifier cannot make a meaningful comparison when there are no
        # representatives yet.  Treat the first confirmed failure as the
        # seed of a behavior cluster deterministically.
        # Besides being the only well-defined answer for an empty reference
        # set, this prevents ``uncertain`` from becoming a permanent
        # cold-start deadlock: an uncertain first judgment would otherwise
        # leave the representatives empty forever.
        if not self._clusters:
            cluster_id = "behavior-{:04d}".format(len(self._clusters) + 1)
            self._clusters.append(BehaviorCluster(cluster_id, signature))
            self._round_new.append(cluster_id)
            return BehaviorLabel.DIFFERENT
        representatives = tuple(cluster.representative for cluster in self._clusters)
        label = BehaviorLabel(classify(signature, representatives))
        if label is BehaviorLabel.DIFFERENT:
            cluster_id = "behavior-{:04d}".format(len(self._clusters) + 1)
            self._clusters.append(BehaviorCluster(cluster_id, signature))
            self._round_new.append(cluster_id)
        return label

    def add_taxonomy(
        self, signature: FailureSignature, taxonomy_branch: str
    ) -> BehaviorLabel:
        """Count taxonomy coverage without an LLM behavior judgment.

        The taxonomy branch is the diversity unit for the simple runtime
        path.  A branch contributes one cluster the first time it produces a
        confirmed failure; later failures in that branch are ``same`` for
        coverage purposes.
        """
        branch = str(taxonomy_branch).strip() or "unknown"
        for cluster in self._clusters:
            if cluster.cluster_id.endswith(":" + branch):
                return BehaviorLabel.SAME
        cluster_id = "behavior-{:04d}:{}".format(len(self._clusters) + 1, branch)
        self._clusters.append(BehaviorCluster(cluster_id, signature))
        self._round_new.append(cluster_id)
        return BehaviorLabel.DIFFERENT

    def add_category(
        self,
        signature: FailureSignature,
        classify: Callable[[FailureSignature, Sequence[FailureSignature]], BehaviorLabel],
    ) -> BehaviorLabel:
        """Add a peer taxonomy category from a novel failure signature.

        ``uncertain`` is deliberately not exposed in the runtime category
        path: an unresolved comparison becomes a new category so discovery
        cannot deadlock or silently discard a failure pattern.  The original
        signature and classifier result remain in the event log for audit.
        """
        if not self._clusters:
            label = self.add(signature, classify)
            self._last_category_id = self._clusters[-1].cluster_id
            return label
        representatives = tuple(cluster.representative for cluster in self._clusters)
        label = BehaviorLabel(classify(signature, representatives))
        if label is BehaviorLabel.SAME:
            # The binary classifier does not return a matching ID.  Keep the
            # category event explicit without inventing a new category ID.
            self._last_category_id = ""
            return label
        # Treat an uncertain comparison as a conservative discovery.  This
        # is the direct-promotion mode: every unresolved novel pattern gets a
        # peer category immediately and can be audited or merged later.
        cluster_id = "behavior-{:04d}".format(len(self._clusters) + 1)
        self._clusters.append(BehaviorCluster(cluster_id, signature))
        self._round_new.append(cluster_id)
        self._last_category_id = cluster_id
        return BehaviorLabel.DIFFERENT

    @property
    def last_category_id(self) -> str:
        return self._last_category_id

    def metrics(self, *, target_calls: int) -> BehavioralMetrics:
        return behavioral_metrics(
            previous_count=len(self._clusters) - len(self._round_new),
            current_labels=self._round_new,
            target_calls=target_calls,
        )

    def finish_round(self) -> None:
        self._round_new.clear()
