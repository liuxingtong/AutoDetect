"""Post-evaluation adaptive analysis separated from experiment scheduling."""

from dataclasses import asdict
from typing import Callable, Sequence, Tuple

from .behavior import BehaviorLabel, BehavioralAnalyzer, FailureSignature
from .branches import BranchManager
from .types import SampleRecord


Event = Tuple[str, dict]


class AdaptiveAnalysisPipeline:
    def __init__(
        self,
        *,
        signature_extractor: Callable[[SampleRecord], FailureSignature],
        behavior_classifier: Callable[[FailureSignature, Sequence[FailureSignature]], BehaviorLabel],
        branch_manager: BranchManager,
    ) -> None:
        self.signature_extractor = signature_extractor
        self.behavior_classifier = behavior_classifier
        self.behavior_analyzer = BehavioralAnalyzer()
        self.branch_manager = branch_manager
        self.target_calls = 0

    def observe(self, record: SampleRecord) -> Tuple[Event, ...]:
        self.target_calls += 1
        if record.status != "valid" or not record.is_weakness:
            return ()
        signature = self.signature_extractor(record)
        label = self.behavior_analyzer.add(signature, self.behavior_classifier)
        branch_id = "{}:{}".format(record.knowledge_point, signature.violation)
        branch = self.branch_manager.record_support(
            branch_id,
            record.knowledge_point,
            signature,
            record.sample_id,
        )
        metrics = self.behavior_analyzer.metrics(target_calls=self.target_calls)
        return (
            ("failure_analysis", {"sample_id": record.sample_id, "signature": signature.to_dict()}),
            ("behavior_cluster_update", {"sample_id": record.sample_id, "label": label.value}),
            ("diversity_metrics", asdict(metrics)),
            (
                "branch_update",
                {
                    "branch_id": branch.branch_id,
                    "status": branch.status,
                    "supporting_samples": list(branch.supporting_samples),
                },
            ),
        )
