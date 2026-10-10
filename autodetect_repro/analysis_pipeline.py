"""Post-evaluation adaptive analysis separated from experiment scheduling."""

from dataclasses import asdict, replace as _dataclass_replace
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .behavior import BehaviorLabel, BehavioralAnalyzer, FailureSignature
from .branches import (
    BranchManager,
    BranchState,
    PromotionDecision,
    PromotionProposal,
)
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

    def record_verification(
        self, branch_id: str, verification_id: str
    ) -> Tuple[Event, ...]:
        """Mark a sample as having been re-run as a controlled verification.

        Verification provides the controlled-comparison evidence required by
        :meth:`BranchManager.propose` before the branch can be promoted to
        the taxonomy.
        """
        if branch_id not in self.branch_manager._branches:
            return ()
        branch = self.branch_manager.record_verification(branch_id, verification_id)
        return (
            (
                "branch_update",
                {
                    "branch_id": branch.branch_id,
                    "status": branch.status,
                    "verifications": list(branch.verifications),
                },
            ),
        )

    def record_counterexample(
        self, branch_id: str, sample_id: str
    ) -> Tuple[Event, ...]:
        """Mark a sample as contradicting the branch's claimed failure.

        Any branch with at least one counterexample is barred from
        promotion until the counterexample is overturned.
        """
        if branch_id not in self.branch_manager._branches:
            return ()
        branch = self.branch_manager.record_counterexample(branch_id, sample_id)
        return (
            (
                "branch_update",
                {
                    "branch_id": branch.branch_id,
                    "status": branch.status,
                    "counterexamples": list(branch.counterexamples),
                },
            ),
        )

    def propose_promotions(
        self,
        *,
        auto_promote: bool,
        semantic_independence: Dict[str, bool],
    ) -> List[Tuple[PromotionProposal, Tuple[str, dict]]]:
        """Ask the BranchManager to propose a decision for every tracked branch.

        ``semantic_independence`` is a mapping from ``branch_id`` to whether
        the supporting samples come from materially different contexts
        (different phases, iterations, or seeds).  When the mapping is
        missing for a branch the conservative default is "not independent"
        so we never promote on insufficient evidence.

        When ``auto_promote`` is True, decisions that BranchManager.propose
        decides to PROMOTE are kept and the branch status becomes
        "promoted".  When False, every decision is still computed and
        emitted as an audit event, but the branch status is rolled back
        so the operator can accept the proposal explicitly.
        """
        results: List[Tuple[PromotionProposal, Tuple[str, dict]]] = []
        for branch_id, branch in list(self.branch_manager._branches.items()):
            semantically_independent = semantic_independence.get(branch_id, False)
            proposal = self.branch_manager.propose(
                branch_id, semantically_independent=semantically_independent
            )
            if not auto_promote:
                # Audit mode: roll back any status change made by propose()
                # so the operator must explicitly accept the proposal.
                if proposal.decision is PromotionDecision.PROMOTE:
                    self.branch_manager._branches[branch_id] = _dataclass_replace(
                        branch, status="verifying"
                    )
            payload = {
                "branch_id": proposal.branch_id,
                "decision": proposal.decision.value,
                "knowledge_point": proposal.knowledge_point,
                "reason": proposal.reason,
            }
            event = ("promotion_proposal", payload)
            results.append((proposal, event))
        return results

    def replay(self, events: Sequence[Dict[str, Any]]) -> None:
        """Rebuild analyzer / branch manager / call counter from a saved log.

        Designed to be called once after a fresh ``AdaptiveAnalysisPipeline``
        is constructed for a resumed run, *before* any new ``observe()`` calls.
        It walks the events in order and replays the ones that contribute
        state: ``failure_analysis`` (signatures), ``branch_update`` (branch
        state), and the implicit ``target_calls`` increment.  Idempotent on
        a clean pipeline.
        """
        for event in events:
            event_type = event.get("event_type")
            if event_type == "failure_analysis":
                self.target_calls += 1
                signature = FailureSignature(**event["signature"])
                self.behavior_analyzer.add(signature, self.behavior_classifier)
            elif event_type == "branch_update":
                branch_id = event["branch_id"]
                samples = tuple(event.get("supporting_samples", []))
                status = event.get("status", "candidate")
                if branch_id in self.branch_manager._branches:
                    branch = self.branch_manager._branches[branch_id]
                    self.branch_manager._branches[branch_id] = _dataclass_replace(
                        branch,
                        supporting_samples=samples,
                        status=status,
                    )
                else:
                    # Reconstruct a minimal branch from the persisted snapshot.
                    # The full signature is unknown; ``record_support`` will
                    # overwrite the fields it needs when the resumed run
                    # observes a fresh failure for this branch.
                    self.branch_manager._branches[branch_id] = BranchState(
                        branch_id=branch_id,
                        knowledge_point="",
                        signature=FailureSignature("", "", "", ""),
                        supporting_samples=samples,
                        status=status,
                    )
        # ``behavior_cluster_update`` and ``diversity_metrics`` are derived
        # state and are not used to rebuild anything; we skip them
        # intentionally so the replay is idempotent.
