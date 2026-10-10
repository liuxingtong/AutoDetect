from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
from autodetect_repro.behavior import BehaviorLabel, FailureSignature
from autodetect_repro.branches import (
    BranchManager,
    PromotionDecision,
)
from tests.test_checkpoint import make_record


def test_pipeline_ignores_non_failures_and_emits_behavior_events():
    pipeline = AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature("v", "b", "c", "e"),
        behavior_classifier=lambda signature, representatives: BehaviorLabel.DIFFERENT,
        branch_manager=BranchManager(min_support=2),
    )
    assert pipeline.observe(make_record().__class__(**{**make_record().__dict__, "is_weakness": False})) == ()
    events = pipeline.observe(make_record())
    assert [event[0] for event in events] == ["failure_analysis", "behavior_cluster_update", "diversity_metrics", "branch_update"]


def test_pipeline_replay_rebuilds_state_from_events():
    branch_manager = BranchManager(min_support=2)

    def classify(signature, representatives):
        return BehaviorLabel.DIFFERENT

    signature_a = FailureSignature("violation-A", "behavior-A", "ctx-A", "evidence-A")
    signature_b = FailureSignature("violation-B", "behavior-B", "ctx-B", "evidence-B")
    signature_c = FailureSignature("violation-A", "behavior-C", "ctx-C", "evidence-C")

    events = [
        {
            "event_type": "failure_analysis",
            "sample_id": "s1",
            "signature": signature_a.to_dict(),
        },
        {
            "event_type": "failure_analysis",
            "sample_id": "s2",
            "signature": signature_b.to_dict(),
        },
        {
            "event_type": "failure_analysis",
            "sample_id": "s3",
            "signature": signature_c.to_dict(),
        },
        {
            "event_type": "branch_update",
            "branch_id": "list format:violation-A",
            "status": "verifying",
            "supporting_samples": ["s1", "s3"],
        },
        {
            "event_type": "behavior_cluster_update",
            "sample_id": "s1",
            "label": "different",
        },
        {
            "event_type": "diversity_metrics",
            "richness": 2,
        },
    ]

    # A fresh pipeline replaying the same events should reach the same
    # behavioral cluster count and target-call counter as the live run.
    fresh = AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature("v", "b", "c", "e"),
        behavior_classifier=classify,
        branch_manager=BranchManager(min_support=2),
    )
    fresh.replay(events)

    assert fresh.target_calls == 3
    # Three DIFFERENT signatures, so three clusters.
    assert len(fresh.behavior_analyzer.clusters) == 3
    # Branch state was restored from the snapshot.
    branch = fresh.branch_manager.get("list format:violation-A")
    assert branch.supporting_samples == ("s1", "s3")
    assert branch.status == "verifying"


def _build_pipeline(branch_manager):
    def signature_for(record):
        return FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        )

    def classify(signature, representatives):
        return BehaviorLabel.DIFFERENT

    return AdaptiveAnalysisPipeline(
        signature_extractor=signature_for,
        behavior_classifier=classify,
        branch_manager=branch_manager,
    )


def test_record_verification_emits_branch_update_event():
    manager = BranchManager(min_support=2)
    pipeline = _build_pipeline(manager)
    record = make_record(prompt="weak prompt 1")
    pipeline.observe(record)
    branch_id = "list format:list constraint"

    events = pipeline.record_verification(branch_id, "ver-1")
    assert [event[0] for event in events] == ["branch_update"]
    assert events[0][1]["branch_id"] == branch_id
    assert events[0][1]["verifications"] == ["ver-1"]
    assert manager.get(branch_id).verifications == ("ver-1",)


def test_record_counterexample_blocks_promotion():
    manager = BranchManager(min_support=2)
    pipeline = _build_pipeline(manager)

    # Two supporting samples trigger the "verifying" status.
    pipeline.observe(make_record(identifier="s1", prompt="weak-1"))
    pipeline.observe(make_record(identifier="s2", prompt="weak-2"))
    branch_id = "list format:list constraint"
    pipeline.record_verification(branch_id, "ver-1")
    pipeline.record_counterexample(branch_id, "ver-1")

    results = pipeline.propose_promotions(
        auto_promote=True,
        semantic_independence={branch_id: True},
    )
    assert len(results) == 1
    proposal, (event_type, payload) = results[0]
    assert proposal.decision is PromotionDecision.CONTINUE
    # Branch status remains "verifying" because counterexample blocks promotion.
    assert manager.get(branch_id).status == "verifying"
    assert event_type == "promotion_proposal"
    assert payload["decision"] == "continue_verification"


def test_propose_promotion_promotes_when_evidence_sufficient():
    manager = BranchManager(min_support=2)
    pipeline = _build_pipeline(manager)

    pipeline.observe(make_record(identifier="s1", prompt="weak-1"))
    pipeline.observe(make_record(identifier="s2", prompt="weak-2"))
    branch_id = "list format:list constraint"
    pipeline.record_verification(branch_id, "ver-1")

    results = pipeline.propose_promotions(
        auto_promote=True,
        semantic_independence={branch_id: True},
    )
    assert len(results) == 1
    proposal, (event_type, payload) = results[0]
    assert proposal.decision is PromotionDecision.PROMOTE
    assert event_type == "promotion_proposal"
    assert payload["decision"] == "promote"
    assert manager.get(branch_id).status == "promoted"


def test_propose_audit_mode_rolls_back_promoted_status():
    manager = BranchManager(min_support=2)
    pipeline = _build_pipeline(manager)

    pipeline.observe(make_record(identifier="s1", prompt="weak-1"))
    pipeline.observe(make_record(identifier="s2", prompt="weak-2"))
    branch_id = "list format:list constraint"
    pipeline.record_verification(branch_id, "ver-1")

    # In audit mode the operator still gets the proposal, but the branch
    # status is rolled back so they must accept the proposal explicitly.
    results = pipeline.propose_promotions(
        auto_promote=False,
        semantic_independence={branch_id: True},
    )
    assert results[0][0].decision is PromotionDecision.PROMOTE
    assert manager.get(branch_id).status == "verifying"


def test_propose_without_independence_returns_continue():
    manager = BranchManager(min_support=2)
    pipeline = _build_pipeline(manager)

    pipeline.observe(make_record(identifier="s1", prompt="weak-1"))
    pipeline.observe(make_record(identifier="s2", prompt="weak-2"))
    branch_id = "list format:list constraint"
    pipeline.record_verification(branch_id, "ver-1")

    results = pipeline.propose_promotions(
        auto_promote=True,
        # Empty mapping = no branch has been judged independent.
        semantic_independence={},
    )
    assert results[0][0].decision is PromotionDecision.CONTINUE
    assert manager.get(branch_id).status == "verifying"
