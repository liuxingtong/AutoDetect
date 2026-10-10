from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
from autodetect_repro.behavior import BehaviorLabel, FailureSignature
from autodetect_repro.branches import BranchManager
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
