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
