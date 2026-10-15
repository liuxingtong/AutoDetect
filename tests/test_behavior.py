from autodetect_repro.behavior import (
    BehaviorLabel,
    BehavioralAnalyzer,
    FailureSignature,
)


def test_failure_signature_contains_observation_not_causal_claim():
    signature = FailureSignature(
        violation="ignored negative constraint",
        observed_behavior="output contained a digit",
        trigger_context="three constraints were active",
        evidence="answer contains 2026",
    )
    assert signature.to_dict()["trigger_context"] == "three constraints were active"


def test_analyzer_adds_only_confirmed_different_clusters():
    analyzer = BehavioralAnalyzer()
    first = FailureSignature("v1", "b1", "c1", "e1")
    second = FailureSignature("v2", "b2", "c2", "e2")
    analyzer.add(first, lambda signature, representatives: BehaviorLabel.DIFFERENT)
    analyzer.add(second, lambda signature, representatives: BehaviorLabel.UNCERTAIN)
    result = analyzer.metrics(target_calls=2)
    assert result.richness == 1
    assert result.gain == 1
    assert result.discovery_rate == 0.5


def test_first_signature_seeds_cluster_without_calling_classifier():
    analyzer = BehavioralAnalyzer()
    first = FailureSignature("v1", "b1", "c1", "e1")
    calls = []

    def classify(signature, representatives):
        calls.append((signature, representatives))
        return BehaviorLabel.UNCERTAIN

    label = analyzer.add(first, classify)

    assert label is BehaviorLabel.DIFFERENT
    assert len(analyzer.clusters) == 1
    assert analyzer.clusters[0].cluster_id == "behavior-0001"
    assert analyzer.clusters[0].representative == first
    assert calls == []


def test_uncertain_after_seed_does_not_recreate_cold_start():
    analyzer = BehavioralAnalyzer()
    first = FailureSignature("v1", "b1", "c1", "e1")
    second = FailureSignature("v2", "b2", "c2", "e2")
    seen_representatives = []

    def classify(signature, representatives):
        seen_representatives.append(representatives)
        return BehaviorLabel.UNCERTAIN

    analyzer.add(first, classify)
    label = analyzer.add(second, classify)

    assert label is BehaviorLabel.UNCERTAIN
    assert len(analyzer.clusters) == 1
    assert seen_representatives == [(analyzer.clusters[0].representative,)]
