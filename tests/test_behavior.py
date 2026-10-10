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
