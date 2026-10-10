from autodetect_repro.behavior import FailureSignature
from autodetect_repro.branches import BranchManager, PromotionDecision


def signature(name):
    return FailureSignature(name, name, "multi-constraint", name)


def test_single_failure_stays_candidate():
    manager = BranchManager(min_support=2)
    branch = manager.record_support("b1", "constraints", signature("ignored negation"), "s1")
    assert branch.status == "candidate"
    assert manager.propose("b1", semantically_independent=True).decision == PromotionDecision.CONTINUE


def test_verified_independent_branch_can_be_promoted():
    manager = BranchManager(min_support=2)
    manager.record_support("b1", "constraints", signature("ignored negation"), "s1")
    manager.record_support("b1", "constraints", signature("ignored negation"), "s2")
    manager.record_verification("b1", "controlled-position-test")
    proposal = manager.propose("b1", semantically_independent=True)
    assert proposal.decision == PromotionDecision.PROMOTE
    assert proposal.knowledge_point == "constraints"
