from autodetect_repro.controller import BranchStats, RuleController


def test_rule_controller_restarts_stalled_branch_with_bounded_budget():
    controller = RuleController(candidate_cap=8, test_cap=3)
    decision = controller.decide(
        knowledge_point="multi-constraint",
        branch_id="b1",
        stats=BranchStats(tested=8, failures=6, novel_failures=0, stagnation_rounds=3),
        remaining_budget=20,
    )
    assert decision.strategy == "restart"
    assert decision.candidate_budget == 8
    assert decision.test_budget == 3


def test_rule_controller_verifies_when_novel_failures_are_found():
    controller = RuleController(candidate_cap=10, test_cap=5)
    decision = controller.decide(
        knowledge_point="negative-constraints",
        branch_id="b2",
        stats=BranchStats(tested=4, failures=2, novel_failures=2, stagnation_rounds=0),
        remaining_budget=2,
    )
    assert decision.strategy == "verify"
    assert decision.test_budget == 2
