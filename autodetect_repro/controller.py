"""Deterministic baseline search controller."""

from dataclasses import dataclass
from typing import Literal


Strategy = Literal["exploit", "explore", "verify", "restart"]


@dataclass(frozen=True)
class BranchStats:
    tested: int
    failures: int
    novel_failures: int
    stagnation_rounds: int


@dataclass(frozen=True)
class SearchDecision:
    knowledge_point: str
    branch_id: str
    strategy: Strategy
    objective: str
    reason: str
    candidate_budget: int
    test_budget: int


class RuleController:
    def __init__(self, *, candidate_cap: int, test_cap: int) -> None:
        if candidate_cap <= 0 or test_cap <= 0:
            raise ValueError("controller caps must be positive")
        self.candidate_cap = candidate_cap
        self.test_cap = test_cap

    def decide(
        self,
        *,
        knowledge_point: str,
        branch_id: str,
        stats: BranchStats,
        remaining_budget: int,
    ) -> SearchDecision:
        if remaining_budget < 0:
            raise ValueError("remaining budget cannot be negative")
        test_budget = min(self.test_cap, remaining_budget)
        candidate_budget = min(self.candidate_cap, max(remaining_budget, 1))
        if stats.stagnation_rounds >= 3:
            strategy: Strategy = "restart"
            reason = "branch has stagnated without confirmed novel failures"
            objective = "sample a new direction with reduced dependence on branch history"
        elif stats.novel_failures > 0:
            strategy = "verify"
            reason = "recent failures include behavior not yet sufficiently verified"
            objective = "run controlled comparisons for the observed failure behavior"
        elif stats.failures > 0:
            strategy = "exploit"
            reason = "existing failures provide a useful local search signal"
            objective = "deepen the current failure pattern while changing one condition"
        else:
            strategy = "explore"
            reason = "no confirmed failure signal is available for exploitation"
            objective = "explore a materially different task context or constraint combination"
        return SearchDecision(
            knowledge_point=knowledge_point,
            branch_id=branch_id,
            strategy=strategy,
            objective=objective,
            reason=reason,
            candidate_budget=candidate_budget,
            test_budget=test_budget,
        )
