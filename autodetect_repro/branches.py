"""Evidence-backed failure branches and taxonomy promotion proposals."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Tuple

from .behavior import FailureSignature


class PromotionDecision(str, Enum):
    PROMOTE = "promote"
    CONTINUE = "continue_verification"
    REJECT = "reject"


@dataclass(frozen=True)
class BranchState:
    branch_id: str
    knowledge_point: str
    signature: FailureSignature
    supporting_samples: Tuple[str, ...] = ()
    counterexamples: Tuple[str, ...] = ()
    verifications: Tuple[str, ...] = ()
    status: str = "candidate"


@dataclass(frozen=True)
class PromotionProposal:
    branch_id: str
    decision: PromotionDecision
    knowledge_point: str
    reason: str


class BranchManager:
    def __init__(self, *, min_support: int = 2) -> None:
        if min_support <= 0:
            raise ValueError("min_support must be positive")
        self.min_support = min_support
        self._branches: Dict[str, BranchState] = {}

    def record_support(
        self,
        branch_id: str,
        knowledge_point: str,
        signature: FailureSignature,
        sample_id: str,
    ) -> BranchState:
        branch = self._branches.get(branch_id)
        if branch is None:
            branch = BranchState(branch_id, knowledge_point, signature)
        samples = branch.supporting_samples
        if sample_id not in samples:
            samples += (sample_id,)
        status = "verifying" if len(samples) >= self.min_support else "candidate"
        branch = replace(branch, supporting_samples=samples, status=status)
        self._branches[branch_id] = branch
        return branch

    def record_verification(self, branch_id: str, verification_id: str) -> BranchState:
        branch = self._branches[branch_id]
        verifications = branch.verifications
        if verification_id not in verifications:
            verifications += (verification_id,)
        branch = replace(branch, verifications=verifications, status="verifying")
        self._branches[branch_id] = branch
        return branch

    def record_counterexample(
        self, branch_id: str, sample_id: str
    ) -> BranchState:
        """Record a sample that contradicts the branch's claimed failure.

        A counterexample is any sample that is "valid" (no error) and
        for which the branch's failure signature is not exhibited.  When
        at least one counterexample is recorded the branch cannot be
        promoted to the taxonomy.
        """
        branch = self._branches[branch_id]
        counterexamples = branch.counterexamples
        if sample_id not in counterexamples:
            counterexamples += (sample_id,)
        branch = replace(branch, counterexamples=counterexamples)
        self._branches[branch_id] = branch
        return branch

    def propose(self, branch_id: str, *, semantically_independent: bool) -> PromotionProposal:
        branch = self._branches[branch_id]
        enough_support = len(branch.supporting_samples) >= self.min_support
        verified = bool(branch.verifications)
        if enough_support and verified and semantically_independent and not branch.counterexamples:
            decision = PromotionDecision.PROMOTE
            reason = "evidence threshold, verification, and semantic independence all passed"
            self._branches[branch_id] = replace(branch, status="promoted")
        else:
            decision = PromotionDecision.CONTINUE
            reason = "more independent support or controlled verification is required"
        return PromotionProposal(branch_id, decision, branch.knowledge_point, reason)

    def get(self, branch_id: str) -> BranchState:
        return self._branches[branch_id]
