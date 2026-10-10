"""AutoDetect Examiner, Questioner, Evaluator, and Assessor roles."""

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from .client import ChatResult, SiliconFlowClient
from .behavior import BehaviorLabel, FailureSignature
from .controller import SearchDecision
from .config import RunConfig
from .novelty import NoveltyLabel
from .parsing import ParseError, parse_json_object, parse_score
from .prompts import (
    adaptive_prompt,
    adaptive_candidates_prompt,
    behavioral_clustering_prompt,
    controller_prompt,
    assessor_prompt,
    examiner_prompt,
    failure_signature_prompt,
    judge_prompt,
    novelty_classify_prompt,
    reference_prompt,
    repair_prompt,
    seed_prompt,
    promotion_prompt,
)
from .types import (
    AssessorDecision,
    KnowledgePoint,
    Question,
    SampleRecord,
    SubtaskConfig,
)


class RoleError(RuntimeError):
    """Raised when a role cannot produce a valid contracted response."""


class LLMController:
    def __init__(self, config: RunConfig, client: SiliconFlowClient) -> None:
        self.config = config
        self.client = client

    def decide(self, *, knowledge_point, branch_id, stats, remaining_budget):
        def validate(payload):
            strategy = str(payload["strategy"]).lower()
            if strategy not in {"exploit", "explore", "verify", "restart"}:
                raise ValueError("invalid strategy")
            candidate_budget = min(int(payload["candidate_budget"]), remaining_budget)
            test_budget = min(int(payload["test_budget"]), remaining_budget)
            if candidate_budget < 0 or test_budget < 0:
                raise ValueError("invalid budget")
            return SearchDecision(knowledge_point, branch_id, strategy, str(payload["objective"]), str(payload["reason"]), candidate_budget, test_budget)
        return _structured_call(self.client, controller_prompt([knowledge_point], stats, {}, remaining_budget), max_tokens=self.config.limits.max_output_tokens, temperature=0.0, validate=validate)


class FailureSignatureAnalyzer:
    def __init__(self, config: RunConfig, client: SiliconFlowClient) -> None:
        self.config = config
        self.client = client

    def __call__(self, record: SampleRecord) -> FailureSignature:
        def validate(payload):
            if payload.get("is_failure") is not True:
                raise ValueError("not a confirmed failure")
            values = [str(payload.get(key, "")).strip() for key in ("violation", "observed_behavior", "trigger_context", "evidence")]
            if not all(values):
                raise ValueError("incomplete signature")
            return FailureSignature(*values)
        prompt = failure_signature_prompt(record.prompt, record.answer, record.reference_answer, record.comparison)
        return _structured_call(self.client, prompt, max_tokens=self.config.limits.max_output_tokens, temperature=0.0, validate=validate)


class BehavioralClusterer:
    def __init__(self, config: RunConfig, client: SiliconFlowClient) -> None:
        self.config = config
        self.client = client

    def __call__(self, signature, representatives):
        def validate(payload):
            return BehaviorLabel(str(payload["label"]).lower())
        return _structured_call(self.client, behavioral_clustering_prompt(signature.to_dict(), [item.to_dict() for item in representatives]), max_tokens=self.config.limits.max_output_tokens, temperature=0.0, validate=validate)


class PromotionJudge:
    def __init__(self, config: RunConfig, client: SiliconFlowClient) -> None:
        self.config = config
        self.client = client

    def judge(self, branch, evidence):
        def validate(payload):
            decision = str(payload["decision"]).lower()
            if decision not in {"promote", "continue_verification", "reject"}:
                raise ValueError("invalid promotion decision")
            return decision
        return _structured_call(self.client, promotion_prompt(branch, evidence), max_tokens=self.config.limits.max_output_tokens, temperature=0.0, validate=validate)


class LLMNoveltyClassifier:
    """Callable LLM-backed novelty classifier for the :class:`NoveltyFilter`.

    Returns a :class:`NoveltyLabel` for one candidate.  Failures fall back to
    ``UNCERTAIN`` so that the filter keeps the candidate rather than silently
    accepting or rejecting it on an erroneous LLM call.
    """

    def __init__(
        self,
        client: SiliconFlowClient,
        *,
        max_tokens: int,
        knowledge_point: str = "",
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.knowledge_point = knowledge_point

    def __call__(
        self, candidate: str, neighbors: Sequence[str]
    ) -> NoveltyLabel:
        def validate(payload: Dict[str, Any]) -> NoveltyLabel:
            label = str(payload["label"]).strip().lower()
            if label not in {"duplicate", "related", "distinct", "uncertain"}:
                raise ValueError("invalid novelty label")
            return NoveltyLabel(label)

        try:
            return _structured_call(
                self.client,
                novelty_classify_prompt(
                    candidate, neighbors, self.knowledge_point
                ),
                max_tokens=self.max_tokens,
                temperature=0.0,
                validate=validate,
            )
        except RoleError:
            return NoveltyLabel.UNCERTAIN


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _structured_call(
    client: SiliconFlowClient,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    validate: Callable[[Dict[str, Any]], Any],
) -> Any:
    latest = ""
    for attempt in range(2):
        current_prompt = (
            prompt if attempt == 0 else repair_prompt(prompt, latest)
        )
        result = client.complete(
            [{"role": "user", "content": current_prompt}],
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
        )
        latest = result.content
        try:
            return validate(parse_json_object(latest))
        except (ParseError, KeyError, TypeError, ValueError):
            continue
    raise RoleError("role failed to return valid structured output after one repair")


class Examiner:
    def __init__(self, config: RunConfig, client: SiliconFlowClient) -> None:
        self.config = config
        self.client = client

    def select_taxonomy(self) -> Tuple[KnowledgePoint, ...]:
        configured = tuple(
            KnowledgePoint(subtask.group, subtask.name, name)
            for subtask in self.config.subtasks
            for name in subtask.knowledge_points
        )

        def validate(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
            selected = payload["selected_points"]
            if not isinstance(selected, list):
                raise ValueError("selected_points must be a list")
            for item in selected:
                if not isinstance(item, dict):
                    raise ValueError("selected point must be an object")
            return selected

        _structured_call(
            self.client,
            examiner_prompt(self.config.subtasks),
            max_tokens=self.config.limits.max_output_tokens,
            temperature=0.0,
            validate=validate,
        )
        return configured


class Questioner:
    def __init__(self, config: RunConfig, client: SiliconFlowClient) -> None:
        self.config = config
        self.client = client

    def generate_seeds(self, point: KnowledgePoint) -> Tuple[Question, ...]:
        count = self.config.search.seed_questions_per_point

        def validate(payload: Dict[str, Any]) -> Tuple[Question, ...]:
            raw_questions = payload["questions"]
            if not isinstance(raw_questions, list) or len(raw_questions) != count:
                raise ValueError("wrong seed count")
            questions = []
            prompts = set()
            for ordinal, raw in enumerate(raw_questions):
                if not isinstance(raw, dict):
                    raise ValueError("seed must be an object")
                key_point = str(raw.get("key_point", "")).strip()
                prompt_text = str(raw.get("prompt", "")).strip()
                normalized_prompt = _normalized(prompt_text)
                if not key_point or not normalized_prompt or normalized_prompt in prompts:
                    raise ValueError("seed prompts must be distinct and non-empty")
                prompts.add(normalized_prompt)
                questions.append(
                    Question(
                        point.group,
                        point.subtask,
                        point.name,
                        key_point,
                        prompt_text,
                        "seed",
                        0,
                        ordinal,
                    )
                )
            return tuple(questions)

        try:
            return _structured_call(
                self.client,
                seed_prompt(point, count),
                max_tokens=self.config.limits.max_output_tokens,
                temperature=0.7,
                validate=validate,
            )
        except RoleError as exc:
            raise RoleError("could not produce distinct seed questions") from exc

    def generate_adaptive(
        self,
        point: KnowledgePoint,
        history: Sequence[SampleRecord],
        iteration: int,
        *,
        strategy: Optional[str] = None,
        objective: Optional[str] = None,
    ) -> Question:
        previous = {_normalized(item.prompt) for item in history}

        def validate(payload: Dict[str, Any]) -> Question:
            key_point = str(payload.get("key_point", "")).strip()
            prompt_text = str(payload.get("prompt", "")).strip()
            if not key_point or not prompt_text or _normalized(prompt_text) in previous:
                raise ValueError("adaptive question must be new and non-empty")
            return Question(
                point.group,
                point.subtask,
                point.name,
                key_point,
                prompt_text,
                "adaptive",
                iteration,
                0,
            )

        return _structured_call(
            self.client,
            adaptive_prompt(point, history, iteration, strategy=strategy, objective=objective),
            max_tokens=self.config.limits.max_output_tokens,
            temperature=0.7,
            validate=validate,
        )

    def generate_adaptive_candidates(
        self,
        point: KnowledgePoint,
        history: Sequence[SampleRecord],
        iteration: int,
        count: int,
        *,
        strategy: Optional[str] = None,
        objective: Optional[str] = None,
    ) -> Tuple[Question, ...]:
        """Generate ``count`` distinct adaptive questions for the same objective.

        The returned tuple is consumed by :class:`NoveltyFilter` in the runner.
        ``count`` is bounded by the controller's ``candidate_budget`` and is
        always at least 1 in the live runner path.
        """
        if count <= 0:
            raise ValueError("candidate count must be positive")
        previous = {_normalized(item.prompt) for item in history}

        def validate(payload: Dict[str, Any]) -> Tuple[Question, ...]:
            raw_questions = payload.get("questions")
            if not isinstance(raw_questions, list) or len(raw_questions) != count:
                raise ValueError("wrong candidate count")
            questions = []
            seen = set(previous)
            for ordinal, raw in enumerate(raw_questions):
                if not isinstance(raw, dict):
                    raise ValueError("candidate must be an object")
                key_point = str(raw.get("key_point", "")).strip()
                prompt_text = str(raw.get("prompt", "")).strip()
                normalized = _normalized(prompt_text)
                if not key_point or not normalized or normalized in seen:
                    raise ValueError("candidate prompt must be new and non-empty")
                seen.add(normalized)
                questions.append(
                    Question(
                        point.group,
                        point.subtask,
                        point.name,
                        key_point,
                        prompt_text,
                        "adaptive",
                        iteration,
                        ordinal,
                    )
                )
            return tuple(questions)

        try:
            return _structured_call(
                self.client,
                adaptive_candidates_prompt(
                    point, history, iteration, count,
                    strategy=strategy, objective=objective,
                ),
                max_tokens=self.config.limits.max_output_tokens,
                temperature=0.7,
                validate=validate,
            )
        except RoleError as exc:
            raise RoleError(
                "could not produce distinct adaptive candidates"
            ) from exc


class Evaluator:
    def __init__(
        self,
        target_client: SiliconFlowClient,
        agent_client: SiliconFlowClient,
        *,
        max_tokens: int,
    ) -> None:
        self.target_client = target_client
        self.agent_client = agent_client
        self.max_tokens = max_tokens

    @staticmethod
    def _totals(results: Sequence[ChatResult]) -> Tuple[int, int, int, int]:
        return (
            sum(item.request_count for item in results),
            sum(item.usage.prompt_tokens for item in results),
            sum(item.usage.completion_tokens for item in results),
            sum(item.usage.total_tokens for item in results),
        )

    def evaluate(self, question: Question) -> SampleRecord:
        target = self.target_client.complete(
            [{"role": "user", "content": question.prompt}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        reference = self.agent_client.complete(
            [{"role": "user", "content": reference_prompt(question)}],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        judgment = self.agent_client.complete(
            [
                {
                    "role": "user",
                    "content": judge_prompt(
                        question, target.content, reference.content
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        score = parse_score(judgment.content)
        request_count, prompt_tokens, completion_tokens, total_tokens = self._totals(
            (target, reference, judgment)
        )
        return SampleRecord(
            sample_id="",
            group=question.group,
            subtask=question.subtask,
            knowledge_point=question.knowledge_point,
            phase=question.phase,
            iteration=question.iteration,
            ordinal=question.ordinal,
            prompt=question.prompt,
            key_point=question.key_point,
            answer=target.content,
            reference_answer=reference.content,
            comparison=judgment.content,
            score=score,
            is_weakness=score < 4,
            status="valid",
            error=None,
            request_count=request_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class Assessor:
    def __init__(self, config: RunConfig, client: SiliconFlowClient) -> None:
        self.config = config
        self.client = client

    def propose(
        self,
        subtask: SubtaskConfig,
        history: Sequence[SampleRecord],
    ) -> AssessorDecision:
        def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
            if payload.get("stop") is True:
                return payload
            if "task_name" not in payload:
                raise ValueError("missing task_name")
            return payload

        payload = _structured_call(
            self.client,
            assessor_prompt(subtask, history),
            max_tokens=self.config.limits.max_output_tokens,
            temperature=0.0,
            validate=validate,
        )
        explanation = str(payload.get("explanation", "")).strip()
        name = str(payload.get("task_name", "")).strip()
        existing = {_normalized(item) for item in subtask.knowledge_points}
        if payload.get("stop") is True or not name or _normalized(name) in existing:
            return AssessorDecision(False, None, explanation)
        return AssessorDecision(
            True,
            KnowledgePoint(subtask.group, subtask.name, name, True),
            explanation,
        )
