"""Capped, resumable AutoDetect experiment scheduling."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .checkpoint import CheckpointError, CheckpointStore, sample_id
from .config import RunConfig
from .controller import BranchStats
from .novelty import NoveltyFilter, NoveltyLabel
from .types import KnowledgePoint, Question, SampleRecord, SubtaskConfig


class RunnerError(RuntimeError):
    """Raised when persisted or generated scheduling state is inconsistent."""


@dataclass(frozen=True)
class RunSummary:
    scheduled: int
    valid: int
    failed: int
    weak: int
    requests: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    interrupted: bool


class ExperimentRunner:
    def __init__(
        self,
        config: RunConfig,
        run_dir: Path,
        *,
        examiner: object,
        questioner: object,
        evaluator: object,
        assessor: object,
        controller: Optional[object] = None,
        analysis_pipeline: Optional[object] = None,
        novelty_filter: Optional[NoveltyFilter] = None,
        novelty_classify: Optional[
            Callable[[str, Sequence[str]], NoveltyLabel]
        ] = None,
        interrupt_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.examiner = examiner
        self.questioner = questioner
        self.evaluator = evaluator
        self.assessor = assessor
        self.controller = controller
        self.analysis_pipeline = analysis_pipeline
        self.novelty_filter = novelty_filter
        # Default classifier rejects exact duplicates only.  Real experiments
        # inject a stricter LLM-backed classifier.
        self.novelty_classify = novelty_classify or (
            lambda candidate, neighbors: NoveltyLabel.DISTINCT
        )
        self.interrupt_check = interrupt_check or (lambda: False)
        self.store: Optional[CheckpointStore] = None
        self._records: List[SampleRecord] = []
        self._by_id: Dict[str, SampleRecord] = {}
        self._halted = False
        self._interrupted = False

    def _can_schedule(self) -> bool:
        if len(self._records) >= self.config.limits.max_total_samples:
            self._halted = True
            return False
        if self.interrupt_check():
            self._halted = True
            self._interrupted = True
            return False
        return True

    def _redacted_error(self, error: BaseException) -> str:
        message = str(error)
        secret = self.config.models.api_key
        if secret:
            message = message.replace(secret, "[REDACTED]")
        return message[:1000]

    @staticmethod
    def _failure_record(identifier: str, question: Question, error: str) -> SampleRecord:
        return SampleRecord(
            sample_id=identifier,
            group=question.group,
            subtask=question.subtask,
            knowledge_point=question.knowledge_point,
            phase=question.phase,
            iteration=question.iteration,
            ordinal=question.ordinal,
            prompt=question.prompt,
            key_point=question.key_point,
            answer="",
            reference_answer="",
            comparison="",
            score=None,
            is_weakness=False,
            status="failed",
            error=error,
            request_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

    def _append(self, record: SampleRecord) -> None:
        if self.store is None:
            raise RunnerError("checkpoint store is not initialized")
        if record.sample_id in self._by_id:
            raise RunnerError("duplicate sample id {}".format(record.sample_id))
        self.store.append(record)
        self._records.append(record)
        self._by_id[record.sample_id] = record

    def _evaluate(self, point: KnowledgePoint, question: Question) -> bool:
        identifier = sample_id(
            point, question.phase, question.iteration, question.ordinal
        )
        if identifier in self._by_id:
            return True
        if not self._can_schedule():
            return False
        try:
            evaluated = self.evaluator.evaluate(question)
            record = replace(
                evaluated,
                sample_id=identifier,
                group=point.group,
                subtask=point.subtask,
                knowledge_point=point.name,
                phase=question.phase,
                iteration=question.iteration,
                ordinal=question.ordinal,
                prompt=question.prompt,
                key_point=question.key_point,
            )
        except Exception as exc:
            record = self._failure_record(
                identifier, question, self._redacted_error(exc)
            )
        self._append(record)
        if self.analysis_pipeline is not None and self.store is not None:
            for event_type, payload in self.analysis_pipeline.observe(record):
                self.store.append_event(event_type, payload)
        return True

    def _record_generation_failure(
        self,
        point: KnowledgePoint,
        phase: str,
        iteration: int,
        ordinal: int,
        error: BaseException,
    ) -> bool:
        identifier = sample_id(point, phase, iteration, ordinal)
        if identifier in self._by_id:
            return True
        if not self._can_schedule():
            return False
        question = Question(
            point.group,
            point.subtask,
            point.name,
            point.name,
            "",
            phase,
            iteration,
            ordinal,
        )
        self._append(
            self._failure_record(identifier, question, self._redacted_error(error))
        )
        return True

    def _point_history(self, point: KnowledgePoint) -> List[SampleRecord]:
        return [
            record
            for record in self._records
            if record.group == point.group
            and record.subtask == point.subtask
            and record.knowledge_point == point.name
        ]

    def _run_point(self, point: KnowledgePoint) -> None:
        seed_count = self.config.search.seed_questions_per_point
        missing_seed_ordinals = [
            ordinal
            for ordinal in range(seed_count)
            if sample_id(point, "seed", 0, ordinal) not in self._by_id
        ]
        if missing_seed_ordinals and not self._halted:
            try:
                questions = self.questioner.generate_seeds(point)
                by_ordinal = {question.ordinal: question for question in questions}
                for ordinal in missing_seed_ordinals:
                    question = by_ordinal.get(ordinal)
                    if question is None:
                        raise RunnerError("questioner omitted seed ordinal {}".format(ordinal))
                    if not self._evaluate(point, question):
                        return
            except Exception as exc:
                for ordinal in missing_seed_ordinals:
                    if not self._record_generation_failure(
                        point, "seed", 0, ordinal, exc
                    ):
                        return

        for iteration in range(
            1, self.config.search.adaptive_iterations_per_point + 1
        ):
            if self._halted:
                return
            identifier = sample_id(point, "adaptive", iteration, 0)
            if identifier in self._by_id:
                continue
            history = self._point_history(point)
            decision = None
            try:
                candidate_budget = 1
                if self.controller is not None:
                    weak_count = sum(1 for item in history if item.status == "valid" and item.is_weakness)
                    recent_iterations = [item for item in history if item.phase == "adaptive" and item.iteration >= max(1, iteration - 2)]
                    recent_weak = sum(1 for item in recent_iterations if item.status == "valid" and item.is_weakness)
                    stats = BranchStats(
                        tested=len(history),
                        failures=weak_count,
                        novel_failures=recent_weak,
                        stagnation_rounds=2 if recent_iterations and recent_weak == 0 else 0,
                    )
                    decision = self.controller.decide(
                        knowledge_point=point.name,
                        branch_id=point.name,
                        stats=stats,
                        remaining_budget=max(0, self.config.limits.max_total_samples - len(self._records)),
                    )
                    candidate_budget = max(1, decision.candidate_budget)
                    self.store.append_event(
                        "search_decision",
                        {
                            "knowledge_point": decision.knowledge_point,
                            "branch_id": decision.branch_id,
                            "strategy": decision.strategy,
                            "objective": decision.objective,
                            "reason": decision.reason,
                            "candidate_budget": decision.candidate_budget,
                            "test_budget": decision.test_budget,
                            "iteration": iteration,
                        },
                    )
                history_prompts = [item.prompt for item in history]
                if self.novelty_filter is not None and candidate_budget > 1:
                    candidates = self.questioner.generate_adaptive_candidates(
                        point, history, iteration, candidate_budget,
                        strategy=decision.strategy if decision is not None else None,
                        objective=decision.objective if decision is not None else None,
                    )
                    filter_result = self.novelty_filter.filter(
                        [c.prompt for c in candidates],
                        history=history_prompts,
                        classify=self.novelty_classify,
                        budget=1,
                    )
                    self.store.append_event(
                        "candidate_filter",
                        {
                            "iteration": iteration,
                            "candidates": [
                                {
                                    "prompt": c.candidate,
                                    "label": c.label.value,
                                    "semantic_novelty": c.semantic_novelty,
                                }
                                for c in (*filter_result.selected, *filter_result.rejected)
                            ],
                        },
                    )
                    if not filter_result.selected:
                        # Filter rejected everything; record a candidate generation
                        # failure so the budget still reflects the attempt.
                        if not self._record_generation_failure(
                            point, "adaptive", iteration, 0,
                            RunnerError("novelty filter rejected all candidates"),
                        ):
                            return
                        continue
                    chosen = next(
                        c for c in candidates
                        if c.prompt == filter_result.selected[0].candidate
                    )
                    question = replace(
                        chosen,
                        group=point.group,
                        subtask=point.subtask,
                        knowledge_point=point.name,
                        phase="adaptive",
                        iteration=iteration,
                        ordinal=0,
                    )
                else:
                    generated = self.questioner.generate_adaptive(
                        point, history, iteration,
                        strategy=decision.strategy if decision is not None else None,
                        objective=decision.objective if decision is not None else None,
                    )
                    question = replace(
                        generated,
                        group=point.group,
                        subtask=point.subtask,
                        knowledge_point=point.name,
                        phase="adaptive",
                        iteration=iteration,
                        ordinal=0,
                    )
                if not self._evaluate(point, question):
                    return
            except Exception as exc:
                if not self._record_generation_failure(
                    point, "adaptive", iteration, 0, exc
                ):
                    return

    def _subtask_history(self, subtask: SubtaskConfig) -> List[SampleRecord]:
        return [
            record
            for record in self._records
            if record.group == subtask.group and record.subtask == subtask.name
        ]

    def _configured_points(self) -> Tuple[KnowledgePoint, ...]:
        return tuple(
            KnowledgePoint(subtask.group, subtask.name, name)
            for subtask in self.config.subtasks
            for name in subtask.knowledge_points
        )

    def _persisted_discovered_point(
        self, subtask: SubtaskConfig
    ) -> Optional[KnowledgePoint]:
        configured = {name.casefold() for name in subtask.knowledge_points}
        for record in self._records:
            if (
                record.group == subtask.group
                and record.subtask == subtask.name
                and record.knowledge_point.casefold() not in configured
            ):
                return KnowledgePoint(
                    subtask.group, subtask.name, record.knowledge_point, True
                )
        return None

    def _load_existing(self) -> None:
        if self.store is None:
            raise RunnerError("checkpoint store is not initialized")
        self._records = self.store.records()
        self._by_id = {}
        for record in self._records:
            if record.sample_id in self._by_id:
                raise CheckpointError(
                    "duplicate sample id in checkpoint: {}".format(record.sample_id)
                )
            self._by_id[record.sample_id] = record

    def _summary(self) -> RunSummary:
        valid = [record for record in self._records if record.status == "valid"]
        return RunSummary(
            scheduled=len(self._records),
            valid=len(valid),
            failed=len(self._records) - len(valid),
            weak=sum(1 for record in valid if record.is_weakness),
            requests=sum(record.request_count for record in self._records),
            prompt_tokens=sum(record.prompt_tokens for record in self._records),
            completion_tokens=sum(
                record.completion_tokens for record in self._records
            ),
            total_tokens=sum(record.total_tokens for record in self._records),
            interrupted=self._interrupted,
        )

    def run(self) -> RunSummary:
        self.store = CheckpointStore(self.run_dir, self.config)
        self._load_existing()
        if len(self._records) >= self.config.limits.max_total_samples:
            return self._summary()
        points: Sequence[KnowledgePoint]
        if self._records:
            points = self._configured_points()
        else:
            points = self.examiner.select_taxonomy()
        point_lookup: Dict[Tuple[str, str], List[KnowledgePoint]] = {}
        for point in points:
            point_lookup.setdefault((point.group, point.subtask), []).append(point)

        for subtask in self.config.subtasks:
            for point in point_lookup.get((subtask.group, subtask.name), []):
                self._run_point(point)
                if self._halted:
                    return self._summary()

            if self._halted:
                return self._summary()
            discovered = self._persisted_discovered_point(subtask)
            if discovered is not None:
                self._run_point(discovered)
            else:
                decision = self.assessor.propose(
                    subtask, self._subtask_history(subtask)
                )
                if decision.accepted and decision.point is not None:
                    self._run_point(decision.point)
                if self._halted:
                    return self._summary()

        return self._summary()
