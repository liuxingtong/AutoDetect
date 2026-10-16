"""Capped, resumable AutoDetect experiment scheduling."""

import json
import os
import random as _random

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
                if (
                    event_type == "taxonomy_category_update"
                    and payload.get("status") == "promoted"
                    and payload.get("category_name")
                ):
                    self._record_promoted_point(
                        point.group,
                        point.subtask,
                        {
                            "knowledge_point": payload["category_name"],
                            "branch_id": payload.get("category_id", ""),
                        },
                    )
            # When a weakness is observed, schedule a controlled
            # verification by re-running the same prompt with a fresh
            # target call (phase="verify").  This provides the
            # ``BranchManager.propose`` evidence required for promotion.
            if (
                record.status == "valid"
                and record.is_weakness
                and self.config.search.auto_promote
            ):
                self._schedule_verification(point, question, record)
            # Independently of verification, re-evaluate the promotion
            # gate after every weakness observation.  Earlier runs only
            # consulted the gate inside _schedule_verification, which
            # meant the second weakness in a branch fired _before_ the
            # branch had accumulated two supporting samples.  The gate
            # could therefore never PROMOTE.  Calling it here, after
            # observe() has added the sample, lets the gate see the
            # full supporting set.
            if (
                record.status == "valid"
                and record.is_weakness
            ):
                self._propose_after_observation(point, record)
        return True

    def _schedule_verification(
        self,
        point: KnowledgePoint,
        question: Question,
        original: SampleRecord,
    ) -> None:
        """Re-evaluate ``question`` to provide verification evidence.

        The verification uses phase="verify" with iteration=-1 and a
        per-sample ordinal.  Negative iteration keeps it out of the
        normal adaptive schedule so a resumed run will not re-run it.
        """
        if not self._can_schedule():
            return
        identifier = sample_id(point, "verify", -1, original.ordinal)
        if identifier in self._by_id:
            return
        try:
            evaluated = self.evaluator.evaluate(question)
            verify_record = replace(
                evaluated,
                sample_id=identifier,
                group=point.group,
                subtask=point.subtask,
                knowledge_point=point.name,
                phase="verify",
                iteration=-1,
                ordinal=original.ordinal,
                prompt=question.prompt,
                key_point=question.key_point,
            )
        except Exception as exc:
            return
        self._append(verify_record)
        if self.analysis_pipeline is not None and self.store is not None:
            branch_id = "{}:{}".format(
                point.name, original.key_point or "violation"
            )
            # If the verification failed (or was not weak), treat the
            # original as a counterexample.  Otherwise record it as
            # verification evidence.  The promotion gate is consulted
            # by ``_propose_after_observation`` after each weakness
            # observation, so we no longer need a second call here.
            if verify_record.status != "valid" or not verify_record.is_weakness:
                for event_type, payload in self.analysis_pipeline.record_counterexample(
                    branch_id, verify_record.sample_id
                ):
                    self.store.append_event(event_type, payload)
            else:
                for event_type, payload in self.analysis_pipeline.record_verification(
                    branch_id, verify_record.sample_id
                ):
                    self.store.append_event(event_type, payload)
            # Re-evaluate the gate after verification so a branch that
            # was waiting for a verification to PROMOTE can do so
            # without waiting for the next weakness observation.
            if (
                self.store is not None
                and self.config.search.auto_promote
            ):
                independence = self._branch_independence_map()
                for _proposal, (event_type, payload) in self.analysis_pipeline.propose_promotions(
                    auto_promote=self.config.search.auto_promote,
                    semantic_independence=independence,
                ):
                    self.store.append_event(event_type, payload)
                    if payload.get("decision") == "promote":
                        self._record_promoted_point(
                            point.group, point.subtask, payload
                        )

    def _propose_after_observation(
        self, point: KnowledgePoint, record: SampleRecord
    ) -> None:
        """Consult the promotion gate after a weakness observation.

        Called from ``_evaluate`` so the gate sees the latest
        supporting_samples set, not just the post-verification set
        (which the original ``_schedule_verification``-only call
        pattern could not).  No-op when auto_promote is off or the
        analysis pipeline is not configured.
        """
        if not self.config.search.auto_promote:
            return
        if self.analysis_pipeline is None or self.store is None:
            return
        independence = self._branch_independence_map()
        for _proposal, (event_type, payload) in self.analysis_pipeline.propose_promotions(
            auto_promote=self.config.search.auto_promote,
            semantic_independence=independence,
        ):
            self.store.append_event(event_type, payload)
            if payload.get("decision") == "promote":
                self._record_promoted_point(
                    point.group, point.subtask, payload
                )

    def _branch_independence_map(self) -> Dict[str, bool]:
        """Return ``{branch_id: semantically_independent}``.

        A branch is independent when its supporting samples come from at
        least two distinct ``(phase, iteration)`` tuples among the
        non-verification observations.  Verification (phase="verify",
        iter=-1) is intentionally excluded so the check measures
        "different material observations", not "observation + its
        re-run".  Branches with fewer than two distinct contexts
        across their supporting samples are not independent.
        """
        if self.analysis_pipeline is None:
            return {}
        if self.store is None:
            return {}
        ids_by_branch: Dict[str, Tuple[str, int]] = {}
        for record in self._records:
            if record.phase == "verify":
                continue
            ids_by_branch[record.sample_id] = (record.phase, record.iteration)
        branch_manager = self.analysis_pipeline.branch_manager
        result: Dict[str, bool] = {}
        for branch_id, branch in branch_manager._branches.items():
            contexts = {
                ids_by_branch[sample_id]
                for sample_id in branch.supporting_samples
                if sample_id in ids_by_branch
            }
            contexts.discard(None)
            result[branch_id] = len(contexts) >= 2
        return result

    def _record_promoted_point(
        self, group: str, subtask: str, payload: Dict[str, object]
    ) -> None:
        """Persist a promoted branch to ``promoted.jsonl`` for resume.

        Written as one JSON object per line so a resumed run can append
        without rewriting.  The file is human-readable and does not
        require any matching ``samples.jsonl`` rows to exist.
        """
        if self.store is None:
            return
        promoted_path = self.run_dir / "promoted.jsonl"
        knowledge_point = str(payload.get("knowledge_point", "")).strip()
        branch_id = str(payload.get("branch_id", "")).strip()
        if not knowledge_point or not branch_id:
            return
        line = {
            "group": group,
            "subtask": subtask,
            "knowledge_point": knowledge_point,
            "branch_id": branch_id,
        }
        promoted_path.touch(exist_ok=True)
        with promoted_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

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
                    # The controller's branch_id should be an actual
                    # branch tracked by the analysis pipeline (the
                    # format is "<point>:<key_point>"), not the bare
                    # point name.  Pick the most recent weakness
                    # record's branch, falling back to the bare name
                    # when no weakness has been observed yet (so the
                    # first adaptive iteration still emits a valid
                    # audit row even before any failure).
                    branch_id = self._most_recent_branch_id(point) or point.name
                    decision = self.controller.decide(
                        knowledge_point=point.name,
                        branch_id=branch_id,
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
                # test_budget bounds how many distinct candidates the controller
                # wants to see evaluated before making the next decision.
                test_budget = (
                    max(1, decision.test_budget) if decision is not None else 1
                )
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
                        budget=test_budget,
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
                    # Random exploration: when the controller's strategy is "explore"
                    # and the exploration ratio is non-zero, randomly replace up to
                    # ratio% of the selected candidates with unselected-but-passing
                    # candidates.  This prevents the search from over-exploiting the
                    # novelty-ranked ordering.
                    selected_list = list(filter_result.selected)
                    if (
                        decision is not None
                        and decision.strategy == "explore"
                        and self.config.search.random_exploration_ratio > 0.0
                    ):
                        _random.seed(
                            self.config.search.random_seed + iteration
                        )
                        n_replace = min(
                            len(selected_list),
                            max(
                                1,
                                int(
                                    len(selected_list)
                                    * self.config.search.random_exploration_ratio
                                ),
                            ),
                        )
                        # Pull candidates from rejected pool that were DISTINCT.
                        rejected_distinct = [
                            c for c in filter_result.rejected
                            if c.label.value == "distinct"
                        ]
                        replacements = _random.sample(
                            rejected_distinct, min(n_replace, len(rejected_distinct))
                        )
                        # Shuffle the selected list and replace the first n_replace.
                        _random.shuffle(selected_list)
                        for i, repl in enumerate(replacements[:n_replace]):
                            selected_list[i] = repl
                        _random.seed()  # restore global state
                    # Evaluate up to test_budget distinct candidates, each with
                    # its own stable ordinal so resume is safe.
                    for ordinal, selected in enumerate(selected_list):
                        chosen = next(
                            c for c in candidates
                            if c.prompt == selected.candidate
                        )
                        question = replace(
                            chosen,
                            group=point.group,
                            subtask=point.subtask,
                            knowledge_point=point.name,
                            phase="adaptive",
                            iteration=iteration,
                            ordinal=ordinal,
                        )
                        if not self._evaluate(point, question):
                            return
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

    def _most_recent_branch_id(
        self, point: KnowledgePoint
    ) -> Optional[str]:
        """Return the real branch_id of the most recent weakness for ``point``.

        Branches are stored in the analysis pipeline as
        ``"{point.name}:{key_point}"``; the controller's decision
        needs this exact form so downstream consumers (e.g. the
        audit log) can correlate with branch_update events.  Returns
        ``None`` when no weakness has been observed for the point
        yet, in which case the caller should fall back to the bare
        point name.
        """
        point_history = self._point_history(point)
        for record in reversed(point_history):
            if (
                record.status == "valid"
                and record.is_weakness
            ):
                return "{}:{}".format(
                    point.name, record.key_point or "violation"
                )
        return None

    def _configured_points(self) -> Tuple[KnowledgePoint, ...]:
        configured = tuple(
            KnowledgePoint(subtask.group, subtask.name, name)
            for subtask in self.config.subtasks
            for name in subtask.knowledge_points
        )
        return configured + self._promoted_points()

    def _promoted_points(self) -> Tuple[KnowledgePoint, ...]:
        """Read ``promoted.jsonl`` and return one point per unique entry.

        A resumed run appends these to the configured taxonomy so a
        previously-promoted knowledge point continues to receive seed
        and adaptive rounds without re-running the discovery phase.
        Entries that already exist in the configured taxonomy are
        de-duplicated by casefolded name to avoid double-scheduling.
        """
        path = self.run_dir / "promoted.jsonl"
        if not path.exists():
            return ()
        configured_names: set = set()
        for subtask in self.config.subtasks:
            for name in subtask.knowledge_points:
                configured_names.add(name.casefold())
        out: List[KnowledgePoint] = []
        seen: set = set()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            group = str(entry.get("group", "")).strip()
            subtask = str(entry.get("subtask", "")).strip()
            name = str(entry.get("knowledge_point", "")).strip()
            if not group or not subtask or not name:
                continue
            if name.casefold() in configured_names:
                continue
            if name.casefold() in seen:
                continue
            seen.add(name.casefold())
            out.append(KnowledgePoint(group, subtask, name, True))
        return tuple(out)

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
        # Replay the analysis pipeline so behavioral clusters, branch state,
        # and the target-call counter match what the live run had at the
        # same checkpoint.  Without this, a resumed run starts from a clean
        # slate and would issue different search decisions.
        if self.analysis_pipeline is not None:
            self.analysis_pipeline.replay(self.store.events())

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
