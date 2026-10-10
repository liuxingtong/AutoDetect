"""Wiring tests for the post-Phase-E runner + NoveltyFilter integration."""

from dataclasses import replace
from pathlib import Path

import pytest

from autodetect_repro.checkpoint import CheckpointStore
from autodetect_repro.config import load_config
from autodetect_repro.controller import BranchStats, RuleController
from autodetect_repro.novelty import NoveltyFilter, NoveltyLabel
from autodetect_repro.runner import ExperimentRunner
from autodetect_repro.types import (
    AssessorDecision,
    KnowledgePoint,
    Question,
    SampleRecord,
    SubtaskConfig,
)


def base_config():
    return load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )


def adaptive_config(*, candidate_budget=3, top_k=2):
    config = base_config()
    knowledge_points = ("point-A",)
    return replace(
        config,
        subtasks=(SubtaskConfig("group", "subtask", knowledge_points),),
        search=replace(
            config.search,
            mode="adaptive_rule",
            seed_questions_per_point=1,
            adaptive_iterations_per_point=1,
            max_initial_points=1,
            candidate_questions_per_round=candidate_budget,
            novelty_top_k=top_k,
        ),
        limits=replace(config.limits, max_total_samples=8),
    )


# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------


class FakeExaminer:
    def __init__(self, points):
        self.points = points
        self.calls = 0

    def select_taxonomy(self):
        self.calls += 1
        return tuple(self.points)


class FakeCandidateQuestioner:
    """A questioner that returns fixed candidate pools per adaptive call."""

    def __init__(self, candidates_by_iteration):
        self.candidates_by_iteration = candidates_by_iteration
        self.seed_calls = []
        self.adaptive_calls = []
        self.candidate_calls = []

    def generate_seeds(self, point):
        self.seed_calls.append(point.name)
        return (
            Question(
                point.group,
                point.subtask,
                point.name,
                "seed-key",
                "seed-prompt-{}".format(point.name),
                "seed",
                0,
                0,
            ),
        )

    def generate_adaptive(self, point, history, iteration):  # pragma: no cover
        raise AssertionError("generate_adaptive should not be used when novelty filter is wired")

    def generate_adaptive_candidates(
        self, point, history, iteration, count, **_,
    ):
        self.candidate_calls.append((point.name, iteration, count))
        prompts = self.candidates_by_iteration[point.name, iteration]
        if len(prompts) != count:
            raise AssertionError(
                "candidate count mismatch: asked {} got {}".format(count, len(prompts))
            )
        return tuple(
            Question(
                point.group,
                point.subtask,
                point.name,
                "key-{}".format(index),
                prompt,
                "adaptive",
                iteration,
                index,
            )
            for index, prompt in enumerate(prompts)
        )


def _evaluated_record(question):
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
        answer="answer",
        reference_answer="reference",
        comparison="Rating: [[3]]",
        score=3.0,
        is_weakness=True,
        status="valid",
        error=None,
        request_count=3,
        prompt_tokens=30,
        completion_tokens=15,
        total_tokens=45,
    )


class FakeEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, question):
        self.calls.append(question)
        return _evaluated_record(question)


class FakeAssessor:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    def propose(self, subtask, history):
        self.calls += 1
        return self.decision


def make_runner_with_filter(config, run_dir, *, candidates, filterer, classify):
    point = KnowledgePoint("group", "subtask", "point-A")
    questioner = FakeCandidateQuestioner(candidates)
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=FakeExaminer([point]),
        questioner=questioner,
        evaluator=FakeEvaluator(),
        assessor=FakeAssessor(AssessorDecision(False, None, "stop")),
        controller=RuleController(
            candidate_cap=config.search.candidate_questions_per_round,
            test_cap=config.search.candidate_questions_per_round,
        ),
        novelty_filter=filterer,
        novelty_classify=classify,
    )
    return runner, questioner


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_runner_uses_novelty_filter_to_pick_top_candidate(tmp_path):
    config = adaptive_config(candidate_budget=3, top_k=2)
    candidates = {
        ("point-A", 1): [
            "alpha-prompt",
            "beta-prompt",
            "gamma-prompt",
        ]
    }

    # Mock embedding: alpha ~ beta, gamma is far away.
    vectors = {
        "alpha-prompt": [1.0, 0.0],
        "beta-prompt": [0.95, 0.31],
        "gamma-prompt": [0.0, 1.0],
        "seed-prompt-point-A": [1.0, 0.0],
    }
    filterer = NoveltyFilter(
        embedding=lambda text: vectors[text], top_k=2
    )

    classify_calls = []

    def classify(candidate, neighbors):
        classify_calls.append((candidate, tuple(neighbors)))
        return NoveltyLabel.DISTINCT

    runner, questioner = make_runner_with_filter(
        config, tmp_path / "run", candidates=candidates,
        filterer=filterer, classify=classify,
    )

    summary = runner.run()

    # The novelty filter must have been consulted.
    assert len(classify_calls) == 3
    # The questioner must have produced a candidate batch of size 3.
    assert questioner.candidate_calls == [("point-A", 1, 3)]
    # test_budget = test_cap = 3; all 3 candidates are DISTINCT,
    # so all 3 get evaluated (1 seed + 3 adaptive = 4 total).
    assert summary.scheduled == 4
    assert summary.valid == 4
    record = CheckpointStore(tmp_path / "run", config).records()
    adaptive_records = [r for r in record if r.phase == "adaptive"]
    assert len(adaptive_records) == 3
    # Ordered by novelty score: gamma is furthest from seed, then beta, then alpha.
    assert [r.prompt for r in adaptive_records] == ["gamma-prompt", "beta-prompt", "alpha-prompt"]
    assert [r.ordinal for r in adaptive_records] == [0, 1, 2]


def test_novelty_filter_rejecting_all_candidates_records_failure(tmp_path):
    config = adaptive_config(candidate_budget=2)
    candidates = {("point-A", 1): ["x", "y"]}
    vectors = {"x": [1.0, 0.0], "y": [1.0, 0.0], "seed-prompt-point-A": [0.0, 1.0]}
    filterer = NoveltyFilter(
        embedding=lambda text: vectors[text], top_k=1
    )

    def classify(candidate, neighbors):
        return NoveltyLabel.DUPLICATE

    runner, _ = make_runner_with_filter(
        config, tmp_path / "run", candidates=candidates,
        filterer=filterer, classify=classify,
    )

    summary = runner.run()

    record = CheckpointStore(tmp_path / "run", config).records()
    adaptive_records = [r for r in record if r.phase == "adaptive"]
    assert len(adaptive_records) == 1
    assert adaptive_records[0].status == "failed"
    assert "novelty filter rejected" in (adaptive_records[0].error or "")
    assert summary.valid == 1  # only the seed succeeded
    assert summary.failed == 1


def test_runner_falls_back_to_single_adaptive_when_filter_is_none(tmp_path):
    config = adaptive_config(candidate_budget=3)
    # When no filter is supplied, candidate_budget=3 is overridden by the
    # runner: the controller still emits the decision, but the legacy
    # ``generate_adaptive`` path is taken so that the test contract holds.
    point = KnowledgePoint("group", "subtask", "point-A")

    class HybridQuestioner(FakeCandidateQuestioner):
        def __init__(self):
            super().__init__({("point-A", 1): ["unused"]})
            self.legacy_calls = 0

        def generate_adaptive(self, point, history, iteration, **kwargs):
            self.legacy_calls += 1
            return Question(
                point.group,
                point.subtask,
                point.name,
                "legacy-key",
                "legacy-prompt",
                "adaptive",
                iteration,
                0,
            )

    questioner = HybridQuestioner()
    runner = ExperimentRunner(
        config,
        tmp_path / "run",
        examiner=FakeExaminer([point]),
        questioner=questioner,
        evaluator=FakeEvaluator(),
        assessor=FakeAssessor(AssessorDecision(False, None, "stop")),
        controller=RuleController(
            candidate_cap=config.search.candidate_questions_per_round,
            test_cap=config.search.candidate_questions_per_round,
        ),
        novelty_filter=None,  # explicit
    )

    summary = runner.run()

    assert summary.scheduled == 2
    assert summary.valid == 2
    assert questioner.legacy_calls == 1
    assert questioner.candidate_calls == []  # never invoked


def test_candidate_filter_event_is_appended_to_events_log(tmp_path):
    config = adaptive_config(candidate_budget=2)
    candidates = {("point-A", 1): ["first", "second"]}
    filterer = NoveltyFilter(
        embedding=lambda text: [1.0, 0.0] if text == "first" else [0.0, 1.0],
        top_k=1,
    )

    def classify(candidate, neighbors):
        return NoveltyLabel.DISTINCT

    runner, _ = make_runner_with_filter(
        config, tmp_path / "run", candidates=candidates,
        filterer=filterer, classify=classify,
    )
    runner.run()

    events = CheckpointStore(tmp_path / "run", config).events()
    filter_events = [e for e in events if e.get("event_type") == "candidate_filter"]
    assert len(filter_events) == 1
    payload = filter_events[0]
    assert payload["iteration"] == 1
    prompts = {c["prompt"] for c in payload["candidates"]}
    assert prompts == {"first", "second"}


class StrategyCapturingQuestioner(FakeCandidateQuestioner):
    """Records (strategy, objective) pairs passed to candidate generation."""

    def __init__(self, candidates):
        super().__init__(candidates)
        self.strategies = []

    def generate_adaptive_candidates(
        self, point, history, iteration, count,
        *, strategy=None, objective=None,
    ):
        self.strategies.append((strategy, objective))
        return super().generate_adaptive_candidates(
            point, history, iteration, count,
        )


def test_controller_decision_strategy_flows_to_questioner(tmp_path):
    config = adaptive_config(candidate_budget=2)
    candidates = {("point-A", 1): ["alpha", "beta"]}
    filterer = NoveltyFilter(
        embedding=lambda text: [1.0, 0.0] if text == "alpha" else [0.0, 1.0],
        top_k=1,
    )

    point = KnowledgePoint("group", "subtask", "point-A")
    questioner = StrategyCapturingQuestioner(candidates)
    runner = ExperimentRunner(
        config,
        tmp_path / "run",
        examiner=FakeExaminer([point]),
        questioner=questioner,
        evaluator=FakeEvaluator(),
        assessor=FakeAssessor(AssessorDecision(False, None, "stop")),
        controller=RuleController(
            candidate_cap=config.search.candidate_questions_per_round,
            test_cap=config.search.candidate_questions_per_round,
        ),
        novelty_filter=filterer,
    )

    runner.run()

    # Controller produced a decision, and the questioner observed the same
    # strategy and objective that the controller emitted.
    assert len(questioner.strategies) == 1
    strategy, objective = questioner.strategies[0]
    assert strategy in {"exploit", "explore", "verify", "restart"}
    assert objective  # non-empty string from RuleController

    events = CheckpointStore(tmp_path / "run", config).events()
    decision_events = [e for e in events if e.get("event_type") == "search_decision"]
    assert len(decision_events) == 1
    assert decision_events[0]["strategy"] == strategy
    assert decision_events[0]["objective"] == objective


def test_test_budget_controls_evaluation_count(tmp_path):
    """RuleController.decide() returns test_budget; the runner must evaluate
    that many distinct candidates (up to the filter's budget) before moving on."""
    config = adaptive_config(candidate_budget=3)
    # Three candidates; RuleController caps at test_cap=2.
    candidates = {("point-A", 1): ["alpha", "beta", "gamma"]}
    filterer = NoveltyFilter(
        embedding=lambda text: [1.0, 0.0] if text == "alpha"
                              else [0.0, 1.0] if text == "beta"
                              else [0.5, 0.5],
        top_k=2,
    )
    point = KnowledgePoint("group", "subtask", "point-A")

    class BudgetAwareEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate(self, question):
            self.calls.append(question.prompt)
            return SampleRecord(
                sample_id="",
                group=question.group,
                subtask=question.subtask,
                knowledge_point=question.knowledge_point,
                phase=question.phase,
                iteration=question.iteration,
                ordinal=question.ordinal,
                prompt=question.prompt,
                key_point="key",
                answer="answer",
                reference_answer="reference",
                comparison="Rating: [[7]]",
                score=7.0,
                is_weakness=False,
                status="valid",
                error=None,
                request_count=1,
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )

    evaluator = BudgetAwareEvaluator()
    runner = ExperimentRunner(
        config,
        tmp_path / "run",
        examiner=FakeExaminer([point]),
        questioner=FakeCandidateQuestioner(candidates),
        evaluator=evaluator,
        assessor=FakeAssessor(AssessorDecision(False, None, "stop")),
        controller=RuleController(
            candidate_cap=config.search.candidate_questions_per_round,
            test_cap=config.search.candidate_questions_per_round,
        ),
        novelty_filter=filterer,
    )

    runner.run()

    # RuleController caps test_budget at test_cap=3 (candidate_questions_per_round),
    # so exactly 3 candidates must have been evaluated.
    assert len(evaluator.calls) == 4  # 1 seed + 3 adaptive = 4 total
    adaptive_calls = [c for c in evaluator.calls if c != "seed-prompt-point-A"]
    # Filter sorts by semantic_novelty descending; gamma is most novel.
    assert set(adaptive_calls) == {"alpha", "beta", "gamma"}
    assert adaptive_calls[0] == "gamma"  # most novel first

    # Verify-phase samples for auto_promote are disabled here.
    events = CheckpointStore(tmp_path / "run", config).events()
    filter_events = [e for e in events if e.get("event_type") == "candidate_filter"]
    assert len(filter_events) == 1
    # All 3 candidates appeared in the event.
    candidate_prompts = {c["prompt"] for c in filter_events[0]["candidates"]}
    assert candidate_prompts == {"alpha", "beta", "gamma"}


def test_random_exploration_does_not_change_evaluation_count(tmp_path):
    """random_exploration_ratio shuffles the selected set but does not
    change how many candidates are evaluated."""
    from dataclasses import replace as _dc_replace
    config = _dc_replace(
        adaptive_config(candidate_budget=3),
        search=_dc_replace(
            adaptive_config(candidate_budget=3).search,
            random_exploration_ratio=0.5,
        ),
    )
    candidates = {("point-A", 1): ["alpha", "beta", "gamma"]}
    filterer = NoveltyFilter(
        embedding=lambda text: [1.0, 0.0] if text == "alpha"
                              else [0.0, 1.0] if text == "beta"
                              else [0.5, 0.5],
        top_k=2,
    )
    point = KnowledgePoint("group", "subtask", "point-A")

    class CountingEvaluator:
        def __init__(self):
            self.calls = []

        def evaluate(self, question):
            self.calls.append(question.prompt)
            return SampleRecord(
                sample_id="", group=question.group, subtask=question.subtask,
                knowledge_point=question.knowledge_point, phase=question.phase,
                iteration=question.iteration, ordinal=question.ordinal,
                prompt=question.prompt, key_point="key", answer="a",
                reference_answer="r", comparison="Rating: [[7]]", score=7.0,
                is_weakness=False, status="valid", error=None, request_count=1,
                prompt_tokens=10, completion_tokens=5, total_tokens=15,
            )

    evaluator = CountingEvaluator()
    runner = ExperimentRunner(
        config, tmp_path / "run",
        examiner=FakeExaminer([point]),
        questioner=FakeCandidateQuestioner(candidates),
        evaluator=evaluator,
        assessor=FakeAssessor(AssessorDecision(False, None, "stop")),
        controller=RuleController(
            candidate_cap=config.search.candidate_questions_per_round,
            test_cap=config.search.candidate_questions_per_round,
        ),
        novelty_filter=filterer,
    )
    runner.run()
    # test_budget=3, so 3 adaptive evaluations + 1 seed = 4 total.
    assert len(evaluator.calls) == 4
    adaptive = [c for c in evaluator.calls if c != "seed-prompt-point-A"]
    assert len(adaptive) == 3
    assert set(adaptive) == {"alpha", "beta", "gamma"}
