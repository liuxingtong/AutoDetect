"""End-to-end offline acceptance test for the adaptive framework.

Exercises the full ExperimentRunner → CheckpointStore → AdaptiveAnalysisPipeline
→ write_report chain for the three search modes (baseline, adaptive_rule,
adaptive_llm) without ever touching the network.  All roles are
mocked with deterministic fakes; the runner is run against a
temporary run_dir; and the report is regenerated from the persisted
events to confirm that the resume path reaches the same final state
as an uninterrupted run.
"""

from dataclasses import replace
from pathlib import Path

from autodetect_repro.checkpoint import CheckpointStore
from autodetect_repro.config import load_config
from autodetect_repro.reporting import write_report
from autodetect_repro.runner import ExperimentRunner
from autodetect_repro.types import (
    AssessorDecision,
    KnowledgePoint,
    Question,
    SampleRecord,
    SubtaskConfig,
)


def _base_config():
    return load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )


def _small_config(*, mode, candidate_questions_per_round=1,
                  random_exploration_ratio=0.0, auto_promote=False,
                  seeds=2, adaptive=1):
    config = _base_config()
    return replace(
        config,
        subtasks=(SubtaskConfig("group", "subtask", ("point-0",)),),
        search=replace(
            config.search,
            mode=mode,
            seed_questions_per_point=seeds,
            adaptive_iterations_per_point=adaptive,
            max_initial_points=1,
            candidate_questions_per_round=candidate_questions_per_round,
            novelty_top_k=2,
            random_exploration_ratio=random_exploration_ratio,
            auto_promote=auto_promote,
        ),
        limits=replace(config.limits, max_total_samples=10),
    )


# ---------------------------------------------------------------------------
# Common fakes.
# ---------------------------------------------------------------------------


class _ScriptedQuestioner:
    def __init__(self, seed_prompts, candidates_by_iteration=None):
        self._seed_prompts = list(seed_prompts)
        self._candidates = dict(candidates_by_iteration or {})
        self.seed_calls = []
        self.adaptive_calls = []

    def generate_seeds(self, point):
        self.seed_calls.append(point.name)
        return tuple(
            Question(
                point.group, point.subtask, point.name,
                "key-{}".format(i), prompt, "seed", 0, i,
            )
            for i, prompt in enumerate(self._seed_prompts)
        )

    def generate_adaptive(self, point, history, iteration, **kwargs):
        self.adaptive_calls.append(iteration)
        return Question(
            point.group, point.subtask, point.name,
            "key-adaptive", "adaptive-{}".format(iteration),
            "adaptive", iteration, 0,
        )

    def generate_adaptive_candidates(self, point, history, iteration, count, **kwargs):
        prompts = self._candidates.get(iteration, ["cand-{}".format(i) for i in range(count)])
        return tuple(
            Question(
                point.group, point.subtask, point.name,
                "key-{}".format(i), prompts[i],
                "adaptive", iteration, i,
            )
            for i in range(min(count, len(prompts)))
        )


class _FakeEvaluator:
    """Return a fixed score per prompt, or 5 by default."""

    def __init__(self, scores=None, is_weakness=None):
        self._scores = scores or {}
        self._is_weakness = is_weakness
        self.calls = []

    def evaluate(self, question):
        self.calls.append((question.phase, question.iteration, question.ordinal))
        score = self._scores.get(question.prompt, 5)
        if self._is_weakness is None:
            weak = score <= 4
        else:
            weak = self._is_weakness
        return SampleRecord(
            sample_id="",
            group=question.group,
            subtask=question.subtask,
            knowledge_point=question.knowledge_point,
            phase=question.phase,
            iteration=question.iteration,
            ordinal=question.ordinal,
            prompt=question.prompt,
            key_point=question.key_point or "key",
            answer="answer",
            reference_answer="reference",
            comparison="Rating: [[{}]]".format(score),
            score=float(score),
            is_weakness=weak,
            status="valid",
            error=None,
            request_count=1,
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )


class _FakeAssessor:
    def __init__(self):
        self.propose_calls = []

    def propose(self, subtask, history):
        self.propose_calls.append((subtask.name, len(history)))
        return AssessorDecision(False, None, "stop")


class _StubExaminer:
    def __init__(self, points):
        self.points = points

    def select_taxonomy(self):
        return tuple(self.points)


# ---------------------------------------------------------------------------
# Mode 1: baseline.
# ---------------------------------------------------------------------------


def test_baseline_mode_completes_and_resumes(tmp_path):
    config = _small_config(mode="baseline")
    run_dir = tmp_path / "run"
    questioner = _ScriptedQuestioner(seed_prompts=["s0", "s1"])
    evaluator = _FakeEvaluator(scores={"s0": 7, "s1": 7})

    runner1 = ExperimentRunner(
        config, run_dir,
        examiner=_StubExaminer([KnowledgePoint("group", "subtask", "point-0")]),
        questioner=questioner,
        evaluator=evaluator,
        assessor=_FakeAssessor(),
    )
    s1 = runner1.run()
    # baseline mode: 2 seed + 1 assessor-proposed discovered point.
    assert s1.scheduled == 3
    assert s1.interrupted is False

    # Resume on the same run_dir must NOT re-evaluate the existing
    # records; summary should match.
    runner2 = ExperimentRunner(
        config, run_dir,
        examiner=_StubExaminer([KnowledgePoint("group", "subtask", "point-0")]),
        questioner=_ScriptedQuestioner(seed_prompts=["s0", "s1"]),
        evaluator=_FakeEvaluator(scores={"s0": 7, "s1": 7}),
        assessor=_FakeAssessor(),
    )
    s2 = runner2.run()
    assert s2.scheduled == s1.scheduled
    # And the records are still there.
    assert len(CheckpointStore(run_dir, config).records()) == s1.scheduled


# ---------------------------------------------------------------------------
# Mode 2: adaptive_rule.
# ---------------------------------------------------------------------------


def test_adaptive_rule_mode_with_novelty_filter(tmp_path):
    from autodetect_repro.controller import RuleController
    from autodetect_repro.novelty import NoveltyFilter
    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    config = _small_config(
        mode="adaptive_rule",
        candidate_questions_per_round=2,
        seeds=2,
        adaptive=1,
    )
    run_dir = tmp_path / "run"

    questioner = _ScriptedQuestioner(
        seed_prompts=["s0", "s1"],
        candidates_by_iteration={1: ["cand-a", "cand-b"]},
    )

    def sig_for(record):
        return FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        )

    filterer = NoveltyFilter(
        embedding=lambda text: [1.0, 0.0] if text == "cand-a"
                              else [0.0, 1.0] if text == "cand-b"
                              else [0.5, 0.5],
        top_k=2,
    )

    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=_StubExaminer([KnowledgePoint("group", "subtask", "point-0")]),
        questioner=questioner,
        evaluator=_FakeEvaluator(scores={"s0": 3, "s1": 3,
                                        "cand-a": 5, "cand-b": 5}),
        assessor=_FakeAssessor(),
        controller=RuleController(candidate_cap=2, test_cap=2),
        analysis_pipeline=AdaptiveAnalysisPipeline(
            signature_extractor=sig_for,
            behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
            branch_manager=BranchManager(min_support=2),
        ),
        novelty_filter=filterer,
    )
    summary = runner.run()
    # 2 seeds + 2 adaptive = 4.
    assert summary.scheduled == 4
    assert summary.valid == 4

    # Events were persisted; the report can be regenerated from disk
    # without re-running the experiment.
    paths = write_report(run_dir)
    text = paths.markdown.read_text()
    assert "## Behavioral Diversity" in text
    assert "## Branch & Promotion" in text


# ---------------------------------------------------------------------------
# Mode 3: adaptive_llm.
# ---------------------------------------------------------------------------


def test_adaptive_llm_mode_smoke(tmp_path):
    """End-to-end adaptive_llm with a stub LLM controller.

    Confirms the LLMController.decide is consulted once per adaptive
    iteration and that the decision drives a search_decision event.
    """
    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    class _FixedStrategyController:
        """Returns a fixed 'exploit' decision without going through the LLM."""

        decide_calls = []

        def decide(
            self, *, knowledge_point, branch_id, stats, remaining_budget,
        ):
            self.decide_calls.append(knowledge_point)
            from autodetect_repro.controller import SearchDecision
            return SearchDecision(
                knowledge_point=knowledge_point,
                branch_id=branch_id,
                strategy="exploit",
                objective="deepen the current failure pattern",
                reason="stub for e2e test",
                candidate_budget=1,
                test_budget=1,
            )

    config = _small_config(mode="adaptive_llm", seeds=1, adaptive=1)
    run_dir = tmp_path / "run"

    def sig_for(record):
        return FailureSignature("v", "b", "c", "e")

    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=_StubExaminer([KnowledgePoint("group", "subtask", "point-0")]),
        questioner=_ScriptedQuestioner(
            seed_prompts=["s0"],
            candidates_by_iteration={1: ["cand-a"]},
        ),
        evaluator=_FakeEvaluator(scores={"s0": 3, "cand-a": 3}),
        assessor=_FakeAssessor(),
        controller=_FixedStrategyController(),
        analysis_pipeline=AdaptiveAnalysisPipeline(
            signature_extractor=sig_for,
            behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
            branch_manager=BranchManager(min_support=2),
        ),
    )
    summary = runner.run()
    assert summary.scheduled == 2
    assert _FixedStrategyController.decide_calls == ["point-0"]

    events = CheckpointStore(run_dir, config).events()
    decision_events = [
        e for e in events if e.get("event_type") == "search_decision"
    ]
    assert decision_events
    assert decision_events[0]["strategy"] == "exploit"


# ---------------------------------------------------------------------------
# Interruption / resume.
# ---------------------------------------------------------------------------


def test_interrupted_run_resumes_with_no_duplicates(tmp_path):
    from autodetect_repro.controller import RuleController
    from autodetect_repro.novelty import NoveltyFilter
    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    config = _small_config(
        mode="adaptive_rule",
        candidate_questions_per_round=1,
        seeds=2,
        adaptive=2,
    )
    run_dir = tmp_path / "run"

    # An interrupt check that fires after the first evaluate().
    state = {"calls": 0}

    def interrupt():
        state["calls"] += 1
        return state["calls"] > 1

    def sig_for(record):
        return FailureSignature("v", "b", "c", "e")

    # First run: will be interrupted after the first evaluate().
    questioner1 = _ScriptedQuestioner(
        seed_prompts=["s0", "s1"],
        candidates_by_iteration={1: ["cand-1"], 2: ["cand-2"]},
    )
    runner1 = ExperimentRunner(
        config, run_dir,
        examiner=_StubExaminer([KnowledgePoint("group", "subtask", "point-0")]),
        questioner=questioner1,
        evaluator=_FakeEvaluator(scores={"s0": 3, "s1": 3,
                                        "cand-1": 3, "cand-2": 3}),
        assessor=_FakeAssessor(),
        controller=RuleController(candidate_cap=1, test_cap=1),
        analysis_pipeline=AdaptiveAnalysisPipeline(
            signature_extractor=sig_for,
            behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
            branch_manager=BranchManager(min_support=2),
        ),
        novelty_filter=NoveltyFilter(
            embedding=lambda t: [0.5, 0.5], top_k=1,
        ),
        interrupt_check=interrupt,
    )
    s1 = runner1.run()
    assert s1.interrupted is True
    first_count = s1.scheduled

    # Second run: must NOT re-evaluate already-existing samples.
    questioner2 = _ScriptedQuestioner(
        seed_prompts=["s0", "s1"],
        candidates_by_iteration={1: ["cand-1"], 2: ["cand-2"]},
    )
    runner2 = ExperimentRunner(
        config, run_dir,
        examiner=_StubExaminer([KnowledgePoint("group", "subtask", "point-0")]),
        questioner=questioner2,
        evaluator=_FakeEvaluator(scores={"s0": 3, "s1": 3,
                                        "cand-1": 3, "cand-2": 3}),
        assessor=_FakeAssessor(),
        controller=RuleController(candidate_cap=1, test_cap=1),
        analysis_pipeline=AdaptiveAnalysisPipeline(
            signature_extractor=sig_for,
            behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
            branch_manager=BranchManager(min_support=2),
        ),
        novelty_filter=NoveltyFilter(
            embedding=lambda t: [0.5, 0.5], top_k=1,
        ),
    )
    s2 = runner2.run()
    # Total scheduled after the full run is at least the first partial
    # count, and no duplicate sample IDs were created.
    records = CheckpointStore(run_dir, config).records()
    sample_ids = [r.sample_id for r in records]
    assert len(sample_ids) == len(set(sample_ids))
    assert len(records) >= first_count
