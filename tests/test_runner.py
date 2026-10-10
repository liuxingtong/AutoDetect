from dataclasses import replace
import json
from pathlib import Path

import pytest

from autodetect_repro.checkpoint import CheckpointError, CheckpointStore, sample_id
from autodetect_repro.config import load_config
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


def small_config(*, points=1, seeds=2, adaptive=2, cap=80):
    config = base_config()
    knowledge_points = tuple("point-{}".format(index) for index in range(points))
    return replace(
        config,
        subtasks=(SubtaskConfig("group", "subtask", knowledge_points),),
        search=replace(
            config.search,
            seed_questions_per_point=seeds,
            adaptive_iterations_per_point=adaptive,
            max_initial_points=points,
        ),
        limits=replace(config.limits, max_total_samples=cap),
    )


class FakeExaminer:
    def __init__(self, points, log):
        self.points = points
        self.log = log
        self.calls = 0

    def select_taxonomy(self):
        self.calls += 1
        self.log.append("examiner")
        return tuple(self.points)


class FakeQuestioner:
    def __init__(self, config, log, shared_key_point=False):
        self.config = config
        self.log = log
        self.seed_calls = []
        self.adaptive_calls = []
        self.shared_key_point = shared_key_point

    def generate_seeds(self, point):
        self.seed_calls.append(point.name)
        self.log.append("generate-seeds:{}".format(point.name))
        return tuple(
            Question(
                point.group,
                point.subtask,
                point.name,
                point.name if self.shared_key_point else "seed-key-{}".format(ordinal),
                "seed-prompt-{}-{}".format(point.name, ordinal),
                "seed",
                0,
                ordinal,
            )
            for ordinal in range(self.config.search.seed_questions_per_point)
        )

    def generate_adaptive(self, point, history, iteration, **kwargs):
        self.adaptive_calls.append((point.name, iteration, len(history)))
        self.log.append("generate-adaptive:{}:{}".format(point.name, iteration))
        return Question(
            point.group,
            point.subtask,
            point.name,
            point.name if self.shared_key_point else "adaptive-key",
            "adaptive-prompt-{}-{}".format(point.name, iteration),
            "adaptive",
            iteration,
            0,
        )


def evaluated_record(question):
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
    def __init__(self, log, fail_prompts=None):
        self.log = log
        self.fail_prompts = set(fail_prompts or ())
        self.calls = []

    def evaluate(self, question):
        self.calls.append(question)
        self.log.append(
            "evaluate:{}:{}:{}".format(
                question.phase, question.iteration, question.ordinal
            )
        )
        if question.prompt in self.fail_prompts:
            raise RuntimeError("provider failed")
        return evaluated_record(question)


class FakeAssessor:
    def __init__(self, decision, log):
        self.decision = decision
        self.log = log
        self.calls = 0

    def propose(self, subtask, history):
        self.calls += 1
        self.log.append("assessor:{}".format(subtask.name))
        return self.decision


def make_runner(config, run_dir, *, decision=None, evaluator=None, interrupt=None):
    log = []
    points = [
        KnowledgePoint(subtask.group, subtask.name, name)
        for subtask in config.subtasks
        for name in subtask.knowledge_points
    ]
    questioner = FakeQuestioner(config, log)
    evaluator = evaluator or FakeEvaluator(log)
    assessor = FakeAssessor(
        decision or AssessorDecision(False, None, "stop"), log
    )
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=FakeExaminer(points, log),
        questioner=questioner,
        evaluator=evaluator,
        assessor=assessor,
        interrupt_check=interrupt,
    )
    return runner, log, questioner, evaluator, assessor


def test_runner_evaluates_all_seeds_before_adaptive_iterations(tmp_path):
    config = small_config(seeds=2, adaptive=2)
    runner, log, _, _, _ = make_runner(config, tmp_path / "run")

    summary = runner.run()

    assert log[:6] == [
        "examiner",
        "generate-seeds:point-0",
        "evaluate:seed:0:0",
        "evaluate:seed:0:1",
        "generate-adaptive:point-0:1",
        "evaluate:adaptive:1:0",
    ]
    assert log[6] == "generate-adaptive:point-0:2"
    assert summary.scheduled == 4
    assert summary.valid == 4


def test_resume_skips_completed_sample_id_without_evaluating_it_again(tmp_path):
    config = small_config(seeds=2, adaptive=1)
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config)
    point = KnowledgePoint("group", "subtask", "point-0")
    completed_question = Question(
        "group", "subtask", "point-0", "key", "completed prompt", "seed", 0, 0
    )
    store.append(
        replace(
            evaluated_record(completed_question),
            sample_id=sample_id(point, "seed", 0, 0),
        )
    )
    runner, _, _, evaluator, _ = make_runner(config, run_dir)

    summary = runner.run()

    assert all(item.ordinal != 0 or item.phase != "seed" for item in evaluator.calls)
    assert summary.scheduled == 3


def test_resume_reuses_persisted_taxonomy_without_examiner_call(tmp_path):
    config = small_config(seeds=1, adaptive=1)
    run_dir = tmp_path / "run"
    first, _, _, _, _ = make_runner(config, run_dir)
    first.run()
    resumed, _, _, evaluator, assessor = make_runner(config, run_dir)

    summary = resumed.run()

    assert resumed.examiner.calls == 0
    assert evaluator.calls == []
    assert assessor.calls == 1
    assert summary.scheduled == 2


def test_resume_reuses_discovered_point_without_assessor_call(tmp_path):
    config = small_config(seeds=1, adaptive=1)
    run_dir = tmp_path / "run"
    discovered = KnowledgePoint("group", "subtask", "new-point", True)
    decision = AssessorDecision(True, discovered, "new failure mode")
    first, _, _, _, _ = make_runner(config, run_dir, decision=decision)
    first.run()
    resumed, _, _, evaluator, assessor = make_runner(
        config, run_dir, decision=decision
    )

    summary = resumed.run()

    assert resumed.examiner.calls == 0
    assert evaluator.calls == []
    assert assessor.calls == 0
    assert summary.scheduled == 4


def test_resume_at_cap_returns_without_any_role_call(tmp_path):
    config = small_config(seeds=1, adaptive=0, cap=1)
    run_dir = tmp_path / "run"
    first, _, _, _, _ = make_runner(config, run_dir)
    first.run()
    resumed, log, _, evaluator, assessor = make_runner(config, run_dir)

    summary = resumed.run()

    assert log == []
    assert resumed.examiner.calls == 0
    assert evaluator.calls == []
    assert assessor.calls == 0
    assert summary.scheduled == 1


def test_manifest_mismatch_stops_before_any_role_call(tmp_path):
    config = small_config()
    run_dir = tmp_path / "run"
    CheckpointStore(run_dir, config)
    altered = replace(config, search=replace(config.search, random_seed=77))
    runner, log, _, _, _ = make_runner(altered, run_dir)

    with pytest.raises(CheckpointError, match="fingerprint"):
        runner.run()

    assert log == []


def test_runner_stops_exactly_at_total_sample_cap(tmp_path):
    config = small_config(points=20, seeds=2, adaptive=3, cap=80)
    runner, _, _, evaluator, _ = make_runner(config, tmp_path / "run")

    summary = runner.run()

    assert len(evaluator.calls) == 80
    assert summary.scheduled == 80


def test_runner_accepts_at_most_one_discovered_point_per_subtask(tmp_path):
    config = small_config(points=1, seeds=1, adaptive=1)
    discovered = KnowledgePoint("group", "subtask", "new-point", True)
    decision = AssessorDecision(True, discovered, "new failure mode")
    runner, _, questioner, evaluator, assessor = make_runner(
        config, tmp_path / "run", decision=decision
    )

    summary = runner.run()

    assert assessor.calls == 1
    assert questioner.seed_calls.count("new-point") == 1
    assert len([q for q in evaluator.calls if q.knowledge_point == "new-point"]) == 2
    assert summary.scheduled == 4


def test_failed_evaluation_is_checkpointed_and_counted(tmp_path):
    config = small_config(points=1, seeds=1, adaptive=1, cap=1)
    log = []
    evaluator = FakeEvaluator(log, {"seed-prompt-point-0-0"})
    runner, _, _, _, _ = make_runner(
        config, tmp_path / "run", evaluator=evaluator
    )

    summary = runner.run()
    records = CheckpointStore(tmp_path / "run", config).records()

    assert summary.scheduled == 1
    assert summary.valid == 0
    assert summary.failed == 1
    assert records[0].status == "failed"
    assert records[0].score is None
    assert records[0].error == "provider failed"


def test_injected_interrupt_stops_cleanly_after_checkpoint(tmp_path):
    config = small_config(points=1, seeds=2, adaptive=2)

    class InterruptAfterFirst:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return self.calls > 1

    runner, _, _, evaluator, _ = make_runner(
        config, tmp_path / "run", interrupt=InterruptAfterFirst()
    )

    summary = runner.run()

    assert summary.interrupted is True
    assert summary.scheduled == 1
    assert len(evaluator.calls) == 1
    assert len(CheckpointStore(tmp_path / "run", config).records()) == 1


class _StubAnalysisPipeline:
    """Records every observe() and replay() call so the resume test can
    assert the runner replays events at startup.  Delegates to a real
    AdaptiveAnalysisPipeline so replay() actually rebuilds state."""

    def __init__(self):
        from autodetect_repro.analysis_pipeline import (
            AdaptiveAnalysisPipeline,
        )
        from autodetect_repro.branches import BranchManager
        self._delegate = AdaptiveAnalysisPipeline(
            signature_extractor=lambda record: type("S", (), {})(),
            behavior_classifier=lambda s, r: "different",
            branch_manager=BranchManager(min_support=2),
        )
        self.observe_calls = []
        self.replay_calls = 0

    def observe(self, record):
        self.observe_calls.append(record)
        return self._delegate.observe(record)

    def replay(self, events):
        self.replay_calls += 1
        return self._delegate.replay(events)

    @property
    def target_calls(self):
        return self._delegate.target_calls

    @property
    def branch_manager(self):
        return self._delegate.branch_manager

    @property
    def behavior_analyzer(self):
        return self._delegate.behavior_analyzer


def test_resume_replays_analysis_pipeline_state_from_events(tmp_path):
    config = small_config(points=1, seeds=1, adaptive=0)
    run_dir = tmp_path / "run"

    # First, complete the seed sample so checkpoint has a record.
    first, _, _, _, _ = make_runner(config, run_dir)
    first.run()

    # Manually write analysis events that a live run would have produced.
    store = CheckpointStore(run_dir, config)
    store.append_event(
        "failure_analysis",
        {
            "sample_id": "manual-1",
            "signature": {
                "violation": "v",
                "observed_behavior": "b",
                "trigger_context": "c",
                "evidence": "e",
            },
        },
    )
    store.append_event(
        "branch_update",
        {
            "branch_id": "point-0:v",
            "status": "verifying",
            "supporting_samples": ["manual-1"],
        },
    )

    # A brand-new pipeline in a brand-new runner process simulates resume.
    fresh_pipeline = _StubAnalysisPipeline()
    resumed, _, _, _, _ = make_runner(config, run_dir)
    # Inject the stub pipeline into the resumed runner.
    resumed.analysis_pipeline = fresh_pipeline
    resumed.run()

    # Replay must have been called exactly once during _load_existing().
    assert fresh_pipeline.replay_calls == 1
    # The replayed failure_analysis event contributed one target_call.
    assert fresh_pipeline.target_calls == 1
    # The branch snapshot was restored.
    branch = fresh_pipeline.branch_manager.get("point-0:v")
    assert branch is not None
    assert branch.supporting_samples == ("manual-1",)
    assert branch.status == "verifying"


def test_runner_schedules_verification_and_promotes_branch_on_reproducible_failure(tmp_path):
    """Phase F: weakness → re-run → promotion_proposal event.

    Two seeds of the same knowledge point both score 3 (a weakness).
    The pipeline should observe both, schedule a verification run for
    the second one, and emit a ``promotion_proposal`` event.  Because
    ``auto_promote`` is on the branch status flips to "promoted".
    """
    config = small_config(points=1, seeds=2, adaptive=0)
    config = replace(config, search=replace(config.search, auto_promote=True))
    run_dir = tmp_path / "run"

    # Reuse the real AdaptiveAnalysisPipeline (not the stub) so we
    # exercise the full Phase F path.
    from autodetect_repro.analysis_pipeline import (
        AdaptiveAnalysisPipeline,
    )
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    pipeline = AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        ),
        behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
        branch_manager=BranchManager(min_support=2),
    )
    questioner = FakeQuestioner(config, [])
    evaluator = FakeEvaluator([])
    log = []
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=FakeExaminer(
            [KnowledgePoint("group", "subtask", "point-0")], log
        ),
        questioner=questioner,
        evaluator=evaluator,
        assessor=FakeAssessor(
            AssessorDecision(False, None, "stop"), []
        ),
        analysis_pipeline=pipeline,
    )
    runner.run()

    events = CheckpointStore(run_dir, config).events()
    promotion_events = [
        e for e in events if e.get("event_type") == "promotion_proposal"
    ]
    # At least one promotion_proposal event must be emitted for the
    # branch that has 2 supporting samples + a verification.
    assert promotion_events, "expected a promotion_proposal event"
    # Find the proposal for the auto_promote branch.
    promote = [
        e for e in promotion_events
        if e.get("decision") == "promote" or e.get("decision") == "continue_verification"
    ]
    assert promote
    # The verify-phase sample is recorded so the budget accounts for it.
    verify_records = [
        r for r in CheckpointStore(run_dir, config).records()
        if r.phase == "verify"
    ]
    assert verify_records, "expected at least one verify-phase sample"


# ---------------------------------------------------------------------------
# Independence map + promoted.jsonl writeback (Phase G wire-up).
# ---------------------------------------------------------------------------


def test_two_distinct_phases_yield_promote_decision(tmp_path):
    """PROMOTE requires semantically independent supporting samples.

    With seeds=1 and adaptive=1, the two weakness samples live in two
    different ``(phase, iteration)`` tuples, so independence is True
    and the deterministic gate returns PROMOTE.
    """
    config = small_config(points=1, seeds=1, adaptive=1)
    config = replace(config, search=replace(config.search, auto_promote=True))
    run_dir = tmp_path / "run"

    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    pipeline = AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        ),
        behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
        branch_manager=BranchManager(min_support=2),
    )
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=FakeExaminer(
            [KnowledgePoint("group", "subtask", "point-0")], []
        ),
        questioner=FakeQuestioner(config, [], shared_key_point=True),
        evaluator=FakeEvaluator([]),
        assessor=FakeAssessor(AssessorDecision(False, None, "stop"), []),
        analysis_pipeline=pipeline,
    )
    runner.run()

    events = CheckpointStore(run_dir, config).events()
    promotion_events = [
        e for e in events if e.get("event_type") == "promotion_proposal"
    ]
    assert promotion_events, "expected a promotion_proposal event"
    promote = [
        e for e in promotion_events if e.get("decision") == "promote"
    ]
    assert promote, "expected decision=promote when phases are distinct"


def test_single_phase_yields_continue_decision(tmp_path):
    """One (phase, iteration) tuple ⇒ independence False ⇒ CONTINUE.

    With seeds=2 and adaptive=0, both weakness samples share the same
    ``(seed, 0)`` context, so independence is False and the gate
    returns CONTINUE regardless of auto_promote.
    """
    config = small_config(points=1, seeds=2, adaptive=0)
    config = replace(config, search=replace(config.search, auto_promote=True))
    run_dir = tmp_path / "run"

    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    pipeline = AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        ),
        behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
        branch_manager=BranchManager(min_support=2),
    )
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=FakeExaminer(
            [KnowledgePoint("group", "subtask", "point-0")], []
        ),
        questioner=FakeQuestioner(config, [], shared_key_point=True),
        evaluator=FakeEvaluator([]),
        assessor=FakeAssessor(AssessorDecision(False, None, "stop"), []),
        analysis_pipeline=pipeline,
    )
    runner.run()

    events = CheckpointStore(run_dir, config).events()
    promotion_events = [
        e for e in events if e.get("event_type") == "promotion_proposal"
    ]
    assert promotion_events
    decisions = {e.get("decision") for e in promotion_events}
    assert "promote" not in decisions, (
        "single-phase samples must not promote: got {}".format(decisions)
    )


def test_promoted_branch_writes_promoted_jsonl(tmp_path):
    """A PROMOTE with auto_promote=True appends to ``promoted.jsonl``."""
    config = small_config(points=1, seeds=1, adaptive=1)
    config = replace(config, search=replace(config.search, auto_promote=True))
    run_dir = tmp_path / "run"

    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    pipeline = AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        ),
        behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
        branch_manager=BranchManager(min_support=2),
    )
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=FakeExaminer(
            [KnowledgePoint("group", "subtask", "point-0")], []
        ),
        questioner=FakeQuestioner(config, [], shared_key_point=True),
        evaluator=FakeEvaluator([]),
        assessor=FakeAssessor(AssessorDecision(False, None, "stop"), []),
        analysis_pipeline=pipeline,
    )
    runner.run()

    promoted_path = run_dir / "promoted.jsonl"
    assert promoted_path.exists(), "expected promoted.jsonl after a PROMOTE"
    lines = promoted_path.read_text(encoding="utf-8").splitlines()
    assert lines, "promoted.jsonl should have at least one line"
    entry = json.loads(lines[0])
    assert entry["knowledge_point"] == "point-0"
    assert entry["group"] == "group"
    assert entry["subtask"] == "subtask"


def test_resume_loads_promoted_points_as_discovered(tmp_path):
    """A resume that finds ``promoted.jsonl`` treats promoted points as
    discovered knowledge points even when no sample carries their name."""
    config = small_config(points=1, seeds=1, adaptive=0)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    promoted_path = run_dir / "promoted.jsonl"
    promoted_path.write_text(
        json.dumps(
            {
                "group": "group",
                "subtask": "subtask",
                "knowledge_point": "promoted-x",
                "branch_id": "promoted-x:v",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner, _, _, _, _ = make_runner(config, run_dir)
    points = runner._configured_points()
    promoted = [point for point in points if point.name == "promoted-x"]
    assert len(promoted) == 1
    assert promoted[0].discovered is True


def test_most_recent_branch_id_uses_most_recent_weakness(tmp_path):
    """The controller's branch_id should match the real
    '<point>:<key_point>' form, taken from the most recent weakness
    for the point, so audit log rows correlate with branch_update
    events.  Before any weakness is observed the helper returns
    None and the runner falls back to the bare point name."""
    config = small_config(points=1, seeds=0, adaptive=0, cap=5)
    run_dir = tmp_path / "run"
    runner, _, _, _, _ = make_runner(config, run_dir)
    point = KnowledgePoint("group", "subtask", "point-0")
    # No records yet: helper returns None.
    assert runner._most_recent_branch_id(point) is None
    # Inject a weakness record with a known key_point.
    record = evaluated_record(
        Question(
            "group", "subtask", "point-0", "shared-key", "p", "seed", 0, 0
        )
    )
    runner._records.append(record)
    assert runner._most_recent_branch_id(point) == "point-0:shared-key"


def test_search_decision_branch_id_matches_real_branch(tmp_path):
    """The search_decision event emitted by the runner carries the
    real branch_id ('<point>:<key_point>') when a weakness has been
    observed, not the bare point name."""
    config = small_config(points=1, seeds=1, adaptive=1)
    run_dir = tmp_path / "run"

    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager
    from autodetect_repro.controller import RuleController

    pipeline = AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        ),
        behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
        branch_manager=BranchManager(min_support=2),
    )
    log = []
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=FakeExaminer(
            [KnowledgePoint("group", "subtask", "point-0")], log
        ),
        questioner=FakeQuestioner(config, [], shared_key_point=True),
        evaluator=FakeEvaluator([]),
        assessor=FakeAssessor(AssessorDecision(False, None, "stop"), []),
        analysis_pipeline=pipeline,
        controller=RuleController(candidate_cap=2, test_cap=2),
    )
    runner.run()

    search_decisions = [
        e for e in CheckpointStore(run_dir, config).events()
        if e.get("event_type") == "search_decision"
    ]
    assert search_decisions, "expected at least one search_decision event"
    # All branch_ids must include the ':<key_point>' suffix, not the
    # bare point name.
    for event in search_decisions:
        branch_id = event["branch_id"]
        assert branch_id != "point-0", (
            "search_decision.branch_id is the bare point name, not "
            "the real <point>:<key_point> form: {}".format(branch_id)
        )
        assert branch_id.startswith("point-0:"), branch_id
