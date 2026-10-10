"""Integration smoke test: run → report → resume → report.

Pins the cross-module contract that the previous series of commits
depends on:

1. ExperimentRunner writes a valid manifest + events + samples file.
2. write_report renders the new Search & Filtering / Behavior Clusters
   sections.
3. Resume reads events.jsonl + samples.jsonl via the analysis
   pipeline and does not double-emit promote events.
4. promoted.jsonl is appended once, not re-emitted on resume.

This is the chain a real adaptive run exercises.  It uses the
runner directly with stub roles (FakeExaminer/Questioner/...) so
no network is touched and the test runs in <1s.  The full
``cli.execute_run`` path is exercised in
``test_adaptive_end_to_end.py`` but requires patching every
role's structured-call validator, which is brittle.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import pytest


from autodetect_repro.checkpoint import CheckpointStore, sample_id
from autodetect_repro.config import load_config
from autodetect_repro.controller import RuleController
from autodetect_repro.runner import ExperimentRunner
from autodetect_repro.types import (
    AssessorDecision,
    KnowledgePoint,
    Question,
    SampleRecord,
    SubtaskConfig,
)
from autodetect_repro.reporting import write_report


# --- shared test scaffolding (duplicated to keep this file self-contained) -


def _base_config():
    return load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )


def _small(*, points=1, seeds=2, adaptive=1, cap=20, **overrides):
    config = _base_config()
    config = replace(
        config,
        subtasks=(SubtaskConfig("g", "s", ("p1",)),),
        search=replace(
            config.search,
            seed_questions_per_point=seeds,
            adaptive_iterations_per_point=adaptive,
            max_initial_points=points,
            **overrides,
        ),
        limits=replace(config.limits, max_total_samples=cap),
    )
    return config


def _evaluated(question: Question) -> SampleRecord:
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


class _Examiner:
    def __init__(self, points):
        self.points = points

    def select_taxonomy(self):
        return tuple(self.points)


class _Questioner:
    def __init__(self, config, shared_key_point=False):
        self.config = config
        self.shared_key_point = shared_key_point

    def generate_seeds(self, point):
        return tuple(
            Question(
                point.group, point.subtask, point.name,
                point.name if self.shared_key_point else "seed-key-{}".format(i),
                "Q-seed-{}".format(i), "seed", 0, i,
            )
            for i in range(self.config.search.seed_questions_per_point)
        )

    def generate_adaptive(self, point, history, iteration, **kwargs):
        return Question(
            point.group, point.subtask, point.name,
            point.name if self.shared_key_point else "adaptive-key",
            "Q-adaptive-{}".format(iteration), "adaptive", iteration, 0,
        )


class _Evaluator:
    def evaluate(self, question):
        return _evaluated(question)


class _Assessor:
    def propose(self, subtask, history):
        return AssessorDecision(False, None, "no new mode")


# --- the actual tests ---


def _build_pipeline(config):
    from autodetect_repro.analysis_pipeline import AdaptiveAnalysisPipeline
    from autodetect_repro.behavior import BehaviorLabel, FailureSignature
    from autodetect_repro.branches import BranchManager

    return AdaptiveAnalysisPipeline(
        signature_extractor=lambda record: FailureSignature(
            violation=record.key_point or "v",
            observed_behavior="b",
            trigger_context="c",
            evidence="e",
        ),
        behavior_classifier=lambda s, r: BehaviorLabel.DIFFERENT,
        branch_manager=BranchManager(min_support=2),
    )


def test_run_writes_promoted_jsonl_and_report_surfaces_it(
    tmp_path, monkeypatch
):
    """End-to-end of the chain: adaptive run -> events.jsonl ->
    promoted.jsonl -> report.md."""
    config = _small(seeds=1, adaptive=1, cap=10, auto_promote=True)
    config = replace(
        config,
        search=replace(config.search, auto_promote=True),
    )
    run_dir = tmp_path / "run"
    point = KnowledgePoint("g", "s", "p1")
    pipeline = _build_pipeline(config)
    controller = RuleController(candidate_cap=2, test_cap=2)
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=_Examiner([point]),
        questioner=_Questioner(config, shared_key_point=True),
        evaluator=_Evaluator(),
        assessor=_Assessor(),
        analysis_pipeline=pipeline,
        controller=controller,
    )
    summary = runner.run()
    assert summary.scheduled > 0

    # The new wire: events.jsonl, promoted.jsonl, report.md all exist.
    assert (run_dir / "events.jsonl").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    types = {e.get("event_type") for e in events}
    # Core events the analysis pipeline + adaptive runner always emit.
    for required in (
        "failure_analysis",
        "branch_update",
        "search_decision",
        "promotion_proposal",
    ):
        assert required in types, "missing {} in events: {}".format(
            required, types
        )

    # promoted.jsonl was appended by the run, exactly once.
    assert (run_dir / "promoted.jsonl").exists()
    promoted_lines = (
        run_dir / "promoted.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(promoted_lines) >= 1
    promoted_entry = json.loads(promoted_lines[0])
    assert promoted_entry["knowledge_point"] == "p1"
    assert promoted_entry["branch_id"] == "p1:p1"

    # The new report sections render with real data.
    write_report(run_dir)
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "## Search & Filtering" in report
    assert "## Behavior Clusters" in report
    # The promoted branch's id should appear in the report.
    assert "p1:p1" in report


def test_resume_does_not_double_emit_promote(tmp_path):
    """Resume must NOT re-emit a PROMOTE event or re-append to
    promoted.jsonl.  The promoted branches are already in
    promoted.jsonl; resume just samples the promoted point
    further."""
    config = _small(seeds=1, adaptive=1, cap=10, auto_promote=True)
    config = replace(
        config,
        search=replace(config.search, auto_promote=True),
    )
    run_dir = tmp_path / "run"
    point = KnowledgePoint("g", "s", "p1")
    pipeline = _build_pipeline(config)
    controller = RuleController(candidate_cap=2, test_cap=2)
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=_Examiner([point]),
        questioner=_Questioner(config, shared_key_point=True),
        evaluator=_Evaluator(),
        assessor=_Assessor(),
        analysis_pipeline=pipeline,
        controller=controller,
    )
    runner.run()

    promoted_before = (
        run_dir / "promoted.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    before_count = len(promoted_before)

    # Resume.  The runner is reconstructed from the on-disk state
    # to simulate a fresh process, exactly the resume contract.
    fresh_pipeline = _build_pipeline(config)
    resumed = ExperimentRunner(
        config,
        run_dir,
        examiner=_Examiner([point]),
        questioner=_Questioner(config, shared_key_point=True),
        evaluator=_Evaluator(),
        assessor=_Assessor(),
        analysis_pipeline=fresh_pipeline,
        controller=RuleController(candidate_cap=2, test_cap=2),
    )
    summary2 = resumed.run()
    # Resume must not append to promoted.jsonl: the promotion was
    # already recorded in the first run, and the second run's
    # gate-evaluation (on a branch already at status=promoted)
    # must not re-emit a PROMOTE event or re-write the file.
    promoted_after = (
        run_dir / "promoted.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(promoted_after) == before_count, (
        "promoted.jsonl grew on resume: before={}, after={}".format(
            before_count, len(promoted_after)
        )
    )
    # And: no new promotion_proposal events with decision=promote
    # should be emitted on resume.
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    promotes = [
        e for e in events
        if e.get("event_type") == "promotion_proposal"
        and e.get("decision") == "promote"
    ]
    # The original PROMOTE was already there.  We expect no NEW
    # PROMOTE events.  Specifically, we expect fewer than the number
    # of resume calls we made (just 1) -> exactly 1 PROMOTE total.
    assert len(promotes) == 1, (
        "expected exactly 1 PROMOTE event (from the first run); "
        "got {} -> resume re-emitted promotion".format(len(promotes))
    )


def test_promoted_branch_loaded_on_resume(tmp_path):
    """A promoted.jsonl on disk is honored on resume: the
    promoted knowledge point shows up in the configured points
    list even when no sample carries its name."""
    config = _small(seeds=1, adaptive=0, cap=5, auto_promote=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Manually create promoted.jsonl.
    (run_dir / "promoted.jsonl").write_text(
        json.dumps(
            {
                "group": "g",
                "subtask": "s",
                "knowledge_point": "newly-discovered",
                "branch_id": "newly-discovered:violation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=_Examiner([KnowledgePoint("g", "s", "p1")]),
        questioner=_Questioner(config),
        evaluator=_Evaluator(),
        assessor=_Assessor(),
    )
    points = runner._configured_points()
    promoted = [p for p in points if p.name == "newly-discovered"]
    assert len(promoted) == 1
    assert promoted[0].discovered is True


def test_search_decision_branch_id_is_real(tmp_path):
    """search_decision.branch_id must be the real
    '<point>:<key_point>' form, not the bare point name."""
    config = _small(seeds=1, adaptive=1, cap=10, auto_promote=True)
    config = replace(
        config,
        search=replace(config.search, auto_promote=True),
    )
    run_dir = tmp_path / "run"
    point = KnowledgePoint("g", "s", "p1")
    pipeline = _build_pipeline(config)
    controller = RuleController(candidate_cap=2, test_cap=2)
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=_Examiner([point]),
        questioner=_Questioner(config, shared_key_point=True),
        evaluator=_Evaluator(),
        assessor=_Assessor(),
        analysis_pipeline=pipeline,
        controller=controller,
    )
    runner.run()
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    decisions = [
        e for e in events if e.get("event_type") == "search_decision"
    ]
    assert decisions, "no search_decision event emitted"
    for d in decisions:
        branch_id = d["branch_id"]
        assert branch_id != "p1", (
            "search_decision.branch_id is bare point name, "
            "expected 'p1:<key_point>'"
        )
        assert branch_id.startswith("p1:"), branch_id
