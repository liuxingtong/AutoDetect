"""Tests for the offline report generator."""

import json
from pathlib import Path

import pytest

from autodetect_repro.checkpoint import CheckpointStore
from autodetect_repro.config import load_config
from autodetect_repro.reporting import (
    _branch_status,
    _behavior_clusters,
    _candidate_filters,
    _diversity_metrics,
    _load_events,
    _promotion_decisions,
    _search_decisions,
    write_report,
)


@pytest.fixture
def config():
    return load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )


def _make_record(identifier, prompt="prompt"):
    from autodetect_repro.types import SampleRecord
    return SampleRecord(
        sample_id=identifier,
        group="format",
        subtask="text format",
        knowledge_point="list format",
        phase="seed",
        iteration=0,
        ordinal=0,
        prompt=prompt,
        key_point="list constraint",
        answer="- one",
        reference_answer="- one\n- two",
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


def test_branch_status_summarizes_latest_branch_update(tmp_path, config):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config)
    store.append(_make_record("rec-1"))

    store.append_event(
        "branch_update",
        {
            "branch_id": "list format:v1",
            "status": "candidate",
            "supporting_samples": ["rec-1"],
        },
    )
    store.append_event(
        "branch_update",
        {
            "branch_id": "list format:v1",
            "status": "verifying",
            "supporting_samples": ["rec-1", "rec-2"],
        },
    )
    store.append_event(
        "branch_update",
        {
            "branch_id": "list format:v2",
            "status": "verifying",
            "supporting_samples": ["rec-3"],
        },
    )

    table = _branch_status(run_dir)
    assert "list format:v1" in table
    assert "list format:v2" in table
    # The latest event wins — v1 should be "verifying" with 2 samples.
    assert "verifying" in table


def test_promotion_decisions_lists_all_proposals(tmp_path, config):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config)
    store.append(_make_record("rec-1"))

    store.append_event(
        "promotion_proposal",
        {
            "branch_id": "list format:v1",
            "decision": "continue_verification",
            "knowledge_point": "list format",
            "reason": "still under support threshold",
        },
    )
    store.append_event(
        "promotion_proposal",
        {
            "branch_id": "list format:v2",
            "decision": "promote",
            "knowledge_point": "list format",
            "reason": "all gates passed",
        },
    )

    table = _promotion_decisions(run_dir)
    assert "list format:v1" in table
    assert "list format:v2" in table
    assert "continue_verification" in table
    assert "promote" in table


def test_diversity_metrics_shows_latest_event(tmp_path, config):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config)
    store.append(_make_record("rec-1"))

    store.append_event(
        "diversity_metrics",
        {
            "richness": 1,
            "gain": 1,
            "discovery_rate": 1.0,
            "target_calls": 1,
        },
    )
    store.append_event(
        "diversity_metrics",
        {
            "richness": 4,
            "gain": 2,
            "discovery_rate": 0.2,
            "target_calls": 12,
        },
    )

    summary = _diversity_metrics(run_dir)
    # Latest event wins.
    assert "4" in summary
    assert "20.00%" in summary
    assert "12" in summary


def test_write_report_includes_branch_and_promotion_sections(tmp_path, config):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config)
    store.append(_make_record("rec-1", prompt="prompt-1"))
    store.append(_make_record("rec-2", prompt="prompt-2"))

    store.append_event(
        "branch_update",
        {
            "branch_id": "list format:violation",
            "status": "verifying",
            "supporting_samples": ["rec-1", "rec-2"],
        },
    )
    store.append_event(
        "promotion_proposal",
        {
            "branch_id": "list format:violation",
            "decision": "continue_verification",
            "knowledge_point": "list format",
            "reason": "still under threshold",
        },
    )
    store.append_event(
        "diversity_metrics",
        {"richness": 2, "gain": 1, "discovery_rate": 0.5, "target_calls": 2},
    )

    paths = write_report(run_dir)
    markdown = paths.markdown.read_text(encoding="utf-8")

    assert "## Behavioral Diversity" in markdown
    assert "## Branch & Promotion" in markdown
    assert "list format:violation" in markdown
    assert "continue_verification" in markdown
    assert "Behavioral richness" in markdown


def test_load_events_handles_missing_file(tmp_path):
    assert _load_events(tmp_path) == []


def test_load_events_skips_malformed_json(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event_type": "x"}\n'
        'this is not json\n'
        '{"event_type": "y"}\n',
        encoding="utf-8",
    )
    events = _load_events(tmp_path)
    assert len(events) == 2
    assert [e["event_type"] for e in events] == ["x", "y"]


def test_search_decisions_renders_table(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_type": "search_decision",
                "knowledge_point": "p1",
                "branch_id": "p1:v",
                "strategy": "verify",
                "objective": "obj",
                "reason": "recent failures include behavior not yet sufficiently verified",
                "candidate_budget": 3,
                "test_budget": 2,
                "iteration": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    md = _search_decisions(tmp_path)
    assert "p1" in md
    assert "verify" in md
    assert "3 / 2" in md
    assert "iteration" in md


def test_search_decisions_empty(tmp_path):
    assert "no search_decision" in _search_decisions(tmp_path)


def test_candidate_filters_renders_summary(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_type": "candidate_filter",
                "iteration": 3,
                "candidates": [
                    {"prompt": "p1"},
                    {"prompt": "p2"},
                    {"prompt": "p3"},
                ],
                "decisions": [
                    {"prompt": "p1", "label": "distinct"},
                    {"prompt": "p2", "label": "duplicate"},
                    {"prompt": "p3", "label": "distinct"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    md = _candidate_filters(tmp_path)
    assert "3" in md
    assert "2" in md  # 2 selected (distinct)
    assert "1" in md  # 1 rejected (duplicate)


def test_behavior_clusters_counts_latest_label_per_sample(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_type": "behavior_cluster_update",
                "sample_id": "s1",
                "label": "different",
            }
        )
        + "\n"
        + json.dumps(
            {
                "event_type": "behavior_cluster_update",
                "sample_id": "s1",
                "label": "same",  # overrides the previous label
            }
        )
        + "\n"
        + json.dumps(
            {
                "event_type": "behavior_cluster_update",
                "sample_id": "s2",
                "label": "different",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    md = _behavior_clusters(tmp_path)
    # s1 ends as "same", s2 as "different" -> 1 each
    assert "| same | 1 |" in md
    assert "| different | 1 |" in md
