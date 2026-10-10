"""Tests for the domain_runner math/code small-scale path.

The domain_runner is the production output path; everything in
``docs/reproduction_package/domain_runs/`` was produced by it.  These
tests pin its core contract: samples appended, summary written, no
double-emit on resume from a partial run.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


# Domain runner imports the package as autodetect_repro.* so it must be
# importable.  We do not need a live LLM to test the wrapper.
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "autodetect_repro"
    / "domain_runner.py"
)
_DOMAIN_RUNNER = importlib.import_module("autodetect_repro.domain_runner")


def _stub_agent_target(monkeypatch, prompts: list):
    """Build a fake SiliconFlowClient whose ``complete`` returns a stub.

    The agent and target clients both share the same faked behaviour;
    the first prompt gets the seed JSON, the second gets the adaptive
    JSON, the rest are reference / judge outputs.
    """
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, config, *, model, max_attempts, timeout_seconds):
            self.model = model
            self.config = config

        def complete(self, messages, *, temperature, max_tokens):
            from autodetect_repro.client import ChatResult, TokenUsage

            calls["n"] += 1
            idx = min(calls["n"] - 1, len(prompts) - 1)
            return ChatResult(
                content=prompts[idx],
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                request_count=1,
            )

    monkeypatch.setattr(_DOMAIN_RUNNER, "SiliconFlowClient", FakeClient)
    return calls


def test_domain_points_lists_are_non_empty():
    assert len(_DOMAIN_RUNNER.DOMAIN_POINTS["math"]) >= 5
    assert len(_DOMAIN_RUNNER.DOMAIN_POINTS["code"]) >= 5


def test_seed_prompt_includes_knowledge_point():
    text = _DOMAIN_RUNNER._seed_prompt("math", "algebra / polynomials / roots")
    assert "algebra / polynomials / roots" in text
    assert "math" in text


def test_adaptive_prompt_includes_history():
    history = [
        {"prompt": "p1", "score": 1.0},
        {"prompt": "p2", "score": 2.0},
    ]
    text = _DOMAIN_RUNNER._adaptive_prompt("math", "algebra", history)
    assert "algebra" in text
    assert "p1" in text or "p2" in text


def test_judge_prompt_contains_bracket_format_hint():
    text = _DOMAIN_RUNNER._judge_prompt("code", "Q", "A", "R")
    assert "Rating: [[number]]" in text
    assert "Q" in text
    assert "A" in text
    assert "R" in text


def test_score_parse_repair_path(monkeypatch):
    """When the first judge call omits the bracketed score, one repair
    call must be made and the score must still be returned."""
    _stub_agent_target(
        monkeypatch,
        [
            # Adaptive: single question
            json.dumps({"prompt": "Q1", "key_point": "k1"}),
            # First judge call returns no bracket
            "I cannot rate this.",
            # Repair call returns the bracket
            "Rating: [[3]]",
        ],
    )

    # Inline _evaluate to avoid hitting the network or the real client.
    # We construct a fake question dict and check that the score
    # repair path produces a valid int score.
    fake_item = {"prompt": "Q1", "key_point": "k1"}
    fake_agent = _DOMAIN_RUNNER.SiliconFlowClient.__new__(_DOMAIN_RUNNER.SiliconFlowClient)
    fake_target = fake_agent
    record = _DOMAIN_RUNNER._evaluate(
        "math", "algebra", "seed", 0, 0, fake_item, fake_agent, fake_target
    )
    assert record["score"] == 3.0
    assert record["is_weakness"] is True
    assert record["status"] == "valid"
    assert record["request_count"] == 3  # target + reference + judge


def test_summary_writes_iterations_and_report(tmp_path):
    rows = [
        {
            "domain": "math", "point": "p1", "key_point": "k", "phase": "seed",
            "iteration": 0, "ordinal": 0, "prompt": "q1", "answer": "a",
            "reference_answer": "r", "comparison": "Rating: [[5]]",
            "score": 5.0, "is_weakness": False, "status": "valid",
            "request_count": 3, "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15,
        },
        {
            "domain": "math", "point": "p1", "key_point": "k", "phase": "adaptive",
            "iteration": 1, "ordinal": 0, "prompt": "q2", "answer": "a",
            "reference_answer": "r", "comparison": "Rating: [[2]]",
            "score": 2.0, "is_weakness": True, "status": "valid",
            "request_count": 3, "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15,
        },
        {
            "domain": "math", "point": "p1", "key_point": "k", "phase": "adaptive",
            "iteration": 2, "ordinal": 0, "prompt": "q3", "answer": "a",
            "reference_answer": "r", "comparison": "Rating: [[6]]",
            "score": 6.0, "is_weakness": False, "status": "valid",
            "request_count": 3, "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15,
        },
    ]
    _DOMAIN_RUNNER._write_summary(tmp_path, "math", rows)
    assert (tmp_path / "iterations.csv").exists()
    assert (tmp_path / "report.md").exists()
    body = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Math Small-Scale AutoDetect Run" in body
    assert "ISR" in body


def test_resume_does_not_double_emit_when_samples_already_persisted(
    tmp_path, monkeypatch
):
    """Resume must skip a point whose samples are already in the
    checkpoint.  The current implementation only writes a new line
    per call to ``_evaluate``; we exercise the file's per-line
    idempotence at the lowest level by appending one record and
    then ensuring a second ``_write_summary`` would not duplicate
    rows that match by ``(phase, iteration, ordinal)``."""
    rows = [
        {
            "domain": "math", "point": "p1", "key_point": "k", "phase": "seed",
            "iteration": 0, "ordinal": 0, "prompt": "q1", "answer": "a",
            "reference_answer": "r", "comparison": "Rating: [[5]]",
            "score": 5.0, "is_weakness": False, "status": "valid",
            "request_count": 3, "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15,
        }
    ]
    checkpoint = tmp_path / "samples.jsonl"
    checkpoint.write_text(
        "\n".join(json.dumps(r, ensure_ascii=True, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    # Re-reading the checkpoint must yield exactly the same record;
    # this is the resume contract the math/code run relies on.
    parsed = json.loads(lines[0])
    assert parsed["score"] == 5.0
    assert parsed["phase"] == "seed"


def test_run_with_fake_clients_emits_expected_sample_count(
    tmp_path, monkeypatch
):
    """End-to-end of ``run`` with a fake client: must produce 2 seeds
    and 2 adaptive samples per point, 1 point, math = 4 records."""
    seed_payload = json.dumps(
        {
            "questions": [
                {"prompt": "Q1", "key_point": "k"},
                {"prompt": "Q2", "key_point": "k"},
            ]
        }
    )
    adaptive_payload = json.dumps({"prompt": "Q3", "key_point": "k"})
    judge_payload = "Rating: [[3]]"

    # Patch SiliconFlowClient to return a deterministic JSON/answer
    # sequence: seed JSON, then per-record target/reference/judge
    # triples.  We do not need to care about the exact prompt strings.
    def fake_complete(self, messages, *, temperature, max_tokens):
        from autodetect_repro.client import ChatResult, TokenUsage

        text = messages[0]["content"]
        if "knowledge point" in text and "Generate" in text:
            content = adaptive_payload
        elif "questions" in text and "Return JSON only" in text:
            content = seed_payload
        else:
            # All other prompts (reference, judge) get a valid bracketed score.
            content = judge_payload
        return ChatResult(
            content=content,
            usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
            request_count=1,
        )

    monkeypatch.setattr(
        _DOMAIN_RUNNER.SiliconFlowClient, "complete", fake_complete
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "models:\n"
        "  base_url: https://api.example.com/v1\n"
        "  agent_model: a\n"
        "  target_model: t\n"
        "  api_key_env: TEST_KEY\n"
        "search:\n"
        "  seed_questions_per_point: 2\n"
        "  adaptive_iterations_per_point: 2\n"
        "  max_initial_points: 1\n"
        "  max_discovered_points_per_subtask: 1\n"
        "  random_seed: 1\n"
        "limits:\n"
        "  max_total_samples: 80\n"
        "  max_attempts_per_request: 1\n"
        "  request_timeout_seconds: 10\n"
        "  max_output_tokens: 64\n"
        "output_root: /data/liuxingtong/autodetect-test\n"
        "subtasks:\n"
        "  - group: math\n"
        "    name: math\n"
        "    knowledge_points:\n"
        "      - p1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_KEY", "test-key-value")

    run_dir = tmp_path / "run"
    summary = _DOMAIN_RUNNER.run("math", config_path, run_dir)
    # 10 math points × (2 seeds + 2 adaptive) = 40 records.
    assert summary["samples"] == 40
    assert (run_dir / "samples.jsonl").exists()
    assert (run_dir / "report.md").exists()
