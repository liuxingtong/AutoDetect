"""Tests for ``scripts/build_calibration_pool.py`` event-aware behavior pool."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# Load the script as a module so we can exercise its functions without
# installing it as a package.
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_calibration_pool.py"
spec = importlib.util.spec_from_file_location("bcp", SCRIPT)
bcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bcp)  # type: ignore[union-attr]


def _write_jsonl(path: Path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_failure_analysis_events_filter(tmp_path):
    events = [
        {"event_type": "failure_analysis", "signature": {"violation": "v1"}},
        {"event_type": "branch_update", "branch_id": "x"},
        {"event_type": "failure_analysis", "signature": {"violation": "v2"}},
        {"event_type": "search_decision"},
    ]
    failures = bcp._failure_analysis_events(events)
    assert len(failures) == 2
    assert {f["signature"]["violation"] for f in failures} == {"v1", "v2"}


def test_behavior_pool_uses_real_signatures_from_events(tmp_path):
    events = [
        {
            "event_type": "failure_analysis",
            "sample_id": "s1",
            "signature": {
                "violation": "alpha violation",
                "observed_behavior": "alpha behavior",
                "trigger_context": "alpha trigger",
                "evidence": "alpha evidence",
            },
        },
        {
            "event_type": "failure_analysis",
            "sample_id": "s2",
            "signature": {
                "violation": "beta violation",
                "observed_behavior": "beta behavior",
                "trigger_context": "beta trigger",
                "evidence": "beta evidence",
            },
        },
        {
            "event_type": "failure_analysis",
            "sample_id": "s3",
            "signature": {
                "violation": "gamma violation",
                "observed_behavior": "gamma behavior",
                "trigger_context": "gamma trigger",
                "evidence": "gamma evidence",
            },
        },
    ]
    pool = bcp.build_behavior_pool_from_events(events, seed=1)
    assert len(pool) == 3  # 3 choose 2
    for pair in pool:
        # Signatures are passed through verbatim — no synthesis from
        # key_point or truncation.  This is the contract the runtime
        # BehaviorJudge is calibrated against.
        assert "alpha" in pair["signature_a"]["violation"] or \
            "beta" in pair["signature_a"]["violation"] or \
            "gamma" in pair["signature_a"]["violation"]
        assert "alpha" in pair["signature_b"]["violation"] or \
            "beta" in pair["signature_b"]["violation"] or \
            "gamma" in pair["signature_b"]["violation"]
        # evidence fields are untouched (not synthesized as "score=…")
        assert "comparison_present" not in pair["signature_a"]["evidence"]


def test_main_prefers_events_when_present(tmp_path, capsys):
    math_path = tmp_path / "math.jsonl"
    code_path = tmp_path / "code.jsonl"
    events_path = tmp_path / "events.jsonl"
    out_dir = tmp_path / "calibration"

    # minimal valid samples.jsonl
    _write_jsonl(
        math_path,
        [
            {
                "domain": "math",
                "point": "p",
                "prompt": "q",
                "key_point": "k",
                "answer": "a",
                "score": 3.0,
                "is_weakness": True,
                "comparison": "Rating: [[3]]",
            }
        ],
    )
    _write_jsonl(code_path, [])
    _write_jsonl(
        events_path,
        [
            {
                "event_type": "failure_analysis",
                "sample_id": "x",
                "signature": {
                    "violation": "from-events",
                    "observed_behavior": "b",
                    "trigger_context": "c",
                    "evidence": "e",
                },
            },
            {
                "event_type": "failure_analysis",
                "sample_id": "y",
                "signature": {
                    "violation": "from-events-2",
                    "observed_behavior": "b",
                    "trigger_context": "c",
                    "evidence": "e",
                },
            },
        ],
    )

    # Patch sys.argv so argparse picks up our paths.
    argv_save = sys.argv
    sys.argv = [
        "bcp",
        "--math", str(math_path),
        "--code", str(code_path),
        "--events", str(events_path),
        "--out", str(out_dir),
    ]
    try:
        rc = bcp.main()
    finally:
        sys.argv = argv_save
    assert rc == 0

    out = capsys.readouterr().out
    assert "events.jsonl" in out
    pool = json.loads((out_dir / "behavior_pool.json").read_text(encoding="utf-8"))
    assert pool[0]["signature_a"]["violation"] in {"from-events", "from-events-2"}


def test_main_falls_back_when_events_missing(tmp_path, capsys):
    math_path = tmp_path / "math.jsonl"
    code_path = tmp_path / "code.jsonl"
    out_dir = tmp_path / "calibration"

    samples = [
        {
            "domain": "math",
            "point": "p1",
            "prompt": "q1",
            "key_point": "k1",
            "answer": "a1",
            "score": 3.0,
            "is_weakness": True,
            "comparison": "Rating: [[3]]",
        },
        {
            "domain": "math",
            "point": "p1",
            "prompt": "q2",
            "key_point": "k2",
            "answer": "a2",
            "score": 2.0,
            "is_weakness": True,
            "comparison": "Rating: [[2]]",
        },
    ]
    _write_jsonl(math_path, samples)
    _write_jsonl(code_path, [])

    argv_save = sys.argv
    sys.argv = [
        "bcp",
        "--math", str(math_path),
        "--code", str(code_path),
        "--events", str(tmp_path / "missing.jsonl"),
        "--out", str(out_dir),
    ]
    try:
        rc = bcp.main()
    finally:
        sys.argv = argv_save
    assert rc == 0

    captured = capsys.readouterr()
    assert "falling back" in captured.err
    pool = json.loads((out_dir / "behavior_pool.json").read_text(encoding="utf-8"))
    # Synthesized signature carries the comparison_present marker.
    assert "comparison_present" in pool[0]["signature_a"]["evidence"]
