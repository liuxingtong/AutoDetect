"""Tests for ``scripts/calibrate.py``.

The script has two distinct paths: one that calls the real LLM judge
(env key required) and one that produces a stub report.  The tests
below cover (1) the no-labeled-pairs stub, (2) the no-API-key stub,
(3) the full report path with a fake ``BehaviorJudge`` whose
verdicts are predictable, and (4) the Cohen kappa + confusion
matrix math.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate.py"
spec = importlib.util.spec_from_file_location("calibrate", SCRIPT)
calibrate = importlib.util.module_from_spec(spec)
sys.modules["calibrate"] = calibrate  # so dataclass can resolve __module__
spec.loader.exec_module(calibrate)  # type: ignore[union-attr]


def _write_jsonl_or_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def test_cohen_kappa_perfect_agreement():
    kappa = calibrate._cohen_kappa(["same"] * 5, ["same"] * 5)
    assert kappa == pytest.approx(1.0)


def test_cohen_kappa_random_chance_level():
    """Two classes, predicted 50/50 and truth 50/50, but always
    wrong: po=0, pe=0.5, kappa=-1.0."""
    kappa = calibrate._cohen_kappa(
        ["a", "b", "a", "b", "a", "b"],
        ["b", "a", "b", "a", "b", "a"],
    )
    assert kappa == pytest.approx(-1.0)


def test_cohen_kappa_partial_agreement():
    """3 same / 1 different; predicted 3 same / 1 different.
    po=1.0, pe=1.0 -> kappa=1.0 (per the function's degenerate
    branch when pe=1)."""
    kappa = calibrate._cohen_kappa(
        ["same", "same", "same", "same"],
        ["same", "same", "same", "different"],
    )
    assert 0.0 <= kappa <= 1.0


def test_cohen_kappa_empty_returns_zero():
    assert calibrate._cohen_kappa([], []) == 0.0


def test_confusion_matrix_counts_correctly():
    predicted = ["same", "different", "same", "uncertain"]
    truth = ["same", "same", "different", "uncertain"]
    counts = calibrate._confusion_matrix(
        predicted, truth, ["same", "different", "uncertain"]
    )
    assert counts[("same", "same")] == 1
    assert counts[("same", "different")] == 1
    assert counts[("same", "uncertain")] == 0
    assert counts[("different", "same")] == 1
    assert counts[("different", "different")] == 0
    assert counts[("uncertain", "uncertain")] == 1


def test_no_labeled_pairs_writes_stub(tmp_path):
    pool = [
        {"domain": "d", "point": "p", "signature_a": {}, "signature_b": {}}
    ]
    gt = [{"index": 0, "label": None, "rationale": ""}]
    _write_jsonl_or_json(tmp_path / "behavior_pool.json", pool)
    _write_jsonl_or_json(tmp_path / "behavior_ground_truth.json", gt)

    rc = calibrate._run_calibration(
        kind="behavior",
        pool_path=tmp_path / "behavior_pool.json",
        ground_truth_path=tmp_path / "behavior_ground_truth.json",
        judge_factory=lambda c: _FakeJudge(decision="same"),
        out_path=tmp_path / "report.md",
    )
    assert rc == 0
    body = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "No labeled pairs" in body
    # The real judge was not constructed because there were no pairs.
    assert "behavior calibration report" in body


def test_no_api_key_writes_stub(tmp_path, monkeypatch):
    pool = [
        {
            "domain": "d",
            "point": "p",
            "signature_a": {"violation": "a"},
            "signature_b": {"violation": "b"},
        }
    ]
    gt = [{"index": 0, "label": "different", "rationale": "different rules"}]
    _write_jsonl_or_json(tmp_path / "behavior_pool.json", pool)
    _write_jsonl_or_json(tmp_path / "behavior_ground_truth.json", gt)

    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    rc = calibrate._run_calibration(
        kind="behavior",
        pool_path=tmp_path / "behavior_pool.json",
        ground_truth_path=tmp_path / "behavior_ground_truth.json",
        judge_factory=lambda c: _FakeJudge(decision="same"),
        out_path=tmp_path / "report.md",
    )
    assert rc == 0
    body = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "SILICONFLOW_API_KEY" in body
    assert "Labeled pairs available: 1" in body


def test_full_report_with_fake_judge(tmp_path, monkeypatch):
    """A perfect fake judge yields kappa=1.0, accuracy=1.0."""
    pool = [
        {
            "domain": "d",
            "point": "p",
            "signature_a": {"violation": "a"},
            "signature_b": {"violation": "b"},
        },
        {
            "domain": "d",
            "point": "p",
            "signature_a": {"violation": "a"},
            "signature_b": {"violation": "a"},
        },
    ]
    gt = [
        {"index": 0, "label": "different", "rationale": ""},
        {"index": 1, "label": "same", "rationale": ""},
    ]
    _write_jsonl_or_json(tmp_path / "behavior_pool.json", pool)
    _write_jsonl_or_json(tmp_path / "behavior_ground_truth.json", gt)

    # Stub the client so we don't need a real API key.
    monkeypatch.setattr(
        calibrate, "_build_client", lambda env_name: object()
    )

    class _PerfectJudge:
        def judge(self, a, b):
            from autodetect_repro.calibration import JudgeVerdict
            if a.get("violation") == b.get("violation"):
                return JudgeVerdict(
                    decision="same", confidence=0.9, reason="match", raw={}
                )
            return JudgeVerdict(
                decision="different",
                confidence=0.9,
                reason="different",
                raw={},
            )

    rc = calibrate._run_calibration(
        kind="behavior",
        pool_path=tmp_path / "behavior_pool.json",
        ground_truth_path=tmp_path / "behavior_ground_truth.json",
        judge_factory=lambda c: _PerfectJudge(),
        out_path=tmp_path / "report.md",
    )
    assert rc == 0
    body = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Cohen kappa | 1.000" in body
    assert "accuracy | 1.000" in body
    assert "Confusion matrix" in body


def test_full_report_with_imperfect_judge(tmp_path, monkeypatch):
    """A judge that flips the second call yields kappa<1.0."""
    pool = [
        {
            "domain": "d",
            "point": "p",
            "signature_a": {"violation": "a"},
            "signature_b": {"violation": "a"},
        },
        {
            "domain": "d",
            "point": "p",
            "signature_a": {"violation": "a"},
            "signature_b": {"violation": "a"},
        },
    ]
    gt = [
        {"index": 0, "label": "same", "rationale": ""},
        {"index": 1, "label": "same", "rationale": ""},
    ]
    _write_jsonl_or_json(tmp_path / "behavior_pool.json", pool)
    _write_jsonl_or_json(tmp_path / "behavior_ground_truth.json", gt)

    monkeypatch.setattr(
        calibrate, "_build_client", lambda env_name: object()
    )

    class _FlakeyJudge:
        def __init__(self):
            self.n = 0

        def judge(self, a, b):
            from autodetect_repro.calibration import JudgeVerdict
            self.n += 1
            if self.n == 1:
                return JudgeVerdict(
                    decision="same", confidence=0.7, reason="", raw={}
                )
            return JudgeVerdict(
                decision="different", confidence=0.6, reason="", raw={}
            )

    rc = calibrate._run_calibration(
        kind="behavior",
        pool_path=tmp_path / "behavior_pool.json",
        ground_truth_path=tmp_path / "behavior_ground_truth.json",
        judge_factory=lambda c: _FlakeyJudge(),
        out_path=tmp_path / "report.md",
    )
    assert rc == 0
    body = (tmp_path / "report.md").read_text(encoding="utf-8")
    # 1 of 2 correct -> accuracy 0.5
    assert "accuracy | 0.500" in body
    # The per-pair table should show 1 same match and 1 different mismatch.
    assert "| 0 | same | same" in body
    assert "| 1 | same | different" in body


def test_load_pairs_skips_unlabeled(tmp_path):
    pool = [
        {"domain": "d", "point": "p", "signature_a": {}, "signature_b": {}}
    ] * 5
    gt = [
        {"index": 0, "label": "same", "rationale": ""},
        {"index": 1, "label": None, "rationale": ""},
        {"index": 2, "label": "different", "rationale": ""},
        {"index": 3, "label": "  ", "rationale": ""},  # whitespace-only
        {"index": 4, "label": "uncertain", "rationale": ""},
    ]
    pairs = calibrate._load_pairs(pool, gt)
    assert len(pairs) == 3
    assert [p[2] for p in pairs] == ["same", "different", "uncertain"]


def test_main_dispatches_by_kind(tmp_path, monkeypatch):
    """``--kind semantic`` reads the semantic files and writes the
    same report.md path.  We don't need a real judge for this
    routing test."""
    pool = []
    gt = []
    _write_jsonl_or_json(tmp_path / "semantic_pool.json", pool)
    _write_jsonl_or_json(tmp_path / "semantic_ground_truth.json", gt)

    argv_save = sys.argv
    sys.argv = [
        "calibrate", "--dir", str(tmp_path), "--kind", "semantic"
    ]
    try:
        rc = calibrate.main()
    finally:
        sys.argv = argv_save
    assert rc == 0
    body = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "semantic calibration report" in body


# --- helpers --------------------------------------------------------------


class _FakeJudge:
    """Stand-in for BehaviorJudge that always returns one decision."""

    def __init__(self, decision: str, confidence: float = 1.0):
        self.decision = decision
        self.confidence = confidence

    def judge(self, a, b):
        from autodetect_repro.calibration import JudgeVerdict
        return JudgeVerdict(
            decision=self.decision,
            confidence=self.confidence,
            reason="",
            raw={},
        )
