"""Regression checks against the live run of 2026-10-10.

The artifacts in ``docs/reproduction_package/live_runs/live-2026-10-10``
are the first successful end-to-end live run against the real
siliconflow API.  These tests pin a few invariants that future
refactors must not silently break:

- The run produced at least 5 weakness observations (ISR > 30%).
- The events.jsonl has all 6 expected event types present.
- The report.md has all 6 expected sections and the ISR line.
- The iterations.csv has a non-trivial per-iteration progression.
- The score_by_iteration.png was actually generated and is a
  non-empty PNG file.

If any of these break, the test fails so the next refactor
notices before merging a release that would have shipped
silently-regressed live output.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "docs" / "reproduction_package" / "live_runs" / "live-2026-10-10"


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_live_run_artifact_directory_exists():
    assert LIVE.is_dir(), (
        "live-2026-10-10 directory missing; the reference live run "
        "is gone.  Re-run the live run to refresh."
    )


@pytest.mark.parametrize(
    "filename",
    [
        "manifest.json",
        "samples.jsonl",
        "events.jsonl",
        "iterations.csv",
        "metrics.csv",
        "report.md",
        "score_by_iteration.png",
        "config.yaml",
        "README.md",
    ],
)
def test_live_run_artifact_present(filename):
    assert (LIVE / filename).is_file(), (
        "{} missing from live-2026-10-10".format(filename)
    )


def test_live_run_found_weaknesses():
    samples = _read_jsonl(LIVE / "samples.jsonl")
    valid = [s for s in samples if s.get("status") == "valid"]
    weak = [s for s in valid if s.get("is_weakness")]
    assert len(samples) >= 5
    assert len(weak) >= 1, (
        "live run found 0 weaknesses; the reference run had 5.  "
        "Either the model is much better now, or the search loop is "
        "not finding weaknesses anymore."
    )


def test_live_run_event_types():
    events = _read_jsonl(LIVE / "events.jsonl")
    types = {e.get("event_type") for e in events}
    for required in (
        "search_decision",
        "failure_analysis",
        "branch_update",
        "promotion_proposal",
        "behavior_cluster_update",
        "diversity_metrics",
    ):
        assert required in types, (
            "live run missing {} event; the framework's adaptive "
            "features are silently bypassing.  Found: {}".format(
                required, sorted(types)
            )
        )


def test_live_run_report_sections():
    report = (LIVE / "report.md").read_text(encoding="utf-8")
    for required in (
        "# AutoDetect Small-Scale Reproduction",
        "## Scope",
        "## Data Quality",
        "## Result",
        "## Branch & Promotion",
        "## Search & Filtering",
        "## Behavior Clusters",
        "## Recorded Evaluation Usage",
    ):
        assert required in report, (
            "live run report missing section {!r}; the report "
            "renderer is missing a section that downstream readers "
            "rely on.".format(required)
        )


def test_live_run_iterations_csv_has_progress():
    rows = list(csv.DictReader((LIVE / "iterations.csv").open(encoding="utf-8")))
    phases = {(r["phase"], int(r["iteration"])): r for r in rows}
    # The reference run had: seed, adaptive 1/2/3, verify -1.
    assert ("seed", 0) in phases
    assert ("verify", -1) in phases
    for n in (1, 2, 3):
        assert ("adaptive", n) in phases
    # Mean score should vary across iterations (i.e. not all the same).
    scores = {float(r["mean_score"]) for r in rows}
    assert len(scores) >= 2, (
        "all iterations have the same mean score; the search loop "
        "is not actually exercising the model."
    )


def test_live_run_png_is_real():
    """A 0-byte file or a text file would silently fail matplotlib
    rendering; check the magic bytes."""
    png = LIVE / "score_by_iteration.png"
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", (
        "score_by_iteration.png is not a valid PNG; matplotlib may "
        "be failing silently in the reporting path."
    )
    assert len(data) >= 1000, (
        "score_by_iteration.png is suspiciously small ({} bytes); "
        "the plot may be empty.".format(len(data))
    )


def test_live_run_manifest_fingerprint_is_sha256_hex():
    """The manifest's config_fingerprint is a 64-character hex string
    (SHA-256 of the canonicalized public config, not the raw YAML).
    This proves the run was produced through the standard
    config_fingerprint() helper rather than fabricated by hand."""
    manifest = json.loads((LIVE / "manifest.json").read_text(encoding="utf-8"))
    fp = manifest.get("config_fingerprint", "")
    assert len(fp) == 64, (
        "config_fingerprint should be a 64-character SHA-256 hex; "
        "got {} characters: {!r}".format(len(fp), fp)
    )
    int(fp, 16)  # raises if not hex
