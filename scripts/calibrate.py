"""Calibrate the LLM-as-judge against human ground truth.

Inputs (default paths under ``docs/reproduction_package/calibration/``):
  * behavior_pool.json       — list of {domain, point, signature_a, signature_b}
  * behavior_ground_truth.json — list of {index, label, rationale} with label in
                                  {same, different, uncertain}.  Pairs whose
                                  label is null are skipped.
  * semantic_pool.json / semantic_ground_truth.json (optional, same shape)

Outputs:
  * report.md — Cohen kappa, accuracy, confusion matrix, per-pair verdicts.

The judge itself is ``autodetect_repro.calibration.BehaviorJudge`` /
``SemanticJudge``.  This script is the only place those classes are
instantiated at runtime, so the calibration module is wired into a real
workflow (not just a definition with no consumer).

Example:
    python scripts/calibrate.py
    python scripts/calibrate.py --pool semantic_pool.json \\
        --ground-truth semantic_ground_truth.json --out report-semantic.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# Make the package importable when run as a script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodetect_repro.calibration import BehaviorJudge, SemanticJudge
from autodetect_repro.client import ApiError, SiliconFlowClient


DEFAULT_DIR = Path("docs/reproduction_package/calibration")


@dataclass(frozen=True)
class PairVerdict:
    index: int
    truth: str
    predicted: str
    confidence: float
    reason: str


def _load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError("calibration input not found: {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pairs(
    pool: Sequence[dict], ground_truth: Sequence[dict]
) -> List[Tuple[int, dict, str]]:
    """Merge pool + ground_truth into (index, pool_entry, label) tuples.

    Ground-truth rows whose label is null or missing are filtered out
    so an operator can fill labels in incrementally and re-run
    calibration on whatever is finished.
    """
    by_index = {row.get("index"): row for row in ground_truth}
    pairs: List[Tuple[int, dict, str]] = []
    for idx, entry in enumerate(pool):
        truth_row = by_index.get(idx)
        if truth_row is None:
            continue
        label = (truth_row.get("label") or "").strip().lower()
        if not label:
            continue
        pairs.append((idx, entry, label))
    return pairs


def _judge_one(
    judge: object, entry: dict
) -> Tuple[str, float, str]:
    """Call judge(signature_a, signature_b) and unpack verdict."""
    verdict = judge.judge(entry["signature_a"], entry["signature_b"])
    return verdict.decision, verdict.confidence, verdict.reason


def _cohen_kappa(predicted: Sequence[str], truth: Sequence[str]) -> float:
    """Cohen kappa with chance agreement from observed marginals.

    Returns 0.0 when there is only one unique class (kappa is
    undefined).  Returns 1.0 when predicted == truth on every pair.
    """
    if len(predicted) != len(truth):
        raise ValueError("predicted/truth length mismatch")
    if not predicted:
        return 0.0
    n = len(predicted)
    po = sum(1 for p, t in zip(predicted, truth) if p == t) / n
    labels = set(predicted) | set(truth)
    pe = 0.0
    for label in labels:
        pe += (sum(1 for x in predicted if x == label) / n) * (
            sum(1 for x in truth if x == label) / n
        )
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _confusion_matrix(
    predicted: Sequence[str], truth: Sequence[str], labels: Sequence[str]
) -> Dict[Tuple[str, str], int]:
    """Return a {truth, predicted} -> count dict for the given label set."""
    counts: Dict[Tuple[str, str], int] = {(t, p): 0 for t in labels for p in labels}
    for t, p in zip(truth, predicted):
        if t in labels and p in labels:
            counts[(t, p)] += 1
    return counts


def _render_table(
    counts: Dict[Tuple[str, str], int], labels: Sequence[str]
) -> str:
    """Render the confusion matrix as a markdown table."""
    header = "| truth \\ pred | " + " | ".join(labels) + " | total |"
    sep = "| --- | " + " | ".join("---" for _ in labels) + " | --- |"
    rows = [header, sep]
    for t in labels:
        cells = [str(counts[(t, p)]) for p in labels]
        total = sum(counts[(t, p)] for p in labels)
        rows.append("| {} | {} | {} |".format(t, " | ".join(cells), total))
    return "\n".join(rows)


def _build_client(env_name: str) -> Optional[SiliconFlowClient]:
    """Build a client from env, returning None if the key is missing."""
    api_key = os.environ.get(env_name)
    if not api_key:
        return None
    base_url = os.environ.get(
        "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
    )
    return SiliconFlowClient(
        {"base_url": base_url, "api_key": api_key, "model": "judge"},
        model="judge",
        max_attempts=1,
        timeout_seconds=60,
    )


def _run_calibration(
    *,
    kind: str,
    pool_path: Path,
    ground_truth_path: Path,
    judge_factory,
    out_path: Path,
) -> int:
    pool = _load_json(pool_path)
    ground_truth = _load_json(ground_truth_path)
    if not isinstance(pool, list) or not isinstance(ground_truth, list):
        raise ValueError("pool and ground truth must be JSON lists")
    pairs = _load_pairs(pool, ground_truth)
    if not pairs:
        out_path.write_text(
            "# {} calibration report\n\n"
            "No labeled pairs found in {}.  Fill in at least one "
            "ground-truth row and re-run.\n".format(kind, ground_truth_path),
            encoding="utf-8",
        )
        print(
            "calibration: 0 labeled pairs, wrote stub to {}".format(out_path)
        )
        return 0

    client = _build_client("SILICONFLOW_API_KEY")
    if client is None:
        out_path.write_text(
            "# {} calibration report\n\n"
            "No `SILICONFLOW_API_KEY` in env, so the judge could not "
            "run.  Set the env var and re-run to populate this "
            "report.\n\nLabeled pairs available: {}.\n".format(
                kind, len(pairs)
            ),
            encoding="utf-8",
        )
        print(
            "calibration: SILICONFLOW_API_KEY missing, wrote stub to {}"
            .format(out_path)
        )
        return 0

    judge = judge_factory(client)
    verdicts: List[PairVerdict] = []
    for idx, entry, label in pairs:
        try:
            decision, confidence, reason = _judge_one(judge, entry)
        except (ApiError, RuntimeError, ValueError) as exc:
            print(
                "pair {}: judge failed: {}".format(idx, exc),
                file=sys.stderr,
            )
            decision, confidence, reason = "uncertain", 0.0, str(exc)
        verdicts.append(
            PairVerdict(
                index=idx,
                truth=label,
                predicted=decision,
                confidence=confidence,
                reason=reason,
            )
        )

    predicted = [v.predicted for v in verdicts]
    truth = [v.truth for v in verdicts]
    labels = sorted(set(predicted) | set(truth))
    kappa = _cohen_kappa(predicted, truth)
    accuracy = sum(1 for v in verdicts if v.truth == v.predicted) / len(verdicts)
    counts = _confusion_matrix(predicted, truth, labels)

    # Confidence-bucketed agreement (agreement at conf >= 0.5 / 0.8).
    def _agreement_at(threshold: float) -> Tuple[int, int]:
        keep = [v for v in verdicts if v.confidence >= threshold]
        if not keep:
            return 0, 0
        agree = sum(1 for v in keep if v.truth == v.predicted)
        return agree, len(keep)

    agree_50, total_50 = _agreement_at(0.5)
    agree_80, total_80 = _agreement_at(0.8)

    # Per-label confidence so the operator can see which label the
    # judge is most uncertain about.
    per_label = {
        label: round(
            sum(v.confidence for v in verdicts if v.predicted == label)
            / max(1, sum(1 for v in verdicts if v.predicted == label)),
            3,
        )
        for label in labels
    }

    lines = [
        "# {} calibration report".format(kind.capitalize()),
        "",
        "Pairs evaluated: **{}**".format(len(verdicts)),
        "Pool source: `{}`".format(pool_path),
        "Ground truth: `{}`".format(ground_truth_path),
        "",
        "## Headline",
        "",
        "| metric | value |",
        "| --- | --- |",
        "| accuracy | {:.3f} |".format(accuracy),
        "| Cohen kappa | {:.3f} |".format(kappa),
        "| agreement @ conf>=0.5 | {} / {} ({:.2f}) |".format(
            agree_50,
            total_50,
            agree_50 / total_50 if total_50 else 0.0,
        ),
        "| agreement @ conf>=0.8 | {} / {} ({:.2f}) |".format(
            agree_80,
            total_80,
            agree_80 / total_80 if total_80 else 0.0,
        ),
        "",
        "## Confusion matrix",
        "",
        _render_table(counts, labels),
        "",
        "## Per-label confidence",
        "",
        "| label | mean judge confidence | count |",
        "| --- | --- | --- |",
    ]
    for label in labels:
        n = sum(1 for v in verdicts if v.predicted == label)
        lines.append("| {} | {} | {} |".format(label, per_label[label], n))
    lines.extend(
        [
            "",
            "## Per-pair verdicts",
            "",
            "| index | truth | predicted | conf | reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for v in verdicts:
        lines.append(
            "| {} | {} | {} | {:.2f} | {} |".format(
                v.index, v.truth, v.predicted, v.confidence, v.reason[:80]
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "calibration: {} pairs, kappa={:.3f}, accuracy={:.3f}, report={}"
        .format(len(verdicts), kappa, accuracy, out_path)
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--kind",
        choices=("behavior", "semantic"),
        default="behavior",
        help="Which pool/ground-truth pair to run.",
    )
    args = parser.parse_args()

    if args.kind == "behavior":
        return _run_calibration(
            kind="behavior",
            pool_path=args.dir / "behavior_pool.json",
            ground_truth_path=args.dir / "behavior_ground_truth.json",
            judge_factory=lambda client: BehaviorJudge(client),
            out_path=args.dir / "report.md",
        )
    return _run_calibration(
        kind="semantic",
        pool_path=args.dir / "semantic_pool.json",
        ground_truth_path=args.dir / "semantic_ground_truth.json",
        judge_factory=lambda client: SemanticJudge(client),
        out_path=args.dir / "report.md",
    )


if __name__ == "__main__":
    raise SystemExit(main())
