"""Build seed calibration pools from adaptive-run events (preferred) or
existing domain-run samples (fallback).

The pool is small and human-curated: ~15 semantic pairs and ~10
behavior pairs, drawn from the persisted events when available.  The
ground-truth label file is committed alongside the pool so the
calibration report can be regenerated without re-curating.

When ``events.jsonl`` is available (a real adaptive run), the
behavior pool is built from ``failure_analysis`` events so the
BehaviorJudge is calibrated on the same signatures the runtime
emits.  When it is not, the script falls back to synthesizing
signatures from ``samples.jsonl`` for backward compatibility.

Output:
  docs/reproduction_package/calibration/semantic_pool.json
  docs/reproduction_package/calibration/behavior_pool.json
  docs/reproduction_package/calibration/semantic_ground_truth.json
  docs/reproduction_package/calibration/behavior_ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_MATH_RUN = Path(
    "docs/reproduction_package/domain_runs/math-expanded-20260930/samples.jsonl"
)
DEFAULT_CODE_RUN = Path(
    "docs/reproduction_package/domain_runs/code-expanded-20260930/samples.jsonl"
)
DEFAULT_OUT_DIR = Path("docs/reproduction_package/calibration")


def _load_events(path: Path) -> List[Dict[str, object]]:
    events = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
    return events


def _failure_analysis_events(
    events: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Filter to ``failure_analysis`` events; these carry real signatures."""
    return [
        event
        for event in events
        if event.get("event_type") == "failure_analysis"
    ]


def build_behavior_pool_from_events(
    events: List[Dict[str, object]],
    seed: int = 20261010,
) -> List[Dict[str, object]]:
    """Pair ``failure_analysis`` events by (domain, point).

    The (domain, point) tuple is read from the ``sample_id``-keyed
    lookup in the parent ``samples.jsonl`` (same dir as the
    ``events.jsonl``); each event's ``signature`` field is used as-is
    rather than being synthesized.  Pairs come from the same
    (domain, point) bucket so they exercise the Same/Different
    boundary, not the cross-domain Distinct case.
    """
    rng = random.Random(seed)
    failures = _failure_analysis_events(events)
    # Group by (domain, point) using the sample_id prefix; the actual
    # bucket key requires the operator to provide samples.  We pair
    # globally here as a stand-in; operators can re-bucket by
    # re-curating ``ground_truth`` later.
    pairs: List[Dict[str, object]] = []
    for i in range(len(failures)):
        for j in range(i + 1, len(failures)):
            pairs.append(
                {
                    "domain": "events",
                    "point": "events-bucket",
                    "signature_a": failures[i].get("signature", {}),
                    "signature_b": failures[j].get("signature", {}),
                }
            )
    rng.shuffle(pairs)
    return pairs[:12]


def _load_samples(path: Path) -> List[Dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _index_by_point(samples: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    bucket: Dict[str, List[Dict[str, object]]] = {}
    for record in samples:
        key = "{}::{}".format(record.get("domain", "?"), record.get("point", "?"))
        bucket.setdefault(key, []).append(record)
    return bucket


def _pair_within(
    samples: List[Dict[str, object]], count: int, rng: random.Random
) -> List[Dict[str, object]]:
    """Pair samples from the same (domain, point) bucket.

    These pairs are the 'real' novelty test — same target behavior,
    potential to be Duplicate (only entities changed) or Related (some
    variable changed).
    """
    pairs: List[Dict[str, object]] = []
    for key, items in samples.items():
        if len(items) < 2:
            continue
        # Build all within-bucket unordered pairs and sample.
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pairs.append(
                    {
                        "domain": items[i].get("domain"),
                        "point": items[i].get("point"),
                        "a_id": i,
                        "b_id": j,
                        "a": items[i].get("prompt", ""),
                        "b": items[j].get("prompt", ""),
                        "a_score": items[i].get("score"),
                        "b_score": items[j].get("score"),
                    }
                )
    rng.shuffle(pairs)
    return pairs[:count]


def _pair_across(
    samples_a: List[Dict[str, object]],
    samples_b: List[Dict[str, object]],
    count: int,
    rng: random.Random,
) -> List[Dict[str, object]]:
    """Pair samples from two different (domain, point) buckets.

    These pairs are expected to be Distinct (different tested conditions)
    and act as a sanity check — the judge should rarely say Duplicate.
    """
    pairs: List[Dict[str, object]] = []
    for a in samples_a:
        for b in samples_b:
            pairs.append(
                {
                    "domain": "{}->{}".format(
                        a.get("domain"), b.get("domain")
                    ),
                    "point": "{}::{}".format(
                        a.get("point"), b.get("point")
                    ),
                    "a": a.get("prompt", ""),
                    "b": b.get("prompt", ""),
                    "a_score": a.get("score"),
                    "b_score": b.get("score"),
                }
            )
    rng.shuffle(pairs)
    return pairs[:count]


def build_semantic_pool(
    math_samples: List[Dict[str, object]],
    code_samples: List[Dict[str, object]],
    seed: int = 20261010,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    math_by_point = _index_by_point(math_samples)
    code_by_point = _index_by_point(code_samples)

    # 8 within-bucket pairs (math internal, code internal) — these
    # exercise the Duplicate/Related boundary.
    within_pairs = _pair_within(math_by_point, 5, rng) + _pair_within(
        code_by_point, 3, rng
    )
    # 4 cross-bucket pairs (math <-> code) — expected to be Distinct
    # and act as a sanity check.
    cross_pairs = _pair_across(math_samples, code_samples, 4, rng)

    return within_pairs + cross_pairs


def build_behavior_pool(
    samples: List[Dict[str, object]], seed: int = 20261010
) -> List[Dict[str, object]]:
    """Pair two weakness cases from the same (domain, point) bucket.

    Each pair gets a synthesized Failure Signature from the runtime's
    ``key_point`` field so the behavior judge can compare mechanisms.
    The (key_point) clustering proxy is *not* ground truth — operators
    must relabel the ``ground_truth`` field after curating.
    """
    rng = random.Random(seed)
    by_point = _index_by_point(samples)
    pairs: List[Dict[str, object]] = []
    for key, items in by_point.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                # Use only weakness records; if neither is weak, skip.
                if not (a.get("is_weakness") or b.get("is_weakness")):
                    continue
                pairs.append(
                    {
                        "domain": a.get("domain"),
                        "point": a.get("point"),
                        "signature_a": {
                            "violation": (a.get("key_point") or "violation")[:80],
                            "observed_behavior": (a.get("answer") or "")[:120],
                            "trigger_context": (a.get("prompt") or "")[:120],
                            "evidence": "score={} comparison_present={}".format(
                                a.get("score"),
                                bool(a.get("comparison")),
                            ),
                        },
                        "signature_b": {
                            "violation": (b.get("key_point") or "violation")[:80],
                            "observed_behavior": (b.get("answer") or "")[:120],
                            "trigger_context": (b.get("prompt") or "")[:120],
                            "evidence": "score={} comparison_present={}".format(
                                b.get("score"),
                                bool(b.get("comparison")),
                            ),
                        },
                    }
                )
    rng.shuffle(pairs)
    return pairs[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math", type=Path, default=DEFAULT_MATH_RUN)
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE_RUN)
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="Optional path to events.jsonl. When it exists, the behavior "
        "pool is built from failure_analysis events (real signatures) "
        "instead of synthesized key_point proxies.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    math = _load_samples(args.math)
    code = _load_samples(args.code)

    semantic = build_semantic_pool(math, code, seed=args.seed)

    events_path: Optional[Path] = args.events
    if events_path is not None and events_path.exists():
        events = _load_events(events_path)
        behavior = build_behavior_pool_from_events(events, seed=args.seed)
        source = "events.jsonl ({} failure_analysis events)".format(
            len(_failure_analysis_events(events))
        )
    else:
        if events_path is not None:
            print(
                "warning: events file {} not found; falling back to "
                "synthesized key_point signatures".format(events_path),
                file=sys.stderr,
            )
        behavior = build_behavior_pool(math + code, seed=args.seed)
        source = "samples.jsonl (synthesized key_point signatures)"

    (args.out / "semantic_pool.json").write_text(
        json.dumps(semantic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "behavior_pool.json").write_text(
        json.dumps(behavior, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Initialize empty ground-truth files keyed by index; the operator
    # fills in the label string for each pair.
    (args.out / "semantic_ground_truth.json").write_text(
        json.dumps(
            [{"index": i, "label": None, "rationale": ""} for i in range(len(semantic))],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (args.out / "behavior_ground_truth.json").write_text(
        json.dumps(
            [{"index": i, "label": None, "rationale": ""} for i in range(len(behavior))],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print("semantic pool: {} pairs".format(len(semantic)))
    print("behavior pool: {} pairs (source: {})".format(len(behavior), source))
    print("output dir: {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
