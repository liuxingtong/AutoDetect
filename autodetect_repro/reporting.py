"""Offline metrics and artifacts for an AutoDetect run directory."""

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .types import SampleRecord


class ReportingError(RuntimeError):
    """Raised when persisted run artifacts cannot be reported safely."""


@dataclass(frozen=True)
class MetricRow:
    level: str
    key: str
    total: int
    valid: int
    failed: int
    weak: int
    isr: Optional[float]
    mean_score: Optional[float]


@dataclass(frozen=True)
class IterationRow:
    phase: str
    iteration: int
    total: int
    valid: int
    weak: int
    isr: Optional[float]
    mean_score: Optional[float]


@dataclass(frozen=True)
class Metrics:
    rows: Tuple[MetricRow, ...]
    iterations: Tuple[IterationRow, ...]
    total_records: int
    valid_records: int
    failed_records: int
    missing_initial_samples: int
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ReportPaths:
    metrics_csv: Path
    iterations_csv: Path
    markdown: Path
    plot: Path


def _metric_row(level: str, key: str, records: Sequence[SampleRecord]) -> MetricRow:
    valid = [
        record
        for record in records
        if record.status == "valid" and record.score is not None
    ]
    weak = sum(1 for record in valid if record.score < 4)
    return MetricRow(
        level=level,
        key=key,
        total=len(records),
        valid=len(valid),
        failed=len(records) - len(valid),
        weak=weak,
        isr=(weak / len(valid)) if valid else None,
        mean_score=mean(record.score for record in valid) if valid else None,
    )


def _grouped_rows(
    level: str,
    records: Sequence[SampleRecord],
    key_func,
) -> Iterable[MetricRow]:
    grouped: Dict[str, List[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(key_func(record), []).append(record)
    for key in sorted(grouped):
        yield _metric_row(level, key, grouped[key])


def compute_metrics(
    records: Sequence[SampleRecord],
    *,
    planned_initial_samples: int,
    completed_initial_samples: Optional[int] = None,
) -> Metrics:
    record_list = list(records)
    rows = [_metric_row("overall", "all", record_list)]
    rows.extend(_grouped_rows("category", record_list, lambda item: item.group))
    rows.extend(
        _grouped_rows(
            "subtask",
            record_list,
            lambda item: "{}/{}".format(item.group, item.subtask),
        )
    )
    rows.extend(
        _grouped_rows(
            "point",
            record_list,
            lambda item: "{}/{}/{}".format(
                item.group, item.subtask, item.knowledge_point
            ),
        )
    )
    rows.extend(_grouped_rows("phase", record_list, lambda item: item.phase))

    iteration_groups: Dict[Tuple[str, int], List[SampleRecord]] = {}
    for record in record_list:
        iteration_groups.setdefault((record.phase, record.iteration), []).append(record)
    phase_rank = {"seed": 0, "adaptive": 1}
    iterations = []
    for (phase, iteration), group in sorted(
        iteration_groups.items(),
        key=lambda item: (phase_rank.get(item[0][0], 2), item[0][1]),
    ):
        metric = _metric_row("iteration", "{}:{}".format(phase, iteration), group)
        iterations.append(
            IterationRow(
                phase,
                iteration,
                metric.total,
                metric.valid,
                metric.weak,
                metric.isr,
                metric.mean_score,
            )
        )

    valid_count = sum(1 for record in record_list if record.status == "valid")
    initial_count = (
        min(len(record_list), planned_initial_samples)
        if completed_initial_samples is None
        else min(completed_initial_samples, planned_initial_samples)
    )
    return Metrics(
        rows=tuple(rows),
        iterations=tuple(iterations),
        total_records=len(record_list),
        valid_records=valid_count,
        failed_records=len(record_list) - valid_count,
        missing_initial_samples=max(0, planned_initial_samples - initial_count),
        request_count=sum(record.request_count for record in record_list),
        prompt_tokens=sum(record.prompt_tokens for record in record_list),
        completion_tokens=sum(record.completion_tokens for record in record_list),
        total_tokens=sum(record.total_tokens for record in record_list),
    )


def _number(value: Optional[float]) -> str:
    return "" if value is None else "{:.6f}".format(value)


def _write_metrics_csv(path: Path, rows: Sequence[MetricRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "key",
                "total",
                "valid",
                "failed",
                "weak",
                "isr",
                "mean_score",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.level,
                    row.key,
                    row.total,
                    row.valid,
                    row.failed,
                    row.weak,
                    _number(row.isr),
                    _number(row.mean_score),
                ]
            )


def _write_iterations_csv(path: Path, rows: Sequence[IterationRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["phase", "iteration", "total", "valid", "weak", "isr", "mean_score"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.phase,
                    row.iteration,
                    row.total,
                    row.valid,
                    row.weak,
                    _number(row.isr),
                    _number(row.mean_score),
                ]
            )


def read_iteration_rows(path: Path) -> List[IterationRow]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                IterationRow(
                    phase=raw["phase"],
                    iteration=int(raw["iteration"]),
                    total=int(raw["total"]),
                    valid=int(raw["valid"]),
                    weak=int(raw["weak"]),
                    isr=float(raw["isr"]) if raw["isr"] else None,
                    mean_score=(
                        float(raw["mean_score"]) if raw["mean_score"] else None
                    ),
                )
            )
    return rows


def _plot_iterations(iterations_csv: Path, output_path: Path) -> None:
    rows = read_iteration_rows(iterations_csv)
    labels = ["{} {}".format(row.phase, row.iteration) for row in rows]
    positions = list(range(len(rows)))
    values = [row.mean_score if row.mean_score is not None else float("nan") for row in rows]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    if rows:
        axis.plot(positions, values, marker="o", linewidth=2)
        axis.set_xticks(positions, labels, rotation=30, ha="right")
    axis.set_ylim(1, 10)
    axis.set_ylabel("Mean judge score (lower exposes more weakness)")
    axis.set_title("AutoDetect score by search iteration")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _load_manifest(run_dir: Path) -> Dict[str, object]:
    try:
        value = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportingError("cannot read manifest") from exc
    if not isinstance(value, dict):
        raise ReportingError("manifest root must be an object")
    return value


def _load_records(run_dir: Path) -> List[SampleRecord]:
    records = []
    try:
        lines = (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReportingError("cannot read samples") from exc
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(SampleRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ReportingError("invalid sample at line {}".format(index)) from exc
    return records


def _planned_initial_samples(manifest: Dict[str, object]) -> int:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ReportingError("manifest has no public config")
    search = config.get("search")
    if not isinstance(search, dict):
        raise ReportingError("manifest has no search config")
    return int(search["max_initial_points"]) * (
        int(search["seed_questions_per_point"])
        + int(search["adaptive_iterations_per_point"])
    )


def _initial_point_keys(manifest: Dict[str, object]) -> set:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ReportingError("manifest has no public config")
    subtasks = config.get("subtasks")
    if not isinstance(subtasks, list):
        raise ReportingError("manifest has no subtask config")
    keys = set()
    for subtask in subtasks:
        if not isinstance(subtask, dict):
            raise ReportingError("manifest subtask is invalid")
        for point in subtask.get("knowledge_points", []):
            keys.add((subtask.get("group"), subtask.get("name"), point))
    return keys


def write_report(run_dir: Path) -> ReportPaths:
    run_dir = Path(run_dir)
    manifest = _load_manifest(run_dir)
    records = _load_records(run_dir)
    planned = _planned_initial_samples(manifest)
    initial_keys = _initial_point_keys(manifest)
    completed_initial = sum(
        1
        for record in records
        if (record.group, record.subtask, record.knowledge_point) in initial_keys
    )
    metrics = compute_metrics(
        records,
        planned_initial_samples=planned,
        completed_initial_samples=completed_initial,
    )

    paths = ReportPaths(
        metrics_csv=run_dir / "metrics.csv",
        iterations_csv=run_dir / "iterations.csv",
        markdown=run_dir / "report.md",
        plot=run_dir / "score_by_iteration.png",
    )
    _write_metrics_csv(paths.metrics_csv, metrics.rows)
    _write_iterations_csv(paths.iterations_csv, metrics.iterations)
    _plot_iterations(paths.iterations_csv, paths.plot)

    public_config = manifest["config"]
    models = public_config["models"]
    overall = next(row for row in metrics.rows if row.level == "overall")
    seed = _phase_row(metrics, "seed")
    adaptive = _phase_row(metrics, "adaptive")
    decreased = (
        seed is not None
        and adaptive is not None
        and seed.mean_score is not None
        and adaptive.mean_score is not None
        and adaptive.mean_score < seed.mean_score
    )
    trend_text = "decreased" if decreased else "did not decrease"
    fraction = (planned / 555.0) * 100
    branch_status = _branch_status(run_dir)
    promotion_decisions = _promotion_decisions(run_dir)
    diversity = _diversity_metrics(run_dir)
    search_decisions_md = _search_decisions(run_dir)
    candidate_filters_md = _candidate_filters(run_dir)
    behavior_clusters_md = _behavior_clusters(run_dir)
    report = """# AutoDetect Small-Scale Reproduction

## Scope

- Agent/reference/judge substitution: `{agent}`
- Target model: `{target}`
- Planned initial volume: {planned}/555 ({fraction:.1f}%) of the released instruction-following search volume
- This is a directional reproduction, not an exact numerical replication.

## Data Quality

- {valid} valid samples
- {failed} failed samples
- {missing} missing initial samples

## Result

- ISR: {isr}
- Mean score: {mean_score}
- Adaptive mean score {trend_text} relative to the seed phase.

## Behavioral Diversity

{behavioral_diversity}

## Branch & Promotion

{branch_summary}

{promotion_summary}

## Search & Filtering

{search_decisions}

{candidate_filters}

## Behavior Clusters

{behavior_clusters}

## Recorded Evaluation Usage

- Requests: {requests}
- Prompt tokens: {prompt_tokens}
- Completion tokens: {completion_tokens}
- Total tokens: {total_tokens}
""".format(
        agent=models["agent_model"],
        target=models["target_model"],
        planned=planned,
        fraction=fraction,
        valid=metrics.valid_records,
        failed=metrics.failed_records,
        missing=metrics.missing_initial_samples,
        isr="N/A" if overall.isr is None else "{:.2%}".format(overall.isr),
        mean_score=(
            "N/A" if overall.mean_score is None else "{:.2f}".format(overall.mean_score)
        ),
        trend_text=trend_text,
        behavioral_diversity=diversity,
        branch_summary=branch_status,
        promotion_summary=promotion_decisions,
        search_decisions=search_decisions_md,
        candidate_filters=candidate_filters_md,
        behavior_clusters=behavior_clusters_md,
        requests=metrics.request_count,
        prompt_tokens=metrics.prompt_tokens,
        completion_tokens=metrics.completion_tokens,
        total_tokens=metrics.total_tokens,
    )
    paths.markdown.write_text(report, encoding="utf-8")
    return paths


def _branch_status(run_dir: Path) -> str:
    """Return a markdown summary of branch_update events from events.jsonl.

    Shows branch_id, status, and supporting sample count for every
    branch_update event.  Returns "(no branch events)" if the file is
    absent or empty.
    """
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return "_(no events.jsonl)_"
    rows = []
    latest_by_branch: Dict[str, Dict[str, Any]] = {}
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "branch_update":
                continue
            latest_by_branch[event["branch_id"]] = event
    if not latest_by_branch:
        return "_(no branch_update events)_"
    lines = ["| branch_id | status | supporting_samples |",
             "| --- | --- | --- |"]
    for branch_id, event in latest_by_branch.items():
        samples = ",".join(event.get("supporting_samples", [])) or "—"
        lines.append(
            "| `{branch}` | {status} | {samples} |".format(
                branch=branch_id, status=event.get("status", "?"), samples=samples
            )
        )
    return "\n".join(lines)


def _promotion_decisions(run_dir: Path) -> str:
    """Return a markdown summary of promotion_proposal events."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return "_(no events.jsonl)_"
    decisions = []
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") != "promotion_proposal":
                continue
            decisions.append(event)
    if not decisions:
        return "_(no promotion_proposal events)_"
    lines = ["| branch_id | decision | knowledge_point | reason |",
             "| --- | --- | --- | --- |"]
    for event in decisions:
        lines.append(
            "| `{branch}` | {decision} | {kp} | {reason} |".format(
                branch=event.get("branch_id", "?"),
                decision=event.get("decision", "?"),
                kp=event.get("knowledge_point", "?"),
                reason=event.get("reason", "?"),
            )
        )
    return "\n".join(lines)


def _diversity_metrics(run_dir: Path) -> str:
    """Return a markdown summary of the latest diversity_metrics event."""
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return "_(no events.jsonl)_"
    latest = None
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "diversity_metrics":
                latest = event
    if latest is None:
        return "_(no diversity_metrics events)_"
    return (
        "- Behavioral richness: **{richness}**\n"
        "- Unique question gain: **{gain}**\n"
        "- Discovery rate: **{rate:.2%}**\n"
        "- Target calls (until report): **{calls}**"
    ).format(
        richness=latest.get("richness", 0),
        gain=latest.get("gain", 0),
        rate=latest.get("discovery_rate", 0.0),
        calls=latest.get("target_calls", 0),
    )


def _phase_row(metrics: Metrics, phase: str) -> Optional[IterationRow]:
    return next(
        (row for row in metrics.iterations if row.phase == phase),
        None,
    )


# ---------------------------------------------------------------------------
# Summary helpers for events that were previously emitted but never
# surfaced in the report.  These are pure read-only filters over
# events.jsonl so they degrade gracefully when the file is absent
# (which is the case for any run driven by domain_runner.py).
# ---------------------------------------------------------------------------


def _load_events(run_dir: Path) -> List[Dict[str, Any]]:
    """Read every JSON object from ``events.jsonl``; return [] on miss."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                out.append(event)
    return out


def _search_decisions(run_dir: Path) -> str:
    """Markdown table of every ``search_decision`` event."""
    events = [
        e for e in _load_events(run_dir)
        if e.get("event_type") == "search_decision"
    ]
    if not events:
        return "_(no search_decision events)_"
    lines = [
        "| knowledge_point | strategy | budget (c/t) | iteration | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for event in events:
        lines.append(
            "| `{kp}` | {strategy} | {cb} / {tb} | {iter} | {reason} |".format(
                kp=event.get("knowledge_point", "?"),
                strategy=event.get("strategy", "?"),
                cb=event.get("candidate_budget", "?"),
                tb=event.get("test_budget", "?"),
                iter=event.get("iteration", "?"),
                reason=(event.get("reason", "") or "")[:80],
            )
        )
    return "\n".join(lines)


def _candidate_filters(run_dir: Path) -> str:
    """Markdown table of every ``candidate_filter`` event."""
    events = [
        e for e in _load_events(run_dir)
        if e.get("event_type") == "candidate_filter"
    ]
    if not events:
        return "_(no candidate_filter events)_"
    lines = [
        "| iteration | in | selected | rejected |",
        "| --- | --- | --- | --- |",
    ]
    for event in events:
        candidates = event.get("candidates", []) or []
        decisions = event.get("decisions", []) or []
        selected = sum(1 for d in decisions if d.get("label") != "duplicate")
        lines.append(
            "| {iter} | {n} | {s} | {r} |".format(
                iter=event.get("iteration", "?"),
                n=len(candidates),
                s=selected,
                r=len(decisions) - selected,
            )
        )
    return "\n".join(lines)


def _behavior_clusters(run_dir: Path) -> str:
    """Markdown table of the most recent label per sample_id."""
    events = [
        e for e in _load_events(run_dir)
        if e.get("event_type") == "behavior_cluster_update"
    ]
    if not events:
        return "_(no behavior_cluster_update events)_"
    latest: Dict[str, str] = {}
    for event in events:
        sample_id = event.get("sample_id")
        if isinstance(sample_id, str):
            latest[sample_id] = event.get("label", "?")
    if not latest:
        return "_(no behavior_cluster_update events)_"
    # Distribution
    counts = Counter(latest.values())
    lines = [
        "| label | samples |",
        "| --- | --- |",
    ]
    for label, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append("| {} | {} |".format(label, n))
    return "\n".join(lines)
