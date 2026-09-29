import csv
from pathlib import Path

from autodetect_repro.checkpoint import CheckpointStore
from autodetect_repro.config import load_config
from autodetect_repro.reporting import compute_metrics, read_iteration_rows, write_report
from autodetect_repro.types import SampleRecord


def config():
    return load_config(
        Path("configs/siliconflow_smallscale.yaml"),
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )


def record(
    identifier,
    *,
    group="format",
    subtask="text format",
    point="list format",
    phase="seed",
    iteration=0,
    score=3.0,
    status="valid",
):
    valid = status == "valid"
    return SampleRecord(
        sample_id=identifier,
        group=group,
        subtask=subtask,
        knowledge_point=point,
        phase=phase,
        iteration=iteration,
        ordinal=0,
        prompt="prompt {}".format(identifier),
        key_point="constraint",
        answer="answer" if valid else "",
        reference_answer="reference" if valid else "",
        comparison="Rating: [[{}]]".format(score) if valid else "",
        score=score if valid else None,
        is_weakness=False,
        status=status,
        error=None if valid else "provider failed",
        request_count=3 if valid else 0,
        prompt_tokens=30 if valid else 0,
        completion_tokens=15 if valid else 0,
        total_tokens=45 if valid else 0,
    )


def fixed_records():
    return [
        record("seed-3", score=3.0),
        record("adaptive-4", phase="adaptive", iteration=1, score=4.0),
        record(
            "adaptive-2",
            group="content",
            subtask="word constraint",
            point="specific words",
            phase="adaptive",
            iteration=1,
            score=2.0,
        ),
        record("failed", phase="adaptive", iteration=2, status="failed"),
    ]


def row_by(metrics, level, key):
    return next(row for row in metrics.rows if row.level == level and row.key == key)


def test_compute_metrics_covers_scopes_boundary_usage_and_missing_samples():
    metrics = compute_metrics(fixed_records(), planned_initial_samples=60)

    assert metrics.total_records == 4
    assert metrics.valid_records == 3
    assert metrics.failed_records == 1
    assert metrics.missing_initial_samples == 56
    assert metrics.request_count == 9
    assert metrics.prompt_tokens == 90
    assert metrics.completion_tokens == 45
    assert metrics.total_tokens == 135

    overall = row_by(metrics, "overall", "all")
    assert overall.valid == 3
    assert overall.weak == 2
    assert overall.isr == 2 / 3
    assert overall.mean_score == 3.0

    format_row = row_by(metrics, "category", "format")
    assert format_row.isr == 0.5
    assert format_row.mean_score == 3.5
    assert row_by(metrics, "point", "content/word constraint/specific words").isr == 1.0
    assert row_by(metrics, "phase", "seed").mean_score == 3.0
    assert row_by(metrics, "phase", "adaptive").mean_score == 3.0


def test_zero_valid_records_produces_none_rates_without_division_error():
    metrics = compute_metrics(
        [record("failed", status="failed")], planned_initial_samples=1
    )

    overall = row_by(metrics, "overall", "all")
    assert overall.valid == 0
    assert overall.isr is None
    assert overall.mean_score is None
    assert metrics.failed_records == 1


def test_write_report_emits_exact_csv_rows_markdown_and_plot(tmp_path):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config())
    for item in fixed_records():
        store.append(item)

    paths = write_report(run_dir)

    assert paths.metrics_csv.is_file()
    assert paths.iterations_csv.is_file()
    assert paths.markdown.is_file()
    assert paths.plot.is_file()
    assert paths.plot.stat().st_size > 0

    with paths.metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    overall = next(row for row in rows if row["level"] == "overall")
    assert overall == {
        "level": "overall",
        "key": "all",
        "total": "4",
        "valid": "3",
        "failed": "1",
        "weak": "2",
        "isr": "0.666667",
        "mean_score": "3.000000",
    }

    iteration_rows = read_iteration_rows(paths.iterations_csv)
    assert [(row.phase, row.iteration, row.mean_score) for row in iteration_rows] == [
        ("seed", 0, 3.0),
        ("adaptive", 1, 3.0),
        ("adaptive", 2, None),
    ]

    report = paths.markdown.read_text(encoding="utf-8")
    assert "deepseek-ai/DeepSeek-V4-Flash" in report
    assert "Qwen/Qwen2.5-7B-Instruct" in report
    assert "3 valid" in report
    assert "1 failed" in report
    assert "did not decrease" in report
    assert "directional reproduction" in report


def test_discovered_point_does_not_reduce_missing_initial_sample_count(tmp_path):
    run_dir = tmp_path / "run"
    store = CheckpointStore(run_dir, config())
    store.append(record("discovered", point="new discovered point"))

    report = write_report(run_dir).markdown.read_text(encoding="utf-8")

    assert "60 missing initial samples" in report
