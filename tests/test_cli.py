from pathlib import Path

import pytest
import yaml

from autodetect_repro.checkpoint import CheckpointStore
from autodetect_repro.cli import EXIT_API, EXIT_CONFIG, EXIT_INCOMPLETE, main
from autodetect_repro.config import load_config
from autodetect_repro.runner import RunSummary


FULL_CONFIG = Path("configs/siliconflow_smallscale.yaml")
SMOKE_CONFIG = Path("configs/siliconflow_smoke.yaml")


def complete_summary(valid=60, failed=0):
    return RunSummary(
        scheduled=valid + failed,
        valid=valid,
        failed=failed,
        weak=10,
        requests=180,
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        interrupted=False,
    )


@pytest.mark.parametrize("subcommand", ["run", "resume"])
def test_network_commands_refuse_without_live_before_client_construction(
    monkeypatch, capsys, subcommand
):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "unit-test-secret")
    constructed = []
    monkeypatch.setattr(
        "autodetect_repro.cli.SiliconFlowClient",
        lambda *args, **kwargs: constructed.append((args, kwargs)),
    )
    argv = [subcommand, "--config", str(FULL_CONFIG)]
    if subcommand == "resume":
        argv += ["--run-dir", "/data/liuxingtong/outputs/autodetect-smallscale/run"]

    exit_code = main(argv)

    assert exit_code == EXIT_CONFIG
    assert constructed == []
    assert "--live" in capsys.readouterr().err


def test_report_never_requires_api_key(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    config = load_config(
        FULL_CONFIG, {"SILICONFLOW_API_KEY": "unit-test-secret"}
    )
    CheckpointStore(run_dir, config)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    exit_code = main(["report", "--run-dir", str(run_dir)])

    assert exit_code == 0
    assert (run_dir / "report.md").is_file()


def test_smoke_config_schedules_one_point_and_two_samples():
    config = load_config(
        SMOKE_CONFIG, {"SILICONFLOW_API_KEY": "unit-test-secret"}
    )

    assert config.search.max_initial_points == 1
    assert config.search.seed_questions_per_point == 1
    assert config.search.adaptive_iterations_per_point == 1
    assert config.limits.max_total_samples == 2


def test_dry_run_prints_models_counts_and_path_but_not_secret(monkeypatch, capsys):
    secret = "unit-test-secret"
    monkeypatch.setenv("SILICONFLOW_API_KEY", secret)

    exit_code = main(
        [
            "run",
            "--config",
            str(FULL_CONFIG),
            "--dry-run",
            "--run-dir",
            "/data/liuxingtong/outputs/autodetect-smallscale/dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "deepseek-ai/DeepSeek-V4-Flash" in output
    assert "Qwen/Qwen2.5-7B-Instruct" in output
    assert "60" in output
    assert "80" in output
    assert "/data/liuxingtong/outputs/autodetect-smallscale/dry-run" in output
    assert secret not in output


def test_invalid_output_path_returns_config_exit_code(tmp_path, monkeypatch, capsys):
    raw = yaml.safe_load(FULL_CONFIG.read_text())
    raw["output_root"] = "/home/liuxingtong/results"
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("SILICONFLOW_API_KEY", "unit-test-secret")

    exit_code = main(["run", "--config", str(path), "--dry-run"])

    assert exit_code == EXIT_CONFIG
    assert "/data/liuxingtong" in capsys.readouterr().err


def test_api_failure_and_incomplete_run_have_distinct_exit_codes(
    monkeypatch, capsys
):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "unit-test-secret")

    from autodetect_repro.client import ApiError

    monkeypatch.setattr(
        "autodetect_repro.cli.execute_run",
        lambda config, run_dir: (_ for _ in ()).throw(
            ApiError("provider failure for unit-test-secret")
        ),
    )
    api_exit = main(
        [
            "run",
            "--config",
            str(FULL_CONFIG),
            "--run-dir",
            "/data/liuxingtong/outputs/autodetect-smallscale/api-error",
            "--live",
        ]
    )
    api_error = capsys.readouterr().err

    monkeypatch.setattr(
        "autodetect_repro.cli.execute_run",
        lambda config, run_dir: complete_summary(valid=40, failed=20),
    )
    incomplete_exit = main(
        [
            "run",
            "--config",
            str(FULL_CONFIG),
            "--run-dir",
            "/data/liuxingtong/outputs/autodetect-smallscale/incomplete",
            "--live",
        ]
    )

    assert api_exit == EXIT_API
    assert "unit-test-secret" not in api_error
    assert incomplete_exit == EXIT_INCOMPLETE
