"""Command-line entry points for safe reproduction runs."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .checkpoint import CheckpointError
from .client import ApiError, SiliconFlowClient
from .config import ConfigError, RunConfig, load_config
from .reporting import ReportingError, write_report
from .roles import Assessor, Evaluator, Examiner, Questioner, RoleError
from .runner import ExperimentRunner, RunSummary


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_API = 3
EXIT_INCOMPLETE = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m autodetect_repro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "resume"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--run-dir", type=Path, required=name == "resume")
        command.add_argument("--live", action="store_true")
        command.add_argument("--dry-run", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("--run-dir", required=True, type=Path)
    return parser


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _run_dir(config: RunConfig, requested: Optional[Path]) -> Path:
    if requested is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        requested = config.output_root / run_id
    path = requested.expanduser()
    if not path.is_absolute() or not _is_below(path, Path("/data/liuxingtong")):
        raise ConfigError("run directory must be below /data/liuxingtong")
    return path


def _client(config: RunConfig, model: str) -> SiliconFlowClient:
    return SiliconFlowClient(
        config.models,
        model=model,
        max_attempts=config.limits.max_attempts_per_request,
        timeout_seconds=config.limits.request_timeout_seconds,
    )


def execute_run(config: RunConfig, run_dir: Path) -> RunSummary:
    agent = _client(config, config.models.agent_model)
    target = _client(config, config.models.target_model)
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=Examiner(config, agent),
        questioner=Questioner(config, agent),
        evaluator=Evaluator(
            target, agent, max_tokens=config.limits.max_output_tokens
        ),
        assessor=Assessor(config, agent),
    )
    summary = runner.run()
    write_report(run_dir)
    return summary


def _redact(message: str, config: Optional[RunConfig]) -> str:
    if config is not None and config.models.api_key:
        return message.replace(config.models.api_key, "[REDACTED]")
    return message


def _dry_run(config: RunConfig, run_dir: Path) -> None:
    planned = config.search.max_initial_points * (
        config.search.seed_questions_per_point
        + config.search.adaptive_iterations_per_point
    )
    print(
        json.dumps(
            {
                "agent_model": config.models.agent_model,
                "target_model": config.models.target_model,
                "planned_initial_samples": planned,
                "max_total_samples": config.limits.max_total_samples,
                "run_dir": str(run_dir),
                "network_calls": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _summary_json(summary: RunSummary, run_dir: Path) -> str:
    return json.dumps(
        {
            "run_dir": str(run_dir),
            "scheduled": summary.scheduled,
            "valid": summary.valid,
            "failed": summary.failed,
            "weak": summary.weak,
            "requests": summary.requests,
            "total_tokens": summary.total_tokens,
            "interrupted": summary.interrupted,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    if args.command == "report":
        try:
            paths = write_report(args.run_dir)
            print(str(paths.markdown))
            return EXIT_OK
        except ReportingError as exc:
            print("report error: {}".format(exc), file=sys.stderr)
            return EXIT_CONFIG

    config = None
    try:
        if not args.live and not args.dry_run:
            raise ConfigError("paid network commands require explicit --live")
        config = load_config(args.config, os.environ)
        run_dir = _run_dir(config, args.run_dir)
        if args.command == "resume" and not args.dry_run and not run_dir.is_dir():
            raise ConfigError("resume run directory does not exist")
        if args.dry_run:
            _dry_run(config, run_dir)
            return EXIT_OK

        summary = execute_run(config, run_dir)
        print(_summary_json(summary, run_dir))
        planned = config.search.max_initial_points * (
            config.search.seed_questions_per_point
            + config.search.adaptive_iterations_per_point
        )
        minimum_valid = int(math.ceil(planned * 0.9))
        if summary.interrupted or summary.valid < minimum_valid:
            return EXIT_INCOMPLETE
        return EXIT_OK
    except (ConfigError, CheckpointError) as exc:
        print("configuration error: {}".format(_redact(str(exc), config)), file=sys.stderr)
        return EXIT_CONFIG
    except (ApiError, RoleError) as exc:
        print("API error: {}".format(_redact(str(exc), config)), file=sys.stderr)
        return EXIT_API
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print("error: {}".format(_redact(str(exc), config)), file=sys.stderr)
        return EXIT_ERROR
