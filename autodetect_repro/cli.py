"""Command-line entry points for safe reproduction runs."""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from .checkpoint import CheckpointError
from .analysis_pipeline import AdaptiveAnalysisPipeline
from .branches import BranchManager
from .client import ApiError, SiliconFlowClient
from .config import ConfigError, RunConfig, load_config
from .controller import RuleController
from .novelty import NoveltyFilter, NoveltyLabel
from .reporting import ReportingError, write_report
from .roles import (
    Assessor, BehavioralClusterer, Evaluator, Examiner, FailureSignatureAnalyzer,
    LLMController, LLMNoveltyClassifier, Questioner, RoleError,
)
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


def _build_novelty_filter(config: RunConfig) -> Optional[NoveltyFilter]:
    """Build a NoveltyFilter if the config requests semantic filtering.

    Returns ``None`` for the baseline path and for any config that does not
    declare an embedding service.  Failures to construct the embedding client
    are surfaced as ``ConfigError`` so the operator sees them up front rather
    than as a silent fallback.
    """
    if config.search.mode == "baseline":
        return None
    if not config.embedding.enabled:
        return None
    if config.search.candidate_questions_per_round <= 1:
        return None

    embedding_fn = _build_embedding_function(config)
    return NoveltyFilter(
        embedding=embedding_fn,
        top_k=config.search.novelty_top_k,
    )


def _build_novelty_classifier(config: RunConfig) -> Optional[
    Callable[[str, Sequence[str]], NoveltyLabel]
]:
    """Build an LLM-backed novelty classifier.

    Returns ``None`` when no embedding service is configured, so the runner
    keeps using its safe "all DISTINCT" default.  When embedding is enabled
    we always wire the classifier — without it the design contract (embed
    + LLM label + sort) collapses to "embedding-only sort".
    """
    if config.search.mode == "baseline":
        return None
    if not config.embedding.enabled:
        return None
    agent = _client(config, config.models.agent_model)
    # A single shared classifier is used across all knowledge points; the
    # prompt includes the knowledge point dynamically if/when the runner
    # forwards it.  For now a context-free call is correct because the
    # neighbors carry enough semantic information.
    return LLMNoveltyClassifier(
        agent,
        max_tokens=config.limits.max_output_tokens,
    )


def _build_embedding_function(config: RunConfig) -> Callable[[str], Sequence[float]]:
    """Construct a single-text embedding function honoring ``EmbeddingConfig``."""
    if config.embedding.type == "remote":
        from .embedding_client import RemoteEmbeddingClient
        client = RemoteEmbeddingClient(
            base_url=config.embedding.base_url,
            api_key=os.environ.get(config.embedding.api_key_env)
            if config.embedding.api_key_env
            else None,
            batch_size=config.embedding.batch_size,
        )
        return client.encode
    if config.embedding.type in {"local", "huggingface"}:
        from .embedding_service import get_embedding_function
        return get_embedding_function(
            model_type=config.embedding.type,
            model_path=config.embedding.model_path or None,
            model_name=config.embedding.model_name or None,
            normalize=config.embedding.normalize,
        )
    raise ConfigError(
        "embedding.type {} is not supported".format(config.embedding.type)
    )


def execute_run(config: RunConfig, run_dir: Path) -> RunSummary:
    agent = _client(config, config.models.agent_model)
    target = _client(config, config.models.target_model)
    controller = None
    analysis_pipeline = None
    novelty_filter = _build_novelty_filter(config)
    novelty_classifier = _build_novelty_classifier(config)
    if config.search.mode == "adaptive_rule":
        controller = RuleController(
            candidate_cap=config.search.candidate_questions_per_round,
            test_cap=config.search.candidate_questions_per_round,
        )
    elif config.search.mode == "adaptive_llm":
        controller = LLMController(config, agent)
    if config.search.mode != "baseline":
        analysis_pipeline = AdaptiveAnalysisPipeline(
            signature_extractor=FailureSignatureAnalyzer(config, agent),
            behavior_classifier=BehavioralClusterer(config, agent),
            branch_manager=BranchManager(),
        )
    runner = ExperimentRunner(
        config,
        run_dir,
        examiner=Examiner(config, agent),
        questioner=Questioner(config, agent),
        evaluator=Evaluator(
            target, agent, max_tokens=config.limits.max_output_tokens
        ),
        assessor=Assessor(config, agent),
        controller=controller,
        analysis_pipeline=analysis_pipeline,
        novelty_filter=novelty_filter,
        novelty_classify=novelty_classifier,
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
