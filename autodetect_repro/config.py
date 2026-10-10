"""Load and validate reproduction configuration."""

import os
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml

from .types import (
    EmbeddingConfig,
    LimitsConfig,
    ModelConfig,
    RunConfig,
    SearchConfig,
    SubtaskConfig,
)


class ConfigError(ValueError):
    """Raised when a run configuration is unsafe or incomplete."""


def _positive_int(section: Mapping[str, Any], field: str) -> int:
    value = section.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError("{} must be a positive integer".format(field))
    return value


def _subtasks(raw_subtasks: Any) -> Tuple[SubtaskConfig, ...]:
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raise ConfigError("subtasks must be a non-empty list")

    result = []
    seen = set()
    for raw in raw_subtasks:
        if not isinstance(raw, dict):
            raise ConfigError("each subtask must be a mapping")
        group = str(raw.get("group", "")).strip()
        name = str(raw.get("name", "")).strip()
        points = raw.get("knowledge_points")
        if not group or not name or not isinstance(points, list) or not points:
            raise ConfigError("each subtask needs group, name, and knowledge_points")

        normalized_points = []
        for point in points:
            point_text = str(point).strip()
            if not point_text:
                raise ConfigError("knowledge points cannot be empty")
            key = point_text.casefold()
            if key in seen:
                raise ConfigError("duplicate knowledge point: {}".format(point_text))
            seen.add(key)
            normalized_points.append(point_text)
        result.append(SubtaskConfig(group, name, tuple(normalized_points)))
    return tuple(result)


def _safe_output_root(raw: Any) -> Path:
    output_root = Path(str(raw)).expanduser()
    # The required root is the run-output sandbox on the original
    # reproduction host.  Operators in other environments can override
    # with AUTODETECT_OUTPUT_ROOT; the default keeps the original
    # behavior for the documented reproduction.
    required_root = Path(
        os.environ.get("AUTODETECT_OUTPUT_ROOT", "/data/liuxingtong")
    )
    if not output_root.is_absolute():
        raise ConfigError(
            "output_root must be an absolute path below {}".format(required_root)
        )
    try:
        output_root.relative_to(required_root)
    except ValueError as exc:
        raise ConfigError(
            "output_root must be below {}".format(required_root)
        ) from exc
    return output_root


def load_config(path: Path, environ: Mapping[str, str]) -> RunConfig:
    """Load a YAML run configuration and inject its named API key."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("cannot load config {}: {}".format(path, exc)) from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    raw_models = raw.get("models")
    raw_search = raw.get("search")
    raw_limits = raw.get("limits")
    if not all(isinstance(item, dict) for item in (raw_models, raw_search, raw_limits)):
        raise ConfigError("models, search, and limits must be mappings")

    api_key_env = str(raw_models.get("api_key_env", "")).strip()
    api_key = environ.get(api_key_env, "") if api_key_env else ""
    if not api_key:
        raise ConfigError("missing API key environment variable {}".format(api_key_env))

    models = ModelConfig(
        base_url=str(raw_models.get("base_url", "")).rstrip("/"),
        agent_model=str(raw_models.get("agent_model", "")).strip(),
        target_model=str(raw_models.get("target_model", "")).strip(),
        api_key_env=api_key_env,
        api_key=api_key,
    )
    if not models.base_url.startswith("https://"):
        raise ConfigError("base_url must use https")
    if not models.agent_model or not models.target_model:
        raise ConfigError("agent_model and target_model are required")

    search = SearchConfig(
        seed_questions_per_point=_positive_int(
            raw_search, "seed_questions_per_point"
        ),
        adaptive_iterations_per_point=_positive_int(
            raw_search, "adaptive_iterations_per_point"
        ),
        max_initial_points=_positive_int(raw_search, "max_initial_points"),
        max_discovered_points_per_subtask=_positive_int(
            raw_search, "max_discovered_points_per_subtask"
        ),
        random_seed=int(raw_search.get("random_seed", 240616714)),
        mode=str(raw_search.get("mode", "baseline")),
        candidate_questions_per_round=int(raw_search.get("candidate_questions_per_round", 1)),
        novelty_top_k=int(raw_search.get("novelty_top_k", 3)),
        random_exploration_ratio=float(raw_search.get("random_exploration_ratio", 0.1)),
        auto_promote=bool(raw_search.get("auto_promote", False)),
    )
    if search.mode not in {"baseline", "adaptive_rule", "adaptive_llm"}:
        raise ConfigError("search.mode must be baseline, adaptive_rule, or adaptive_llm")
    if search.candidate_questions_per_round <= 0 or search.novelty_top_k <= 0:
        raise ConfigError("candidate_questions_per_round and novelty_top_k must be positive")
    if not 0.0 <= search.random_exploration_ratio <= 1.0:
        raise ConfigError("random_exploration_ratio must be between 0 and 1")
    limits = LimitsConfig(
        max_total_samples=_positive_int(raw_limits, "max_total_samples"),
        max_attempts_per_request=_positive_int(
            raw_limits, "max_attempts_per_request"
        ),
        request_timeout_seconds=_positive_int(
            raw_limits, "request_timeout_seconds"
        ),
        max_output_tokens=_positive_int(raw_limits, "max_output_tokens"),
    )
    subtasks = _subtasks(raw.get("subtasks"))
    point_count = sum(len(subtask.knowledge_points) for subtask in subtasks)
    if point_count != search.max_initial_points:
        raise ConfigError(
            "max_initial_points must equal configured knowledge point count"
        )

    embedding = _embedding_config(raw.get("embedding"))

    return RunConfig(
        models=models,
        search=search,
        limits=limits,
        subtasks=subtasks,
        output_root=_safe_output_root(raw.get("output_root", "")),
        embedding=embedding,
    )


def _embedding_config(raw_embedding: Any) -> EmbeddingConfig:
    """Parse the optional ``embedding`` section.  Missing or null disables it."""
    if not raw_embedding:
        return EmbeddingConfig(enabled=False, type="none")
    if not isinstance(raw_embedding, dict):
        raise ConfigError("embedding must be a mapping when provided")

    raw_type = str(raw_embedding.get("type", "")).strip().lower()
    if raw_type not in {"remote", "local", "huggingface"}:
        raise ConfigError(
            "embedding.type must be remote, local, or huggingface"
        )

    base_url = str(raw_embedding.get("base_url", "")).strip()
    model_path = str(raw_embedding.get("model_path", "")).strip()
    model_name = str(raw_embedding.get("model_name", "")).strip()

    if raw_type == "remote" and not base_url:
        raise ConfigError("embedding.base_url is required for remote type")
    if raw_type in {"local", "huggingface"} and not model_path and not model_name:
        raise ConfigError(
            "embedding.model_path or embedding.model_name is required for local/huggingface"
        )

    api_key_env = str(raw_embedding.get("api_key_env", "")).strip()
    try:
        batch_size = int(raw_embedding.get("batch_size", 32))
        max_length = int(raw_embedding.get("max_length", 512))
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "embedding.batch_size and max_length must be integers"
        ) from exc
    if batch_size <= 0 or max_length <= 0:
        raise ConfigError(
            "embedding.batch_size and max_length must be positive"
        )

    return EmbeddingConfig(
        enabled=True,
        type=raw_type,
        base_url=base_url,
        model_path=model_path,
        model_name=model_name,
        api_key_env=api_key_env,
        batch_size=batch_size,
        max_length=max_length,
        normalize=bool(raw_embedding.get("normalize", True)),
    )
