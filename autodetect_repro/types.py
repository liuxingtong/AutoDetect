"""Immutable configuration and experiment record types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    agent_model: str
    target_model: str
    api_key_env: str
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class SearchConfig:
    seed_questions_per_point: int
    adaptive_iterations_per_point: int
    max_initial_points: int
    max_discovered_points_per_subtask: int
    random_seed: int


@dataclass(frozen=True)
class LimitsConfig:
    max_total_samples: int
    max_attempts_per_request: int
    request_timeout_seconds: int
    max_output_tokens: int


@dataclass(frozen=True)
class SubtaskConfig:
    group: str
    name: str
    knowledge_points: Tuple[str, ...]


@dataclass(frozen=True)
class RunConfig:
    models: ModelConfig
    search: SearchConfig
    limits: LimitsConfig
    subtasks: Tuple[SubtaskConfig, ...]
    output_root: Path
