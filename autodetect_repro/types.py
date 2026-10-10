"""Immutable configuration and experiment record types."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


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
    mode: str = "baseline"
    candidate_questions_per_round: int = 1
    novelty_top_k: int = 3
    random_exploration_ratio: float = 0.1
    auto_promote: bool = False


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
class EmbeddingConfig:
    """Optional embedding-service configuration for the semantic novelty filter."""

    enabled: bool
    type: str  # "remote" | "local" | "huggingface" | "none"
    base_url: str = ""
    model_path: str = ""
    model_name: str = ""
    api_key_env: str = ""
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True


@dataclass(frozen=True)
class RunConfig:
    models: ModelConfig
    search: SearchConfig
    limits: LimitsConfig
    subtasks: Tuple[SubtaskConfig, ...]
    output_root: Path
    embedding: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(enabled=False, type="none")
    )


@dataclass(frozen=True)
class KnowledgePoint:
    group: str
    subtask: str
    name: str
    discovered: bool = False


@dataclass(frozen=True)
class Question:
    group: str
    subtask: str
    knowledge_point: str
    key_point: str
    prompt: str
    phase: str
    iteration: int
    ordinal: int


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    group: str
    subtask: str
    knowledge_point: str
    phase: str
    iteration: int
    ordinal: int
    prompt: str
    key_point: str
    answer: str
    reference_answer: str
    comparison: str
    score: Optional[float]
    is_weakness: bool
    status: str
    error: Optional[str]
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class AssessorDecision:
    accepted: bool
    point: Optional[KnowledgePoint]
    explanation: str
