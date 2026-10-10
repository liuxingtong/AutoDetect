"""Pin the contract that ``embedding.enabled: false`` is honored.

Without this, the config loader unconditionally enables the
embedding service whenever an ``embedding.type`` is set, even
when the operator explicitly says ``enabled: false``.  The
consequence is silent: the runner builds a NoveltyFilter that
calls the embedding API on every adaptive round, and any sample
without an ``embedding.base_url`` reachable to the cluster
fails with a 400 Bad Request (the API rejects empty
``input``).  The runner then writes a ``failed`` sample whose
``error`` is the embedding 400, not a real model failure.

The 2026-10-10 live run hit exactly this bug: 2 of 4 samples
failed with
``Failed to encode texts: 400 Client Error: Bad Request for
url: https://api.siliconflow.cn/v1/embeddings``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from autodetect_repro.config import ConfigError, load_config


def _write_yaml(text: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as handle:
        handle.write(text)
        return Path(handle.name)


_BASE = """
models:
  base_url: https://api.example.com/v1
  agent_model: a
  target_model: t
  api_key_env: SILICONFLOW_API_KEY
search:
  seed_questions_per_point: 1
  adaptive_iterations_per_point: 1
  max_initial_points: 1
  max_discovered_points_per_subtask: 1
  random_seed: 1
limits:
  max_total_samples: 4
  max_attempts_per_request: 1
  request_timeout_seconds: 10
  max_output_tokens: 256
output_root: /tmp/embedding-flag-test
subtasks:
  - group: g
    name: s
    knowledge_points:
      - p1
"""


def test_embedding_enabled_false_is_honored_when_type_set(monkeypatch):
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", "/tmp/embedding-flag-test")
    yaml = _BASE + """
embedding:
  enabled: false
  type: remote
  base_url: https://api.example.com/v1
"""
    path = _write_yaml(yaml)
    config = load_config(path, {"SILICONFLOW_API_KEY": "k"})
    assert config.embedding.enabled is False, (
        "embedding.enabled was set to false in the YAML, but the "
        "config loader returned enabled=True.  The runner will "
        "still build a NoveltyFilter and call the embedding API "
        "on every adaptive round (this is the bug from the "
        "2026-10-10 live run)."
    )


def test_embedding_enabled_true_is_default_when_type_set(monkeypatch):
    """The default should remain enabled=True so existing configs
    that omit ``enabled`` keep working."""
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", "/tmp/embedding-flag-test")
    yaml = _BASE + """
embedding:
  type: remote
  base_url: https://api.example.com/v1
"""
    path = _write_yaml(yaml)
    config = load_config(path, {"SILICONFLOW_API_KEY": "k"})
    assert config.embedding.enabled is True, (
        "embedding section is configured with a type, so the "
        "default should still be enabled=True.  A future change "
        "to the default would silently break the novelty filter "
        "for all existing configs."
    )


def test_embedding_omitted_means_disabled(monkeypatch):
    """No embedding section at all -> enabled=False (the original
    behavior the missing-section branch was designed for)."""
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", "/tmp/embedding-flag-test")
    yaml = _BASE
    path = _write_yaml(yaml)
    config = load_config(path, {"SILICONFLOW_API_KEY": "k"})
    assert config.embedding.enabled is False
    assert config.embedding.type == "none"


def test_embedding_enabled_false_does_not_build_novelty_filter(monkeypatch):
    """End-to-end: the CLI helper ``_build_novelty_filter`` returns
    None when ``embedding.enabled: false`` even if a base_url is
    present.  This is the wire that the runner uses, and the one
    that 2026-10-10 run was failing through."""
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", "/tmp/embedding-flag-test")
    from autodetect_repro.cli import _build_novelty_filter
    from dataclasses import replace
    yaml = _BASE + """
embedding:
  enabled: false
  type: remote
  base_url: https://api.example.com/v1
"""
    path = _write_yaml(yaml)
    config = load_config(path, {"SILICONFLOW_API_KEY": "k"})
    # Even with mode=adaptive_rule, the filter is None.
    config = replace(
        config,
        search=replace(
            config.search, mode="adaptive_rule", candidate_questions_per_round=3
        ),
    )
    assert _build_novelty_filter(config) is None
