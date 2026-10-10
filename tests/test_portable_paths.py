"""Tests for the AUTODETECT_OUTPUT_ROOT and AUTODETECT_EMBEDDING_MODEL
environment variable overrides that make the package portable off the
original reproduction host.

Previously /data/liuxingtong was hard-coded in 4 places. Operators on
other machines (or running in CI) can now point those at a different
root via env vars; the defaults preserve the original reproduction
contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def test_output_root_override(monkeypatch, tmp_path):
    """_safe_output_root accepts paths under an override env var."""
    from autodetect_repro import config

    override = tmp_path / "alt_root"
    override.mkdir()
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", str(override))
    result = config._safe_output_root(str(override / "runs"))
    assert result == (override / "runs").expanduser()


def test_output_root_rejects_paths_outside_override(monkeypatch, tmp_path):
    """Override only widens the *allowed* root, never narrows it."""
    from autodetect_repro import config
    from autodetect_repro.config import ConfigError

    override = tmp_path / "alt_root"
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", str(override))
    with pytest.raises(ConfigError, match="output_root must be below"):
        config._safe_output_root("/somewhere/else")


def test_output_root_default_unchanged(monkeypatch):
    """When AUTODETECT_OUTPUT_ROOT is unset, the original /data/liuxingtong
    rule still applies (and still rejects paths elsewhere)."""
    from autodetect_repro import config
    from autodetect_repro.config import ConfigError

    monkeypatch.delenv("AUTODETECT_OUTPUT_ROOT", raising=False)
    with pytest.raises(ConfigError, match="/data/liuxingtong"):
        config._safe_output_root("/home/someone/elsewhere")


def test_run_dir_accepts_override(monkeypatch, tmp_path):
    """cli._run_dir honors the same env var."""
    from autodetect_repro import cli

    override = tmp_path / "alt_root"
    override.mkdir()
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", str(override))
    # Build a stub config object that has the only attribute
    # _run_dir reads.
    from types import SimpleNamespace
    cfg = SimpleNamespace(output_root=override)
    requested = override / "explicit_run"
    requested.mkdir()
    out = cli._run_dir(cfg, requested)
    assert out == requested


def test_run_dir_rejects_paths_outside_override(monkeypatch, tmp_path):
    """_run_dir with an override still rejects paths above the override."""
    from autodetect_repro import cli
    from autodetect_repro.config import ConfigError
    from types import SimpleNamespace

    override = tmp_path / "alt_root"
    monkeypatch.setenv("AUTODETECT_OUTPUT_ROOT", str(override))
    cfg = SimpleNamespace(output_root=override)
    with pytest.raises(ConfigError, match="run directory must be below"):
        cli._run_dir(cfg, Path("/tmp/somewhere_else"))


def test_embedding_model_default(monkeypatch):
    """DEFAULT_MODEL_PATH still points to the original path when no
    env var is set."""
    monkeypatch.delenv("AUTODETECT_EMBEDDING_MODEL", raising=False)
    from autodetect_repro import embedding_service
    assert embedding_service.DEFAULT_MODEL_PATH == (
        "/data/liuxingtong/models/Qwen3-Embedding-0.6B"
    )


def test_embedding_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTODETECT_EMBEDDING_MODEL", str(tmp_path / "model"))
    # Re-import the module so the module-level constant is recomputed.
    import importlib
    from autodetect_repro import embedding_service
    importlib.reload(embedding_service)
    try:
        assert embedding_service.DEFAULT_MODEL_PATH == str(tmp_path / "model")
    finally:
        # Restore for the rest of the test session.
        monkeypatch.delenv("AUTODETECT_EMBEDDING_MODEL", raising=False)
        importlib.reload(embedding_service)
