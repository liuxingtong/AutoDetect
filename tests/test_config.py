from pathlib import Path

import pytest
import yaml

from autodetect_repro.config import ConfigError, load_config


CONFIG_PATH = Path("configs/siliconflow_smallscale.yaml")


def test_checked_in_config_loads_exact_models_limits_and_taxonomy():
    config = load_config(
        CONFIG_PATH,
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )

    assert config.models.base_url == "https://api.siliconflow.cn/v1"
    assert config.models.agent_model == "deepseek-ai/DeepSeek-V4-Flash"
    assert config.models.target_model == "Qwen/Qwen2.5-7B-Instruct"
    assert config.search.seed_questions_per_point == 2
    assert config.search.adaptive_iterations_per_point == 3
    assert config.search.max_initial_points == 12
    assert config.search.max_discovered_points_per_subtask == 1
    assert config.limits.max_total_samples == 80
    assert config.limits.max_attempts_per_request == 4
    assert config.limits.request_timeout_seconds == 120
    assert config.output_root == Path(
        "/data/liuxingtong/outputs/autodetect-smallscale"
    )
    assert [subtask.name for subtask in config.subtasks] == [
        "text format",
        "word constraint",
        "scenario simulation",
        "emotional tone",
    ]
    assert sum(len(subtask.knowledge_points) for subtask in config.subtasks) == 12


def test_missing_api_key_variable_is_rejected():
    with pytest.raises(ConfigError, match="SILICONFLOW_API_KEY"):
        load_config(CONFIG_PATH, {})


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("search", "seed_questions_per_point"),
        ("search", "adaptive_iterations_per_point"),
        ("limits", "max_total_samples"),
        ("limits", "max_attempts_per_request"),
        ("limits", "request_timeout_seconds"),
    ],
)
def test_non_positive_limits_are_rejected(tmp_path, section, field):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw[section][field] = 0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match=field):
        load_config(path, {"SILICONFLOW_API_KEY": "unit-test-secret"})


def test_duplicate_knowledge_points_are_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    duplicate = raw["subtasks"][0]["knowledge_points"][0]
    raw["subtasks"][1]["knowledge_points"].append(duplicate)
    path = tmp_path / "duplicate.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="duplicate knowledge point"):
        load_config(path, {"SILICONFLOW_API_KEY": "unit-test-secret"})


def test_config_repr_redacts_api_key():
    secret = "unit-test-secret"
    config = load_config(CONFIG_PATH, {"SILICONFLOW_API_KEY": secret})

    assert secret not in repr(config)


def test_output_root_must_be_below_data_directory(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["output_root"] = "/home/liuxingtong/results"
    path = tmp_path / "home-output.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="/data/liuxingtong"):
        load_config(path, {"SILICONFLOW_API_KEY": "unit-test-secret"})


def test_embedding_section_is_optional_and_disabled_by_default():
    config = load_config(
        CONFIG_PATH,
        {"SILICONFLOW_API_KEY": "unit-test-secret"},
    )
    assert config.embedding.enabled is False
    assert config.embedding.type == "none"


def test_remote_embedding_config_is_parsed(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["embedding"] = {
        "type": "remote",
        "base_url": "http://115.182.62.246:8080",
        "batch_size": 16,
        "max_length": 256,
        "normalize": False,
    }
    path = tmp_path / "embedding.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_config(path, {"SILICONFLOW_API_KEY": "unit-test-secret"})

    assert config.embedding.enabled is True
    assert config.embedding.type == "remote"
    assert config.embedding.base_url == "http://115.182.62.246:8080"
    assert config.embedding.batch_size == 16
    assert config.embedding.max_length == 256
    assert config.embedding.normalize is False


def test_remote_embedding_without_base_url_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["embedding"] = {"type": "remote"}
    path = tmp_path / "embedding-bad.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="base_url"):
        load_config(path, {"SILICONFLOW_API_KEY": "unit-test-secret"})


def test_unknown_embedding_type_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["embedding"] = {"type": "imaginary", "base_url": "http://x"}
    path = tmp_path / "embedding-bad-type.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ConfigError, match="embedding.type"):
        load_config(path, {"SILICONFLOW_API_KEY": "unit-test-secret"})
