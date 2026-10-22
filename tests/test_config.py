from pathlib import Path

import pytest

from thetaserve.config import ConfigError, load_catalog


def test_builtin_catalog_loads() -> None:
    catalog = load_catalog()
    assert {"qwen3.8-27b", "gemma4-31b"} <= catalog.models.keys()
    assert {"coding", "chat", "research"} <= catalog.presets.keys()
    assert catalog.providers.names() == ["runpod", "aws"]


def test_user_file_replaces_and_adds_entries(user_dir: Path) -> None:
    (user_dir / "presets.yaml").write_text(
        "chat: {min_context: 8192, tools: false, min_bits: 4}\n"
        "agent-long: {min_context: 262144, tools: true, min_bits: 8}\n",
        encoding="utf-8",
    )
    catalog = load_catalog()
    assert catalog.presets["chat"].min_context == 8192
    assert catalog.presets["agent-long"].min_bits == 8
    assert "coding" in catalog.presets  # built-in entries the user did not touch remain


def test_validation_error_names_file_and_key(user_dir: Path) -> None:
    (user_dir / "presets.yaml").write_text(
        "bad: {min_context: lots, tools: true, min_bits: 8}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=r"presets\.yaml: bad\.min_context: "):
        load_catalog()


def test_unknown_model_field_rejected(user_dir: Path) -> None:
    (user_dir / "models.yaml").write_text("tiny: {hf_family: x/y, tiers: {}}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"models\.yaml: tiny\.tiers: "):
        load_catalog()


def test_wrong_schema_version_rejected(user_dir: Path) -> None:
    (user_dir / "models.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="schema_version must be 2"):
        load_catalog()


def test_invalid_yaml_rejected(user_dir: Path) -> None:
    (user_dir / "providers.yaml").write_text("runpod: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_catalog()
