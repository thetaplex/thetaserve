from pathlib import Path

import pytest

from thetaserve.config import ConfigError, load_catalog


def test_builtin_catalog_loads() -> None:
    catalog = load_catalog()
    assert {"qwen3.8-27b", "gemma4-31b"} <= catalog.models.keys()
    assert catalog.devices["l40s-48g"].providers["aws"] == {"instance_type": "g6e.xlarge"}
    assert catalog.providers.names() == ["runpod", "aws"]


def test_user_file_replaces_and_adds_entries(user_dir: Path) -> None:
    (user_dir / "devices.yaml").write_text(
        "a6000-48g: {vram_gb: 48, arch: ampere, providers: {runpod: {gpu_type_id: X}}}\n"
        "b200-180g: {vram_gb: 180, arch: blackwell, providers: {runpod: {gpu_type_id: B}}}\n",
        encoding="utf-8",
    )
    catalog = load_catalog()
    assert catalog.devices["a6000-48g"].providers["runpod"] == {"gpu_type_id": "X"}
    assert catalog.devices["b200-180g"].vram_gb == 180
    assert "h100-80g" in catalog.devices  # built-in entries the user did not touch remain


def test_validation_error_names_file_and_key(user_dir: Path) -> None:
    (user_dir / "devices.yaml").write_text(
        "bad-gpu: {vram_gb: lots, arch: ampere, providers: {}}\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match=r"devices\.yaml: bad-gpu\.vram_gb: "):
        load_catalog()


def test_invalid_tier_key_rejected(user_dir: Path) -> None:
    (user_dir / "models.yaml").write_text(
        "tiny: {hf_family: x, min_vllm: '1', kv_bytes_per_token: 1,"
        " tiers: {big: {repo: r, weights_gb: 1, max_model_len: 8, kv_cache_dtype: auto}}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="tier key 'big'"):
        load_catalog()


def test_wrong_schema_version_rejected(user_dir: Path) -> None:
    (user_dir / "models.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="schema_version must be 1"):
        load_catalog()


def test_invalid_yaml_rejected(user_dir: Path) -> None:
    (user_dir / "providers.yaml").write_text("runpod: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_catalog()
