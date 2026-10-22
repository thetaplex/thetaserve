"""Catalogue models and loading: built-in YAML merged with the user's config directory."""

from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

import platformdirs
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
    ValidationError,
)

SCHEMA_VERSION = 2

type Arch = Literal["ampere", "ada", "hopper", "blackwell"]
type KvCacheDtype = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]


class ConfigError(Exception):
    """A catalogue file is missing, unreadable or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Model(_Strict):
    hf_family: str  # base Hugging Face repository
    common_args: list[str] = Field(default_factory=list)  # family vLLM flags
    kv_scale: PositiveFloat = 1.0  # KV cache per token vLLM needs / the size config.json gives


class RunPodConfig(_Strict):
    api_key_env: str = "RUNPOD_API_KEY"
    cloud_type: Literal["SECURE", "COMMUNITY"] = "SECURE"
    network_volume: str | None  # None: no weights cache, download on every start
    network_volume_gb: PositiveInt = 100
    container_disk_gb: PositiveInt


class AwsConfig(_Strict):
    profile: str
    region: str
    ami_ssm: str
    root_volume_gb: PositiveInt
    ssh_key_name: str
    allow_cidr: str = "auto"


class Providers(_Strict):
    runpod: RunPodConfig
    aws: AwsConfig | None = None

    def names(self) -> list[str]:
        return [name for name in type(self).model_fields if getattr(self, name) is not None]


class Preset(_Strict):
    min_context: PositiveInt | None  # None: the model's maximum
    tools: bool
    min_bits: PositiveInt


class Catalog(_Strict):
    vllm_image: str
    models: dict[str, Model]
    providers: Providers
    presets: dict[str, Preset]
    bandwidth: dict[str, PositiveFloat]  # GB/s per provider GPU ID
    publishers: dict[str, str]  # trusted Hugging Face organisation -> why


def user_config_dir() -> Path:
    return Path(platformdirs.user_config_dir("thetaserve", appauthor=False))


def load_catalog(user_dir: Path | None = None) -> Catalog:
    """Load the built-in catalogues, then let files in `user_dir` replace or add entries by key."""
    user_dir = user_config_dir() if user_dir is None else user_dir

    models, model_src = _merged("models.yaml", user_dir)
    providers, provider_src = _merged("providers.yaml", user_dir)
    presets, preset_src = _merged("presets.yaml", user_dir)
    bandwidth, bandwidth_src = _merged("bandwidth.yaml", user_dir)
    publishers, publisher_src = _merged("publishers.yaml", user_dir)

    version = models.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"{model_src.get('schema_version', 'models.yaml')}: schema_version must be "
            f"{SCHEMA_VERSION}, got {version!r}"
        )
    image = models.pop("vllm_image", None)
    if not isinstance(image, str):
        raise ConfigError(f"{model_src.get('vllm_image', 'models.yaml')}: vllm_image is required")

    return Catalog(
        vllm_image=image,
        models=_validate(dict[str, Model], models, model_src),
        providers=_validate(Providers, providers, provider_src),
        presets=_validate(dict[str, Preset], presets, preset_src),
        bandwidth=_validate(dict[str, PositiveFloat], bandwidth, bandwidth_src),
        publishers=_validate(dict[str, str], publishers, publisher_src),
    )


def _merged(name: str, user_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Top-level keys of the built-in file, replaced or extended by the user's file.

    Returns the merged mapping and, for each key, the file it came from (for error messages).
    """
    builtin = files("thetaserve.catalog").joinpath(name)
    data = _read_yaml(builtin.read_text(encoding="utf-8"), f"<built-in>/{name}")
    source = dict.fromkeys(data, f"<built-in>/{name}")

    user_file = user_dir / name
    if user_file.is_file():
        user = _read_yaml(user_file.read_text(encoding="utf-8"), str(user_file))
        data |= user
        source |= dict.fromkeys(user, str(user_file))
    return data, source


def _read_yaml(text: str, origin: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{origin}: invalid YAML: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{origin}: top level must be a mapping")
    return data


def _validate[T](kind: type[T], data: dict[str, Any], source: dict[str, str]) -> T:
    try:
        return TypeAdapter(kind).validate_python(data)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = [str(part) for part in err["loc"]]
            origin = source.get(loc[0], "?") if loc else "?"
            lines.append(f"{origin}: {'.'.join(loc)}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from exc
