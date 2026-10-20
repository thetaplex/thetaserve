import shlex
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from thetaserve.cli import app, parse_duration
from thetaserve.config import load_catalog
from thetaserve.profiles import Overrides, resolve

runner = CliRunner()
# Built-in catalogue only; collection runs before the autouse user-dir fixture.
_catalog = load_catalog(Path(__file__).parent / "no-user-config")
COMBOS = [
    (provider, device, model)
    for device, d in sorted(_catalog.devices.items())
    for provider in sorted(d.providers)
    for model in sorted(_catalog.models)
]


@pytest.mark.parametrize(("provider", "device", "model"), COMBOS)
def test_every_combination_renders_a_valid_command(provider: str, device: str, model: str) -> None:
    spec = resolve(
        _catalog,
        provider=provider,
        device=device,
        model=model,
        port=8123,
        idle_s=1800,
        max_life_s=36000,
        overrides=Overrides(),
    )
    assert spec.fit.fits, f"default profile must fit: {spec.fit}"

    argv = shlex.split(spec.command())
    assert argv[:3] == ["vllm", "serve", spec.repo]
    flags = dict(zip(argv[3::2], argv[4::2], strict=False))
    assert flags["--port"] == "8123"
    assert flags["--api-key"] == "$VLLM_API_KEY"
    assert flags["--max-model-len"] == str(spec.max_model_len)
    assert flags["--kv-cache-dtype"] == spec.kv_cache_dtype
    assert flags["--gpu-memory-utilization"] == "0.92"
    for arg in _catalog.models[model].common_args:
        assert arg in argv


def test_api_key_stays_a_shell_variable() -> None:
    spec = resolve(
        _catalog,
        provider="runpod",
        device="a6000-48g",
        model="qwen3.8-27b",
        port=8000,
        idle_s=60,
        max_life_s=60,
    )
    assert '--api-key "$VLLM_API_KEY"' in spec.command()


def test_cli_dry_run_prints_command() -> None:
    result = runner.invoke(
        app,
        ["up", "--provider", "runpod", "--device", "a6000-48g", "--model", "qwen3.8-27b",
         "--port", "8000", "--dry-run"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "vllm serve TODO-FP8 --port 8000" in result.output
    assert "VLLM_API_KEY=****" in result.output


def test_cli_without_dry_run_refuses() -> None:
    result = runner.invoke(
        app, ["up", "--provider", "runpod", "--device", "a6000-48g", "--model", "qwen3.8-27b"]
    )
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_cli_resolve_error_exits_nonzero() -> None:
    result = runner.invoke(
        app, ["up", "--provider", "aws", "--device", "a6000-48g", "--model", "qwen3.8-27b",
              "--dry-run"],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "not offered on 'aws'" in result.output


@pytest.mark.parametrize("kind", ["models", "devices", "providers"])
def test_cli_list(kind: str) -> None:
    result = runner.invoke(app, ["list", kind])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(("text", "seconds"), [("90s", 90), ("30m", 1800), ("1h30m", 5400)])
def test_parse_duration(text: str, seconds: int) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "10", "5d", "1h 30m", "m30"])
def test_parse_duration_rejects(text: str) -> None:
    with pytest.raises(typer.BadParameter, match="not a duration"):
        parse_duration(text)
