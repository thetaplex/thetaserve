import shlex

import pytest
import typer
from typer.testing import CliRunner

from thetaserve.cli import app, parse_duration
from thetaserve.spec import LaunchSpec, estimate

runner = CliRunner()


def test_kv_cache_estimate_matches_design_table() -> None:
    # DESIGN.md section 6.2: Qwen3.8-27B at 128K is 8.0 GiB in BF16, 4.0 GiB in FP8.
    assert estimate(28.75, 65536, 0, 131072, "auto", 48).kv_cache_gb == pytest.approx(8.0)
    assert estimate(28.75, 65536, 0, 131072, "fp8", 48).kv_cache_gb == pytest.approx(4.0)


def test_command_is_valid_and_keeps_api_key_a_shell_variable(spec: LaunchSpec) -> None:
    assert '--api-key "$VLLM_API_KEY"' in spec.command()
    argv = shlex.split(spec.command())
    assert argv[:3] == ["vllm", "serve", spec.repo]
    flags = dict(zip(argv[3::2], argv[4::2], strict=False))
    assert flags["--port"] == "8000"
    assert flags["--max-model-len"] == str(spec.max_model_len)
    assert flags["--kv-cache-dtype"] == spec.kv_cache_dtype
    assert flags["--gpu-memory-utilization"] == "0.92"
    assert flags["--revision"] == "a" * 40


@pytest.mark.parametrize("kind", ["models", "providers"])
def test_cli_list(kind: str) -> None:
    result = runner.invoke(app, ["list", kind])
    assert result.exit_code == 0, result.output


def test_unknown_provider_exits_nonzero() -> None:
    result = runner.invoke(app, ["up", "--provider", "vast", "--model", "qwen3.8-27b", "--dry-run"])
    assert result.exit_code == 1
    assert "unknown provider 'vast'" in result.output


@pytest.mark.parametrize(("text", "seconds"), [("90s", 90), ("30m", 1800), ("1h30m", 5400)])
def test_parse_duration(text: str, seconds: int) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "10", "5d", "1h 30m", "m30"])
def test_parse_duration_rejects(text: str) -> None:
    with pytest.raises(typer.BadParameter, match="not a duration"):
        parse_duration(text)
