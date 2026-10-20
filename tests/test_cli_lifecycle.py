"""End to end through the CLI: up, status, env, logs, stop, start, down, all with Runpod mocked."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from thetaserve import secrets, state
from thetaserve.cli import app
from thetaserve.providers.runpod import API_URL

runner = CliRunner()
PROXY = "https://pod1-8000.proxy.runpod.net/v1"
GPU = "NVIDIA%20RTX%20A6000"
TESTED_PROFILE = """
vllm_image: vllm/vllm-openai:v0.17.1
qwen3.8-27b:
  hf_family: Qwen/Qwen3.8-27B
  min_vllm: "0.17"
  kv_bytes_per_token: 65536
  tiers:
    48g: { repo: Qwen/Qwen3.8-27B-FP8, weights_gb: 28, max_model_len: 131072, kv_cache_dtype: auto }
"""
UP = ["up", "--provider", "runpod", "--device", "a6000-48g", "--model", "qwen3.8-27b",
      "--name", "dev"]  # fmt: skip


def _pod(status: str) -> dict[str, object]:
    return {"id": "pod1", "name": "thetaserve-dev", "status": status, "cost": 0.49,
            "startedAt": "2026-10-20T12:00:00Z"}  # fmt: skip


@pytest.fixture
def mocked_runpod(user_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[respx.MockRouter]:
    (user_dir / "models.yaml").write_text(TESTED_PROFILE, encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(
            json={
                "id": "NVIDIA RTX A6000",
                "price": {"secure": 0.49, "community": 0.33},
                "dataCenters": [{"id": "US-TX-3", "name": "Texas", "availability": "HIGH"}],
            }
        )
        router.get(f"{API_URL}/v2/network-volumes").respond(
            json={
                "networkVolumes": [
                    {"id": "vol1", "name": "thetaserve-cache", "dataCenter": "US-TX-3", "size": 100}
                ],
                "pagination": {"nextCursor": None, "hasNextPage": False},
            }
        )
        router.post(f"{API_URL}/v2/pods", name="create").respond(201, json=_pod("PROVISIONING"))
        router.get(f"{API_URL}/v2/pods/pod1").respond(json=_pod("RUNNING"))
        router.get(f"{PROXY}/models", name="models").respond(
            json={"data": [{"id": "Qwen/Qwen3.8-27B-FP8"}]}
        )
        router.get(f"{API_URL}/v2/pods/pod1/logs").respond(
            text='data: {"line": "GPU KV cache size: 150,000 tokens"}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
        router.post(f"{API_URL}/v2/pods/pod1/action", name="action").respond(json=_pod("EXITED"))
        router.delete(f"{API_URL}/v2/pods/pod1", name="delete").respond(204)
        yield router


def test_full_lifecycle(mocked_runpod: respx.MockRouter) -> None:
    result = runner.invoke(app, [*UP, "--yes"])
    assert result.exit_code == 0, result.output
    assert "$0.49/h" in result.output
    assert f"serving at {PROXY}" in result.output
    assert "150,000 tokens" in result.output

    # The generated key went to the pod and to the health check, never to state or output.
    body = json.loads(mocked_runpod["create"].calls.last.request.content)
    key = body["env"]["VLLM_API_KEY"]
    assert mocked_runpod["models"].calls.last.request.headers["Authorization"] == f"Bearer {key}"
    assert key not in result.output
    assert key not in state.path().read_text(encoding="utf-8")
    d = state.load().deployments["dev"]
    assert secrets.get(d.secret_name) == key
    assert d.max_context == 131072  # capped by the configured context, not the KV cache

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "RUNNING" in result.output

    result = runner.invoke(app, ["env", "--shell", "posix"])
    assert result.output.splitlines() == [
        f"export OPENAI_BASE_URL='{PROXY}'",
        f"export OPENAI_API_KEY='{key}'",
    ]

    result = runner.invoke(app, ["logs"])
    assert "GPU KV cache size" in result.output

    result = runner.invoke(app, ["stop", "dev"])
    assert result.exit_code == 0, result.output
    assert json.loads(mocked_runpod["action"].calls.last.request.content) == {"action": "stop"}
    assert state.load().deployments["dev"].status == "EXITED"

    result = runner.invoke(app, ["down", "dev", "--yes"])
    assert result.exit_code == 0, result.output
    assert mocked_runpod["delete"].call_count == 1
    assert state.load().deployments == {}
    assert secrets.get(d.secret_name) is None


def test_up_refuses_duplicate_name(mocked_runpod: respx.MockRouter) -> None:
    assert runner.invoke(app, [*UP, "--yes"]).exit_code == 0
    result = runner.invoke(app, [*UP, "--yes"])
    assert result.exit_code == 1
    assert "exists" in result.output
    assert mocked_runpod["create"].call_count == 1


def test_up_without_yes_in_non_interactive_shell_creates_nothing(
    mocked_runpod: respx.MockRouter,
) -> None:
    result = runner.invoke(app, UP)
    assert result.exit_code == 1
    assert "--yes" in result.output
    assert mocked_runpod["create"].call_count == 0


def test_up_reports_running_host_when_model_never_loads(mocked_runpod: respx.MockRouter) -> None:
    mocked_runpod["models"].respond(401)
    result = runner.invoke(app, [*UP, "--yes"])
    assert result.exit_code == 1
    assert "still running and billing" in result.output
    assert "dev" in state.load().deployments  # kept, so `down` can clean it up


def test_commands_without_deployments_explain() -> None:
    result = runner.invoke(app, ["env"])
    assert result.exit_code == 1
    assert "no deployments" in result.output
