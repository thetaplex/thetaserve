"""Runpod serverless adapter and its CLI flow against a mocked REST API v2."""

import json
import shlex
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import respx
from typer.testing import CliRunner

from thetaserve import healthcheck, secrets, state
from thetaserve.cli import app
from thetaserve.config import Preset, load_catalog
from thetaserve.options import ModelFacts, Weights, suggest, to_spec
from thetaserve.providers.base import Gpu
from thetaserve.providers.runpod import API_URL
from thetaserve.providers.runpod_serverless import RunPodServerless
from thetaserve.spec import LaunchSpec
from thetaserve.state import Deployment

runner = CliRunner()
EP = f"{API_URL}/v2/serverless/ep1"
BASE = "https://ep1.api.runpod.ai"
GPU = "NVIDIA%20RTX%20A6000"
LISTING = {
    "gpus": [
        {"id": "NVIDIA RTX A6000", "manufacturer": "NVIDIA", "memory": 48, "pool": "AMPERE_48",
         "price": {"secure": 0.49, "community": 0.33, "serverless": 0.85}, "availability": "HIGH"},
        {"id": "NVIDIA RTX 3090", "manufacturer": "NVIDIA", "memory": 24, "pool": None,
         "price": {"secure": 0.22}, "availability": "HIGH"},
    ]
}  # fmt: skip
UP = ["up", "--provider", "runpod-serverless", "--model", "qwen3.8-27b",
      "--device", "NVIDIA RTX A6000", "--name", "dev"]  # fmt: skip


def _endpoint(max_workers: int = 1) -> dict[str, Any]:
    return {
        "id": "ep1",
        "name": "thetaserve-dev",
        "type": "LOAD_BALANCER",
        "requestUrls": {"base": BASE, "health": f"{BASE}/ping"},
        "workers": {"min": 0, "max": max_workers, "idleTimeout": 1800},
    }


def _workers(*statuses: str) -> dict[str, Any]:
    return {
        "workers": [
            {"id": f"w{i}", "status": s, "isStale": False, "startedAt": f"2026-10-25T12:0{i}:00Z"}
            for i, s in enumerate(statuses)
        ]
    }


@pytest.fixture
def sl_spec() -> LaunchSpec:
    gpu = Gpu("NVIDIA RTX A6000", 48, "ampere", 0.85, 3, "AMPERE_48")
    weights = Weights("Qwen/Qwen3.8-27B-FP8", "fp8", 8, 28.75, revision="a" * 40)
    facts = ModelFacts("Qwen/Qwen3.8-27B", 262144, 65536, 0)
    option = suggest(facts, [weights], [gpu], Preset(min_context=131072, tools=True, min_bits=8))
    return to_spec(
        option[0],
        load_catalog(),
        provider="runpod-serverless",
        model="qwen3.8-27b",
        common_args=[],
        extra_args=[],
        port=8000,
        idle_s=1800,
        max_life_s=36000,
        name="dev",
    )


@pytest.fixture
def adapter() -> RunPodServerless:
    return RunPodServerless(load_catalog().providers.runpod, "rp-key", sleep=lambda _s: None)


@pytest.fixture
def deployed(sl_spec: LaunchSpec) -> Deployment:
    return Deployment(
        name="dev",
        provider="runpod-serverless",
        provider_id="ep1",
        spec=sl_spec,
        status="SCALED_DOWN",
        base_url=f"{BASE}/v1",
        extra={"rate": 0.85},
    )


def test_spec_has_no_vllm_key_and_names_the_pool(sl_spec: LaunchSpec) -> None:
    # Runpod's gateway checks the key; vLLM's own key would reject the Runpod key clients send.
    assert "--api-key" not in sl_spec.vllm_args
    assert "--middleware" not in sl_spec.vllm_args
    assert sl_spec.secret_env == []
    assert sl_spec.sku == {"gpu_type_id": "NVIDIA RTX A6000", "gpu_pool": "AMPERE_48"}


@respx.mock(assert_all_called=False)
def test_plan_is_a_load_balancer_endpoint_that_scales_to_zero(
    adapter: RunPodServerless, sl_spec: LaunchSpec
) -> None:
    body = adapter.plan(sl_spec)
    assert not respx.calls
    assert body["type"] == "LOAD_BALANCER"
    assert body["gpu"] == {"pools": ["AMPERE_48"], "count": 1, "minCudaVersion": "13.0"}
    assert body["workers"] == {"min": 0, "max": 1, "idleTimeout": 1800}
    assert body["env"] == {"PORT": "8000", "PORT_HEALTH": "8001"}
    assert body["ports"] == ["8000/http", "8001/http"]
    assert body["disk"] >= sl_spec.fit.weights_gb  # no cache volume: weights on the disk
    script = body["cmd"][0]
    assert "thetaserve_healthcheck.py 8000 8001 &" in script
    assert "watchdog" not in script
    assert shlex.split(script.splitlines()[-1])[:3] == ["exec", "vllm", "serve"]


def test_idle_timeout_is_capped_at_the_api_limit(
    adapter: RunPodServerless, sl_spec: LaunchSpec
) -> None:
    body = adapter.plan(sl_spec.model_copy(update={"idle_s": 10 * 3600}))
    assert body["workers"]["idleTimeout"] == 3600


@respx.mock
def test_create_keeps_the_gateway_url_and_rate(
    adapter: RunPodServerless, sl_spec: LaunchSpec
) -> None:
    create = respx.post(f"{API_URL}/v2/serverless").respond(201, json=_endpoint())
    respx.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(json=LISTING["gpus"][0])

    d = adapter.create(sl_spec, {"HF_TOKEN": "hf-secret"})

    assert json.loads(create.calls.last.request.content)["env"]["HF_TOKEN"] == "hf-secret"  # noqa: S105
    assert d.base_url == f"{BASE}/v1"
    assert d.status == "SCALED_DOWN"
    assert d.cost_per_hr == 0.0  # nothing runs yet
    assert d.extra == {"rate": 0.85}


@respx.mock
@pytest.mark.parametrize(
    ("max_workers", "statuses", "status", "cost"),
    [
        (1, ("RUNNING",), "RUNNING", 0.85),
        (1, ("IDLE", "INITIALIZING"), "SCALED_DOWN", 0.0),  # prepared, not billed
        (1, ("UNHEALTHY",), "UNHEALTHY", 0.0),
        (0, (), "STOPPED", 0.0),
    ],
)
def test_refresh_reads_status_from_workers(
    adapter: RunPodServerless,
    deployed: Deployment,
    max_workers: int,
    statuses: tuple[str, ...],
    status: str,
    cost: float,
) -> None:
    respx.get(EP).respond(json=_endpoint(max_workers))
    respx.get(f"{EP}/workers").respond(json=_workers(*statuses))

    d = adapter.refresh(deployed)

    assert (d.status, d.cost_per_hr) == (status, cost)


@respx.mock
def test_refresh_of_deleted_endpoint_is_terminated(
    adapter: RunPodServerless, deployed: Deployment
) -> None:
    respx.get(EP).respond(404, json={"detail": "not found"})
    assert adapter.refresh(deployed).status == "TERMINATED"


@respx.mock
def test_stop_and_start_set_max_workers(adapter: RunPodServerless, deployed: Deployment) -> None:
    patch = respx.patch(EP).respond(json=_endpoint())

    assert adapter.stop(deployed).status == "STOPPED"
    assert json.loads(patch.calls.last.request.content) == {"workers": {"max": 0}}
    assert adapter.start(deployed).status == "SCALED_DOWN"
    assert json.loads(patch.calls.last.request.content) == {"workers": {"max": 1}}


@respx.mock
def test_logs_follow_the_running_worker(adapter: RunPodServerless, deployed: Deployment) -> None:
    respx.get(f"{EP}/workers").respond(json=_workers("RUNNING", "INITIALIZING"))
    logs = respx.get(f"{EP}/workers/w0/logs").respond(
        text='data: {"line": "loading"}\n\n', headers={"Content-Type": "text/event-stream"}
    )

    assert list(adapter.logs(deployed, follow=False)) == ["loading"]
    assert logs.calls.last.request.url.params["source"] == "container"


@respx.mock
def test_logs_without_workers_are_empty(adapter: RunPodServerless, deployed: Deployment) -> None:
    respx.get(f"{EP}/workers").respond(json={"workers": []})
    assert list(adapter.logs(deployed, follow=False)) == []


@respx.mock
def test_gpus_are_pooled_at_the_serverless_rate(adapter: RunPodServerless) -> None:
    route = respx.get(f"{API_URL}/v2/catalog/gpus").respond(json=LISTING)

    gpus = adapter.gpus()

    assert [(g.id, g.price, g.pool) for g in gpus] == [("NVIDIA RTX A6000", 0.85, "AMPERE_48")]
    assert route.calls.last.request.url.params["product"] == "SERVERLESS"


@pytest.mark.parametrize(("answer", "expected"), [(200, 200), (503, 204), (None, 204)])
def test_healthcheck_says_initializing_until_vllm_answers(
    monkeypatch: pytest.MonkeyPatch, answer: int | None, expected: int
) -> None:
    class Resp:
        status = answer

        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *_: object) -> None:
            pass

    def urlopen(url: str, timeout: float) -> Resp:
        if answer is None:
            raise ConnectionRefusedError
        if answer != 200:
            raise urllib.error.HTTPError(url, answer, "down", {}, None)  # type: ignore[arg-type]
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert healthcheck.status("http://127.0.0.1:8000/health") == expected


def test_lifecycle_through_the_cli(
    user_dir: Path, monkeypatch: pytest.MonkeyPatch, hub: respx.MockRouter
) -> None:
    (user_dir / "models.yaml").write_text("vllm_image: vllm/vllm-openai:v0.32.0\n")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    hub.get(f"{API_URL}/v2/catalog/gpus", name="gpus").respond(json=LISTING)
    hub.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(json=LISTING["gpus"][0])
    create = hub.post(f"{API_URL}/v2/serverless").respond(201, json=_endpoint())
    hub.get(EP).respond(json=_endpoint())
    hub.get(f"{EP}/workers").respond(json=_workers("RUNNING"))
    models = hub.get(f"{BASE}/v1/models").respond(json={"data": [{"id": "Qwen/Qwen3.8-27B-FP8"}]})
    hub.get(f"{EP}/workers/w0/logs").respond(
        text='data: {"line": "GPU KV cache size: 150,000 tokens"}\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    delete = hub.delete(EP).respond(204)

    result = runner.invoke(app, [*UP, "--yes"])
    assert result.exit_code == 0, result.output
    assert "$0.85/h while a worker runs" in result.output
    assert "restricted key" in result.output
    assert f"serving at {BASE}/v1" in result.output

    # Clients and the readiness poll use the Runpod key; no vLLM key is made or sent.
    assert "VLLM_API_KEY" not in json.loads(create.calls.last.request.content)["env"]
    assert models.calls.last.request.headers["Authorization"] == "Bearer rp-key"
    assert "rp-key" not in state.path().read_text(encoding="utf-8")
    d = state.load().deployments["dev"]
    assert secrets.get(d.secret_name) == "rp-key"

    result = runner.invoke(app, ["env", "--shell", "posix"])
    assert result.stdout.splitlines() == [
        f"export OPENAI_BASE_URL='{BASE}/v1'",
        "export OPENAI_API_KEY='rp-key'",
    ]

    result = runner.invoke(app, ["down", "dev", "--yes"])
    assert result.exit_code == 0, result.output
    assert delete.call_count == 1
    assert secrets.get(d.secret_name) is None
