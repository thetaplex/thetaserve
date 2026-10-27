"""Runpod adapter against a mocked REST API v2. No request ever leaves the process."""

import json
from typing import Any

import httpx
import pytest
import respx

from thetaserve.config import load_catalog
from thetaserve.providers.base import ProviderError
from thetaserve.providers.runpod import API_URL, RunPod, parse_gpus
from thetaserve.spec import LaunchSpec
from thetaserve.state import Deployment

GPU = "NVIDIA%20RTX%20A6000"
VOLUMES = f"{API_URL}/v2/network-volumes"


def _pod(status: str = "PROVISIONING", **extra: Any) -> dict[str, Any]:
    return {"id": "pod1", "name": "thetaserve-x", "status": status, "cost": 0.59, **extra}


def _volumes(*items: dict[str, Any]) -> dict[str, Any]:
    return {"networkVolumes": list(items), "pagination": {"nextCursor": None, "hasNextPage": False}}


def _stock(**levels: str) -> dict[str, Any]:
    return {
        "id": "NVIDIA RTX A6000",
        "price": {"secure": 0.49, "community": 0.33},
        "dataCenters": [{"id": dc, "name": dc, "availability": lvl} for dc, lvl in levels.items()],
    }


@pytest.fixture
def runpod() -> RunPod:
    return RunPod(load_catalog().providers.runpod, "rp-key", sleep=lambda _s: None)


@pytest.fixture
def deployed(spec: LaunchSpec) -> Deployment:
    return Deployment(
        name="dev",
        provider="runpod",
        provider_id="pod1",
        spec=spec,
        status="RUNNING",
        extra={"volume_id": "vol1", "data_center": "US-TX-3"},
    )


@respx.mock(assert_all_called=False)
def test_plan_makes_no_requests_and_masks_secrets(runpod: RunPod, spec: LaunchSpec) -> None:
    body = runpod.plan(spec)
    assert not respx.calls
    assert body["env"]["VLLM_API_KEY"] == "****"
    assert body["gpu"] == {"id": "NVIDIA RTX A6000", "count": 1}
    assert body["ports"] == ["8000/http"]
    assert body["entrypoint"] == ["/bin/bash", "-c"]
    assert body["cmd"] == [spec.script()]


def test_plan_works_without_credentials(spec: LaunchSpec) -> None:
    assert RunPod(load_catalog().providers.runpod, None).plan(spec)["name"] == "thetaserve-dev"


@respx.mock
def test_create_reuses_volume_where_gpu_is_in_stock(runpod: RunPod, spec: LaunchSpec) -> None:
    respx.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(
        json=_stock(**{"EU-RO-1": "NONE", "US-TX-3": "LOW"})
    )
    respx.get(VOLUMES).respond(
        json=_volumes(
            {"id": "vol-eu", "name": "thetaserve-cache", "dataCenter": "EU-RO-1", "size": 100},
            {"id": "vol-us", "name": "thetaserve-cache", "dataCenter": "US-TX-3", "size": 100},
            {"id": "other", "name": "someone-else", "dataCenter": "US-TX-3", "size": 10},
        )
    )
    create = respx.post(f"{API_URL}/v2/pods").respond(201, json=_pod())

    d = runpod.create(spec, {"VLLM_API_KEY": "secret-key"})

    body = json.loads(create.calls.last.request.content)
    assert body["mounts"]["network"] == [{"volumeId": "vol-us", "path": "/runpod-volume"}]
    assert body["dataCenterIds"] == ["US-TX-3"]
    assert body["env"] == {
        "HF_HOME": "/runpod-volume/huggingface",
        "VLLM_CACHE_ROOT": "/runpod-volume/vllm",
        "VLLM_API_KEY": "secret-key",
    }
    assert create.calls.last.request.headers["Authorization"] == "Bearer rp-key"
    assert d.provider_id == "pod1"
    assert d.base_url == "https://pod1-8000.proxy.runpod.net/v1"
    assert d.extra == {"volume_id": "vol-us", "data_center": "US-TX-3"}


@respx.mock
def test_create_makes_volume_in_best_stocked_data_centre(runpod: RunPod, spec: LaunchSpec) -> None:
    respx.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(
        json=_stock(**{"EU-RO-1": "LOW", "US-TX-3": "HIGH"})
    )
    respx.get(VOLUMES).respond(json=_volumes())
    make = respx.post(VOLUMES).respond(
        201,
        json={"id": "vol-new", "name": "thetaserve-cache", "dataCenter": "US-TX-3", "size": 100},
    )
    respx.post(f"{API_URL}/v2/pods").respond(201, json=_pod())

    d = runpod.create(spec, {})

    assert json.loads(make.calls.last.request.content) == {
        "name": "thetaserve-cache",
        "size": 100,
        "dataCenter": "US-TX-3",
    }
    assert d.extra["volume_id"] == "vol-new"


@respx.mock
def test_create_fails_clearly_when_sold_out_at_volume(runpod: RunPod, spec: LaunchSpec) -> None:
    respx.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(json=_stock(**{"EU-RO-1": "NONE"}))
    respx.get(VOLUMES).respond(
        json=_volumes({"id": "v", "name": "thetaserve-cache", "dataCenter": "EU-RO-1", "size": 1})
    )
    with pytest.raises(ProviderError, match=r"sold out where your 'thetaserve-cache' volume"):
        runpod.create(spec, {})


@respx.mock
def test_volume_listing_follows_pagination(runpod: RunPod, spec: LaunchSpec) -> None:
    respx.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(json=_stock(**{"US-TX-3": "HIGH"}))
    respx.get(VOLUMES, params={"cursor": "p2"}).respond(
        json=_volumes({"id": "v2", "name": "thetaserve-cache", "dataCenter": "US-TX-3", "size": 1})
    )
    respx.get(VOLUMES).respond(
        json={"networkVolumes": [], "pagination": {"nextCursor": "p2", "hasNextPage": True}}
    )
    respx.post(f"{API_URL}/v2/pods").respond(201, json=_pod())
    assert runpod.create(spec, {}).extra["volume_id"] == "v2"


@respx.mock
def test_wait_running_polls_until_running(runpod: RunPod, deployed: Deployment) -> None:
    respx.get(f"{API_URL}/v2/pods/pod1").mock(
        side_effect=[
            httpx.Response(200, json=_pod("PROVISIONING")),
            httpx.Response(200, json=_pod("STARTING")),
            httpx.Response(200, json=_pod("RUNNING", startedAt="2026-10-20T12:00:00Z")),
        ]
    )
    d = runpod.wait_running(deployed, 600)
    assert d.status == "RUNNING"
    assert d.started_at is not None
    assert d.started_at.year == 2026


@respx.mock
def test_wait_running_fails_on_error_status(runpod: RunPod, deployed: Deployment) -> None:
    respx.get(f"{API_URL}/v2/pods/pod1").respond(json=_pod("ERROR"))
    with pytest.raises(ProviderError, match="pod pod1 is ERROR"):
        runpod.wait_running(deployed, 600)


@respx.mock
def test_refresh_treats_missing_pod_as_terminated(runpod: RunPod, deployed: Deployment) -> None:
    respx.get(f"{API_URL}/v2/pods/pod1").respond(404, json={"title": "Not Found", "status": 404})
    assert runpod.refresh(deployed).status == "TERMINATED"


@respx.mock
@pytest.mark.parametrize(("action", "status"), [("stop", "EXITED"), ("start", "STARTING")])
def test_stop_and_start_send_actions(
    runpod: RunPod, deployed: Deployment, action: str, status: str
) -> None:
    route = respx.post(f"{API_URL}/v2/pods/pod1/action").respond(json=_pod(status))
    d = getattr(runpod, action)(deployed)
    assert json.loads(route.calls.last.request.content) == {"action": action}
    assert d.status == status


@respx.mock
def test_invalid_action_surfaces_runpod_detail(runpod: RunPod, deployed: Deployment) -> None:
    respx.post(f"{API_URL}/v2/pods/pod1/action").respond(
        409, json={"title": "Conflict", "status": 409, "detail": "pod is not running"}
    )
    with pytest.raises(ProviderError, match="Runpod API 409: pod is not running"):
        runpod.stop(deployed)


@respx.mock
def test_destroy_keeps_volume_unless_purge(runpod: RunPod, deployed: Deployment) -> None:
    pod = respx.delete(f"{API_URL}/v2/pods/pod1").respond(204)
    vol = respx.delete(f"{VOLUMES}/vol1").respond(204)
    runpod.destroy(deployed)
    assert pod.call_count == 1
    assert vol.call_count == 0
    runpod.destroy(deployed, purge=True)
    assert vol.call_count == 1


@respx.mock
def test_logs_parse_server_sent_events(runpod: RunPod, deployed: Deployment) -> None:
    sse = (
        'id: 1\nevent: log\ndata: {"source": "container", "line": "INFO starting", "ts": "1"}\n\n'
        'data: {"source": "container", "line": "GPU KV cache size: 10 tokens", "ts": "2"}\n\n'
        ": keep-alive comment\n\n"
    )
    route = respx.get(f"{API_URL}/v2/pods/pod1/logs").respond(
        text=sse, headers={"Content-Type": "text/event-stream"}
    )
    assert list(runpod.logs(deployed, follow=False, tail=50)) == [
        "INFO starting",
        "GPU KV cache size: 10 tokens",
    ]
    assert route.calls.last.request.url.params["tail"] == "50"
    assert route.calls.last.request.url.params["source"] == "container"


@respx.mock
def test_rate_limit_is_retried(runpod: RunPod, deployed: Deployment) -> None:
    respx.get(f"{API_URL}/v2/pods/pod1").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json=_pod("RUNNING")),
        ]
    )
    assert runpod.refresh(deployed).status == "RUNNING"


@respx.mock
def test_bad_key_message(runpod: RunPod, spec: LaunchSpec) -> None:
    respx.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(401)
    with pytest.raises(ProviderError, match=r"rejected the API key \(401\)"):
        runpod.price_per_hour(spec)


def test_missing_key_message(spec: LaunchSpec) -> None:
    with pytest.raises(ProviderError, match="no Runpod API key: set RUNPOD_API_KEY"):
        RunPod(load_catalog().providers.runpod, None).price_per_hour(spec)


@respx.mock
def test_price_per_hour_uses_configured_cloud(runpod: RunPod, spec: LaunchSpec) -> None:
    respx.get(f"{API_URL}/v2/catalog/gpus/{GPU}").respond(json=_stock())
    assert runpod.price_per_hour(spec) == pytest.approx(0.49)


@respx.mock
def test_community_cloud_launches_without_the_cache_volume(spec: LaunchSpec) -> None:
    # The built-in config names a cache volume; Community Cloud pods cannot mount one.
    cfg = load_catalog().providers.runpod.model_copy(update={"cloud_type": "COMMUNITY"})
    create = respx.post(f"{API_URL}/v2/pods").respond(201, json=_pod(cost=0.33))

    d = RunPod(cfg, "rp-key").create(spec, {"VLLM_API_KEY": "secret-key"})

    assert d.cost_per_hr == pytest.approx(0.33)  # community tier, not the secure price
    body = json.loads(create.calls.last.request.content)
    assert body["cloud"] == "COMMUNITY"
    assert "mounts" not in body
    assert "dataCenterIds" not in body
    assert body["disk"] >= spec.fit.weights_gb  # the weights go on the container disk
    assert d.extra == {}


@respx.mock
def test_create_without_cache_volume_lets_runpod_place_the_pod(spec: LaunchSpec) -> None:
    cfg = load_catalog().providers.runpod.model_copy(update={"network_volume": None})
    create = respx.post(f"{API_URL}/v2/pods").respond(201, json=_pod())

    d = RunPod(cfg, "rp-key").create(spec, {"VLLM_API_KEY": "secret-key"})

    body = json.loads(create.calls.last.request.content)
    assert len(respx.calls) == 1
    assert body["env"] == {"VLLM_API_KEY": "secret-key"}
    assert "mounts" not in body
    assert "dataCenterIds" not in body
    assert d.extra == {}
    assert "mounts" not in RunPod(cfg, None).plan(spec)


def test_gpus_lists_only_types_the_cloud_offers() -> None:
    # Runpod quotes a community price for Secure-only types such as the A40.
    a40 = {"id": "NVIDIA A40", "manufacturer": "NVIDIA", "memory": 48, "pool": "AMPERE_48",
           "secure": True, "community": False,
           "price": {"secure": 0.59, "community": 0.35}}  # fmt: skip
    listing = {"gpus": [a40]}
    assert [g.id for g in parse_gpus(listing, "SECURE")] == ["NVIDIA A40"]
    assert parse_gpus(listing, "COMMUNITY") == []
