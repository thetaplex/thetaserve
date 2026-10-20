"""Runpod adapter, against the Runpod REST API v2 (https://api.runpod.io/v2/openapi.json)."""

import json
import re
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from thetaserve.config import Arch, RunPodConfig
from thetaserve.profiles import LaunchSpec
from thetaserve.providers.base import Endpoint, Gpu, ProviderError
from thetaserve.state import Deployment

API_URL = "https://api.runpod.io"
VOLUME_PATH = "/runpod-volume"
HF_HOME = f"{VOLUME_PATH}/huggingface"
MASK = "****"
POLL_S = 5.0
_AVAILABILITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
_FAILED = {"ERROR", "TERMINATED", "EXITED"}
_POOL_ARCH: dict[str, Arch] = {
    "AMPERE": "ampere", "ADA": "ada", "HOPPER": "hopper", "BLACKWELL": "blackwell"
}  # fmt: skip


class RunPod:
    name = "runpod"

    def __init__(
        self,
        cfg: RunPodConfig,
        api_key: str | None,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self._api_key = api_key
        self._client = client
        self._sleep = sleep

    # -- no network ---------------------------------------------------------------------

    def plan(self, spec: LaunchSpec) -> dict[str, Any]:
        if not self.cfg.network_volume:
            return self._body(spec, env=dict.fromkeys(spec.secret_env, MASK))
        return self._body(
            spec,
            env=dict.fromkeys(spec.secret_env, MASK),
            volume_id=f"<{self.cfg.network_volume} volume, created on first use>",
            data_center="<data centre of the volume>",
        )

    def endpoint(self, d: Deployment) -> Endpoint:
        return Endpoint(f"https://{d.provider_id}-{d.spec.port}.proxy.runpod.net/v1", False)

    # -- lifecycle ----------------------------------------------------------------------

    def create(self, spec: LaunchSpec, secrets: Mapping[str, str]) -> Deployment:
        volume_id = data_center = None
        if self.cfg.network_volume:
            volume_id, data_center = self._ensure_volume(_gpu_id(spec))
        body = self._body(spec, env=dict(secrets), volume_id=volume_id, data_center=data_center)
        pod = self._request("POST", "/v2/pods", json=body).json()
        d = Deployment(
            name=spec.name,
            provider=self.name,
            provider_id=pod["id"],
            spec=spec,
            status=pod["status"],
            cost_per_hr=pod.get("cost"),
            extra={"volume_id": volume_id, "data_center": data_center} if volume_id else {},
        )
        d.base_url = self.endpoint(d).base_url
        return d

    def wait_running(self, d: Deployment, timeout_s: int) -> Deployment:
        deadline = time.monotonic() + timeout_s
        while True:
            d = self.refresh(d)
            if d.status == "RUNNING":
                return d
            if d.status in _FAILED:
                raise ProviderError(
                    f"pod {d.provider_id} is {d.status}; see `thetaserve logs {d.name}`"
                )
            if time.monotonic() >= deadline:
                raise ProviderError(
                    f"pod {d.provider_id} still {d.status} after {timeout_s} s; "
                    f"check `thetaserve logs {d.name}` or `thetaserve down {d.name}`"
                )
            self._sleep(POLL_S)

    def refresh(self, d: Deployment) -> Deployment:
        resp = self._request("GET", f"/v2/pods/{d.provider_id}", ok=(404,))
        if resp.status_code == 404:
            return d.model_copy(update={"status": "TERMINATED", "cost_per_hr": 0.0})
        return _apply(d, resp.json())

    def stop(self, d: Deployment) -> Deployment:
        return self._action(d, "stop")

    def start(self, d: Deployment) -> Deployment:
        # ponytail: if the GPU type is gone from this data centre the start fails with
        # Runpod's message; offering a fresh pod on the same volume (DESIGN.md 8.1) comes later.
        return self._action(d, "start")

    def destroy(self, d: Deployment, purge: bool = False) -> None:
        self._request("DELETE", f"/v2/pods/{d.provider_id}", ok=(404,))
        if purge and (volume_id := d.extra.get("volume_id")):
            self._request("DELETE", f"/v2/network-volumes/{volume_id}", ok=(404,))

    def logs(self, d: Deployment, follow: bool, tail: int = 100) -> Iterator[str]:
        """Container log lines. Without `follow`, stops once the backlog has been sent."""
        # Runpod has no "backlog done" marker; a few quiet seconds means we have it all.
        timeout = httpx.Timeout(30.0, read=None if follow else 3.0)
        params = {"source": "container", "tail": str(tail)}
        try:
            with self._http().stream(
                "GET", f"/v2/pods/{d.provider_id}/logs", params=params, timeout=timeout
            ) as resp:
                if not resp.is_success:
                    resp.read()
                    _raise(resp)
                for raw in resp.iter_lines():
                    if raw.startswith("data:") and (line := _log_line(raw[5:])) is not None:
                        yield line
        except httpx.ReadTimeout:
            if follow:
                raise
        except httpx.HTTPError as exc:
            raise ProviderError(f"Runpod logs request failed: {exc}") from exc

    def gpus(self) -> list[Gpu]:
        """NVIDIA GPU types offered for one-GPU pods, with this cloud's price and stock."""
        listing = self._request(
            "GET",
            "/v2/catalog/gpus",
            params={"include": "AVAILABILITY", "product": "POD", "cloud": self.cfg.cloud_type},
        ).json()
        found = []
        for g in listing["gpus"]:
            price = (g.get("price") or {}).get(self.cfg.cloud_type.lower())
            arch = _arch(g)
            if g.get("manufacturer") == "NVIDIA" and price and arch and g.get("memory"):
                stock = _AVAILABILITY_RANK.get(g.get("availability", "NONE"), 0)
                found.append(Gpu(g["id"], float(g["memory"]), arch, float(price), stock))
        return found

    def price_per_hour(self, spec: LaunchSpec) -> float | None:
        gpu = self._request("GET", f"/v2/catalog/gpus/{quote(_gpu_id(spec), safe='')}").json()
        price = gpu.get("price", {}).get(self.cfg.cloud_type.lower())
        return float(price) if price is not None else None

    # -- helpers ------------------------------------------------------------------------

    def _body(
        self,
        spec: LaunchSpec,
        *,
        env: dict[str, str],
        volume_id: str | None = None,
        data_center: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "name": f"thetaserve-{spec.name}",
            "image": spec.image,
            # A shell so "$VLLM_API_KEY" in the command expands from the pod's environment;
            # exec makes vLLM the process that receives stop signals.
            "entrypoint": ["/bin/bash", "-c"],
            "cmd": [f"exec {spec.command()}"],
            "env": env,
            "ports": [f"{spec.port}/http"],
            "disk": self.cfg.container_disk_gb,
            "cloud": self.cfg.cloud_type,
            "gpu": {"id": _gpu_id(spec), "count": 1},
        }
        if volume_id:
            body["env"] = {"HF_HOME": HF_HOME, **env}
            body["mounts"] = {"network": [{"volumeId": volume_id, "path": VOLUME_PATH}]}
            body["dataCenterIds"] = [data_center]
        return body

    def _ensure_volume(self, gpu_id: str) -> tuple[str, str]:
        """The cache volume in a data centre that has this GPU, creating one if needed."""
        stock = self._gpu_stock(gpu_id)
        volumes = [v for v in self._list_volumes() if v["name"] == self.cfg.network_volume]
        for v in sorted(volumes, key=lambda v: -stock.get(v["dataCenter"], 0)):
            if stock.get(v["dataCenter"], 0) > 0:
                return v["id"], v["dataCenter"]
        if volumes:
            # ponytail: no automatic second volume; DESIGN.md 8.1 says offer one in another
            # data centre (and try fallback devices). Add when users hit it.
            where = ", ".join(sorted(v["dataCenter"] for v in volumes))
            raise ProviderError(
                f"{gpu_id} is sold out where your {self.cfg.network_volume!r} volume lives "
                f"({where}). Try another --device, or try again later."
            )
        if not stock or max(stock.values()) == 0:
            raise ProviderError(f"{gpu_id} has no {self.cfg.cloud_type} capacity right now")
        data_center = max(stock, key=lambda dc: stock[dc])
        created = self._request(
            "POST",
            "/v2/network-volumes",
            json={
                "name": self.cfg.network_volume,
                "size": self.cfg.network_volume_gb,
                "dataCenter": data_center,
            },
        ).json()
        return created["id"], created["dataCenter"]

    def _gpu_stock(self, gpu_id: str) -> dict[str, int]:
        """Data centre -> availability rank (0 = none) for one GPU of this type."""
        gpu = self._request(
            "GET",
            f"/v2/catalog/gpus/{quote(gpu_id, safe='')}",
            params={"include": "AVAILABILITY", "product": "POD", "cloud": self.cfg.cloud_type},
        ).json()
        return {
            dc["id"]: _AVAILABILITY_RANK.get(dc.get("availability", "NONE"), 0)
            for dc in gpu.get("dataCenters") or []
        }

    def _list_volumes(self) -> list[dict[str, Any]]:
        volumes: list[dict[str, Any]] = []
        params: dict[str, str] = {}
        while True:
            page = self._request("GET", "/v2/network-volumes", params=params).json()
            volumes += page["networkVolumes"]
            cursor = page["pagination"].get("nextCursor")
            if not page["pagination"].get("hasNextPage") or not cursor:
                return volumes
            params = {"cursor": cursor}

    def _action(self, d: Deployment, action: str) -> Deployment:
        resp = self._request("POST", f"/v2/pods/{d.provider_id}/action", json={"action": action})
        return _apply(d, resp.json())

    def _http(self) -> httpx.Client:
        if self._client is None:
            if not self._api_key:
                raise ProviderError(
                    f"no Runpod API key: set {self.cfg.api_key_env} or run `thetaserve setup`"
                )
            self._client = httpx.Client(
                base_url=API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(30.0),
                transport=httpx.HTTPTransport(retries=2),
            )
        return self._client

    def _request(
        self, method: str, url: str, *, ok: tuple[int, ...] = (), **kwargs: Any
    ) -> httpx.Response:
        for attempt in range(4):
            try:
                resp = self._http().request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                raise ProviderError(f"Runpod request failed: {exc}") from exc
            if resp.status_code == 429 and attempt < 3:
                self._sleep(_retry_after(resp))
                continue
            if resp.is_success or resp.status_code in ok:
                return resp
            _raise(resp)
        raise AssertionError("unreachable")


def _arch(gpu: dict[str, Any]) -> Arch | None:
    # ponytail: the pool name is Runpod's serverless grouping, not a spec sheet; some cards sit
    # in another architecture's pool. Blackwell is fixed up by name because FP4 weights need it;
    # other mislabels only make the estimate more cautious. Use a real lookup if that bites.
    if re.search(r"Blackwell|\bB[23]00\b", gpu.get("id", "")):
        return "blackwell"
    return _POOL_ARCH.get(str(gpu.get("pool")).split("_")[0])


def _gpu_id(spec: LaunchSpec) -> str:
    gpu_id = spec.sku.get("gpu_type_id")
    if not gpu_id:
        raise ProviderError(f"device {spec.device!r} has no runpod.gpu_type_id in devices.yaml")
    return gpu_id


def _apply(d: Deployment, pod: dict[str, Any]) -> Deployment:
    started = pod.get("startedAt")
    return d.model_copy(
        update={
            "status": pod["status"],
            "cost_per_hr": pod.get("cost"),
            "started_at": datetime.fromisoformat(started) if started else None,
        }
    )


def _log_line(payload: str) -> str | None:
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return payload.strip() or None
    line = event.get("line") if isinstance(event, dict) else None
    return line if isinstance(line, str) else None


def _retry_after(resp: httpx.Response) -> float:
    try:
        return min(float(resp.headers.get("Retry-After", "5")), 60.0)
    except ValueError:
        return 5.0


def _raise(resp: httpx.Response) -> None:
    if resp.status_code == 401:
        raise ProviderError(
            "Runpod rejected the API key (401). Check it, or run `thetaserve setup`."
        )
    try:
        body = resp.json()
        detail = body.get("detail") or body.get("title") or resp.reason_phrase
        if errors := body.get("errors"):
            detail = f"{detail}: {'; '.join(errors)}"
    except ValueError:
        detail = resp.text[:200] or resp.reason_phrase
    raise ProviderError(f"Runpod API {resp.status_code}: {detail}")
