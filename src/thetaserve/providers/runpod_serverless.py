"""Runpod serverless adapter: vLLM behind a load-balancing endpoint (DESIGN.md section 8.3).

Same REST API v2 and key as the pod adapter. An endpoint costs nothing while it has no worker
running; a request starts one, and Runpod scales it to zero after the idle timeout.
"""

from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import Any

from thetaserve.providers.base import Endpoint, Gpu, ProviderError
from thetaserve.providers.runpod import MASK, MIN_CUDA, RunPod, _disk_for_weights, parse_gpus
from thetaserve.spec import LaunchSpec
from thetaserve.state import Deployment

NAME = "runpod-serverless"
MAX_IDLE_S = 3600  # the API's limit for workers.idleTimeout


class RunPodServerless(RunPod):
    name = NAME

    # -- no network ---------------------------------------------------------------------

    def plan(self, spec: LaunchSpec) -> dict[str, Any]:
        """The endpoint create `create` would send, secrets masked. No network calls."""
        return self._endpoint_body(spec, env=dict.fromkeys(spec.secret_env, MASK))

    def endpoint(self, d: Deployment) -> Endpoint:
        return Endpoint(d.base_url or f"https://{d.provider_id}.api.runpod.ai/v1", False)

    # -- lifecycle ----------------------------------------------------------------------

    def create(self, spec: LaunchSpec, secrets: Mapping[str, str]) -> Deployment:
        body = self._endpoint_body(spec, env=dict(secrets))
        ep = self._request("POST", "/v2/serverless", json=body).json()
        d = Deployment(
            name=spec.name,
            provider=self.name,
            provider_id=ep["id"],
            spec=spec,
            status="SCALED_DOWN",
            cost_per_hr=0.0,
            extra={"rate": self.price_per_hour(spec)},
        )
        base = (ep.get("requestUrls") or {}).get("base")
        d.base_url = f"{base.rstrip('/')}/v1" if base else self.endpoint(d).base_url
        return d

    def wait_running(self, d: Deployment, timeout_s: int) -> Deployment:
        # The endpoint exists once created; the first request (the readiness poll) starts a worker.
        return self.refresh(d)

    def refresh(self, d: Deployment) -> Deployment:
        resp = self._request("GET", f"/v2/serverless/{d.provider_id}", ok=(404,))
        if resp.status_code == 404:
            return d.model_copy(update={"status": "TERMINATED", "cost_per_hr": 0.0})
        if not resp.json().get("workers", {}).get("max"):
            return d.model_copy(update={"status": "STOPPED", "cost_per_hr": 0.0})
        # Only workers that are up are listed; RUNNING is the one billed state.
        workers = self._workers(d)
        running = [w for w in workers if w.get("status") == "RUNNING"]
        if not running:
            unhealthy = any(w.get("status") == "UNHEALTHY" for w in workers)
            status = "UNHEALTHY" if unhealthy else "SCALED_DOWN"
            return d.model_copy(update={"status": status, "cost_per_hr": 0.0})
        started = running[0].get("startedAt")
        return d.model_copy(
            update={
                "status": "RUNNING",
                "cost_per_hr": d.extra.get("rate"),
                "started_at": datetime.fromisoformat(started) if started else d.started_at,
            }
        )

    def stop(self, d: Deployment) -> Deployment:
        """No worker can start until `start`; a running worker is stopped."""
        self._request("PATCH", f"/v2/serverless/{d.provider_id}", json={"workers": {"max": 0}})
        return d.model_copy(update={"status": "STOPPED", "cost_per_hr": 0.0})

    def start(self, d: Deployment) -> Deployment:
        """Allow a worker again; the next request starts it."""
        self._request("PATCH", f"/v2/serverless/{d.provider_id}", json={"workers": {"max": 1}})
        return d.model_copy(update={"status": "SCALED_DOWN"})

    def destroy(self, d: Deployment, purge: bool = False) -> None:
        self._request("DELETE", f"/v2/serverless/{d.provider_id}", ok=(404,))

    def logs(self, d: Deployment, follow: bool, tail: int = 100) -> Iterator[str]:
        """The log of the running worker, else of the newest one; no lines while none is up."""
        workers = sorted(
            self._workers(d),
            key=lambda w: (w.get("status") == "RUNNING", w.get("startedAt") or ""),
        )
        if not workers:
            return iter(())
        route = f"/v2/serverless/{d.provider_id}/workers/{workers[-1]['id']}/logs"
        return self._sse(route, follow, tail)

    def gpus(self, listing: dict[str, Any] | None = None) -> list[Gpu]:
        """NVIDIA GPU types in a serverless pool, with the pool's rate and stock."""
        return [g for g in parse_gpus(listing or self.gpu_listing(), self.price_key) if g.pool]

    @property
    def price_key(self) -> str:
        return "serverless"  # USD per GPU-hour while a worker runs

    @property
    def _product(self) -> dict[str, str]:
        return {"product": "SERVERLESS"}

    def client_key(self) -> str | None:
        """The key clients send. Runpod's gateway accepts only a Runpod API key."""
        return self._api_key

    # -- helpers ------------------------------------------------------------------------

    def _endpoint_body(self, spec: LaunchSpec, *, env: dict[str, str]) -> dict[str, Any]:
        pool = spec.sku.get("gpu_pool")
        if not pool:
            raise ProviderError(f"launch spec {spec.name!r} has no Runpod serverless pool")
        health_port = spec.port - 1 if spec.port == 65535 else spec.port + 1
        return {
            "name": f"thetaserve-{spec.name}",
            "type": "LOAD_BALANCER",
            "image": spec.image,
            "entrypoint": ["/bin/bash", "-c"],
            "cmd": [spec.worker_script(health_port)],
            # The load balancer sends requests to PORT and polls PORT_HEALTH.
            "env": {**env, "PORT": str(spec.port), "PORT_HEALTH": str(health_port)},
            "ports": [f"{spec.port}/http", f"{health_port}/http"],
            "disk": _disk_for_weights(spec, self.cfg),
            # The pool is every card with this memory size and architecture, so the worker may
            # get a sibling of the chosen GPU, at the same rate.
            "gpu": {"pools": [pool], "count": 1, "minCudaVersion": MIN_CUDA},
            # One worker at most, none while idle: billed only while a worker runs.
            "workers": {"min": 0, "max": 1, "idleTimeout": min(spec.idle_s, MAX_IDLE_S)},
            "scaling": {"type": "REQUEST_COUNT", "requestCount": 1},
            "flashboot": "FLASHBOOT",
        }

    def _workers(self, d: Deployment) -> list[dict[str, Any]]:
        resp = self._request("GET", f"/v2/serverless/{d.provider_id}/workers", ok=(404,))
        workers: list[dict[str, Any]] = [] if resp.status_code == 404 else resp.json()["workers"]
        return workers
