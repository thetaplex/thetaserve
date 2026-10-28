"""The interface every provider adapter implements (DESIGN.md section 8)."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from thetaserve.config import Arch
from thetaserve.spec import LaunchSpec
from thetaserve.state import Deployment


class ProviderError(Exception):
    """A provider call failed; the message is safe to show the user."""


@dataclass(frozen=True)
class Gpu:
    """One GPU type a provider offers for a single-GPU host."""

    id: str  # provider's GPU type ID, e.g. "NVIDIA A40"
    vram_gb: float
    arch: Arch
    price: float  # USD per hour
    stock: int  # 0 none, 1 low, 2 medium, 3 high
    pool: str | None = None  # Runpod serverless pool, e.g. AMPERE_48


@dataclass(frozen=True)
class Endpoint:
    base_url: str  # OpenAI-compatible base URL, ending in /v1
    needs_tunnel: bool


class Provider(Protocol):
    name: str

    def plan(self, spec: LaunchSpec) -> dict[str, Any]:
        """The request `create` would send, with secrets masked. No network calls."""
        ...

    def create(self, spec: LaunchSpec, secrets: Mapping[str, str]) -> Deployment:
        """Start a host. `secrets` become environment variables of the server process."""
        ...

    def wait_running(self, d: Deployment, timeout_s: int) -> Deployment: ...

    def refresh(self, d: Deployment) -> Deployment:
        """Current status and cost from the provider."""
        ...

    def endpoint(self, d: Deployment) -> Endpoint: ...

    def logs(self, d: Deployment, follow: bool, tail: int = 100) -> Iterator[str]: ...

    def stop(self, d: Deployment) -> Deployment: ...

    def start(self, d: Deployment) -> Deployment: ...

    def destroy(self, d: Deployment, purge: bool = False) -> None: ...

    def price_per_hour(self, spec: LaunchSpec) -> float | None: ...
