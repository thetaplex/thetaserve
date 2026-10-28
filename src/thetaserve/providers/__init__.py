"""Provider adapters, looked up by name."""

from thetaserve import secrets
from thetaserve.config import Catalog
from thetaserve.providers.base import Endpoint, Provider, ProviderError
from thetaserve.providers.runpod import RunPod
from thetaserve.providers.runpod_serverless import RunPodServerless

__all__ = ["Endpoint", "Provider", "ProviderError", "get_provider"]

RUNPOD_SECRET = "runpod"  # noqa: S105 - name of a secrets-store entry, not a secret


def get_provider(name: str, catalog: Catalog) -> Provider:
    """The adapter for `name`. Credentials are looked up lazily, so `plan` works without them."""
    match name:
        case "runpod":
            cfg = catalog.providers.runpod
            return RunPod(cfg, secrets.get(RUNPOD_SECRET, env=cfg.api_key_env))
        case "runpod-serverless":  # same account, key and providers.yaml entry as runpod
            cfg = catalog.providers.runpod
            return RunPodServerless(cfg, secrets.get(RUNPOD_SECRET, env=cfg.api_key_env))
        case "aws":
            raise ProviderError("the AWS adapter comes after v0.1; use --provider runpod")
        case _:
            raise ProviderError(f"unknown provider {name!r}")
