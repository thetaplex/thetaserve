import sys

import pytest

from thetaserve import state
from thetaserve.config import load_catalog
from thetaserve.profiles import resolve
from thetaserve.state import Deployment, DeploymentNotFound


def _deployment(name: str) -> Deployment:
    spec = resolve(
        load_catalog(),
        provider="runpod",
        device="a6000-48g",
        model="qwen3.8-27b",
        port=8000,
        idle_s=60,
        max_life_s=60,
        name=name,
    )
    return Deployment(
        name=name, provider="runpod", provider_id=f"pod-{name}", spec=spec, status="RUNNING"
    )


def test_round_trip_and_private_mode() -> None:
    state.put(_deployment("a"))
    loaded = state.load()
    assert loaded.deployments["a"].provider_id == "pod-a"
    if sys.platform != "win32":
        assert state.path().stat().st_mode & 0o777 == 0o600


def test_state_holds_no_secret_values() -> None:
    state.put(_deployment("a"))
    text = state.path().read_text(encoding="utf-8")
    assert "api_key" not in text.replace('"--api-key"', "")


def test_remove() -> None:
    state.put(_deployment("a"))
    state.remove("a")
    state.remove("a")  # idempotent
    assert state.load().deployments == {}


def test_resolve_name() -> None:
    with pytest.raises(DeploymentNotFound, match="no deployments"):
        state.resolve_name(state.load(), None)
    state.put(_deployment("a"))
    assert state.resolve_name(state.load(), None).name == "a"
    state.put(_deployment("b"))
    with pytest.raises(DeploymentNotFound, match="several deployments; pass a name: a, b"):
        state.resolve_name(state.load(), None)
    with pytest.raises(DeploymentNotFound, match="no deployment named 'c'; known: a, b"):
        state.resolve_name(state.load(), "c")
