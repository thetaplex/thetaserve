import sys
from collections.abc import Callable

import pytest

from thetaserve import state
from thetaserve.spec import LaunchSpec
from thetaserve.state import Deployment, DeploymentNotFound


@pytest.fixture
def make(spec: LaunchSpec) -> Callable[[str], Deployment]:
    def _deployment(name: str) -> Deployment:
        return Deployment(
            name=name,
            provider="runpod",
            provider_id=f"pod-{name}",
            spec=spec.model_copy(update={"name": name}),
            status="RUNNING",
        )

    return _deployment


def test_round_trip_and_private_mode(make: Callable[[str], Deployment]) -> None:
    state.put(make("a"))
    loaded = state.load()
    assert loaded.deployments["a"].provider_id == "pod-a"
    if sys.platform != "win32":
        assert state.path().stat().st_mode & 0o777 == 0o600


def test_state_holds_no_secret_values(make: Callable[[str], Deployment]) -> None:
    state.put(make("a"))
    text = state.path().read_text(encoding="utf-8")
    assert "api_key" not in text.replace('"--api-key"', "")


def test_remove(make: Callable[[str], Deployment]) -> None:
    state.put(make("a"))
    state.remove("a")
    state.remove("a")  # idempotent
    assert state.load().deployments == {}


def test_resolve_name(make: Callable[[str], Deployment]) -> None:
    with pytest.raises(DeploymentNotFound, match="no deployments"):
        state.resolve_name(state.load(), None)
    state.put(make("a"))
    assert state.resolve_name(state.load(), None).name == "a"
    state.put(make("b"))
    with pytest.raises(DeploymentNotFound, match="several deployments; pass a name: a, b"):
        state.resolve_name(state.load(), None)
    with pytest.raises(DeploymentNotFound, match="no deployment named 'c'; known: a, b"):
        state.resolve_name(state.load(), "c")
