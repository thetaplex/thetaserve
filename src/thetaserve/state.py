"""Local record of deployments. Never holds secrets (DESIGN.md section 4)."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import platformdirs
from pydantic import BaseModel, Field

from thetaserve._files import write_private
from thetaserve.spec import LaunchSpec

FILE_NAME = "state.json"


class DeploymentNotFound(Exception):
    """No deployment matches; the message says what exists."""


class Deployment(BaseModel):
    name: str
    provider: str
    provider_id: str
    spec: LaunchSpec
    status: str
    base_url: str | None = None
    needs_tunnel: bool = False
    cost_per_hr: float | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    max_context: int | None = None
    # Provider-specific details the adapter needs later, e.g. a volume ID.
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def secret_name(self) -> str:
        """Secrets-store key for this deployment's vLLM API key."""
        return f"deployment/{self.name}/api-key"


class State(BaseModel):
    schema_version: int = 1
    deployments: dict[str, Deployment] = Field(default_factory=dict)


def state_dir() -> Path:
    return Path(platformdirs.user_state_dir("thetaserve", appauthor=False))


def path() -> Path:
    return state_dir() / FILE_NAME


def load() -> State:
    p = path()
    if not p.is_file():
        return State()
    return State.model_validate_json(p.read_text(encoding="utf-8"))


# ponytail: load-modify-save without a lock; two thetaserve runs changing state at the same
# moment can lose one update. Add a file lock if concurrent use becomes real.
def save(state: State) -> None:
    write_private(path(), state.model_dump_json(indent=2))


def put(deployment: Deployment) -> None:
    state = load()
    state.deployments[deployment.name] = deployment
    save(state)


def remove(name: str) -> None:
    state = load()
    if state.deployments.pop(name, None) is not None:
        save(state)


def resolve_name(state: State, name: str | None) -> Deployment:
    """The named deployment, or the only one when no name is given."""
    if name is not None:
        if name not in state.deployments:
            known = ", ".join(sorted(state.deployments)) or "none"
            raise DeploymentNotFound(f"no deployment named {name!r}; known: {known}")
        return state.deployments[name]
    if len(state.deployments) == 1:
        return next(iter(state.deployments.values()))
    if not state.deployments:
        raise DeploymentNotFound("no deployments; launch one with `thetaserve up`")
    raise DeploymentNotFound(
        f"several deployments; pass a name: {', '.join(sorted(state.deployments))}"
    )
