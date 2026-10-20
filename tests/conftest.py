from collections.abc import Iterator
from pathlib import Path

import keyring
import keyring.backend
import keyring.errors
import pytest

from thetaserve import config, state


class MemoryKeyring(keyring.backend.KeyringBackend):
    """In-process keyring so tests never read or write the real system keyring."""

    priority = 1

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.items.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.items[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self.items.pop((service, username), None) is None:
            raise keyring.errors.PasswordDeleteError(username)


@pytest.fixture(autouse=True)
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point config and state at empty temp dirs so real user files never leak in."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    monkeypatch.setattr(config, "user_config_dir", lambda: cfg)
    monkeypatch.setattr(state, "state_dir", lambda: tmp_path / "state")
    for var in ("RUNPOD_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return cfg


@pytest.fixture(autouse=True)
def memory_keyring() -> Iterator[MemoryKeyring]:
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    yield backend
    keyring.set_keyring(previous)
