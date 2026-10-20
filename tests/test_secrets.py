import os
import sys
from pathlib import Path

import keyring
import pytest
from keyring.backends import fail

from tests.conftest import MemoryKeyring
from thetaserve import secrets


def test_env_wins_over_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets.put("runpod", "from-keyring")
    monkeypatch.setenv("RUNPOD_API_KEY", "from-env")
    assert secrets.get("runpod", env="RUNPOD_API_KEY") == "from-env"


def test_keyring_used_when_available(memory_keyring: MemoryKeyring, user_dir: Path) -> None:
    assert secrets.put("runpod", "k") == "keyring"
    assert memory_keyring.items[("thetaserve", "runpod")] == "k"
    assert secrets.get("runpod") == "k"
    assert not (user_dir / secrets.FILE_NAME).exists()


def test_file_fallback_without_keyring(user_dir: Path) -> None:
    keyring.set_keyring(fail.Keyring())  # type: ignore[no-untyped-call]
    assert secrets.put("runpod", "k") == "file"
    assert secrets.get("runpod") == "k"
    path = user_dir / secrets.FILE_NAME
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600
    secrets.delete("runpod")
    assert secrets.get("runpod") is None


def test_delete_missing_is_fine() -> None:
    secrets.delete("never-stored")


def test_api_keys_are_long_and_unique() -> None:
    keys = {secrets.new_api_key() for _ in range(100)}
    assert len(keys) == 100
    assert all(len(k) >= 43 for k in keys)  # 32 bytes, base64url


def test_mask_hides_most_of_the_value() -> None:
    assert secrets.mask("short") == "****"
    masked = secrets.mask("abcdefghijklmnop")
    assert masked == "abcd…op"
    assert "efgh" not in masked


def test_no_real_keyring_touched() -> None:
    assert isinstance(keyring.get_keyring(), MemoryKeyring)
    assert "RUNPOD_API_KEY" not in os.environ
