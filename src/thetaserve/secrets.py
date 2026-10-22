"""Secret lookup and storage (DESIGN.md section 4).

Lookup order, first hit wins: environment variable, keyring (when a real backend exists),
then `secrets.json` in the user config directory. Writes go to keyring when available,
otherwise to that file.
"""

import contextlib
import json
import os
import secrets as _random
from pathlib import Path
from typing import Literal

import keyring
import keyring.errors
from keyring.backends import fail, null

from thetaserve import config
from thetaserve._files import write_private

SERVICE = "thetaserve"
FILE_NAME = "secrets.json"

type Store = Literal["keyring", "file"]


def new_api_key() -> str:
    """A fresh per-launch key: 32 random bytes as hex.

    Not base64url: a key that starts with "-" is read by `vllm serve` as an option.
    """
    return _random.token_hex(32)


def mask(value: str) -> str:
    return "****" if len(value) < 12 else f"{value[:4]}…{value[-2:]}"


def get(name: str, env: str | None = None) -> str | None:
    if env and (value := os.environ.get(env)):
        return value
    if _keyring_ok():
        try:
            if (value := keyring.get_password(SERVICE, name)) is not None:
                return value
        except keyring.errors.KeyringError:
            pass
    return _read_file().get(name)


def put(name: str, value: str) -> Store:
    """Store a secret; returns where it went so the CLI can say so."""
    if _keyring_ok():
        try:
            keyring.set_password(SERVICE, name, value)
            return "keyring"
        except keyring.errors.KeyringError:
            pass
    data = _read_file()
    data[name] = value
    write_private(_file(), json.dumps(data, indent=2))
    return "file"


def delete(name: str) -> None:
    if _keyring_ok():
        # PasswordDeleteError when the entry is absent, which is fine.
        with contextlib.suppress(keyring.errors.KeyringError):
            keyring.delete_password(SERVICE, name)
    data = _read_file()
    if data.pop(name, None) is not None:
        write_private(_file(), json.dumps(data, indent=2))


def file_path() -> Path:
    return _file()


def _file() -> Path:
    return config.user_config_dir() / FILE_NAME


def _keyring_ok() -> bool:
    """False for keyring's 'no backend' stand-ins, e.g. on headless Linux without D-Bus."""
    return not isinstance(keyring.get_keyring(), fail.Keyring | null.Keyring)


def _read_file() -> dict[str, str]:
    path = _file()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return {str(k): str(v) for k, v in data.items()}
