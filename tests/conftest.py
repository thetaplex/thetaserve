from pathlib import Path

import pytest

from thetaserve import config


@pytest.fixture(autouse=True)
def user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the user config directory at an empty temp dir so real user files never leak in."""
    monkeypatch.setattr(config, "user_config_dir", lambda: tmp_path)
    return tmp_path
