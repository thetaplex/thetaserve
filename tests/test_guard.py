import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import thetaserve.guard
from thetaserve.guard import RequireKey
from thetaserve.spec import GUARD_DIR, LaunchSpec


def call(path: str, auth: bytes | None = None, method: str = "GET") -> int:
    async def app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    headers = [(b"authorization", auth)] if auth else []
    scope = {"type": "http", "method": method, "path": path, "headers": headers}
    asyncio.run(RequireKey(app)(scope, None, send))
    status: int = sent[0]["status"]
    return status


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "k")


@pytest.mark.parametrize("path", ["/metrics", "/load", "/version", "/v1/models"])
def test_guard_requires_key(path: str) -> None:
    assert call(path) == 401
    assert call(path, b"Bearer wrong") == 401
    assert call(path, b"Bearer k") == 200


def test_guard_leaves_health_and_preflight_open() -> None:
    assert call("/health") == 200
    assert call("/metrics", method="OPTIONS") == 200


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="needs bash")
def test_script_installs_guard_and_execs_vllm(spec: LaunchSpec, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "vllm"  # prints what vLLM would get
    fake.write_text('#!/bin/sh\necho "$PYTHONPATH"\necho "$@"\n')
    fake.chmod(0o755)
    guard_dir = tmp_path / "guard"
    script = spec.script().replace(GUARD_DIR, str(guard_dir))
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "VLLM_API_KEY": "k"}
    out = subprocess.run(  # noqa: S603
        ["bash", "-c", script],  # noqa: S607
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert out[0] == str(guard_dir)
    assert "--middleware thetaserve_guard.RequireKey" in out[1]
    assert "--api-key k " in out[1]

    assert (guard_dir / "thetaserve_guard.py").read_text() == Path(
        thetaserve.guard.__file__
    ).read_text()
