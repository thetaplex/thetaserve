# thetaserve

Open-source CLI (PyPI) that launches open-weight models with vLLM on rented GPUs (RunPod core, AWS extra) and returns a secured OpenAI-compatible endpoint. Part of the ThetaPlex organisation.

Read `docs/DESIGN.md` before making changes; it is the source of truth. Build in the milestone order in its section 12, starting with M0.

## Stack
Python 3.13+, uv, hatchling, typer + rich, pydantic v2, pyyaml, httpx, keyring, platformdirs. Must run on Linux (including headless), macOS and Windows. Extras: `aws` (boto3), `dev` (pytest, respx, moto, ruff, mypy). Apache-2.0.

## Commands
- `uv sync --all-extras` - install
- `uv run pytest` - tests
- `uv run ruff check . && uv run ruff format .` - lint and format
- `uv run thetaserve up --provider runpod --device a6000-48g --model qwen3.8-27b --port 8000 --dry-run`

## Rules
- Tests never hit real provider APIs; mock with respx and moto.
- Never print or log secrets; mask them in output. Persist them only to keyring or, when no keyring backend exists, the mode-600 secrets file (see DESIGN.md section 4).
- No hard-coded POSIX paths; use platformdirs.
- Do not add SkyPilot or other heavy dependencies to core.
- Profile values marked TODO stay TODO until measured on real hardware.
- Before writing the RunPod adapter, check RunPod's current REST API reference for exact endpoints and fields.
