# thetaserve

Open-source CLI (PyPI) that launches open-weight models with vLLM on rented GPUs (RunPod core, AWS extra) and returns a secured OpenAI-compatible endpoint. Part of the ThetaPlex organisation.

Read `docs/DESIGN.md` before making changes; it is the source of truth. Build in the milestone order in its section 12, starting with M0.

## Stack
Python 3.13+, uv, hatchling, typer + rich, pydantic v2, pyyaml, httpx, keyring, platformdirs. Must run on Linux (including headless), macOS and Windows. Extra: `aws` (boto3). Dev tools live in the `dev` dependency group. Apache-2.0.

## Commands
- `just setup` - install from the lockfile
- `just check` - lint, format check, mypy, tests (what CI runs)
- `just fmt` - format and apply safe lint fixes
- `just run up --provider runpod --device a6000-48g --model qwen3.8-27b --port 8000 --dry-run`

## Rules
- Tests never hit real provider APIs; mock with respx and moto.
- Never print or log secrets; mask them in output. Persist them only to keyring or, when no keyring backend exists, the mode-600 secrets file (see DESIGN.md section 4).
- No hard-coded POSIX paths; use platformdirs.
- Do not add SkyPilot or other heavy dependencies to core.
- Profile values marked TODO stay TODO until measured on real hardware.
- Before writing the RunPod adapter, check RunPod's current REST API reference for exact endpoints and fields.
