# thetaserve

Open-source CLI (PyPI) that launches open-weight models with vLLM on rented GPUs (Runpod in core; the AWS extra comes after v0.1) and returns a secured OpenAI-compatible endpoint. Part of the ThetaPlex organisation.

Read `docs/DESIGN.md` before making changes; it is the source of truth. Its section 12 gives the milestone order and status.

## Stack
Python 3.13+, uv, hatchling, typer + rich, pydantic v2, pyyaml, httpx, keyring, platformdirs. Must run on Linux (including headless), macOS and Windows. Extra: `aws` (boto3). Dev tools live in the `dev` dependency group. Apache-2.0.

## Commands
- `just setup` - install from the lockfile
- `just check` - lint, format check, mypy, tests (what CI runs)
- `just fmt` - format and apply safe lint fixes
- `just run up --provider runpod --model qwen3.8-27b --dry-run` - options and the launch command; offline from packaged samples without a Runpod key
- `just samples` - refresh `catalog/samples.json` (needs the Runpod key)

## Rules
- Tests never hit real provider APIs; mock with respx and moto.
- Never print or log secrets; mask them in output. Persist them only to keyring or, when no keyring backend exists, the mode-600 secrets file (see DESIGN.md section 4).
- No hard-coded POSIX paths; use platformdirs.
- Do not add SkyPilot or other heavy dependencies to core.
- Hand-written values (overhead, family flags, vLLM image) marked TODO stay TODO until measured on real hardware.
- Before changing the Runpod adapter, check the current Runpod REST API v2 spec (https://api.runpod.io/v2/openapi.json) for exact endpoints and fields.
- User docs in `docs/` follow Diátaxis and ASD-STE100 (see `docs/README.md`). When a change alters a command, option, default or file, update `docs/reference/` and CHANGELOG.md in the same PR.
