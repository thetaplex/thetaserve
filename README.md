# thetaserve

[![CI](https://github.com/thetaplex/thetaserve/actions/workflows/ci.yml/badge.svg)](https://github.com/thetaplex/thetaserve/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)

One command to launch an open-weight model with [vLLM](https://docs.vllm.ai) on a rented GPU and get back a secured, OpenAI-compatible endpoint.

> **Status: pre-alpha.** Launching on Runpod is implemented (milestone M1) but not yet run on real hardware, and the built-in profiles still hold `TODO` values until the M2 pilot, so `up` refuses them; the [tutorial](docs/tutorials/first-deployment.md) shows the settings to add. Automatic idle stop arrives in M3: stop pods yourself. See the [build order](docs/DESIGN.md#12-build-order).

## Why

Strong open models such as Qwen3.8-27B and Gemma 4 31B run well on one 24-80 GB GPU, but getting the vLLM settings right for each model and card takes trial and error. thetaserve keeps those settings as tested **profiles**, so you only pick three things: a **provider**, a **device** (GPU class) and a **model**.

## Install

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/) and, for development, [just](https://just.systems/man/en/packages.html).

There is no PyPI release yet. Install from source:

```sh
git clone https://github.com/thetaplex/thetaserve.git
cd thetaserve
just setup
```

## Quickstart

Show what would be launched for Qwen3.8-27B on a Runpod RTX A6000:

```sh
uv run thetaserve up --provider runpod --device a6000-48g --model qwen3.8-27b --port 8000 --dry-run
```

This prints the resolved profile, the memory estimate, the exact `vllm serve` command and the Runpod request body. It needs no credentials and creates nothing.

Leave out `--device` to see the cheapest GPUs in stock that fit the model instead (needs your Runpod key; see `thetaserve setup`):

```sh
uv run thetaserve up --provider runpod --model qwen3.8-27b --use coding --dry-run
```

## Usage

```sh
uv run thetaserve list models      # also: devices, providers
uv run thetaserve setup            # store your Runpod API key (and optional Hugging Face token)
uv run thetaserve up --provider runpod --model qwen3.8-27b   # pick from the options
uv run thetaserve status           # endpoint, uptime, cost so far
eval "$(uv run thetaserve env)"    # OPENAI_BASE_URL and OPENAI_API_KEY for your tools
uv run thetaserve logs -f
uv run thetaserve stop             # pause compute, keep the weights cache
uv run thetaserve down             # destroy the pod; add --purge to delete the cache volume
```

On Windows, `thetaserve env` prints PowerShell syntax; run `uv run thetaserve env | Invoke-Expression`.

Secrets are read from `RUNPOD_API_KEY` and `HF_TOKEN` first, then the system keyring. Where no keyring exists (headless Linux, containers), `setup` stores them in a mode-600 file and says so.

`up` options: `--context` overrides the profile's context length, `--kv-cache-dtype` the KV-cache precision, and `--extra-arg` (repeatable) passes extra flags to vLLM. `--force` skips the memory-fit check.

To add or override devices, models or providers, put a `devices.yaml`, `models.yaml` or `providers.yaml` in your user config directory (on Linux `~/.config/thetaserve/`). Entries there replace built-in entries with the same key. See [Override the catalogue](docs/how-to/override-catalogue.md).

## Links

- [Documentation](docs/README.md): tutorial, how-to guides, reference, explanation
- [Design](docs/DESIGN.md) and [decision records](docs/adr/)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Security policy](SECURITY.md)

## License

[Apache-2.0](LICENSE)
