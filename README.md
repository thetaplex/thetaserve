# thetaserve

[![CI](https://github.com/thetaplex/thetaserve/actions/workflows/ci.yml/badge.svg)](https://github.com/thetaplex/thetaserve/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](pyproject.toml)

One command to launch an open-weight model with [vLLM](https://docs.vllm.ai) on a rented GPU and get back a secured, OpenAI-compatible endpoint.

> **Status: pre-alpha.** Milestone M0 is done: the catalogues, the resolver and `up --dry-run` work. Launching on a provider arrives in M1. See the [build order](docs/DESIGN.md#12-build-order).

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

Show what would be launched for Qwen3.8-27B on a RunPod RTX A6000:

```sh
uv run thetaserve up --provider runpod --device a6000-48g --model qwen3.8-27b --port 8000 --dry-run
```

This prints the resolved profile, the memory estimate and the exact `vllm serve` command. It needs no credentials and creates nothing.

## Usage

```sh
uv run thetaserve list models      # also: devices, providers
uv run thetaserve up --help
```

`up` options: `--context` overrides the profile's context length, `--kv-cache-dtype` the KV-cache precision, and `--extra-arg` (repeatable) passes extra flags to vLLM. `--force` skips the memory-fit check.

To add or override devices, models or providers, put a `devices.yaml`, `models.yaml` or `providers.yaml` in your user config directory (on Linux `~/.config/thetaserve/`). Entries there replace built-in entries with the same key.

## Links

- [Design](docs/DESIGN.md)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)
- [Security policy](SECURITY.md)

## License

[Apache-2.0](LICENSE)
