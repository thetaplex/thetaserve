# thetaserve

[![CI](https://github.com/thetaplex/thetaserve/actions/workflows/ci.yml/badge.svg)](https://github.com/thetaplex/thetaserve/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/thetaplex/thetaserve/blob/main/LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://github.com/thetaplex/thetaserve/blob/main/pyproject.toml)

thetaserve launches an open-weight model with [vLLM](https://docs.vllm.ai) on a rented GPU. One command gives you a secured, OpenAI-compatible endpoint.

> **Status: beta (v0.1.0).** Runpod only. On 2026-10-22 it served Qwen3.8-27B from a 48 GB A40 at $0.59/h to a coding agent. Pods stop themselves when idle (`--idle`, default 30 min) or after `--max-life` (default 10 h). See the [build order](https://github.com/thetaplex/thetaserve/blob/main/docs/DESIGN.md#12-build-order).

## Why

Strong open models such as Qwen3.8-27B and Gemma 4 31B run well on one 24-80 GB GPU. But it takes trial and error to find the correct vLLM settings for each model and GPU. You name a **model** and a **use case**. thetaserve reads the model's Hugging Face metadata and the provider's GPU prices and stock. Then it shows the cheapest GPU and settings that fit, and you select one.

## Install

Prerequisite: [uv](https://docs.astral.sh/uv/getting-started/installation/). To contribute, see [CONTRIBUTING.md](https://github.com/thetaplex/thetaserve/blob/main/CONTRIBUTING.md).

```sh
uv tool install thetaserve
```

## Quickstart

Show the options for Qwen3.8-27B as a coding model, and what thetaserve would launch:

```sh
thetaserve up --provider runpod --model qwen3.8-27b --dry-run
```

This shows the cheapest GPU for each weight and KV-cache precision, the memory estimate, the exact `vllm serve` command and the Runpod request body. It creates nothing. It needs no credentials and no network: without a Runpod key, it uses a dated sample of Runpod's GPU prices and stock and of the model's Hugging Face data, and tells you so. The sample holds the built-in models only. Run `thetaserve setup` for live options.

## Usage

```sh
thetaserve list models      # also: providers
thetaserve setup            # store your Runpod API key (and optional Hugging Face token)
thetaserve up --provider runpod --model qwen3.8-27b   # pick from the options
thetaserve status           # endpoint, uptime, cost so far
eval "$(thetaserve env)"    # OPENAI_BASE_URL and OPENAI_API_KEY for your tools
thetaserve logs -f
thetaserve stop             # stop the pod, keep the weights cache; `start` resumes
thetaserve down             # destroy the pod; add --purge to delete the cache volume
```

On Windows, `thetaserve env` prints PowerShell syntax; run `thetaserve env | Invoke-Expression`.

thetaserve reads secrets from `RUNPOD_API_KEY` and `HF_TOKEN` first, then from the system keyring. When no keyring is available (headless Linux, containers), `setup` stores them in a mode-600 file and tells you.

`up` options: `--use` sets the use case (`coding`, `chat`, `research`). `--context` sets the context length. `--kv-cache-dtype` sets the KV-cache precision. `--extra-arg` (repeatable) gives extra flags to vLLM. `--device "NVIDIA A40"` shows only the options on that GPU. `--idle` and `--max-life` set when the pod stops itself. See the [CLI reference](https://github.com/thetaplex/thetaserve/blob/main/docs/reference/cli.md).

To add or change presets, models or providers, put a YAML file with the same name in your user config directory (on Linux `~/.config/thetaserve/`). Its entries replace built-in entries with the same key. See [Override the catalogue](https://github.com/thetaplex/thetaserve/blob/main/docs/how-to/override-catalogue.md).

## Links

- [Documentation](https://github.com/thetaplex/thetaserve/blob/main/docs/README.md): tutorial, how-to guides, reference, explanation
- [Design and decision log](https://github.com/thetaplex/thetaserve/blob/main/docs/DESIGN.md)
- [Contributing](https://github.com/thetaplex/thetaserve/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/thetaplex/thetaserve/blob/main/CHANGELOG.md)
- [Security policy](https://github.com/thetaplex/thetaserve/blob/main/SECURITY.md)

## License

[Apache-2.0](https://github.com/thetaplex/thetaserve/blob/main/LICENSE)
