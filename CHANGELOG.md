# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Built-in presets, models and providers, overridable from the user config directory.
- `thetaserve list models|providers`.
- `thetaserve up --dry-run`, printing the options, the `vllm serve` command and the provider request. Without a Runpod key it uses a packaged, dated sample of Runpod's GPU catalogue and the built-in models' Hugging Face data, so it works offline; `just samples` refreshes it.
- Runpod adapter (REST API v2): pods on a shared network volume for the weights cache, placed where the GPU is in stock.
- `thetaserve setup`, `up`, `status`, `env`, `logs`, `stop`, `start` and `down`.
- A fresh API key for every launch; secrets from environment variables, the system keyring, or a mode-600 file when no keyring exists.
- Local deployment state in the platform's state directory, without secrets.
- `network_volume: null` in `providers.yaml` launches Runpod pods without a weights cache volume.
- Documentation in `docs/`: tutorial, how-to guides, reference and explanation.
- Design decisions D1 (accepted, derive launch options from live data), D2 (accepted, offer every precision) and D3 (accepted, trusted weight publishers) and D4 (accepted, remove device and tier files), in the decision log of `docs/DESIGN.md`.
- `up` pins the chosen variant to the commit its sizes were read from (`vllm serve --revision`).
- `up` offers weights only from the base model's organisation and the publishers in `publishers.yaml` (`RedHatAI`, `nvidia`); `--any-publisher` lists the rest, marked.
- `up` lists the cheapest GPU in stock for every weight precision (16, 8, 4 bits) and KV-cache precision (16-bit, FP8) that fits the model, cheapest first, from Hugging Face metadata and the Runpod GPU catalogue, and asks which to launch. `--use coding|chat|research` (presets in `presets.yaml`) sets the context and the precision floor for the default; `--min-precision` hides lower precisions; each row shows estimated tokens per second from GPU memory bandwidth (`bandwidth.yaml`); `--model` also takes a Hugging Face repo; `--device` takes a Runpod GPU ID and limits the options to it.
- `env` also shows the model, the model name for requests, the context, the device and the endpoint URLs, on stderr; `env --shell json` prints all values, the key included, as JSON.
- `up` keeps a crashed server's log, key masked, in the state directory's `logs/`.
- `up` stops waiting and reports the error when the vLLM log shows a crash, instead of waiting 40 minutes.
- The built-in vLLM image is `vllm/vllm-openai:v0.32.0`, and Qwen3.8-27B launches with `--max-num-seqs 32`, as run in the M1 pilot on an A40 (`docs/DESIGN.md` section 13.1).
- Install with `uv tool install git+https://github.com/thetaplex/thetaserve`.
- Launched servers require the API key on every path except `/health`, including `/metrics`, which vLLM's `--api-key` leaves open.
