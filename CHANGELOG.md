# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Built-in device, model and provider catalogues, overridable from the user config directory.
- Resolver that turns provider x device x model into a launch spec, with a memory-fit check.
- `thetaserve list models|devices|providers`.
- `thetaserve up --dry-run`, printing the resolved profile, the `vllm serve` command and the provider request.
- Runpod adapter (REST API v2): pods on a shared network volume for the weights cache, placed where the GPU is in stock.
- `thetaserve setup`, `up`, `status`, `env`, `logs`, `stop`, `start` and `down`.
- A fresh API key for every launch; secrets from environment variables, the system keyring, or a mode-600 file when no keyring exists.
- Local deployment state in the platform's state directory, without secrets.
- `network_volume: null` in `providers.yaml` launches Runpod pods without a weights cache volume.
- Documentation in `docs/`: tutorial, how-to guides, reference and explanation.
- ADR 0001 (accepted): derive launch options from live model and GPU data.
- `up` without `--device` lists the cheapest GPU in stock for every weight precision (16, 8, 4 bits) and KV-cache precision (16-bit, FP8) that fits the model, from Hugging Face metadata and the Runpod GPU catalogue, marks a recommended row, and asks which to launch. `--use coding|chat|research` (presets in `presets.yaml`) sets the context and the precision floor for the recommendation; `--min-precision` hides lower precisions; each row shows estimated tokens per second from GPU memory bandwidth (`bandwidth.yaml`) and how many full-context sequences the KV cache holds; `--model` also takes a Hugging Face repo.
