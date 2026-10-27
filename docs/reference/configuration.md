# Configuration and files reference

## Locations

Resolved with [platformdirs](https://platformdirs.readthedocs.io/).

| Purpose | Linux | macOS | Windows |
| --- | --- | --- | --- |
| User config (catalogue overrides, `secrets.json`) | `~/.config/thetaserve/` | `~/Library/Application Support/thetaserve/` | `%LOCALAPPDATA%\thetaserve\` |
| State (`state.json`, crash logs in `logs/`) | `~/.local/state/thetaserve/` | `~/Library/Application Support/thetaserve/` | `%LOCALAPPDATA%\thetaserve\` |

The built-in files are in the package at `src/thetaserve/catalog/`. A user file with the same
name overrides them. Its top-level keys replace all of the built-in entry with the same key.

## models.yaml

Top level: `schema_version: 2` (built-in only) and `vllm_image`, the image and tag for all
launches. `up` refuses an image that contains `TODO`.

| Field | Type | Meaning |
| --- | --- | --- |
| `<key>` | string | Short model name for `--model` |
| `hf_family` | string | Base Hugging Face repository |
| `common_args` | list of strings | `vllm serve` arguments for this model family, for example the tool-call and reasoning parsers |
| `kv_scale` | positive number, default 1 | Measured factor on the KV cache per token from `config.json`. Gemma 4 31B: 1.5 |

`--model` also takes a Hugging Face repository. If it is the `hf_family` of an entry, `up` uses
that entry's `common_args` and `kv_scale`. Otherwise `up` launches without family arguments.

## presets.yaml

Use cases for `up --use`.

| Field | Type | Meaning |
| --- | --- | --- |
| `<key>` | string | Preset name |
| `min_context` | positive integer or `null` | Context in tokens; `null` for the model's maximum |
| `tools` | boolean | Warn when no tool-call parser is known for the model |
| `min_bits` | positive integer | Precision floor for the default option. The table still shows lower rows |

Built in: `coding` (131072, tools, 8), `chat` (32768, no tools, 4), `research` (`null`, tools, 8).

## bandwidth.yaml

GPU memory bandwidth in GB/s from vendor data sheets, keyed by Runpod GPU ID. Used only for the
`tok/s` estimate. A MIG slice gets its parent's bandwidth in proportion to its memory.

## publishers.yaml

Hugging Face organisations whose quantised weights `up` offers, keyed by organisation name;
the value says why it is trusted. The base model's own organisation is always trusted. Built
in: `RedHatAI`, `nvidia`. `up --any-publisher` also lists other publishers.

## samples.json

Built in only. A dated sample of the Runpod GPU catalogue (NVIDIA GPUs only) and of the
Hugging Face data of each built-in model, trusted publishers only. `up --dry-run` uses it when no Runpod key is
available. `just samples` fetches it again.

## providers.yaml

`runpod`:

| Field | Built-in value | Required in a user file | Meaning |
| --- | --- | --- | --- |
| `api_key_env` | `RUNPOD_API_KEY` | no (default `RUNPOD_API_KEY`) | Environment variable read first for the key |
| `cloud_type` | `SECURE` | no (default `SECURE`) | `SECURE` or `COMMUNITY` |
| `network_volume` | `thetaserve-cache` | yes | Cache volume name; `null` for no cache volume |
| `network_volume_gb` | `100` | no (default `100`) | Size when the volume is created, GB |
| `container_disk_gb` | `40` | yes | Pod container disk, GB |

A user `runpod` entry replaces all of the built-in entry, so it must contain the required fields.

`aws` (the adapter comes after v0.1): `profile`, `region`, `ami_ssm`, `root_volume_gb`,
`ssh_key_name`, `allow_cidr` (default `auto`).

## Secrets

Lookup order, first hit wins: environment variable, system keyring (service `thetaserve`),
`secrets.json` (mode 600) in the user config directory. thetaserve writes to the keyring when a
backend is available, else to the file.

| Entry | Environment variable | Written by |
| --- | --- | --- |
| `runpod` | `RUNPOD_API_KEY` | `setup` |
| `hf` | `HF_TOKEN` | `setup` |
| `deployment/<name>/api-key` | none | `up`; deleted by `down` |

thetaserve never writes secrets to `state.json` and never shows them unmasked.

## Pod contents (Runpod)

| Item | Value |
| --- | --- |
| Command | `/bin/bash -c` with a script that writes `/tmp/thetaserve/thetaserve_guard.py` and `/tmp/thetaserve/thetaserve_watchdog.py`, starts the watchdog, then runs `exec vllm serve <repo> --port <port> --api-key "$VLLM_API_KEY" --middleware thetaserve_guard.RequireKey ...` |
| Watchdog | Reads `/metrics` each minute. Stops the pod after `--idle` without activity, or `--max-life` after the container starts, with the `RUNPOD_API_KEY` and `RUNPOD_POD_ID` that Runpod puts in every pod |
| Unauthenticated paths | `/health` only |
| Environment | `VLLM_API_KEY`; `HF_TOKEN` if stored; `HF_HOME=/runpod-volume/huggingface` and `VLLM_CACHE_ROOT=/runpod-volume/vllm` (compile cache) with a cache volume |
| Port | `<port>/http`, served at `https://<pod-id>-<port>.proxy.runpod.net/v1` |
| Runpod API | REST v2, `https://api.runpod.io/v2`, Bearer authentication |
