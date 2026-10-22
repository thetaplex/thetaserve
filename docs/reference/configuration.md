# Configuration and files reference

## Locations

Resolved with [platformdirs](https://platformdirs.readthedocs.io/).

| Purpose | Linux | macOS | Windows |
| --- | --- | --- | --- |
| User config (catalogue overrides, `secrets.json`) | `~/.config/thetaserve/` | `~/Library/Application Support/thetaserve/` | `%LOCALAPPDATA%\thetaserve\` |
| State (`state.json`) | `~/.local/state/thetaserve/` | `~/Library/Application Support/thetaserve/` | `%LOCALAPPDATA%\thetaserve\` |

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

`--model` also takes a Hugging Face repository. If it is the `hf_family` of an entry, `up` uses
that entry's `common_args`. Otherwise `up` launches without family arguments.

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

## samples.json

Built in only. A dated sample of the Runpod GPU catalogue (NVIDIA GPUs only) and of the
Hugging Face data of each built-in model. `up --dry-run` uses it when no Runpod key is
available. `just samples` fetches it again.

## providers.yaml

`runpod`:

| Field | Default | Meaning |
| --- | --- | --- |
| `api_key_env` | `RUNPOD_API_KEY` | Environment variable read first for the key |
| `cloud_type` | `SECURE` | `SECURE` or `COMMUNITY` |
| `network_volume` | `thetaserve-cache` | Cache volume name; `null` for no cache volume |
| `network_volume_gb` | `100` | Size when the volume is created, GB |
| `container_disk_gb` | `40` | Pod container disk, GB |

`aws` (adapter not implemented yet): `profile`, `region`, `ami_ssm`, `root_volume_gb`,
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
| Command | `/bin/bash -c` with a script that writes `/tmp/thetaserve/thetaserve_guard.py`, then runs `exec vllm serve <repo> --port <port> --api-key "$VLLM_API_KEY" --middleware thetaserve_guard.RequireKey ...` |
| Unauthenticated paths | `/health` only |
| Environment | `VLLM_API_KEY`; `HF_TOKEN` if stored; `HF_HOME=/runpod-volume/huggingface` with a cache volume |
| Port | `<port>/http`, served at `https://<pod-id>-<port>.proxy.runpod.net/v1` |
| Runpod API | REST v2, `https://api.runpod.io/v2`, Bearer authentication |
