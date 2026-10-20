# Configuration and files reference

## Locations

Resolved with [platformdirs](https://platformdirs.readthedocs.io/).

| Purpose | Linux | macOS | Windows |
| --- | --- | --- | --- |
| User config (catalogue overrides, `secrets.json`) | `~/.config/thetaserve/` | `~/Library/Application Support/thetaserve/` | `%LOCALAPPDATA%\thetaserve\` |
| State (`state.json`) | `~/.local/state/thetaserve/` | `~/Library/Application Support/thetaserve/` | `%LOCALAPPDATA%\thetaserve\` |

Built-in catalogues ship in the package at `src/thetaserve/catalog/`. A user file's top-level
keys replace built-in keys whole.

## devices.yaml

| Field | Type | Meaning |
| --- | --- | --- |
| `<key>` | string | Device name used by `--device`, e.g. `a6000-48g` |
| `vram_gb` | positive integer | GPU memory, GiB |
| `arch` | `ampere`, `ada`, `hopper`, `blackwell` | Selects `arch_overrides` |
| `providers.runpod.gpu_type_id` | string | Runpod GPU ID, e.g. `NVIDIA A40` |
| `providers.aws.instance_type` | string | EC2 instance type (AWS adapter not implemented yet) |

## models.yaml

Top level: `schema_version: 1` (built-in only), `vllm_image` (image and tag for every profile).

| Field | Type | Meaning |
| --- | --- | --- |
| `<key>` | string | Model name used by `--model` |
| `hf_family` | string | Base Hugging Face repository |
| `min_vllm` | string | Lowest vLLM version known to serve it |
| `common_args` | list of strings | `vllm serve` arguments for every tier |
| `kv_bytes_per_token` | integer ≥ 0 | KV-cache bytes per token at BF16; halved for FP8 KV |
| `kv_fixed_bytes` | integer ≥ 0 | Context-independent KV state, bytes (default 0) |
| `tiers.<NNg>` | mapping | Profile for GPUs with at least NN GiB |
| `tiers.*.repo` | string | Repository to serve |
| `tiers.*.weights_gb` | positive number | Weight size, GiB |
| `tiers.*.max_model_len` | positive integer | Context length, tokens |
| `tiers.*.kv_cache_dtype` | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` | KV-cache precision |
| `tiers.*.extra_args` | list of strings | Extra `vllm serve` arguments |
| `tiers.*.arch_overrides.<arch>` | mapping | Any tier field except `arch_overrides`, per architecture |

Tier selection: the largest tier not above the device's `vram_gb`.

Fit check: `weights_gb + KV(max_model_len) + 2.0` ≤ `vram_gb × 0.92`, else `up` fails unless
`--force`.

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
`secrets.json` (mode 600) in the user config directory. Writes go to the keyring when a backend
exists, else to the file.

| Entry | Environment variable | Written by |
| --- | --- | --- |
| `runpod` | `RUNPOD_API_KEY` | `setup` |
| `hf` | `HF_TOKEN` | `setup` |
| `deployment/<name>/api-key` | none | `up`; deleted by `down` |

Secrets are never written to `state.json` or printed unmasked.

## Pod contents (Runpod)

| Item | Value |
| --- | --- |
| Command | `/bin/bash -c "exec vllm serve <repo> --port <port> --api-key $VLLM_API_KEY ..."` |
| Environment | `VLLM_API_KEY`; `HF_TOKEN` if stored; `HF_HOME=/runpod-volume/huggingface` with a cache volume |
| Port | `<port>/http`, served at `https://<pod-id>-<port>.proxy.runpod.net/v1` |
| Runpod API | REST v2, `https://api.runpod.io/v2`, Bearer authentication |
