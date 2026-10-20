# CLI reference

`thetaserve <command> [options]`. Commands that take `[name]` default to the only deployment
when there is one.

## setup

Store the Runpod API key and an optional Hugging Face token. Skips a prompt when
`RUNPOD_API_KEY` or `HF_TOKEN` is set. Needs an interactive terminal.

## list

`thetaserve list {models|devices|providers}`. Print the merged catalogue.

## up

Launch a model. Without `--device`, `up` lists, for each weight precision (16, 8, 4 bits) and
KV-cache precision (16-bit, FP8), the cheapest GPU in stock that fits the model at the
`--use` context, and asks which to launch. The default is the recommended row: the cheapest
that meets the use case's precision floor. `--yes` and `--dry-run` take it. With `--device`,
`up` resolves the catalogue profile instead.

| Option | Default | Meaning |
| --- | --- | --- |
| `--provider` | required | `runpod` |
| `--model` | required | Model key from `list models`, or a Hugging Face repo such as `Qwen/Qwen3.8-27B` |
| `--device` | none | Pin a device key from `list devices` and skip the options |
| `--use` | `coding` | Preset from `presets.yaml`: `coding`, `chat`, `research` |
| `--min-precision` | none | `4bit`, `8bit` or `16bit`; hide lower weight precisions |
| `--port` | `8000` | Port vLLM listens on, 1-65535 |
| `--context` | profile or preset | Maximum context length in tokens |
| `--kv-cache-dtype` | profile | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` |
| `--extra-arg` | none | Extra `vllm serve` argument; repeatable |
| `--idle` | `30m` | Idle stop (watchdog not implemented yet, M3) |
| `--max-life` | `10h` | Lifetime stop (watchdog not implemented yet, M3) |
| `--name` | `<model>-<device>` or `<model>-<gpu>` | Deployment name |
| `--dry-run` | off | Print the spec, `vllm serve` command and masked request; create nothing |
| `--force` | off | Skip the memory-fit check |
| `--yes`, `-y` | off | Do not ask before launching; required when not interactive |

Options are computed from the model's Hugging Face `config.json`, its quantised variants
(GGUF, MLX and unsupported formats skipped) and their `.safetensors` sizes, and the Runpod GPU
catalogue (NVIDIA only). FP8 KV cache is not offered on Ampere and FP4 weights only on
Blackwell. Options need the Runpod key; a dry run
with `--device` does not.

Steps: resolve and fit-check, print rate and confirm, generate and store a per-launch API key,
create the pod, wait for `RUNNING` (20 min limit), wait for `GET /v1/models` (40 min limit),
record vLLM's KV-cache capacity from the logs. Refuses profiles whose image or repo contains
`TODO`, and names already in use.

## status

`thetaserve status [name]`. Status, endpoint, uptime and cost so far for the current run.

## env

`thetaserve env [name] [--shell posix|fish|powershell|cmd]`. Print `OPENAI_BASE_URL` and
`OPENAI_API_KEY` lines. Default shell: PowerShell on Windows, POSIX elsewhere.

## logs

`thetaserve logs [name] [-f|--follow] [--tail N]`. vLLM container log; `--tail` 0-5000,
default 100.

## stop / start

`thetaserve stop [name]` pauses compute and keeps the disk. `thetaserve start [name]` resumes.
Without a cache volume the weights are downloaded again on start.

## down

`thetaserve down [name] [--purge] [-y|--yes]`. Delete the pod and the deployment's API key.
`--purge` also deletes the cache volume, unless another deployment uses it.

## Exit status

`0` on success, `1` on an error or a declined confirmation, `2` on invalid command-line usage.
