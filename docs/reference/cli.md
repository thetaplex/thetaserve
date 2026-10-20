# CLI reference

`thetaserve <command> [options]`. Commands that take `[name]` default to the only deployment
when there is one.

## setup

Store the Runpod API key and an optional Hugging Face token. Skips a prompt when
`RUNPOD_API_KEY` or `HF_TOKEN` is set. Needs an interactive terminal.

## list

`thetaserve list {models|devices|providers}`. Print the merged catalogue.

## up

Launch a model.

| Option | Default | Meaning |
| --- | --- | --- |
| `--provider` | required | `runpod` |
| `--device` | required | Device key from `list devices` |
| `--model` | required | Model key from `list models` |
| `--port` | `8000` | Port vLLM listens on, 1-65535 |
| `--context` | profile | Maximum context length in tokens |
| `--kv-cache-dtype` | profile | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` |
| `--extra-arg` | none | Extra `vllm serve` argument; repeatable |
| `--idle` | `30m` | Idle stop (watchdog not implemented yet, M3) |
| `--max-life` | `10h` | Lifetime stop (watchdog not implemented yet, M3) |
| `--name` | `<model>-<device>` | Deployment name |
| `--dry-run` | off | Print the spec, `vllm serve` command and masked request; create nothing |
| `--force` | off | Skip the memory-fit check |
| `--yes`, `-y` | off | Do not ask before launching; required when not interactive |

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
