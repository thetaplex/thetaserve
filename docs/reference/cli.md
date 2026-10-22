# CLI reference

`thetaserve <command> [options]`. If you have only one deployment, commands with `[name]` use
it when you give no name.

## setup

Stores the Runpod API key and an optional Hugging Face token. Skips a prompt when
`RUNPOD_API_KEY` or `HF_TOKEN` is set. Needs an interactive terminal.

## list

`thetaserve list {models|providers}`. Shows the model names with their base repository and
vLLM flags, or the providers.

## up

Launches a model. `up` shows a table of options. Each row is the cheapest GPU in stock that
fits the model at the `--use` context, for one weight precision (16, 8, 4 bits) and one
KV-cache precision (16-bit, FP8). The cheapest row is first. `up` asks which row to launch.
The default is the cheapest row that meets the precision floor of the use case. `--yes` and
`--dry-run` take the default.

| Option | Default | Meaning |
| --- | --- | --- |
| `--provider` | required | `runpod` |
| `--model` | required | Model key from `list models`, or a Hugging Face repository such as `Qwen/Qwen3.8-27B` |
| `--device` | none | Runpod GPU ID, such as `NVIDIA A40`. Shows only the options on that GPU. Error if the GPU is unknown or sold out |
| `--use` | `coding` | Preset from `presets.yaml`: `coding`, `chat`, `research` |
| `--min-precision` | none | `4bit`, `8bit` or `16bit`. Removes lower weight precisions |
| `--port` | `8000` | Port vLLM listens on, 1-65535 |
| `--context` | from `--use` | Context length in tokens |
| `--kv-cache-dtype` | `auto` and `fp8` | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`. Shows only rows with this KV-cache precision |
| `--any-publisher` | off | Also lists weights from publishers not in `publishers.yaml`, marked `(untrusted publisher)` |
| `--extra-arg` | none | Extra `vllm serve` argument; repeatable |
| `--idle` | `30m` | Idle stop (watchdog not implemented yet, M3) |
| `--max-life` | `10h` | Lifetime stop (watchdog not implemented yet, M3) |
| `--name` | `<model>-<gpu>` | Deployment name |
| `--dry-run` | off | Show the spec, the `vllm serve` command and the masked request. Create nothing |
| `--yes`, `-y` | off | Do not ask before the launch. Necessary when the terminal is not interactive |

Sources of the options:

- the model's Hugging Face `config.json`;
- its quantised variants, without GGUF, MLX and unsupported formats;
- the `.safetensors` sizes of each variant at the current commit. `vllm serve --revision` pins
  the launch to that commit;
- the Runpod GPU catalogue, NVIDIA only.

No row has FP8 KV cache on Ampere. Only Blackwell rows have FP4 weights.

Columns:

- `memory`: weights + KV cache for one full-context sequence + 3.2 GiB overhead, against
  VRAM × 0.92.
- `tok/s`: estimate for one request at short context, 0.6 × memory bandwidth
  (`bandwidth.yaml`) ÷ weight size. Rows below 8 bits use the size of the smallest 8-bit
  variant. `?` when the bandwidth of the GPU is not known.

These values are estimates, not measurements.

A launch needs the Runpod key. A dry run without the key uses the packaged `samples.json`
instead of live GPU prices, stock and model data, and shows its date. For a model that is not
in the sample, it reads the Hugging Face Hub.

Steps:

1. Find the options. Select one (the prompt, or the default with `--yes`).
2. Show the rate.
3. Make and store a per-launch API key.
4. Create the pod.
5. Wait for `RUNNING` (20 min limit).
6. Wait for `GET /v1/models` (40 min limit).
7. Record vLLM's KV-cache capacity from the logs.

`up` refuses a vLLM image that contains `TODO`, and a name that is already in use.

## status

`thetaserve status [name]`. Shows the status, endpoint, uptime and cost so far of the current
run.

## env

`thetaserve env [name] [--shell posix|fish|powershell|cmd|json]`. Shows `OPENAI_BASE_URL`
and `OPENAI_API_KEY` lines on stdout. Default shell: PowerShell on Windows, POSIX on other
systems. On stderr it shows the other values: the model name for requests, weights, revision,
context, device, and the URLs (base, `/chat/completions`, `/completions`, `/models`,
`/metrics`, `/health`). Thus `eval` reads only the exports. Every URL except `/health` needs
the key. `--shell json` prints all values, the key included, as one JSON object on stdout.

## logs

`thetaserve logs [name] [-f|--follow] [--tail N]`. Shows the vLLM container log. `--tail`
0-5000, default 100.

## stop / start

`thetaserve stop [name]` pauses compute and keeps the disk. `thetaserve start [name]` starts
it again. Without a cache volume, `start` downloads the weights again.

## down

`thetaserve down [name] [--purge] [-y|--yes]`. Deletes the pod and the API key of the
deployment. `--purge` also deletes the cache volume, unless a different deployment uses it.

## Exit status

`0` on success, `1` on an error or a declined confirmation, `2` on invalid command-line usage.
