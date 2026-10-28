# CLI reference

`thetaserve <command> [options]`. If you have only one deployment, commands with `[name]` use
it when you give no name. `thetaserve --version` shows the version.

Durations (`--idle`, `--max-life`) use the units `s`, `m` and `h`, for example `90s`, `45m`,
`2h` or `1h30m`.

## setup

Stores the Runpod API key and an optional Hugging Face token. Skips a prompt when
`RUNPOD_API_KEY` or `HF_TOKEN` is set. Needs an interactive terminal.

## list

`thetaserve list {models|providers}`. Shows the model names with their base repository and
vLLM flags, or the providers in `providers.yaml`. The list shows `aws`, but the AWS adapter
comes after v0.1.

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
| `--cloud` | `cloud_type` (`SECURE`) | `SECURE` or `COMMUNITY`, in any case. Runpod cloud tier for the GPU types, the prices, the stock and the pod. Community Cloud has no cache volume |
| `--port` | `8000` | Port vLLM listens on, 1-65535 |
| `--context` | from `--use` | Context length in tokens |
| `--kv-cache-dtype` | `auto` and `fp8` | `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2`. Shows only rows with this KV-cache precision |
| `--any-publisher` | off | Also lists weights from publishers not in `publishers.yaml`, marked `(untrusted publisher)` |
| `--extra-arg` | none | Extra `vllm serve` argument. Repeatable |
| `--idle` | `30m` | Stop the pod after this duration with no request completed or running |
| `--max-life` | `10h` | Stop the pod this duration after the container starts, with or without activity |
| `--name` | `<model>-<gpu>` | Deployment name, for example `qwen3.8-27b-a40` |
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
4. With a cache volume, find the volume in a data centre that has the GPU in stock. If no
   volume exists, create one in the data centre with the most stock.
5. Create the pod.
6. Wait for `RUNNING` (20 min limit).
7. Wait for `GET /v1/models` (40 min limit). Each minute, read the vLLM log. If the log shows a
   crash, stop the wait, keep the log in the state directory's `logs/`, and show the error.
8. Record vLLM's KV-cache capacity from the logs.

`up` refuses a vLLM image that contains `TODO`, and a name that is already in use.

## status

`thetaserve status [name]`. Shows the status, endpoint, uptime, rate and cost so far of each
deployment, or of the named deployment. The cost counts only the current run.

## env

`thetaserve env [name] [--shell posix|fish|powershell|cmd]`. Shows `OPENAI_BASE_URL`
and `OPENAI_API_KEY` lines on stdout. Default shell: PowerShell on Windows, POSIX on other
systems. On stderr it shows the other values: the model name for requests (`model`), weights,
revision, context, device, and the URLs (base, `/chat/completions`, `/completions`, `/models`,
`/metrics`, `/health`). Thus `eval` reads only the exports. Every URL except `/health` needs
the key.

## logs

`thetaserve logs [name] [-f|--follow] [--tail N]`. Shows the vLLM container log. `--tail`
0-5000, default 100.

## stop / start

`thetaserve stop [name]` stops the pod. The cache volume stays, but Runpod erases the
container disk. `thetaserve start [name]` starts the pod again. Without a cache volume, `start`
downloads the weights again. If the data centre has no GPU of that type, `start` fails with
Runpod's message.

## down

`thetaserve down [name] [--purge] [-y|--yes]`. Deletes the pod and the API key of the
deployment. `--purge` also deletes the cache volume. If a different deployment uses the same
volume, `--purge` refuses and deletes nothing.

## bench

`thetaserve bench --provider runpod --model M --device D [--use U] [--precision P] [--cloud C]
[--context N] [--kv-cache-dtype K] [--out results] [--keep] [-y|--yes]`.

Launches the option that `up --device D` recommends. With `--precision 4bit|8bit|16bit`, it
selects only from options with that weight precision. Then it runs the fixed workload and
writes `results/<model>/<device>-<weights>-<provider>-<date>.json`. It removes the deployment
unless you give `--keep`, also when the run fails.

`--cloud` selects the cloud tier, the same as for `up`. The default is `SECURE`.

Bench runs use no cache volume. The container disk holds the weights. The pod has `--idle 30m`
and `--max-life 3h`, and uses only trusted publishers. A run takes approximately 15-25 minutes
of GPU time (estimate). The [benchmark plan](benchmark-plan.md) lists the runs and the metrics.

## Exit status

`0` on success. `1` on an error or a declined confirmation. `2` on invalid command-line usage
or an invalid catalogue file.
