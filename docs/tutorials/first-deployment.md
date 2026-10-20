# Your first deployment on Runpod

In this tutorial you launch Qwen3.8-27B on a Runpod A40 (48 GB), point a client at it, and
remove it. It takes about 30 minutes and costs roughly $0.59 per hour of pod time.

> **Pre-alpha.** As of 2026-10-21 this path has not yet completed on real hardware; it is the M1
> pilot. If a step fails, keep the error output and open an issue.

## Before you start

- [uv](https://docs.astral.sh/uv/getting-started/installation/) and
  [just](https://just.systems/man/en/packages.html) installed.
- A Runpod account with credit, a spend limit set in the console, and an API key (see
  [Store credentials](../how-to/store-credentials.md) for the permissions it needs).

## 1. Install

```sh
git clone https://github.com/thetaplex/thetaserve.git
cd thetaserve
just setup
```

## 2. Store your Runpod key

Run this in a real terminal; the prompt hides what you type.

```sh
uv run thetaserve setup
```

Paste the Runpod key, then press Enter to skip the Hugging Face token.

## 3. Add the pilot settings

The built-in vLLM image is still `TODO` until the pilot pins it, and `up` refuses it. Create two
files in your user config directory (`~/.config/thetaserve/` on Linux,
`~/Library/Application Support/thetaserve/` on macOS, `%LOCALAPPDATA%\thetaserve\` on Windows).

`models.yaml`:

```yaml
vllm_image: vllm/vllm-openai:v0.32.0
```

`providers.yaml` (no cache volume, so nothing is billed after you remove the pod):

```yaml
runpod:
  api_key_env: RUNPOD_API_KEY
  cloud_type: SECURE
  network_volume: null
  container_disk_gb: 50
```

## 4. See the options

```sh
uv run thetaserve up --provider runpod --model qwen3.8-27b --dry-run
```

thetaserve reads the model's metadata from Hugging Face and Runpod's GPU prices and stock, and
lists the three cheapest GPUs in stock that fit the `coding` use case (128K context, at least
8-bit weights). A typical first row is an NVIDIA A40 (48 GB) with `Qwen/Qwen3.8-27B-FP8`,
38.7 of 44.2 GiB, $0.59/h. Below the table is the dry run for option 1: the `vllm serve` command
and the request body with the API key shown as `****`. Nothing is created.

## 5. Launch

```sh
uv run thetaserve up --provider runpod --model qwen3.8-27b
```

The same table appears, followed by `Launch which option`. Press Enter for option 1, or type
its number. thetaserve then creates the pod, waits for it to start, and waits while vLLM
downloads about 29 GiB of weights and loads them. Expect 5-15 minutes. When it prints `Ready.`
you have a working endpoint.

## 6. Use it

```sh
eval "$(uv run thetaserve env)"
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

The response lists `Qwen/Qwen3.8-27B-FP8`. Any OpenAI-compatible client now works with these two
variables; see [Connect a client](../how-to/connect-a-client.md).

## 7. Remove it

Idle stop is not implemented yet, so the pod bills until you remove it.

```sh
uv run thetaserve down
```

Answer `y`. `uv run thetaserve status` now reports no deployments.

## Next

- [Run without a cache volume](../how-to/run-without-cache-volume.md) explains what you set in
  step 3.
- [How launch settings are chosen](../explanation/launch-settings.md) explains the options
  table.
