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

The built-in model profiles still contain `TODO` values, and `up` refuses them. Create three
files in your user config directory (`~/.config/thetaserve/` on Linux,
`~/Library/Application Support/thetaserve/` on macOS, `%LOCALAPPDATA%\thetaserve\` on Windows).

`devices.yaml`:

```yaml
a40-48g:
  vram_gb: 48
  arch: ampere
  providers:
    runpod: { gpu_type_id: "NVIDIA A40" }
```

`models.yaml`:

```yaml
vllm_image: vllm/vllm-openai:v0.32.0

qwen3.8-27b:
  hf_family: Qwen/Qwen3.8-27B
  min_vllm: "0.17"
  common_args: [--enable-prefix-caching, --reasoning-parser, qwen3,
                --enable-auto-tool-choice, --tool-call-parser, qwen3_xml]
  kv_bytes_per_token: 65536
  kv_fixed_bytes: 0
  tiers:
    48g: { repo: Qwen/Qwen3.8-27B-FP8, weights_gb: 28.75, max_model_len: 131072, kv_cache_dtype: auto }
```

`providers.yaml` (no cache volume, so nothing is billed after you remove the pod):

```yaml
runpod:
  api_key_env: RUNPOD_API_KEY
  cloud_type: SECURE
  network_volume: null
  container_disk_gb: 50
```

## 4. Check what will be launched

```sh
uv run thetaserve up --provider runpod --device a40-48g --model qwen3.8-27b --port 8000 --dry-run
```

You see the memory estimate (`weights 28.8 + KV 8.0 + overhead 2.0 = 38.8 of 44.2 GiB usable`),
the `vllm serve` command and the request body with the API key shown as `****`. Nothing is
created.

## 5. Launch

```sh
uv run thetaserve up --provider runpod --device a40-48g --model qwen3.8-27b --port 8000
```

thetaserve prints the hourly rate and asks `Launch?`. Answer `y`. It then creates the pod, waits
for it to start, and waits while vLLM downloads about 29 GiB of weights and loads them. Expect
5-15 minutes. When it prints `Ready.` you have a working endpoint.

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
- [How launch settings are chosen](../explanation/launch-settings.md) explains the profile you
  wrote.
