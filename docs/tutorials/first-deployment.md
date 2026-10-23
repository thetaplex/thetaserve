# Your first deployment on Runpod

In this tutorial, you launch Qwen3.8-27B on a Runpod A40 (48 GB). You connect a client to it.
Then you remove it. This takes approximately 30 minutes. The pod costs approximately $0.59 per
hour.

> **Pre-alpha.** This procedure ran on 2026-10-22 (the M1 pilot). If a step fails, keep the
> error output and open an issue.

## Before you start

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.
- A Runpod account with credit, a spend limit set in the console, and an API key (see
  [Store credentials](../how-to/store-credentials.md) for the permissions it needs).

## 1. Install

```sh
uv tool install git+https://github.com/thetaplex/thetaserve
```

## 2. Store your Runpod key

Use a real terminal. The prompt does not show what you type.

```sh
thetaserve setup
```

Paste the Runpod key. Press Enter to skip the Hugging Face token.

## 3. Turn off the cache volume

By default, `up` creates a network volume that keeps the weights between launches. Runpod bills
it until you delete it. For this tutorial, turn it off: create `providers.yaml` in your user
config directory (`~/.config/thetaserve/` on Linux, `~/Library/Application Support/thetaserve/`
on macOS, `%LOCALAPPDATA%\thetaserve\` on Windows):

```yaml
runpod:
  api_key_env: RUNPOD_API_KEY
  cloud_type: SECURE
  network_volume: null
  container_disk_gb: 50
```

## 4. See the options

```sh
thetaserve up --provider runpod --model qwen3.8-27b --dry-run
```

thetaserve reads the model's metadata from Hugging Face and the GPU prices and stock from
Runpod. It shows a table. Each row is the cheapest GPU in stock for one weight precision (16,
8, 4 bits) and one KV-cache precision (16-bit, FP8), at the `coding` context of 128K tokens.
The cheapest row is first.

The dry run takes the cheapest row with 8-bit weights or more. This is usually an NVIDIA A40
(48 GB) with `Qwen/Qwen3.8-27B-FP8`, 38.7 of 44.2 GiB, $0.59/h. Below the table, you see the
`vllm serve` command and the request body. The API key shows as `****`. The dry run creates
nothing.

## 5. Launch

```sh
thetaserve up --provider runpod --model qwen3.8-27b
```

The same table shows, then the prompt `Launch which option`. Press Enter to select the default
(the row of the dry run), or type the number of a different row.

thetaserve creates the pod and waits for it to start. Then vLLM downloads approximately 29 GiB
of weights, loads them and prepares the GPU. In the pilot this took 4 minutes after the pod
started. Runpod can take 10 minutes more to start the pod. When thetaserve shows `Ready.`, the
endpoint is available.

## 6. Use it

```sh
eval "$(thetaserve env)"
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

The response shows `Qwen/Qwen3.8-27B-FP8`. All OpenAI-compatible clients work with these two
variables. See [Connect a client](../how-to/connect-a-client.md).

## 7. Remove it

The pod stops itself after 30 minutes without requests, or after 10 hours in total. A stopped
pod still costs a little for its disk. Remove it when you are done:

```sh
thetaserve down
```

Type `y`. Run `thetaserve status`. It shows no deployments.

## Next

- [Run without a cache volume](../how-to/run-without-cache-volume.md) explains the setting of
  step 3.
- [How launch settings are chosen](../explanation/launch-settings.md) explains the options
  table.
