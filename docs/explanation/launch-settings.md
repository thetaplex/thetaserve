# How launch settings are chosen

Serving a model well on one GPU means choosing a weight format, a context length and a KV-cache
precision that together fit in the GPU's memory, plus model-specific vLLM flags. This page
explains how thetaserve makes those choices today and why that is changing.

## Pinned launches: tiers

With `--device`, the M0/M1 resolver ([DESIGN.md](../DESIGN.md) section 6) works from two hand-written
catalogues. `devices.yaml` gives each GPU its memory and architecture. `models.yaml` gives each
model a profile per memory **tier** (24g, 48g, 80g): a repository, a context length and a
KV-cache type. `up` picks the largest tier that fits the GPU, applies architecture overrides
(for example, no FP8 KV cache on Ampere until verified), then your flags, then checks the fit:

weights + KV cache for the full context + 2 GiB overhead ≤ 92 % of VRAM.

For Qwen3.8-27B FP8 on a 48 GB card that is 28.8 + 8.0 + 2.0 = 38.8 of 44.2 GiB.

## Why tiers are going

Preparing the first real run showed three problems:

- **The catalogues go stale.** The A6000 named in the plan was sold out. An A40, same price and
  architecture, was in stock but invisible because nobody had written it down.
- **Tiers bundle independent choices.** Weight format, context and KV precision are tied to a
  memory bucket, so combinations between buckets are unreachable.
- **The facts are already published.** Weight sizes and the attention layout are in each model's
  Hugging Face metadata; the 64 KiB-per-token KV figure for Qwen3.8-27B computed from
  `config.json` matches the hand-written one. GPU memory, architecture, price and stock come from
  Runpod's catalogue.

## Options: what replaces them

[ADR 0001](../adr/0001-derive-launch-options.md) derives options instead, and `up` does this
whenever `--device` is omitted. You name a model and a use case; thetaserve computes which
weight variants fit on which GPUs in stock, and shows the cheapest GPU for every combination
of weight precision and KV-cache precision ([ADR 0002](../adr/0002-options-for-every-precision.md)).

Use cases replace tiers as the only opinionated table:

| Preset | Minimum context | Tool calling | Lowest weight precision |
| --- | --- | --- | --- |
| `coding` (default) | 131072 | yes | 8-bit |
| `chat` | 32768 | no | 4-bit |
| `research` | model maximum | yes | 8-bit |

The 8-bit floor for coding is a judgement, not a measurement: an agent's long chains of exact
tool calls suffer more from small errors than conversation does, and 4-bit would save only about
$0.26/h on Qwen3.8-27B. Because it is unmeasured, the floor only picks the default row;
4-bit rows are still listed, so you can make the trade yourself.
`bench` is planned to measure 4-bit against 8-bit; the floor drops for a model family where
4-bit holds up.

## What 4-bit buys: room, not speed

4-bit weights rarely make the cheaper GPU fit for coding. For Qwen3.8-27B at 128K they need
24–30 GiB, more than the 22.1 GiB usable on a 24 GB card, and 24 GB cards on Runpod cost no
less than a 48 GB A40. What 4-bit does buy is room on the same GPU. On an A40 at 128K, FP8 weights use
38.7 of 44.2 GiB and 4-bit uses 29.6, so about 9 GiB more is left for vLLM's KV cache. More room
means more parallel agent sessions, more of each repeated prompt served from vLLM's prefix
cache, or the model's full context on a cheap card. The `memory` column shows the headroom.

4-bit does not buy speed. Decoding is limited by memory bandwidth, and fewer bytes per weight
should mean faster tokens, but unpacking 4-bit weights costs about as much as the smaller read
saves. imec's October 2026 study of Qwen3.8-27B on vLLM, run as a coding agent on an H200,
measured 985 tok/s for FP8, 857 for BF16 and 808 for INT4 AWQ. The `tok/s` column therefore
times 4-bit rows as the 8-bit variant.

Quality is the open question. In the same study, FP8 and INT4 scored within a few tasks of BF16
on SWE-Bench-Pro (BF16 33, FP8 30, INT4 32 of 64 tasks), but Jang et al. (arXiv 2607.27275) found 4-bit
models make up to 2.5× more malformed tool calls while benchmark scores stay flat. This is why
the coding floor is still 8-bit: choose a 4-bit row when you need room more than reliability.

Two things stay hand-written because no metadata holds them: vLLM's tool-call and reasoning
parsers per model family, and the memory overhead, which is calibrated from measured runs.
