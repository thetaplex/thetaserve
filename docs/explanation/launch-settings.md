# How launch settings are chosen

Serving a model well on one GPU means choosing a weight format, a context length and a KV-cache
precision that together fit in the GPU's memory, plus model-specific vLLM flags. This page
explains how thetaserve makes those choices today and why that is changing.

## Today: tiers

The M0/M1 resolver ([DESIGN.md](../DESIGN.md) section 6) works from two hand-written
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

## What replaces them

[ADR 0001](../adr/0001-derive-launch-options.md) (status: proposed) derives options instead.
You name a model and a use case; thetaserve computes which weight variants fit on which GPUs in
stock and suggests the cheapest few.

Use cases replace tiers as the only opinionated table:

| Preset | Minimum context | Tool calling | Lowest weight precision |
| --- | --- | --- | --- |
| `coding` (default) | 131072 | yes | 8-bit |
| `chat` | 32768 | no | 4-bit |
| `research` | model maximum | yes | 8-bit |

The 8-bit floor for coding is a judgement, not a measurement: an agent's long chains of exact
tool calls suffer more from small errors than conversation does, and 4-bit would save only about
$0.26/h on Qwen3.8-27B. `bench` is planned to measure 4-bit against 8-bit; the floor drops for a
model family where 4-bit holds up.

Two things stay hand-written because no metadata holds them: vLLM's tool-call and reasoning
parsers per model family, and the memory overhead, which is calibrated from measured runs.
