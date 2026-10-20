# 1. Derive launch options from live model and GPU data

Date: 2026-10-20

## Status

Accepted. Points 2 (precision floor) and 6 (ranking) amended by
[ADR 0002](0002-options-for-every-precision.md).

## Context

The design brief (`docs/DESIGN.md` v0.2, sections 5.1, 5.2 and 6) resolves a launch from two
hand-written catalogues. `devices.yaml` lists GPUs with their VRAM and architecture. `models.yaml`
lists each model's settings per memory tier (24g, 48g, 80g): a repository, a context length and a
KV-cache type. The user names a device and a model, and the resolver picks the largest tier that
fits the device.

Preparing the first real run on 2026-10-20 showed the costs of this:

- The catalogues go stale. The RTX A6000 named in M1 was sold out on both RunPod clouds. The A40,
  at the same price ($0.59/h Secure) with high stock, was invisible because it is not in
  `devices.yaml`.
- Tiers are artificial. They tie three independent choices (weight format, context length,
  KV-cache type) to a VRAM bucket, so any combination between buckets is unreachable.
- The user has to know which GPU to ask for. What they actually know is the model and what they
  want it for.

The facts the tiers encode are available from the source, checked on 2026-10-20:

| Fact | Source | Example (Qwen/Qwen3.8-27B) |
| --- | --- | --- |
| Weight size | Hugging Face Hub API, `safetensors` parameter counts per dtype | 27.8 B params, BF16 |
| KV bytes per token | `config.json`: `layer_types`, `num_key_value_heads`, `head_dim` | 16 full-attention layers of 64, 4 KV heads, head dim 256 → 64 KiB at BF16, matching DESIGN.md section 6 |
| Maximum context | `config.json` `max_position_embeddings` | 262144 |
| Quantised variants | Hub filter `base_model:quantized:<repo>`, then each variant's `quantization_config` | `Qwen/Qwen3.8-27B-FP8` (`quant_method: fp8`) |
| GPU VRAM, architecture, price, stock | RunPod `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD` | A40: 48 GB, pool `AMPERE_48`, $0.59/h, stock HIGH |

Some things are not in any metadata: vLLM's tool-call and reasoning parsers for a model family,
vLLM's real memory overhead, and whether a kernel (such as FP8 KV cache on Ampere) runs well.
These can only be written by hand or measured.

Alternatives considered:

- **Keep the catalogues and add entries as needed.** This is the smallest change, but every new
  GPU or model needs a release, and the A40 case will recur.
- **Derive devices from RunPod but keep model tiers.** This fixes the GPU staleness but keeps the
  artificial buckets.

## Decision

thetaserve derives launch options from live metadata and suggests them; it no longer resolves
fixed profiles.

1. **Input.** A model (a Hugging Face repository, or a short alias for it), a use case, and
   optional pins. `thetaserve up qwen3.8-27b --use coding`; `coding` is the default.
2. **Use-case presets.** A small packaged YAML file maps each use case to its requirements.
   These presets are the only opinionated table and replace tiers:

   | Preset | Minimum context | Tool calling | Lowest weight precision |
   | --- | --- | --- | --- |
   | `coding` | 131072 | yes | 8-bit |
   | `chat` | 32768 | no | 4-bit |
   | `research` | the model's maximum | yes | 8-bit |

   The precision floors are judgements, not measurements: coding and research run long chains
   of tool calls where one wrong token breaks a step, and chat does not. They stay defaults until
   `bench`'s coding-quality check (DESIGN.md section 10) compares 4-bit with 8-bit for a model
   family; where 4-bit holds up, that family's floor drops. `--min-precision` overrides the floor
   for one launch. Reasoning is not a preset setting: thinking is switched per request by the
   client. Users can add presets in their config directory (DESIGN.md section 3).
3. **Model candidates.** The base repository plus its quantised variants from the Hub, keeping
   only formats vLLM serves (BF16/FP16, `fp8`, `awq`, `gptq`, `compressed-tensors`) and dropping
   GGUF and MLX. The Hugging Face token from `setup` is used for gated repositories.
4. **Memory estimate.** weights (from safetensors counts and dtypes) + KV per token × context
   (counting only full-attention layers; sliding-window and linear-attention layers add a fixed
   amount) + overhead ≤ VRAM × `gpu_memory_utilization`. The overhead is a setting with a default,
   calibrated from measured runs (point 8).
5. **GPU candidates.** The RunPod catalogue's VRAM, price and per-data-centre stock, with the
   architecture read from the pool name (`AMPERE_48` → ampere). GPUs whose pool cannot be mapped
   are skipped.
6. **Ranking.** Keep the (variant, GPU, data centre) combinations that fit, meet the use case and
   are in stock. Rank data centres that already hold the cache volume first, so weights are not
   downloaded again, then by price. Show the top three with GPU, weight format, context,
   data centre and $/h, and ask which one to use. `--yes` takes the first.
7. **Family settings.** The vLLM flags that cannot be derived (parsers, and settings such as
   disabling image input) live in a small packaged file keyed by `architectures[0]` from
   `config.json` (for example `Qwen3_8ForCausalLM`), not by repository. An unknown family
   launches without extra flags and a warning.
8. **Calibration.** Every `bench` result records estimated and measured KV capacity. The
   overhead default and any per-architecture corrections come from those results.
9. **Pins.** `--device` (a provider GPU ID such as `"NVIDIA A40"`), `--repo`, `--context` and
   `--kv-cache-dtype` fix a choice and skip the search for it. Benchmarks and bug reports pin all
   of them.
10. **AWS.** v0.1 uses one instance type, so the mapping from GPU to instance type (for example
    L40S → `g6e.xlarge`) moves to the `aws` section of `providers.yaml`. Deriving it from EC2's
    `DescribeInstanceTypes` waits until more instance types are supported.

This supersedes DESIGN.md sections 5.1 (`devices.yaml`), the tier structure in 5.2 and
resolution rules 1 and 2 in section 6. When this ADR is accepted, those sections get a pointer
here and the files are removed.

## Consequences

- New GPUs and quantised variants are usable without a release, and the cheapest suitable GPU in
  stock is found automatically.
- `up` and `--dry-run` need network access. The RunPod catalogue rejects requests without a key
  (`401`), so DESIGN.md section 7's rule that a dry run works with no credentials holds only when
  every choice is pinned.
- A Hugging Face or RunPod outage blocks suggestions. Fully pinned launches still work for
  RunPod, but not while the RunPod API itself is down.
- Estimates can be wrong, and a wrong estimate means vLLM fails at start-up after billing has
  begun. Mitigations: a conservative overhead default, calibration from `bench`, and `up`
  reporting vLLM's out-of-memory error from the logs instead of waiting for the health check to
  time out.
- The rule that every profile number comes from a recorded run (DESIGN.md section 14,
  `CLAUDE.md`) now applies to the overhead calibration and the family settings. Sizes computed
  from metadata need no measurement.
- Milestones change. M0's "done when" (a correct command for every model × device) no longer
  applies. M2 becomes: family settings verified and overhead calibrated for both launch models.
- Tests mock the Hugging Face Hub and RunPod catalogue with respx, as they already do for the
  RunPod adapter.
