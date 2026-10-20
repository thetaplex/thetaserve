# 2. Offer every weight and KV-cache precision

Date: 2026-10-21

## Status

Accepted. Amends [ADR 0001](0001-derive-launch-options.md), points 2 (precision floor) and
6 (ranking).

## Context

ADR 0001 lists, for each GPU in stock, the highest-precision weights that fit, drops weights
below the use case's precision floor, and shows the three cheapest GPUs. For Qwen3.8-27B with
`coding` that meant three 48 GB cards, all with FP8 weights and BF16 KV cache. The user never
saw that 4-bit weights or an FP8 KV cache were possible, or what they would cost, so they could
not trade quality for price themselves.

The floor is a judgement that `bench` has not yet measured (ADR 0001, point 2), which makes it a
weak reason to hide options.

## Decision

1. `up` without `--device` shows one row per combination of weight precision (16, 8 and 4 bits,
   as available for the model) and KV-cache precision (16-bit, FP8): the cheapest GPU in stock
   that fits it. Rows are ordered by weight precision, then KV precision. A combination that
   fits no GPU in stock is left out.
2. Within a precision, the first variant that fits is used, in Hub download order with the base
   repository first. FP4 weights are offered only on Blackwell, FP8 KV cache never on Ampere.
3. The preset's precision floor no longer hides rows. It picks the **recommended** row: the
   cheapest that meets the floor (higher precision on a tie), or the cheapest overall if none
   does. Rows below the floor are labelled. The prompt defaults to the recommended row, and
   `--yes` and `--dry-run` take it.
4. `--min-precision` still removes lower precisions, for users who want them hidden.

## Consequences

- Up to six rows instead of three, all at the preset's context length.
- Some rows cost more than a higher-precision row (for example FP8 KV cache needs a non-Ampere
  card). They stay, because FP8 KV also doubles how many tokens the cache holds.
- The recommendation still rests on the unmeasured floor; `bench` results replace it as before.
