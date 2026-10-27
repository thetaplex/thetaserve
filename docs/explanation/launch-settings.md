# How launch settings are chosen

To serve a model well on one GPU, you must choose a weight format, a context length and a
KV-cache precision. Together they must fit in the GPU's memory. The model also needs
model-specific vLLM flags. This page explains how thetaserve makes these choices, and why.

## Options

`up` calculates the options from published data:

- **The model's Hugging Face metadata.** `config.json` gives the attention layout, from which
  thetaserve calculates the KV-cache size per token. For Qwen3.8-27B this is 64 KiB at BF16.
  The quantised variants of the model and their `.safetensors` sizes give the weight size of
  each precision.
- **Runpod's GPU catalogue.** It gives the memory, architecture, price and stock of each GPU.

A combination fits a GPU when:

weights + KV cache for the full context + 3.2 GiB overhead ≤ 92 % of VRAM.

For each combination of weight precision (16, 8, 4 bits) and KV-cache precision (16-bit, FP8),
`up` shows the cheapest GPU in stock that fits. Thus a new GPU or a new quantised variant is
available as soon as Runpod or Hugging Face lists it.

## Use cases

The use case (`--use`) sets the context length of all rows and the precision floor of the
default row:

| Preset | Context | Tool calling | Precision floor |
| --- | --- | --- | --- |
| `coding` (default) | 131072 | yes | 8-bit |
| `chat` | 32768 | no | 4-bit |
| `research` | model maximum | yes | 8-bit |

The 8-bit floor for coding is a judgement, not a measurement. An agent makes long chains of
exact tool calls, and small errors cause more damage there than in conversation. For
Qwen3.8-27B, 4-bit saves no money: in the sample of 2026-10-27, the cheapest 4-bit and 8-bit
rows are on the same A40 at $0.59/h.

Because the floor is not measured, it only selects the default row. The table also shows the
rows below the floor, so you can compare quality and price yourself. `--min-precision` removes
them.

## What 4-bit gives: memory, not speed

4-bit weights seldom let the model fit on a cheaper GPU for coding. For Qwen3.8-27B at 128K,
4-bit needs 24–30 GiB. A 24 GB card has only 22.1 GiB usable, so even at 4 bits the model does
not fit.

4-bit gives free memory on the same GPU. On an A40 at 128K, FP8 weights use 39.9 of 44.2 GiB
and 4-bit uses 29.3 GiB. Thus approximately 10 GiB more is available for vLLM's KV cache. More
memory gives more parallel agent sessions, more prefix-cache hits on repeated prompts, or the
model's full context on a cheap card. The `memory` column shows the free memory.

4-bit does not give speed. Memory bandwidth limits decoding, so fewer bytes per weight can
give faster tokens. But the unpacking of 4-bit weights costs approximately the time that the
smaller read saves. imec's October 2026 study of Qwen3.8-27B on vLLM, run as a coding agent on
an H200, measured 985 tok/s for FP8, 857 for BF16 and 808 for INT4 AWQ. Thus the `tok/s`
column uses the 8-bit size for 4-bit rows.

Quality is the open question. In the same study, FP8 and INT4 scores were near BF16 on
SWE-Bench-Pro (BF16 33, FP8 30, INT4 32 of 64 tasks). But Jang et al. (arXiv 2607.27275) found
that 4-bit models make up to 2.5× more malformed tool calls, while benchmark scores stay
the same. For this reason the coding floor is 8-bit. Select a 4-bit row when you need memory
more than reliability.

## Where the weights come from

The quantised variants come from the Hugging Face Hub, most downloaded first. Anyone can
publish a repository with a quantisation tag, and download counts are easy to inflate. Thus
`up` offers only variants from the base model's own organisation and from the organisations in
`publishers.yaml` (`RedHatAI` and `nvidia` built in). `--any-publisher` lists the rest too,
marked `(untrusted publisher)`. Fewer 4-bit rows is the cost: some AWQ variants come only from
individuals.

A `.safetensors` file holds only tensors, and thetaserve never passes `--trust-remote-code`.
Thus a variant cannot run code on the host. But a publisher can change the weights or the chat
template, and no metadata shows this. A trusted publisher limits whom you trust; it does not
verify the weights.

`up` pins each launch to the commit whose sizes it read (`vllm serve --revision`). The
tokenizer and chat template are pinned with it. Thus a push to the repository after you select
an option does not change what runs.

## One GPU

`--device "NVIDIA A40"` shows only the options on that GPU. Use it when you must repeat a
launch on a specific GPU, for example for a benchmark or a bug report. Add `--context` and
`--kv-cache-dtype` to set the row too.

## What is hand-written

No metadata gives two things: the vLLM tool-call and reasoning parsers for each model family,
and the memory overhead. The parsers come from the `common_args` of the model's entry in
`models.yaml`. A model without an entry launches without them, with a warning. The 3.2 GiB
overhead is the larger of two measurements (Qwen3.8-27B 2.2 GiB, Gemma 4 31B 3.1 GiB), rounded
up. For Gemma 4, vLLM also
needs 1.5× the KV cache that `config.json` gives; `kv_scale` in `models.yaml` records that.
