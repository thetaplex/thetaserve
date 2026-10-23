# M4 benchmark plan

The runs and metrics of milestone M4 (`thetaserve bench`, [DESIGN.md section 10](../DESIGN.md#10-benchmark-harness-thetaserve-bench)).
Status: running, 2026-10-23.

## Runs

All runs use Runpod Secure Cloud. Each run records every metric in the next table.

| # | Model | Weights (repository) | GPU | Context, KV cache | Compares |
| --- | --- | --- | --- | --- | --- |
| 1 | Qwen3.8-27B | 8-bit FP8 (`Qwen/Qwen3.8-27B-FP8`) | A40 48 GB, Ampere | 128K, 16-bit | Baseline, cheapest card. Trial run first |
| 2 | Qwen3.8-27B | 4-bit INT4 (`RedHatAI/Qwen3.8-27B-INT4`) | A40 48 GB, Ampere | 128K, 16-bit | 4-bit against 8-bit on the same GPU (with 1) |
| 3 | Qwen3.8-27B | 8-bit FP8 (`Qwen/Qwen3.8-27B-FP8`) | L40S 48 GB, Ada | 128K, 16-bit | Ampere against Ada, same weights (with 1) |
| 4 | Qwen3.8-27B | 16-bit BF16 (`Qwen/Qwen3.8-27B`) | A100-SXM4 80 GB, Ampere | 128K, 16-bit | 16-bit weights on a high-bandwidth 80 GB card |
| 5 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | A40 48 GB, Ampere | 64K, 16-bit | Cheapest card; 128K does not fit |
| 6 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | L40S 48 GB, Ada | 128K, FP8 | Full context on 48 GB with an FP8 KV cache |
| 7 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | A100-SXM4 80 GB, Ampere | 128K, 16-bit | 80 GB card with a 16-bit KV cache |
| 8 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | RTX PRO 6000 Blackwell 96 GB | 128K, 16-bit | Baseline for 9 |
| 9 | Gemma 4 31B | 4-bit NVFP4 (`nvidia/Gemma-4-31B-IT-NVFP4`) | RTX PRO 6000 Blackwell 96 GB | 128K, 16-bit | 4-bit against 8-bit on the same GPU (with 8) |

## Metrics

All times are measured on the caller's machine, so they include the Runpod proxy's latency.
"Public" means known before a run, from the model files or Runpod's catalogue.

| # | Metric | Public before a run? | Source, or why not | Qwen3.8-27B runs | Gemma 4 31B runs |
| --- | --- | --- | --- | --- | --- |
| 1-3 | Decode speed: 1 request; 4 requests per request and in total | No; estimable | vLLM and NVIDIA publish H100-class figures only. Bandwidth estimate; the pilots measured approximately 1.4× it | 1-4 | 5-9 |
| 4-5 | Time to first token, 8K and 32K prompts | No | Depends on GPU compute, the FP8 path and vLLM prefill | 1-4 | 5-9 |
| 6 | Time to first token, 128K prompt | No | Same | 1-4 | 6-9 (not 5: 64K context) |
| 7 | Weights | Yes | Sum of `.safetensors` on Hugging Face; the pilots matched within 0.1 GiB | 1-4 | 5-9 |
| 8-9 | Peak activation, CUDA graphs | No | Depends on vLLM version, `--max-num-seqs` and context | 1-4 | 5-9 |
| 10-11 | KV-cache memory and capacity | In theory | `config.json` gives the KV size per token; Gemma needed 1.5× | 1-4 | 5-9 |
| 12 | Estimate | Computed from public data | thetaserve's fit check | 1-4 | 5-9 |
| 13-16 | Weight download time; time to ready, first start and warm restart | No | Runpod hosts, Hugging Face speed, cache volume | 1-4 | 5-9 |
| 17 | Price | Yes | Runpod's live catalogue | 1-4 | 5-9 |
| 18-19 | Cost per million output tokens, 1 and 4 requests | No | Public price ÷ measured speed | 1-4 | 5-9 |
| 20 | Tool-call pass rate (10 cases × 5 at temperature 0.7) | Partly | Model cards report BFCL-type scores for the original weights; rarely for quantised variants | 1-4; 4-bit against 8-bit: 2 with 1 | 5-9; 4-bit against 8-bit: 9 with 8 |
| 21-25 | Setup: configuration hash, versions, image digest, weights and revision, launch settings, hardware | Yes | Recorded for reproducibility | 1-4 | 5-9 |
