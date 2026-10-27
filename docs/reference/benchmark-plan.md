# M4 benchmark plan

The runs and metrics of milestone M4 (`thetaserve bench`, [DESIGN.md section 10](../DESIGN.md#10-benchmark-harness-thetaserve-bench)).
Status: the harness is done, the runs are pending (DESIGN.md section 12, 2026-10-27).

## Runs

All runs use Runpod Secure Cloud, without a cache volume. Each run records every metric in the
next table.

| # | Model | Weights (repository) | GPU | Context | KV cache | Compares |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Qwen3.8-27B | 8-bit FP8 (`Qwen/Qwen3.8-27B-FP8`) | A40 48 GB, Ampere | 128K | 16-bit | Baseline, cheapest card. Trial run first |
| 2 | Qwen3.8-27B | 4-bit INT4 (`RedHatAI/Qwen3.8-27B-INT4`) | A40 48 GB, Ampere | 128K | 16-bit | 4-bit against 8-bit on the same GPU (with 1) |
| 3 | Qwen3.8-27B | 8-bit FP8 (`Qwen/Qwen3.8-27B-FP8`) | L40S 48 GB, Ada | 128K | 16-bit | Ampere against Ada, same weights (with 1) |
| 4 | Qwen3.8-27B | 16-bit BF16 (`Qwen/Qwen3.8-27B`) | A100-SXM4 80 GB, Ampere | 128K | 16-bit | 16-bit weights on a high-bandwidth 80 GB card |
| 5 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | A40 48 GB, Ampere | 64K | 16-bit | Cheapest card; 128K does not fit |
| 6 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | L40S 48 GB, Ada | 128K | 8-bit FP8 | Full context on 48 GB with an FP8 KV cache |
| 7 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | A100-SXM4 80 GB, Ampere | 128K | 16-bit | 80 GB card with a 16-bit KV cache |
| 8 | Gemma 4 31B | 8-bit FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) | RTX PRO 6000 Blackwell 96 GB | 128K | 16-bit | Baseline for 9 |
| 9 | Gemma 4 31B | 4-bit NVFP4 (`nvidia/Gemma-4-31B-IT-NVFP4`) | RTX PRO 6000 Blackwell 96 GB | 128K | 16-bit | 4-bit against 8-bit on the same GPU (with 8) |

## Metrics

All times are measured on the caller's machine, so they include the Runpod proxy's latency.
"Public" means known before a run, from the model files or Runpod's catalogue.

| # | Group | Metric | How it is measured | Unit | Public before a run? | Qwen3.8-27B runs | Gemma 4 31B runs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Speed | Decode, 1 request | 256 output tokens, 3 rounds, median of (tokens − 1) ÷ time after the first token | tok/s | No; estimated from memory bandwidth | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 2 | Speed | Decode, 4 requests: per request | 4 parallel requests, 3 rounds, median per request | tok/s | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 3 | Speed | Decode, 4 requests: total | All tokens ÷ wall time, median of 3 rounds | tok/s | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 4 | Speed | Time to first token, 8K prompt | One streamed request, fresh random prefix | s | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 5 | Speed | Time to first token, 32K prompt | Same | s | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 6 | Speed | Time to first token, 128K prompt | Same; skipped when above the run's context | s | No | 1, 2, 3, 4 | 6, 7, 8, 9 |
| 7 | Memory | Weights | vLLM start-up log | GiB | Yes: sum of `.safetensors` files | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 8 | Memory | Peak activation | vLLM start-up log | GiB | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 9 | Memory | CUDA graphs | vLLM start-up log | GiB | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 10 | Memory | KV-cache memory | vLLM start-up log | GiB | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 11 | Memory | KV-cache capacity | vLLM start-up log | tokens | In theory, from `config.json`; Gemma needed 1.5× | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 12 | Memory | Estimate | thetaserve's fit check: weights, KV cache, overhead, budget, `kv_scale` | GiB | Computed from public data | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 13 | Start-up | Weights download | vLLM start-up log | s | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 14 | Start-up | Time to ready | Pod creation to `/v1/models` answering | s | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 15 | Cost | Price | Runpod's rate for the pod | $/h | Yes: Runpod catalogue | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 16 | Cost | Per million output tokens, 1 request | $/h ÷ (metric 1 × 3,600) × 10⁶ | $ | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 17 | Cost | Per million output tokens, 4 requests | $/h ÷ (metric 3 × 3,600) × 10⁶ | $ | No | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 18 | Quality | Tool-call pass rate | 10 cases × 5 at temperature 0.7 = 50 calls; passes only with the exact function and arguments | passed / 50 | Partly: BFCL-type scores for the original weights | 1, 2, 3, 4; 4-bit against 8-bit: 2 with 1 | 5, 6, 7, 8, 9; 4-bit against 8-bit: 9 with 8 |
| 19 | Setup | Configuration hash | Image, repository, vLLM arguments, GPU, cloud type | hash | Yes | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 20 | Setup | Software | thetaserve version, vLLM version, image and its Docker Hub digest | text | Yes | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 21 | Setup | Weights | Repository, pinned revision, format | text | Yes | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 22 | Setup | Launch settings | Context, KV-cache type, full `vllm serve` arguments | text | Yes | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |
| 23 | Setup | Hardware | GPU, VRAM, cloud type, data centre | text | Yes | 1, 2, 3, 4 | 5, 6, 7, 8, 9 |

Not measured: a warm restart. Without a cache volume, a stopped pod loses its container disk, so
a restart is a second cold start.
