# Benchmark results

- Generated: `2026-10-15T07:58:59+00:00`
- Host CPU: `arm` · OS: `Darwin 27.0.0` · WSL: `False`
- GPU: none detected (no CUDA device) · driver: `NOT FOUND` · torch CUDA: `None`
- Index: 204 chunks · embeddings `sentence-transformers/all-MiniLM-L6-v2` (sentence_transformers) · build 0.715 s
- Workload: 12 prompts × 3 reps (1 warm-up discarded), max_new_tokens=128

| Config | Status | Device | tokens/s (mean ± stdev) | TTFT s (mean) | latency ms (p50 / p95) | peak VRAM (MB) |
| --- | --- | --- | --- | --- | --- | --- |
| `extractive_cpu` | measured | cpu | 110411.286 ± 16313.4531 | 0.0 | 0.019 / 0.026 | not_measured |
| `hf_cpu_fp32` | measured | cpu | 22.081 ± 2.523 | 0.2093 | 5602.794 / 9286.15 | not_measured |
| `hf_cuda_fp16` | not_measured_on_this_host | — | — | — | — | — |
| `hf_cuda_fp16_batch4` | not_measured_on_this_host | — | — | — | — | — |
| `endpoint_vllm_or_nim` | not_measured_on_this_host | — | — | — | — | — |

## Skipped configs

| Config | Reason |
| --- | --- |
| `hf_cuda_fp16` | no CUDA device on this host |
| `hf_cuda_fp16_batch4` | no CUDA device on this host |
| `endpoint_vllm_or_nim` | LOCALRAG_ENDPOINT is not set (no vLLM/NIM server configured) |

