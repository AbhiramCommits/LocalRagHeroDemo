# LocalRAG Hero Demo — Agentic Shop-Floor Assistant on a Single GPU Workstation

A fully local, GPU-accelerated **agentic RAG** system for manufacturing
maintenance and QA. It ingests a corpus of shop-floor documents — maintenance
manuals, machine spec sheets, QA work instructions and incident/change notes —
and answers natural-language technician questions through a tool-calling agent
that returns **citation-grounded** answers ("per `MANUAL-CNC-07 §4`"). Inference
is pluggable across local Hugging Face Transformers (CPU or CUDA), any
OpenAI-compatible endpoint (vLLM / NVIDIA NIM), or a deterministic extractive
fallback used in CI. It measures itself — retrieval recall/MRR, answer accuracy,
citation precision, tokens/sec, time-to-first-token, p50/p95 latency and peak
VRAM — and every number in this README comes from a JSON file this build produced.

> **Host honesty.** The build host is an Apple-Silicon Mac with **no NVIDIA GPU**.
> CPU configs were run for real; GPU configs are reported as
> `not_measured_on_this_host` with the reason. No tokens/sec, device name or VRAM
> figure is fabricated anywhere in this repository.

---

## 60-second try it

```bash
conda env create -f environment.yml && conda activate localrag   # python 3.11
pip install -e ".[all]"
make corpus index demo
```

`make demo` builds the corpus and index (if needed) and answers five scripted
shop-floor questions with Rich output, including a supersede case and a refusal
case. To see the full agent trace:

```bash
localrag ask "What is the current torque spec for the CNC-07 spindle clamp bolts?" --show-trace
```

---

## Architecture

```
  data/corpus/  (40 Markdown docs: manuals · specs · QA work instructions · incident notes)
        │
        ▼
  section-aware chunker            keeps tables intact, emits citable §sections
  (chunking.py)                    Chunk(doc_id, section, revision, effective_date, …)
        │
        ├──────────────► embeddings (sentence-transformers all-MiniLM-L6-v2, 384-d)
        │                    │  cuda when available, else cpu; on-disk content-hash cache
        │                    ▼
        │              FAISS IndexFlatIP (L2-normalised; pure-numpy fallback)
        │                    │
        │                    ▼
        └────────────► BM25 (rank_bm25) ──► reciprocal-rank fusion + recency/supersede boost
                                                  │
                                     ┌────────────▼────────────┐
                                     │  agentic loop (ReAct)   │  max 4 tool calls,
                                     │  search_docs            │  every step recorded
                                     │  lookup_asset           │
                                     │  get_spec_value         │
                                     │  check_superseded       │
                                     └────────────┬────────────┘
                                                  ▼
                                     grounding verifier  (every citation + number checked
                                     (grounding.py)       against the retrieved chunks)
                                                  ▼
                                          answer + citations   (or INSUFFICIENT_EVIDENCE)
                                                  ▲
                                                  │  tokens from one of:
   ┌──────────────────┬───────────────────────────┬──────────────────────────┬───────────────────┐
   │ Transformers     │ OpenAI-compatible endpoint│ NVIDIA NIM container     │ Extractive        │
   │ CPU fp32 / CUDA  │ vLLM  (/v1/chat/…)        │ (/v1/chat/…, NGC key)    │ (no weights, CI)  │
   │ fp16             │                           │                          │                   │
   └──────────────────┴───────────────────────────┴──────────────────────────┴───────────────────┘
```

**Backend swap matrix**

| Backend | How to select | Where it runs | Notes |
| --- | --- | --- | --- |
| `transformers` | `--backend transformers` | Local CPU fp32 / CUDA fp16 | Qwen2.5-0.5B-Instruct default; real streaming TTFT |
| `openai_compat` | `LOCALRAG_ENDPOINT=http://host:8000/v1` | vLLM server | `LOCALRAG_API_KEY` optional |
| `openai_compat` (NIM) | same env vars, NIM image in compose | NVIDIA NIM container | identical agent path |
| `extractive` | `--backend extractive` | anywhere | offline, deterministic, CI default |
| `auto` | default | best available | endpoint → transformers → extractive |

---

## How to run

**Step 0 — preflight (every environment):**

```bash
bash scripts/bringup_check.sh | tee results/bringup_report.txt
localrag doctor | tee results/host_profile.txt
```

### Linux (native)

```bash
conda env create -f environment.yml && conda activate localrag
pip install -e ".[all]"
localrag corpus build && localrag index build
localrag ask "What is the current torque spec for the CNC-07 spindle clamp bolts?" --show-trace
```

### Docker / Compose

```bash
# GPU machine (needs nvidia-container-toolkit + a CUDA driver):
docker build -t localrag-hero:gpu .
docker run --rm --gpus all localrag-hero:gpu demo
# or, with the optional vLLM endpoint service:
docker compose --profile vllm up -d vllm

# CPU-only machine:
docker build -t localrag-hero:cpu --build-arg BASE=python:3.11-slim .
docker run --rm localrag-hero:cpu demo
```

To point the agent at a vLLM server or an **NVIDIA NIM** container, set
`LOCALRAG_ENDPOINT` / `LOCALRAG_API_KEY` (see the NIM substitution block in
`docker-compose.yml`).

### Conda

```bash
conda env create -f environment.yml    # env "localrag", python 3.11
conda activate localrag && pip install -e ".[all]"
make corpus index demo
```

### WSL2

Install WSL2, install the **Windows** NVIDIA driver (never a driver inside the
distro), enable Docker Desktop WSL integration, then follow Linux above. Full
steps: [`scripts/wsl_setup.md`](scripts/wsl_setup.md).

---

## Results

All figures below were produced on the build host and are copied from the JSON in
[`results/`](results/). Host profile: **Apple Silicon (arm64), macOS 27.0.1 /
Darwin 27.0.0, Python 3.11.17, torch 2.14.1, MPS available, no NVIDIA GPU, NVML
unavailable** (`results/host_profile.txt`). Index: **204 chunks, 384-d
`all-MiniLM-L6-v2`, FAISS, built in 0.715 s**.

### (a) Retrieval — dense vs BM25 vs hybrid

Recall@k and MRR over the 30 answerable gold questions (the 5 refusal questions
have no gold document and are excluded). Source: `results/retrieval_eval.json`.

| Config | Recall@1 | Recall@3 | Recall@5 | MRR | Mean latency (ms) | P95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_only` | 0.633 | 0.800 | 0.900 | **0.729** | 23.007 | 39.686 |
| `bm25_only` | 0.567 | 0.867 | **0.967** | 0.727 | 0.558 | 0.730 |
| `hybrid_rrf` | 0.533 | 0.867 | 0.933 | 0.704 | 0.694 | 1.000 |
| `hybrid_rrf_recency` | 0.567 | 0.833 | 0.933 | 0.703 | 0.790 | 1.185 |

**Honest note.** On this small synthetic corpus the best config by MRR is
`dense_only` (0.729) and the best by recall@5 is `bm25_only` (0.967); the hybrid
is competitive but not best. This is reported as measured, not tuned to look
good. The one permitted fusion improvement (run 1 applied the supersede penalty to
a whole document; run 2 targets only the superseded section) is recorded side by
side in `results/retrieval_eval.md` — hybrid MRR rose from 0.506 to 0.703.

### (b) Agentic vs naive RAG

Over all 35 gold questions, per category. Sources:
`results/agent_eval.json` / `results/agent_eval.md`.

**Backend `transformers` (Qwen2.5-0.5B-Instruct, CPU fp32):**

| Mode | Answer substring match | Citation precision | Expected-tool coverage | Correct refusal | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| **agentic** | 0.700 | **0.714** | 0.986 | **1.000** | 3.343 | 21.097 |
| naive | 0.767 | 0.586 | 0.000 | 0.000 | 0.000 | 6.821 |

**Backend `extractive` (offline, deterministic — the CI path):**

| Mode | Answer substring match | Citation precision | Expected-tool coverage | Correct refusal | Mean tool calls | Mean latency (s) |
| --- | --- | --- | --- | --- | --- | --- |
| **agentic** | 0.933 | **0.795** | 0.986 | **1.000** | 3.286 | 0.004 |
| naive | 0.867 | 0.610 | 0.000 | 0.200 | 0.000 | 0.001 |

Per-category (agentic, transformers): lookup 0.400/0.650, numeric 0.750/0.375,
multi_hop 1.000/0.857, supersede 0.800/0.900, refusal —/1.000 (answer match /
citation precision).

**Honest note.** The agentic loop wins clearly on **citation precision**
(0.714 vs 0.586) and **correct-refusal** (1.000 vs 0.000), and covers the expected
tools 98.6% of the time. On this tiny 0.5B model the naive baseline's *raw
substring* score is occasionally higher (0.767 vs 0.700) because the agentic
answer is more concise; the agentic advantage is accuracy of *provenance*, not
verbosity. Both numbers are reported.

### (c) Throughput / latency benchmark

12-prompt shop-floor workload, 3 repetitions, 1 warm-up discarded. Source:
`results/benchmarks.json` / `results/benchmarks.md`.

| Config | Status | Device | tokens/s (mean ± stdev) | TTFT (s) | latency p50 / p95 (ms) | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| `extractive_cpu` | measured | cpu | 110411.286 ± 16313.453 | 0.000 | 0.019 / 0.026 | not_measured |
| `hf_cpu_fp32` | measured | cpu | 22.081 ± 2.523 | 0.209 | 5602.794 / 9286.150 | not_measured |
| `hf_cuda_fp16` | **not_measured_on_this_host** | — | — | — | — | — |
| `hf_cuda_fp16_batch4` | **not_measured_on_this_host** | — | — | — | — | — |
| `endpoint_vllm_or_nim` | **not_measured_on_this_host** | — | — | — | — | — |

Skipped-config reasons: `hf_cuda_fp16` and `hf_cuda_fp16_batch4` —
*"no CUDA device on this host"*; `endpoint_vllm_or_nim` — *"LOCALRAG_ENDPOINT is
not set (no vLLM/NIM server configured)"*. Peak VRAM is `not_measured` because
NVML is unavailable (`NVMLError_LibraryNotFound`). The `extractive_cpu` tokens/s
is a real measured value but is not comparable to a neural backend — it is a
rule-based span composer, not a token generator.

### (d) Tests + coverage

Source: `results/test_report.txt`.

| Metric | Value |
| --- | --- |
| Tests collected / passed | 67 / 67 (`pytest -m "not slow"`) |
| Coverage (`--cov=localrag`) | 68 % (1993 stmts, 635 missed) |
| Lint | `ruff check .` → clean |

### How these numbers were produced

```bash
conda activate localrag
localrag corpus build                                   # 40 docs + 35 gold questions
localrag index build                                    # 204 chunks → FAISS + BM25
localrag eval retrieval --out results/retrieval_eval.json
localrag eval agent --out results/agent_eval.json --backend transformers --backend extractive
localrag bench run --out results/benchmarks.json
localrag bench table
bash scripts/bringup_check.sh | tee results/bringup_report.txt
localrag doctor --out results/host_profile.txt
make lint test > results/test_report.txt
```

---

## Limitations & next steps

- **Small model.** The default local model is Qwen2.5-0.5B-Instruct — chosen so the
  demo runs on one workstation. It is weakest on `lookup` and `numeric`
  categories (answer match 0.40 / 0.75); a larger local model or a vLLM/NIM
  endpoint will lift these.
- **Synthetic corpus.** The 40 documents are fictional but internally consistent.
  Real corpora will need format handling (PDF/DWG/IFC) and a domain-specific gold
  set.
- **Configs not measurable on this host.** All three GPU/endpoint configs were
  recorded as `not_measured_on_this_host`; on a CUDA host they run unchanged, and
  the harness will fill the same table.
- **TensorRT-LLM left as a documented path.** For maximum single-GPU throughput,
  an engine build via TensorRT-LLM behind the same OpenAI-compatible endpoint is
  the next step; the agent and tools require no change.
- **Better fusion.** The hybrid fusion could be improved (learned weights,
  cross-encoder re-ranking); it was intentionally left at one principled iteration
  and reported honestly.

## Repository layout

```
localrag/     corpus · chunking · embeddings · index · retrieve · agent · backends ·
              grounding · bench · telemetry · evaluate · cli · config
config/       default.yaml (env-overridable via LOCALRAG_*)
data/         corpus/ (40 docs) · gold/questions.jsonl (35 questions)
results/      host_profile · retrieval_eval · agent_eval · benchmarks · bringup · test_report
scripts/      bringup_check.sh · smoke_test.sh · wsl_setup.md
docs/         ONE_PAGER.md · PLAYBOOK.md · DEMO_SCRIPT.md · USE_CASES.md
tests/        67 tests (CPU-only, offline)
```

## License

MIT — see [LICENSE](LICENSE).
