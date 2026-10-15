# LocalRAG Hero Demo — one-pager

**An agentic, citation-grounded shop-floor assistant that runs entirely on one
workstation.**

---

## The problem

Maintenance and QA engineers spend hours hunting through manuals, spec sheets,
work instructions and change notes for a single torque value or fault-code
meaning — and the answer often lives in a *newer* document that supersedes the
manual they were reading. Cloud RAG is not an option when that documentation is
export-controlled or simply cannot leave the plant. What is needed is a
retrieval system that runs locally, reasons over document *revisions*, and — when
it does not know — says so instead of guessing.

## The solution

A fully local pipeline: a 40-document synthetic shop-floor corpus is chunked
section-aware, embedded, and indexed with a **hybrid FAISS + BM25** retriever
fused by reciprocal rank with a **recency/supersede boost**. An **agentic loop**
(max 4 tool calls) drives four real retrieval tools — `search_docs`,
`lookup_asset`, `get_spec_value`, `check_superseded` — and every answer is passed
through a **grounding verifier** that checks each citation and each number
against the chunks that were actually retrieved. Inference is pluggable: local
Hugging Face Transformers (CPU or CUDA), any OpenAI-compatible endpoint
(vLLM / **NVIDIA NIM**), or a deterministic extractive backend for CI.

## Proof points (measured on the build host)

1. **Supersede reasoning, grounded and cited.** Asked for the *current* CNC-07
   spindle-clamp torque, the agent returns **52 N·m citing `INC-2026-001`** — not
   the stale 45 N·m in the manual. Agentic citation precision **0.714**
   (Qwen2.5-0.5B on CPU) / **0.795** (extractive) versus **0.586 / 0.610** for a
   naive single-shot RAG baseline, and **correct-refusal rate 1.00** versus
   **0.00 / 0.20** for naive. The agent refuses the unanswerable questions.

2. **Hybrid retrieval that works on a tiny index.** 204 chunks from 40 documents:
   **recall@5 = 0.933 and MRR = 0.703** for the hybrid (recency-aware) config;
   BM25 alone recalls **0.967 @5**; index build is **0.715 s**. Recall@1/@3/@5 and
   MRR are reported for dense-only, BM25-only and hybrid, with the fusion
   improvement (run 1 → run 2) recorded side by side.

3. **Runs on one workstation, measured honestly.** Local Qwen2.5-0.5B-Instruct on
   **CPU fp32 at 22.08 ± 2.52 tokens/s**, **TTFT 0.209 s**, **p95 latency
   9.29 s** over a 12-prompt workload (3 reps, warm-up discarded). GPU configs
   (`hf_cuda_fp16`, `hf_cuda_fp16_batch4`) and the `endpoint_vllm_or_nim` config
   are recorded as **`not_measured_on_this_host`** with the reason (no CUDA device
   / no endpoint) — no number is invented.

## Deployment footprint

| Concern | What ships |
| --- | --- |
| Code | Python 3.11 package `localrag` (CLI: `localrag`), ~2 000 statements, 67 tests, 68 % coverage, ruff-clean. |
| Container | `Dockerfile` GPU default (`nvidia/cuda:12.4.1-runtime-ubuntu22.04`) + documented CPU path (`--build-arg BASE=python:3.11-slim`); **CPU image built and run on the build host (10.5 GB)**. |
| Orchestration | `docker-compose.yml` with GPU device reservation and an optional vLLM service; drop-in NVIDIA NIM container documented with its env vars. |
| One command | `make setup && make demo` (corpus → index → 5 scripted questions, Rich output). |
| Bring-up | `scripts/bringup_check.sh` preflight (driver, toolkit, WSL2, disk, VRAM, HF cache) + `scripts/wsl_setup.md`. |
| Measurement | Retrieval, agent, and throughput results written to `results/*.json` and `*.md`. |

## What this proves about local, GPU-accelerated enterprise AI

- **A single workstation is enough** to run an agentic, citation-grounded RAG
  assistant — no cloud round-trips, no data egress.
- **The inference layer is portable**: the same agent and tools run against local
  Transformers, vLLM, or an NVIDIA NIM container by changing a base URL and key.
- **Revision-aware retrieval is the differentiator** for industrial documents:
  the supersede logic is what lets a technician trust the answer.
- **Measurement and honesty are designed in**, not bolted on: unsupported configs
  are reported as not-measured, and every published figure traces to a JSON file
  the build produced.
