# Demo video script — LocalRAG Hero Demo (3–5 minutes)

**Format:** screen recording, 1920×1080, terminal font ≥ 16 pt, light-on-dark.
**Tools:** [asciinema](https://asciinema.org/) for the terminal, or OBS Studio for
a composed walkthrough with a webcam inset.
**Save the recording to:** `recordings/localrag_hero_demo.mp4` (or `.cast`).
**Length target:** 3:45–4:30.

> Everything shown is run on the actual host. If a GPU is not present, the GPU
> rows in the benchmark table visibly read `not_measured_on_this_host` — say so
> out loud; do not imply a GPU that is not there.

---

## 0:00–0:30 — The problem

**On screen:** the title card, then `README.md` scrolled to the top.

**Narration:**
> "This is LocalRAG — an agentic, citation-grounded assistant that runs entirely
> on one workstation. The job: a maintenance technician on a factory floor asks a
> question in plain English, and gets an answer grounded in the shop-floor
> documents — with the exact document and revision cited. No cloud. No data
> leaves the machine."

## 0:30–1:15 — Bring-up and honest host check

**Type:**
```bash
bash scripts/bringup_check.sh
localrag doctor
```

**Narration:**
> "Step zero is a preflight. It checks the NVIDIA driver, the container toolkit,
> whether we are inside WSL2, free disk and free VRAM — and tells you exactly how
> to fix anything that fails. Then `doctor` prints the real host profile: OS,
> Python, torch, CUDA, GPU and free VRAM. It's probed, never hard-coded. On this
> machine" *(point at the GPU line)* "there is no NVIDIA GPU, so GPU configs will
> be reported as *not measured on this host* — we never fabricate a number."

## 1:15–2:45 — Live agentic queries

**Type:**
```bash
localrag corpus build
localrag index build
localrag search "current torque spec for the CNC-07 spindle clamp bolts" --k 5
localrag ask "What is the current torque spec for the CNC-07 spindle clamp bolts?" --show-trace
```

**Narration:**
> "Forty synthetic shop-floor documents — manuals, spec sheets, QA work
> instructions and incident notes — are chunked section-aware and indexed with a
> hybrid of FAISS vectors and BM25, fused by reciprocal rank. Now the agent.
> Watch the trace: it calls `search_docs`, then `lookup_asset`, then
> `get_spec_value`, and finally `check_superseded`. That last tool is the point:
> the manual says 45 newton-metres, but a newer incident note supersedes it. The
> agent prefers the newest applicable revision and answers 52 newton-metres,
> citing `INC-2026-001`. This is the supersede reasoning test."

**Type (refusal case):**
```bash
localrag ask "What is the serial number of the CNC-07 spindle motor?" --show-trace
```

**Narration:**
> "And when the answer genuinely is not in the corpus, it does not hallucinate —
> it returns INSUFFICIENT_EVIDENCE and cites nothing. That refusal is scored in
> the evaluation."

## 2:45–3:45 — Benchmark walkthrough

**On screen:** `results/benchmarks.md`, then `results/retrieval_eval.md` and
`results/agent_eval.md`.

**Narration:**
> "Every number in the README comes from a JSON file this build produced.
> Retrieval: recall@1, recall@3, recall@5 and MRR for dense-only, BM25-only and
> the hybrid. Agent quality: agentic versus a naive single-shot RAG baseline, on
> answer accuracy, citation precision, correct-refusal rate and latency, broken
> out per question category. Throughput: tokens per second, time to first token,
> p50 and p95 latency — with the host profile printed above the table. Rows the
> host can't run are shown as not-measured with the reason."

## 3:45–4:30 — Deployment swap: vLLM / NVIDIA NIM

**Type:**
```bash
docker compose --profile vllm up -d vllm
export LOCALRAG_ENDPOINT=http://localhost:8000/v1
export LOCALRAG_BACKEND=openai_compat
localrag ask "What is the current assist-gas pressure for the LASER-02?"
```

**Narration:**
> "The inference backend is pluggable. The same agent, the same tools, the same
> citations — but the tokens now come from a vLLM server. Swap the compose
> service for an NVIDIA NIM container and only the base URL and API key change.
> One workstation, or a shared inference server, with no change to the RAG logic."

**Closing card:** repo summary, `make demo` one-liner, and the measured headline
numbers.

---

## Recording checklist

- [ ] Terminal cleared; `cd` into the repo root.
- [ ] `results/` populated (`make eval bench` run beforehand, but show it live where possible).
- [ ] `recordings/` directory exists and is git-ignored.
- [ ] Font large enough to read on a laptop; hide personal shell prompt/env.
- [ ] Say the honest host status out loud (GPU present or not).
- [ ] Export to `recordings/localrag_hero_demo.mp4`; do not commit the binary.
