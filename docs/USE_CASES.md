# Use cases — the same local RAG pipeline across three verticals

The pipeline is deliberately domain-agnostic: ingest a document corpus → chunk it
section-aware → build a hybrid (vector + keyword) index → let an agentic loop call
retrieval tools → answer with citations → verify grounding. Only three things
change between verticals: the **corpus**, the **tools**, and the **metadata**.

Everything below reuses the same `localrag` package, the same backends (local
Transformers / vLLM / NIM / extractive), the same telemetry and the same eval
harness.

---

## 1. Manufacturing QA & maintenance — *built here*

| Aspect | What it is in this repo |
| --- | --- |
| Corpus | 40 shop-floor docs: maintenance manuals, spec sheets, QA work instructions, incident/change notes for "Northline Precision — Cell 4". |
| Tools | `search_docs`, `lookup_asset(asset_id)`, `get_spec_value(asset_id, spec)`, `check_superseded(doc_id)`. |
| Metadata | `doc_id`, `asset`, `revision`, `effective_date`, `supersedes` (+ superseded section). |
| Hard problem solved | **Supersede reasoning** — a newer incident note overrides an older manual value; the retriever recency-boosts the new revision and the `check_superseded` tool surfaces it. |
| Proof point | The agent answers "current torque for CNC-07 spindle clamp bolts = 52 N·m" with a citation to `INC-2026-001`, not the stale 45 N·m manual value. |

## 2. AECO — architecture, engineering, construction & operations drawings/specs

| Aspect | What changes |
| --- | --- |
| Corpus | Drawing registers, specification sections (MasterFormat/Uniclass), RFI logs, addenda, submittals (PDF/IFC/DWG exports → text + tables). |
| Tools | `search_drawings`, `get_detail(sheet_id)`, `resolve_revision(sheet_id)` (like `check_superseded`), `list_scope(spec_section)`. |
| Metadata | `sheet_id` / `spec_section`, `discipline`, `revision` (P1/C1…), `issue_date`, `supersedes`, `level`/`grid_ref`. |
| What stays | Section-aware chunking, hybrid retrieval, revision-aware recency, citation-grounded answers, the backend swap matrix, the benchmark harness. |
| Hard problem | "Which revision of detail 3/A-501 governs this connection, and does the addendum change it?" — the same supersede logic, with sheet/issue metadata instead of asset/revision. |

## 3. Media & entertainment — asset-metadata search

| Aspect | What changes |
| --- | --- |
| Corpus | Shot/scene metadata, camera & lens specs, LUT/colour pipeline notes, VFX turnover sheets, rights/clearance records, EDL/AAF summaries. |
| Tools | `search_assets`, `get_camera_spec(project, camera)`, `check_version(asset_id)` (supersede analogue), `list_rights(asset_id)`. |
| Metadata | `asset_id` / `shot_id`, `project`, `version`, `created_date`, `supersedes`, `codec`/`resolution`/`fps`. |
| What stays | Chunking (here: timecoded sections), hybrid FAISS+BM25 retrieval, recency/version preference, local GPU inference, NVML telemetry, JSON + Markdown eval outputs. |
| Hard problem | "What colour space and LUT did shot 104 take to delivery, and has the grade been revised since?" — version-aware retrieval with citation back to the turnover sheet. |

---

## What is reusable vs. bespoke

**Reusable as-is:** `chunking.py`, `embeddings.py`, `index.py`, `retrieve.py`,
`backends.py`, `bench.py`, `telemetry.py`, the CLI shape, the Docker/compose
deploy, the WSL playbook, and the evaluation harness.

**Bespoke per vertical:** the corpus generator, the tool *implementations* (the
four tool names above are examples), the front-matter schema (the fields the
recency/supersede logic reads), and the gold question set used to score quality.

The design keeps the boundary clean so a team can stand up a new vertical by
swapping the corpus + the tool layer while inheriting the retrieval, inference,
telemetry and evaluation machinery unchanged.
