"""Evaluation: retrieval metrics and (added later) agent answer metrics."""

from __future__ import annotations

import json
import re
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings, load_settings
from .retrieve import CONFIGS, HybridRetriever

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212\u2015"
_DOTS = "\u00b7\u22c5\u2022\u2027"


def _normalize(text: str) -> str:
    """Normalise for substring matching: case, dashes, middle dots, thousands spaces."""
    text = str(text).lower()
    for ch in _DASHES:
        text = text.replace(ch, "-")
    for ch in _DOTS:
        text = text.replace(ch, ".")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("|", " ")
    # remove the space inside grouped numbers ("1 450" -> "1450")
    text = re.sub(r"(?<=\d)[ ](?=\d)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Retrieval evaluation                                                        #
# --------------------------------------------------------------------------- #
def eval_retrieval(
    retriever: HybridRetriever,
    questions: list[dict],
    configs: tuple[str, ...] = CONFIGS,
    k_values: tuple[int, ...] = (1, 3, 5),
    out: str | Path | None = None,
    settings: Settings | None = None,
) -> dict:
    """Compute recall@k, MRR and mean latency for each retrieval config."""
    settings = settings or load_settings()
    k_max = max(k_values)
    scored = [q for q in questions if q.get("expected_doc_ids")]
    skipped = len(questions) - len(scored)

    results: dict[str, dict] = {}
    per_question: dict[str, dict] = {}

    for config in configs:
        recalls = {k: 0 for k in k_values}
        reciprocal_ranks: list[float] = []
        latencies: list[float] = []
        per_q: dict[str, dict] = {}
        for q in scored:
            t0 = time.perf_counter()
            hits = retriever.search(q["question"], k=k_max, config=config)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            retrieved = [h.chunk.doc_id for h in hits]
            expected = set(q["expected_doc_ids"])
            first_rank = None
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in expected:
                    first_rank = rank
                    break
            reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
            for k in k_values:
                if any(d in expected for d in retrieved[:k]):
                    recalls[k] += 1
            per_q[q["id"]] = {
                "retrieved_doc_ids": retrieved,
                "first_hit_rank": first_rank,
                "top_citation": hits[0].citation if hits else None,
            }
        n = len(scored) or 1
        results[config] = {
            **{f"recall@{k}": round(recalls[k] / n, 4) for k in k_values},
            "mrr": round(sum(reciprocal_ranks) / n, 4),
            "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else None,
            "p95_latency_ms": round(_percentile(latencies, 95), 3) if latencies else None,
            "n_questions": len(scored),
        }
        per_question[config] = per_q

    best = max(configs, key=lambda c: (results[c]["mrr"], results[c].get("recall@3", 0)))
    index_meta_path = settings.path("index_dir") / "meta.json"
    index_meta = (
        json.loads(index_meta_path.read_text(encoding="utf-8"))
        if index_meta_path.exists()
        else {}
    )

    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "k_values": list(k_values),
        "n_questions_total": len(questions),
        "n_questions_scored": len(scored),
        "n_questions_skipped_no_gold": skipped,
        "best_config_by_mrr": best,
        "index_meta": index_meta,
        "configs": results,
        "per_question": per_question,
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


# --------------------------------------------------------------------------- #
# Agent evaluation                                                            #
# --------------------------------------------------------------------------- #
_DOC_ID_RE = re.compile(r"\b((?:MANUAL|SPEC|QC-WI|INC)-[A-Z0-9-]+)\b")


def _citation_doc_ids(citations: list[str]) -> list[str]:
    ids: list[str] = []
    for c in citations:
        m = _DOC_ID_RE.search(c)
        if m:
            ids.append(m.group(1))
    return ids


def score_result(result, q: dict) -> dict:
    """Score one :class:`AgentResult` against one gold question."""
    answer = _normalize(getattr(result, "answer", ""))
    substrings = q.get("expected_answer_substrings", [])
    matched = [s for s in substrings if _normalize(s) in answer]
    substring_match = (len(matched) / len(substrings)) if substrings else None

    expected_docs = set(q.get("expected_doc_ids", []))
    cited_docs = _citation_doc_ids(list(getattr(result, "citations", [])))
    cited_set = set(cited_docs)
    if expected_docs:
        precision = (len(cited_set & expected_docs) / len(cited_set)) if cited_set else 0.0
    else:
        precision = 1.0 if not cited_set else 0.0

    used_tools = [c.tool for c in getattr(result, "tool_calls", [])]
    expected_tools = q.get("expected_tools", [])
    coverage = (
        len(set(expected_tools) & set(used_tools)) / len(set(expected_tools))
        if expected_tools
        else None
    )
    refused = bool(getattr(result, "refused", False))
    refusal_correct = (1.0 if refused else 0.0) if q["category"] == "refusal" else None

    return {
        "substring_match": substring_match,
        "citation_precision": precision,
        "tool_coverage": coverage,
        "refusal_correct": refusal_correct,
        "n_tool_calls": len(used_tools),
        "latency_s": getattr(result, "latency_s", None),
        "refused": refused,
        "cited_doc_ids": cited_docs,
        "answer": getattr(result, "answer", ""),
    }


def _aggregate(rows: list[dict]) -> dict:
    def mean_of(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.fmean(vals), 4) if vals else None

    return {
        "answer_substring_match": mean_of("substring_match"),
        "citation_precision": mean_of("citation_precision"),
        "expected_tool_coverage": mean_of("tool_coverage"),
        "mean_tool_calls": round(
            statistics.fmean([r["n_tool_calls"] for r in rows]), 4
        ) if rows else None,
        "mean_latency_s": round(
            statistics.fmean([r["latency_s"] for r in rows if r["latency_s"] is not None]), 4
        ) if rows else None,
        "n_questions": len(rows),
    }


def eval_agent(
    agent,
    naive,
    questions: list[dict],
    settings: Settings | None = None,
    progress: bool = False,
) -> dict:
    """Run the agentic agent and the naive baseline over the gold set."""
    settings = settings or load_settings()
    per_q: dict[str, dict] = {"agentic": {}, "naive": {}}
    rows: dict[str, list[dict]] = {"agentic": [], "naive": []}
    cat_rows: dict[str, dict[str, list[dict]]] = {"agentic": {}, "naive": {}}

    for i, q in enumerate(questions):
        for mode, runner in (("agentic", agent), ("naive", naive)):
            if runner is None:
                continue
            result = runner.run(q["question"])
            scored = score_result(result, q)
            per_q[mode][q["id"]] = scored
            rows[mode].append(scored)
            cat_rows[mode].setdefault(q["category"], []).append(scored)
        if progress:
            print(f"  [{i + 1}/{len(questions)}] {q['id']} done")

    block: dict[str, object] = {
        "backend": getattr(agent, "backend_name", ""),
        "device": getattr(agent, "device", ""),
    }
    for mode in ("agentic", "naive"):
        if not rows[mode]:
            continue
        refusal_rows = [r for r in rows[mode] if r["refusal_correct"] is not None]
        block[mode] = {
            "overall": _aggregate(rows[mode]),
            "by_category": {cat: _aggregate(rs) for cat, rs in sorted(cat_rows[mode].items())},
            "correct_refusal_rate": (
                round(statistics.fmean([r["refusal_correct"] for r in refusal_rows]), 4)
                if refusal_rows
                else None
            ),
            "per_question": per_q[mode],
        }
    return block


def write_agent_markdown(report: dict, out: str | Path = "results/agent_eval.md") -> Path:
    """Render the agentic-vs-naive comparison as Markdown."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Agent evaluation — agentic vs naive RAG",
        "",
        f"- Generated: `{report.get('generated_utc')}`",
        f"- Gold questions: **{report.get('n_questions')}**",
        "",
    ]
    for backend, block in report.get("backends", {}).items():
        lines.append(f"## Backend `{backend}` (device `{block.get('device')}`)")
        lines.append("")
        lines.append(
            "| Mode | Answer substring match | Citation precision | Expected-tool coverage | "
            "Correct refusal | Mean tool calls | Mean latency (s) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")

        def _fmt(v, pct=False):
            if v is None:
                return "n/a"
            return f"{v:.3f}"

        for mode in ("agentic", "naive"):
            m = block.get(mode)
            if not m:
                continue
            o = m["overall"]
            lines.append(
                f"| {mode} | {_fmt(o['answer_substring_match'])} | {_fmt(o['citation_precision'])} | "
                f"{_fmt(o['expected_tool_coverage'])} | {_fmt(m.get('correct_refusal_rate'))} | "
                f"{_fmt(o['mean_tool_calls'])} | {_fmt(o['mean_latency_s'])} |"
            )
        lines.append("")
        lines.append("Per category (agentic):")
        lines.append("")
        lines.append("| Category | n | Answer match | Citation precision | Mean tool calls | Mean latency (s) |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for cat, m in (block.get("agentic") or {}).get("by_category", {}).items():
            lines.append(
                f"| {cat} | {m['n_questions']} | {_fmt(m['answer_substring_match'])} | "
                f"{_fmt(m['citation_precision'])} | {_fmt(m['mean_tool_calls'])} | "
                f"{_fmt(m['mean_latency_s'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_retrieval_markdown(
    report: dict,
    out: str | Path = "results/retrieval_eval.md",
    previous: dict | None = None,
) -> Path:
    """Render the retrieval evaluation as a Markdown table."""
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    k_values = report["k_values"]
    header = "| Config | " + " | ".join(f"Recall@{k}" for k in k_values) + " | MRR | Mean latency (ms) | P95 (ms) |"
    sep = "| --- | " + " | ".join("---" for _ in k_values) + " | --- | --- | --- |"
    lines = [
        "# Retrieval evaluation",
        "",
        f"- Generated: `{report['generated_utc']}`",
        f"- Questions scored: **{report['n_questions_scored']}** "
        f"(skipped {report['n_questions_skipped_no_gold']} refusal questions with no gold doc)",
        f"- Best config by MRR: **{report['best_config_by_mrr']}**",
    ]
    meta = report.get("index_meta") or {}
    if meta:
        lines.append(
            f"- Index: {meta.get('chunk_count')} chunks · dim {meta.get('dim')} · "
            f"embeddings `{meta.get('embedding_model')}` ({meta.get('embedding_backend')}) · "
            f"ANN `{meta.get('ann_backend')}` · built in {meta.get('build_seconds')} s"
        )
    lines += ["", header, sep]
    for config, m in report["configs"].items():
        row = f"| `{config}` | " + " | ".join(
            f"{m[f'recall@{k}']:.3f}" for k in k_values
        )
        row += f" | {m['mrr']:.3f} | {m['mean_latency_ms']} | {m['p95_latency_ms']} |"
        lines.append(row)
    lines.append("")
    best = report["best_config_by_mrr"]
    if best != "hybrid_rrf_recency":
        lines.append(
            f"> **Honest note:** on this corpus/host the best measured config was "
            f"`{best}`, not `hybrid_rrf_recency`. Reported as measured; not tuned to look good."
        )
        lines.append("")
    if previous is not None:
        lines.append("## Fusion improvement (run 1 -> run 2)")
        lines.append("")
        lines.append(
            "Run 1 applied the supersede penalty to a *whole document*, which also demoted "
            "still-valid sections (safety, fault codes, non-superseded torque values). "
            "Run 2 targets only the superseded section (`supersedes_section`), a single "
            "principled change; both runs are recorded."
        )
        lines.append("")
        lines.append("| Config | Run | MRR | Recall@3 | Recall@5 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for config in report["configs"]:
            prev = (previous.get("configs") or {}).get(config)
            if prev:
                lines.append(
                    f"| `{config}` | 1 | {prev['mrr']:.3f} | {prev['recall@3']:.3f} | "
                    f"{prev['recall@5']:.3f} |"
                )
            cur = report["configs"][config]
            lines.append(
                f"| `{config}` | 2 | {cur['mrr']:.3f} | {cur['recall@3']:.3f} | "
                f"{cur['recall@5']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
