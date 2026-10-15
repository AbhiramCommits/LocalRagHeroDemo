"""Evaluation: retrieval metrics and (added later) agent answer metrics."""

from __future__ import annotations

import json
import re
import statistics
import time
from datetime import datetime, timezone
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
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
