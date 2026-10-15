"""Hybrid retriever: dense cosine + BM25, fused with reciprocal-rank fusion and a
recency/supersede boost.

Available configurations (all scored against the same gold set):

    dense_only            cosine similarity over the FAISS/numpy index
    bm25_only             Okapi BM25 over the chunk text
    hybrid_rrf            reciprocal-rank fusion of the two, no metadata
    hybrid_rrf_recency    RRF + recency boost + superseded-document penalty
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .chunking import Chunk
from .config import Settings, load_settings
from .embeddings import Embedder
from .index import VectorIndex

_TOKEN_RE = re.compile(r"[a-z0-9]+")

CONFIGS = ("dense_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_recency")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    superseded: bool = False
    boosted: float = 1.0

    @property
    def citation(self) -> str:
        return self.chunk.citation()


class HybridRetriever:
    """Combine a dense index with BM25 using reciprocal-rank fusion."""

    def __init__(
        self,
        index: VectorIndex,
        embedder: Embedder,
        settings: Settings | None = None,
        rrf_k: int | None = None,
        recency_weight: float | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.index = index
        self.embedder = embedder
        self.chunks: list[Chunk] = index.chunks
        self.rrf_k = rrf_k if rrf_k is not None else self.settings.rrf_k
        self.recency_weight = (
            recency_weight if recency_weight is not None else self.settings.recency_weight
        )
        self._build_bm25()
        self._build_supersede_map()
        self._build_recency()

    # ------------------------------------------------------------------ #
    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        self._tokenized = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def _build_supersede_map(self) -> None:
        """Map ``doc_id`` -> the sections that a newer document supersedes.

        A value of ``None`` means the whole document is superseded. Targeting the
        superseded *section* (e.g. ``MANUAL-CNC-07 §4``) rather than the whole
        document keeps the still-valid sections (safety, fault codes, other
        torque values) fully rankable.
        """
        sup: dict[str, set[str] | None] = {}
        for c in self.chunks:
            target = getattr(c, "supersedes", None)
            if not target:
                continue
            target = str(target)
            section = getattr(c, "supersedes_section", None)
            if section and (target not in sup or sup[target] is not None):
                if target not in sup:
                    sup[target] = set()
                if sup[target] is not None:
                    sup[target].add(str(section))
            elif not section:
                sup[target] = None
        self._superseded_docs = sup

    def _build_recency(self) -> None:
        """Map each doc_id to a normalised recency weight in [0, 1]."""
        dates: dict[str, date] = {}
        for c in self.chunks:
            d = _parse_date(c.effective_date)
            if d is None:
                continue
            prev = dates.get(c.doc_id)
            if prev is None or d > prev:
                dates[c.doc_id] = d
        self._doc_dates = dates
        if dates:
            lo, hi = min(dates.values()), max(dates.values())
            span = (hi - lo).days or 1
            self._doc_recency = {
                doc: (d - lo).days / span for doc, d in dates.items()
            }
        else:
            self._doc_recency = {}

    # ------------------------------------------------------------------ #
    def dense_ranking(self, query: str, n: int) -> list[tuple[int, float]]:
        qv = self.embedder.encode([query], use_cache=True)[0]
        return self.index.search(qv, n)

    def bm25_ranking(self, query: str, n: int) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [(int(i), float(scores[i])) for i in order]

    # ------------------------------------------------------------------ #
    def _boost(self, idx: int) -> tuple[bool, float]:
        chunk = self.chunks[idx]
        superseded = False
        if chunk.doc_id in self._superseded_docs:
            secs = self._superseded_docs[chunk.doc_id]
            superseded = secs is None or chunk.section in secs
        factor = 0.45 if superseded else 1.0
        recency = self._doc_recency.get(chunk.doc_id, 0.0)
        factor *= 1.0 + self.recency_weight * recency
        return superseded, factor

    def search(
        self,
        query: str,
        k: int = 5,
        config: str = "hybrid_rrf_recency",
        pool: int | None = None,
    ) -> list[Hit]:
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}; choose from {CONFIGS}")
        if not self.chunks:
            return []
        n = pool or max(k * 4, 20)

        dense = self.dense_ranking(query, n)
        sparse = self.bm25_ranking(query, n)
        dense_scores = dict(dense)
        bm25_scores = dict(sparse)
        dense_ranks = {i: r + 1 for r, (i, _) in enumerate(dense)}
        bm25_ranks = {i: r + 1 for r, (i, _) in enumerate(sparse)}

        if config == "dense_only":
            ranked = [(i, s) for i, s in dense]
            fused = {i: s for i, s in dense}
        elif config == "bm25_only":
            ranked = [(i, s) for i, s in sparse]
            fused = {i: s for i, s in sparse}
        else:
            fused: dict[int, float] = {}
            for i in dense_ranks:
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + dense_ranks[i])
            for i in bm25_ranks:
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + bm25_ranks[i])
            if config == "hybrid_rrf_recency":
                boosted: dict[int, float] = {}
                for i, s in fused.items():
                    _, factor = self._boost(i)
                    boosted[i] = s * factor
                fused = boosted
            ranked = sorted(fused.items(), key=lambda kv: -kv[1])

        hits: list[Hit] = []
        for i, s in ranked[:k]:
            superseded, factor = self._boost(i)
            hits.append(
                Hit(
                    chunk=self.chunks[i],
                    score=float(s),
                    dense_score=float(dense_scores.get(i, 0.0)),
                    bm25_score=float(bm25_scores.get(i, 0.0)),
                    dense_rank=dense_ranks.get(i),
                    bm25_rank=bm25_ranks.get(i),
                    superseded=superseded,
                    boosted=factor,
                )
            )
        return hits


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- #
# Convenience loaders                                                         #
# --------------------------------------------------------------------------- #
def load_retriever(
    index_dir: str = "artifacts/index",
    settings: Settings | None = None,
) -> HybridRetriever:
    settings = settings or load_settings()
    index = VectorIndex.load(index_dir)
    embedder = Embedder(
        model_name=settings.embedding_model,
        device=settings.device,
        cache_dir=settings.path("cache_dir"),
        allow_download=settings.allow_download,
    )
    return HybridRetriever(index, embedder, settings)
