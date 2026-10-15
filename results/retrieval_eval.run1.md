# Retrieval evaluation

- Generated: `2026-10-15T07:27:48+00:00`
- Questions scored: **30** (skipped 5 refusal questions with no gold doc)
- Best config by MRR: **dense_only**
- Index: 204 chunks · dim 384 · embeddings `sentence-transformers/all-MiniLM-L6-v2` (sentence_transformers) · ANN `faiss` · built in 0.715 s

| Config | Recall@1 | Recall@3 | Recall@5 | MRR | Mean latency (ms) | P95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_only` | 0.633 | 0.800 | 0.900 | 0.729 | 9.559 | 7.565 |
| `bm25_only` | 0.567 | 0.867 | 0.967 | 0.727 | 0.233 | 0.304 |
| `hybrid_rrf` | 0.533 | 0.867 | 0.933 | 0.704 | 0.229 | 0.3 |
| `hybrid_rrf_recency` | 0.433 | 0.600 | 0.600 | 0.506 | 0.233 | 0.308 |

> **Honest note:** on this corpus/host the best measured config was `dense_only`, not `hybrid_rrf_recency`. Reported as measured; not tuned to look good.

