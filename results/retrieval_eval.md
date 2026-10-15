# Retrieval evaluation

- Generated: `2026-10-15T08:27:55+00:00`
- Questions scored: **30** (skipped 5 refusal questions with no gold doc)
- Best config by MRR: **dense_only**
- Index: 204 chunks · dim 384 · embeddings `sentence-transformers/all-MiniLM-L6-v2` (sentence_transformers) · ANN `faiss` · built in 0.715 s

| Config | Recall@1 | Recall@3 | Recall@5 | MRR | Mean latency (ms) | P95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| `dense_only` | 0.633 | 0.800 | 0.900 | 0.729 | 23.007 | 39.686 |
| `bm25_only` | 0.567 | 0.867 | 0.967 | 0.727 | 0.558 | 0.73 |
| `hybrid_rrf` | 0.533 | 0.867 | 0.933 | 0.704 | 0.694 | 1.0 |
| `hybrid_rrf_recency` | 0.567 | 0.833 | 0.933 | 0.703 | 0.79 | 1.26 |

> **Honest note:** on this corpus/host the best measured config was `dense_only`, not `hybrid_rrf_recency`. Reported as measured; not tuned to look good.

## Fusion improvement (run 1 -> run 2)

Run 1 applied the supersede penalty to a *whole document*, which also demoted still-valid sections (safety, fault codes, non-superseded torque values). Run 2 targets only the superseded section (`supersedes_section`), a single principled change; both runs are recorded.

| Config | Run | MRR | Recall@3 | Recall@5 |
| --- | --- | --- | --- | --- |
| `dense_only` | 1 | 0.729 | 0.800 | 0.900 |
| `dense_only` | 2 | 0.729 | 0.800 | 0.900 |
| `bm25_only` | 1 | 0.727 | 0.867 | 0.967 |
| `bm25_only` | 2 | 0.727 | 0.867 | 0.967 |
| `hybrid_rrf` | 1 | 0.704 | 0.867 | 0.933 |
| `hybrid_rrf` | 2 | 0.704 | 0.867 | 0.933 |
| `hybrid_rrf_recency` | 1 | 0.506 | 0.600 | 0.600 |
| `hybrid_rrf_recency` | 2 | 0.703 | 0.833 | 0.933 |

