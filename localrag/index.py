"""Dense vector index: FAISS ``IndexFlatIP`` with a numpy brute-force fallback.

Vectors are stored L2-normalised, so inner product == cosine similarity. The
index persists to ``artifacts/index/``:

    index.faiss    FAISS binary (only when faiss is importable)
    vectors.npy    raw float32 vectors (used by the numpy fallback)
    chunks.json    the chunk records, in vector order
    meta.json      build metadata (dim, count, model, backend, timing)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class IndexMeta:
    dim: int
    chunk_count: int
    ann_backend: str  # "faiss" | "numpy"
    embedding_model: str
    embedding_backend: str
    device: str
    build_seconds: float
    corpus_dir: str
    created_utc: str
    load_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VectorIndex:
    dim: int
    backend: str
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    chunks: list[Chunk] = field(default_factory=list)
    _faiss: object | None = None

    # ------------------------------------------------------------------ #
    def build(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.chunks = chunks
        self.dim = int(self.vectors.shape[1]) if self.vectors.size else self.dim
        if faiss_available():
            import faiss

            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self.vectors)
            self.backend = "faiss"
        else:
            self._faiss = None
            self.backend = "numpy"

    # ------------------------------------------------------------------ #
    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Return ``[(chunk_index, score), ...]`` ranked by descending cosine."""
        if self.vectors.size == 0 or not self.chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        k = max(1, min(k, len(self.chunks)))
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, k)  # type: ignore[union-attr]
            return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]
        scores = (self.vectors @ q.T).ravel()
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    # ------------------------------------------------------------------ #
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self.vectors)
        (out / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        if self._faiss is not None:
            import faiss

            faiss.write_index(self._faiss, str(out / "index.faiss"))

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, index_dir: str | Path) -> VectorIndex:
        d = Path(index_dir)
        vectors = np.load(d / "vectors.npy") if (d / "vectors.npy").exists() else np.zeros((0, 0), np.float32)
        raw = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in raw]
        dim = int(vectors.shape[1]) if vectors.size else 0
        idx = cls(dim=dim, backend="numpy", vectors=vectors, chunks=chunks)
        fpath = d / "index.faiss"
        if faiss_available() and fpath.exists():
            import faiss

            idx._faiss = faiss.read_index(str(fpath))
            idx.backend = "faiss"
        return idx


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    out_dir: str | Path = "artifacts/index",
    corpus_dir: str = "data/corpus",
) -> IndexMeta:
    """Embed ``chunks``, build the index, persist it and return the metadata."""
    start = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunks], batch_size=32)
    index = VectorIndex(dim=embedder.dim, backend="numpy")
    index.build(vectors, chunks)
    elapsed = time.perf_counter() - start
    index.save(out_dir)

    meta = IndexMeta(
        dim=int(index.dim),
        chunk_count=len(chunks),
        ann_backend=index.backend,
        embedding_model=embedder.model_name,
        embedding_backend=embedder.backend,
        device=embedder.device,
        build_seconds=round(elapsed, 3),
        corpus_dir=str(corpus_dir),
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        load_error=embedder.load_error,
    )
    (Path(out_dir) / "meta.json").write_text(
        json.dumps(meta.to_dict(), indent=2), encoding="utf-8"
    )
    return meta


if __name__ == "__main__":  # pragma: no cover
    from .chunking import build_chunks

    cs = build_chunks()
    print(build_index(cs, Embedder()).to_dict())
