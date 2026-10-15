"""Text embeddings with an on-disk cache and an offline hashing fallback.

Primary backend is ``sentence-transformers`` (``all-MiniLM-L6-v2`` by default) on
``cuda`` when a GPU is available, otherwise CPU. If the model cannot be loaded
(for example there is no network to download the weights), we fall back to a
deterministic, dependency-free hashed bag-of-words embedder so the rest of the
pipeline — and the CI — keeps working. The active backend is recorded as
``embedding_backend`` in the index metadata.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def resolve_device(preference: str = "auto") -> str:
    """Resolve ``auto`` to ``cuda`` when available, otherwise ``cpu``."""
    if preference and preference != "auto":
        return preference
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _hash_token(token: str) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)


class Embedder:
    """Encode text to L2-normalised vectors, with caching and a fallback."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "auto",
        cache_dir: str | Path = ".cache",
        allow_download: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.allow_download = allow_download
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "embeddings.npz"
        self.backend = "uninitialized"
        self.dim = 0
        self.load_error: str | None = None
        self._model = None
        self._cache: dict[str, np.ndarray] = self._load_cache()
        self._init_model()

    # ------------------------------------------------------------------ #
    def _init_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            get_dim = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension"
            )
            self.dim = int(get_dim() or 0)
            self.backend = "sentence_transformers"
        except Exception as exc:  # pragma: no cover - depends on network/host
            self._model = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.dim = HASH_DIM
            self.backend = "hashing_fallback"

    # ------------------------------------------------------------------ #
    def _load_cache(self) -> dict[str, np.ndarray]:
        if not self.cache_path.exists():
            return {}
        try:
            with np.load(self.cache_path, allow_pickle=False) as data:
                keys = [str(k) for k in data["keys"]]
                vecs = data["vectors"]
            return {k: vecs[i] for i, k in enumerate(keys)}
        except Exception:
            return {}

    def _save_cache(self) -> None:
        if not self._cache:
            return
        keys = list(self._cache.keys())
        vectors = np.vstack([self._cache[k] for k in keys]).astype(np.float32)
        np.savez(self.cache_path, keys=np.array(keys), vectors=vectors)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model_name}|{text}".encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ #
    def _embed_hashing(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), HASH_DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                out[i, _hash_token(token) % HASH_DIM] += 1.0
        return out

    def _embed_model(self, texts: list[str], batch_size: int) -> np.ndarray:
        vecs = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Encode ``texts`` to a ``(len(texts), dim)`` float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        keys = [self._key(t) for t in texts]
        result: list[np.ndarray | None] = [None] * len(texts)
        missing_idx: list[int] = []
        for i, key in enumerate(keys):
            if use_cache and key in self._cache:
                result[i] = self._cache[key]
            else:
                missing_idx.append(i)

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            if self.backend == "sentence_transformers" and self._model is not None:
                fresh = self._embed_model(missing_texts, batch_size)
            else:
                fresh = self._embed_hashing(missing_texts)
            fresh = np.asarray(fresh, dtype=np.float32)
            if normalize:
                fresh = _l2_normalize(fresh)
            for pos, i in enumerate(missing_idx):
                result[i] = fresh[pos]
                if use_cache:
                    self._cache[keys[i]] = fresh[pos].astype(np.float32)
            if use_cache:
                self._save_cache()

        matrix = np.vstack([r for r in result if r is not None]).astype(np.float32)
        if normalize:
            matrix = _l2_normalize(matrix)
        return matrix

    # ------------------------------------------------------------------ #
    def signature(self) -> dict[str, object]:
        return {
            "embedding_model": self.model_name,
            "embedding_backend": self.backend,
            "device": self.device,
            "dim": self.dim,
            "load_error": self.load_error,
        }


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


if __name__ == "__main__":  # pragma: no cover
    emb = Embedder()
    print(emb.signature())
