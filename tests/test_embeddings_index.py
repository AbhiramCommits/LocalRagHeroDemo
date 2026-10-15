"""Embeddings and vector-index tests (offline hashing embedder)."""

from __future__ import annotations

import numpy as np
import pytest

from localrag.embeddings import HASH_DIM, Embedder, resolve_device
from localrag.index import VectorIndex, build_index, faiss_available


def test_hashing_embedder_is_deterministic(tmp_path):
    e1 = Embedder("m", cache_dir=tmp_path / "a", allow_download=False)
    e2 = Embedder("m", cache_dir=tmp_path / "b", allow_download=False)
    assert np.allclose(e1.encode(["hello world"]), e2.encode(["hello world"]))


def test_embedding_backend_is_hashing(embedder):
    assert embedder.backend == "hashing_fallback"
    assert embedder.dim == HASH_DIM


def test_embeddings_are_l2_normalised(embedder):
    vecs = embedder.encode(["alpha beta gamma"])
    assert abs(float(np.linalg.norm(vecs[0])) - 1.0) < 1e-5


def test_encoding_shape(embedder):
    vecs = embedder.encode(["one", "two", "three"])
    assert vecs.shape == (3, embedder.dim)


def test_cache_roundtrip(tmp_path):
    e1 = Embedder("m", cache_dir=tmp_path, allow_download=False)
    e1.encode(["cached text"])
    assert (tmp_path / "embeddings.npz").exists()
    e2 = Embedder("m", cache_dir=tmp_path, allow_download=False)
    assert len(e2._cache) >= 1


def test_resolve_device_preference():
    assert resolve_device("cpu") == "cpu"


def test_index_search_is_sorted(real_chunks, embedder):
    vectors = embedder.encode([c.text for c in real_chunks])
    index = VectorIndex(dim=embedder.dim, backend="numpy")
    index.build(vectors, real_chunks)
    scores = [s for _, s in index.search(vectors[0], k=5)]
    assert scores == sorted(scores, reverse=True)


def test_index_persist_and_load(tmp_path, real_chunks, embedder):
    build_index(real_chunks[:20], embedder, tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert len(loaded.chunks) == 20
    assert loaded.dim == embedder.dim
    assert (tmp_path / "meta.json").exists()


def test_index_meta_fields(tmp_path, real_chunks, embedder):
    meta = build_index(real_chunks[:10], embedder, tmp_path)
    assert meta.chunk_count == 10
    assert meta.embedding_backend == "hashing_fallback"
    assert meta.embedding_model == "offline"


def test_numpy_fallback_matches_faiss_ordering(real_chunks, embedder):
    if not faiss_available():
        pytest.skip("faiss not installed")
    vectors = embedder.encode([c.text for c in real_chunks[:60]])
    np_index = VectorIndex(dim=embedder.dim, backend="numpy")
    np_index.build(vectors, real_chunks[:60])
    np_index._faiss = None  # force the numpy path

    faiss_index = VectorIndex(dim=embedder.dim, backend="numpy")
    faiss_index.build(vectors, real_chunks[:60])

    query = vectors[7]
    np_order = [i for i, _ in np_index.search(query, 5)]
    faiss_order = [i for i, _ in faiss_index.search(query, 5)]
    assert np_order == faiss_order
