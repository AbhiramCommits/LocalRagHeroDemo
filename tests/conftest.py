"""Shared pytest fixtures.

All fixtures are CPU-only and offline: they use the deterministic hashing
embedder (``allow_download=False``) so the suite never touches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localrag.chunking import build_chunks
from localrag.config import load_settings
from localrag.embeddings import Embedder
from localrag.index import VectorIndex

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def settings():
    return load_settings(allow_download=False)


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    return ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def real_chunks(corpus_dir):
    return build_chunks(corpus_dir)


@pytest.fixture(scope="session")
def embedder(tmp_path_factory):
    cache = tmp_path_factory.mktemp("embed_cache")
    return Embedder(model_name="offline", device="cpu", cache_dir=cache, allow_download=False)


@pytest.fixture(scope="session")
def retriever(real_chunks, embedder, settings):
    from localrag.retrieve import HybridRetriever

    vectors = embedder.encode([c.text for c in real_chunks])
    index = VectorIndex(dim=embedder.dim, backend="numpy")
    index.build(vectors, real_chunks)
    return HybridRetriever(index, embedder, settings)


@pytest.fixture(scope="session")
def tools(retriever, settings):
    from localrag.agent import ShopFloorTools

    return ShopFloorTools(retriever, settings)


@pytest.fixture(scope="session")
def agent(retriever, settings):
    from localrag.agent import Agent
    from localrag.backends import ExtractiveBackend

    return Agent(retriever, ExtractiveBackend(settings), settings)


@pytest.fixture(scope="session")
def built_index_dir(tmp_path_factory, real_chunks, embedder):
    """A persisted index built with the offline embedder (for CLI tests)."""
    from localrag.index import build_index

    out = tmp_path_factory.mktemp("index")
    build_index(real_chunks, embedder, out)
    return out


@pytest.fixture
def tiny_corpus(tmp_path: Path) -> Path:
    """A tiny, self-contained corpus with one supersede case."""
    manual = tmp_path / "MANUAL-FIX-01.md"
    manual.write_text(
        "---\n"
        'doc_id: "MANUAL-FIX-01"\n'
        'title: "Fixture manual"\n'
        'asset: "FIX-01"\n'
        'doc_type: "manual"\n'
        'revision: "A"\n'
        'effective_date: "2025-01-01"\n'
        "supersedes: null\n"
        "---\n"
        "# Fixture manual\n\n"
        "## §1 Safety\n- Lock out the 480 V disconnect.\n\n"
        "## §4 Torque Specs\n\n"
        "| Fastener | Value | Unit |\n| --- | --- | --- |\n"
        "| Clamp bolt | 10 | N·m |\n| Drawbar force | 5 | kN |\n",
        encoding="utf-8",
    )
    incident = tmp_path / "INC-2099-001.md"
    incident.write_text(
        "---\n"
        'doc_id: "INC-2099-001"\n'
        'title: "Fixture supersede"\n'
        'asset: "FIX-01"\n'
        'doc_type: "incident"\n'
        'revision: "A"\n'
        'effective_date: "2099-01-01"\n'
        'supersedes: "MANUAL-FIX-01"\n'
        'supersedes_section: "§4"\n'
        "---\n"
        "# Fixture supersede\n\n"
        "## §2 Change Record\n\n"
        "| Item | Previous value | New value |\n| --- | --- | --- |\n"
        "| Clamp bolt | 10 N·m | 14 N·m |\n",
        encoding="utf-8",
    )
    return tmp_path
