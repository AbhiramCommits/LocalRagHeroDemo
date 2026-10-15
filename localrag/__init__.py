"""LocalRAG Hero Demo — a fully local, GPU-accelerated agentic RAG system."""

from __future__ import annotations

import os as _os

# --------------------------------------------------------------------------- #
# Guard against the duplicate-OpenMP runtime clash between faiss-cpu and torch
# (libomp), which on macOS/arm64 can cause an intermittent segfault when a FAISS
# search runs after a Transformers model has loaded. Setting these before either
# library is imported makes the process deterministic. Users can override
# OMP_NUM_THREADS for benchmarking by exporting it themselves.
# --------------------------------------------------------------------------- #
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.1.0"
__all__ = ["__version__"]
