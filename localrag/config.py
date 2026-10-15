"""Configuration for the LocalRAG Hero Demo.

`Settings` is a plain dataclass populated from (in priority order):

  1. Explicit keyword arguments
  2. ``LOCALRAG_*`` environment variables
  3. ``config/default.yaml``
  4. Field defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

# Repository root = the parent of the ``localrag`` package directory.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


@dataclass
class Settings:
    """Runtime settings for the whole pipeline."""

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"

    chunk_size: int = 900
    chunk_overlap: int = 120

    top_k: int = 5
    rrf_k: int = 60
    recency_weight: float = 0.15

    backend: str = "auto"
    endpoint: str = ""
    device: str = "auto"

    max_tool_calls: int = 4
    max_new_tokens: int = 320
    temperature: float = 0.0
    seed: int = 0

    corpus_dir: str = "data/corpus"
    gold_path: str = "data/gold/questions.jsonl"
    index_dir: str = "artifacts/index"
    cache_dir: str = ".cache"
    results_dir: str = "results"

    api_key: str = ""
    allow_download: bool = True

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, config_path: str | Path | None = None, **overrides: object) -> Settings:
        """Build settings from YAML + environment + explicit overrides."""
        raw: dict[str, object] = {}
        path = Path(config_path) if config_path else DEFAULT_CONFIG
        if path.exists():
            raw.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

        known = {f.name for f in fields(cls)}
        env_raw: dict[str, object] = {}
        for name in known:
            env_key = f"LOCALRAG_{name.upper()}"
            if env_key in os.environ:
                env_raw[name] = os.environ[env_key]
        # A couple of conventional aliases.
        if "LOCALRAG_ENDPOINT" in os.environ:
            env_raw["endpoint"] = os.environ["LOCALRAG_ENDPOINT"]
        if "LOCALRAG_API_KEY" in os.environ:
            env_raw["api_key"] = os.environ["LOCALRAG_API_KEY"]

        merged = {**raw, **env_raw, **overrides}
        # Coerce types based on the dataclass defaults.
        typed: dict[str, object] = {}
        for f in fields(cls):
            if f.name not in merged:
                continue
            value = merged[f.name]
            default = f.default
            if isinstance(default, bool):
                value = (
                    value
                    if isinstance(value, bool)
                    else str(value).strip().lower() in {"1", "true", "yes", "on"}
                )
            elif isinstance(default, int) and not isinstance(default, bool):
                value = int(value)  # type: ignore[arg-type]
            elif isinstance(default, float):
                value = float(value)  # type: ignore[arg-type]
            typed[f.name] = value
        return cls(**typed)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    def path(self, attr: str) -> Path:
        """Resolve a path-like attribute against the repository root."""
        value = getattr(self, attr)
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    def as_dict(self) -> dict[str, object]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_settings(**overrides: object) -> Settings:
    """Convenience wrapper around :meth:`Settings.load`."""
    return Settings.load(**overrides)
