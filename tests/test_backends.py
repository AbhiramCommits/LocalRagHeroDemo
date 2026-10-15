"""Backend factory and extractive backend tests."""

from __future__ import annotations

import pytest

from localrag.backends import ExtractiveBackend, backend_status, get_backend


def test_extractive_refuses_without_context(settings):
    backend = ExtractiveBackend(settings)
    gen = backend.generate("### QUESTION\nWhat is the torque?\n### INSTRUCTIONS\nanswer\n")
    assert "INSUFFICIENT_EVIDENCE" in gen.text


def test_extractive_answers_from_context(settings):
    backend = ExtractiveBackend(settings)
    prompt = (
        "### CONTEXT\n"
        "[1] MANUAL-CNC-07 §4 (rev C, 2025-06-15)\n"
        "| Clamp bolt | 45 | N·m |\n\n"
        "### QUESTION\nWhat is the clamp bolt torque?\n"
        "### INSTRUCTIONS\nanswer\n"
    )
    gen = backend.generate(prompt)
    assert "45" in gen.text


def test_extractive_reports_stats(settings):
    backend = ExtractiveBackend(settings)
    gen = backend.generate("### QUESTION\nhello\n")
    stats = gen.stats.as_dict()
    assert stats["model"] == "extractive-rule-based"
    assert stats["total_s"] >= 0


def test_backend_status_lists_all(settings):
    names = {row["backend"] for row in backend_status(settings)}
    assert names == {"transformers", "openai_compat", "extractive"}


def test_get_backend_extractive(settings):
    assert get_backend("extractive", settings).name == "extractive"


def test_get_backend_unknown_raises(settings):
    with pytest.raises(ValueError):
        get_backend("does_not_exist", settings)
