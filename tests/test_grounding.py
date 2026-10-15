"""Grounding verifier tests."""

from __future__ import annotations

from localrag.chunking import Chunk
from localrag.grounding import verify


def _chunk(text: str, doc_id: str = "MANUAL-CNC-07", section: str = "§4") -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}#1",
        doc_id=doc_id,
        section=section,
        section_title="Torque Specs",
        text=text,
        revision="A",
        effective_date="2025-01-01",
    )


def test_fabricated_citation_is_flagged():
    report = verify("See INC-2026-999 §2 for the value 45 N·m.", [_chunk("Clamp bolt 45 N·m")])
    assert "INC-2026-999" in report.ungrounded_citations
    assert report.grounded is False


def test_fabricated_number_is_flagged():
    report = verify(
        "The clamp torque is 99 N·m (MANUAL-CNC-07 §4).", [_chunk("Clamp bolt 45 N·m")]
    )
    assert "99" in report.ungrounded_numbers
    assert report.grounded is False


def test_grounded_answer_passes():
    report = verify(
        "The clamp torque is 45 N·m (MANUAL-CNC-07 §4).", [_chunk("Clamp bolt 45 N·m")]
    )
    assert report.grounded is True
    assert report.cited_doc_ids == ["MANUAL-CNC-07"]


def test_reports_retrieved_doc_ids():
    report = verify("nothing numeric here", [_chunk("text", doc_id="SPEC-CNC-07")])
    assert report.retrieved_doc_ids == ["SPEC-CNC-07"]
