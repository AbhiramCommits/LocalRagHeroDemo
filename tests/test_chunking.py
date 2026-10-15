"""Section-aware chunker tests."""

from __future__ import annotations

from localrag.chunking import build_chunks, chunk_document, parse_document


def test_front_matter_parsing(tiny_corpus):
    meta, body = parse_document(tiny_corpus / "MANUAL-FIX-01.md")
    assert meta["doc_id"] == "MANUAL-FIX-01"
    assert meta["revision"] == "A"
    assert body.lstrip().startswith("# Fixture manual")


def test_heading_split_creates_sections(tiny_corpus):
    sections = {(c.doc_id, c.section) for c in build_chunks(tiny_corpus)}
    assert ("MANUAL-FIX-01", "§1") in sections
    assert ("MANUAL-FIX-01", "§4") in sections
    assert ("INC-2099-001", "§2") in sections


def test_section_code_and_title(tiny_corpus):
    chunks = build_chunks(tiny_corpus)
    sec4 = next(c for c in chunks if c.doc_id == "MANUAL-FIX-01" and c.section == "§4")
    assert sec4.section_title == "Torque Specs"


def test_table_preserved_intact(tiny_corpus):
    chunks = build_chunks(tiny_corpus)
    sec4 = next(c for c in chunks if c.doc_id == "MANUAL-FIX-01" and c.section == "§4")
    assert "| Clamp bolt | 10 | N·m |" in sec4.text
    assert "| --- | --- | --- |" in sec4.text


def test_metadata_carried_through(tiny_corpus):
    chunk = next(c for c in build_chunks(tiny_corpus) if c.doc_id == "MANUAL-FIX-01")
    assert chunk.revision == "A"
    assert chunk.effective_date == "2025-01-01"
    assert chunk.asset == "FIX-01"
    assert chunk.doc_type == "manual"


def test_supersede_metadata(tiny_corpus):
    chunk = next(c for c in build_chunks(tiny_corpus) if c.doc_id == "INC-2099-001")
    assert chunk.supersedes == "MANUAL-FIX-01"
    assert chunk.supersedes_section == "§4"


def test_citation_format(tiny_corpus):
    chunk = next(c for c in build_chunks(tiny_corpus) if c.section == "§4")
    assert chunk.citation() == "MANUAL-FIX-01 §4 (rev A, 2025-01-01)"


def test_preamble_chunk(tiny_corpus):
    chunks = [c for c in build_chunks(tiny_corpus) if c.doc_id == "MANUAL-FIX-01"]
    assert any(c.section == "" for c in chunks)


def test_long_section_is_split():
    body = "## §X Big\n\n" + "\n\n".join(f"Paragraph {i} " + "x" * 200 for i in range(10))
    chunks = chunk_document(
        {"doc_id": "D", "revision": "A", "effective_date": "2025-01-01"},
        body,
        chunk_size=300,
        chunk_overlap=50,
    )
    assert len(chunks) > 1


def test_char_span_recorded():
    body = "lead in\n\n## §1 One\nhello\n\n## §2 Two\nworld\n"
    chunks = chunk_document({"doc_id": "D"}, body)
    assert all(isinstance(c.char_span, list) and len(c.char_span) == 2 for c in chunks)
