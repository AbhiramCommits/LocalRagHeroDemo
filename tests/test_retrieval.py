"""Hybrid retrieval tests."""

from __future__ import annotations

import pytest

from localrag.retrieve import CONFIGS


def test_configs_available():
    for name in ("dense_only", "bm25_only", "hybrid_rrf", "hybrid_rrf_recency"):
        assert name in CONFIGS


def test_known_answer_hits_right_doc(retriever):
    hits = retriever.search("DRLL-08 collet nut torque", k=5)
    assert "MANUAL-DRLL-08" in [h.chunk.doc_id for h in hits]


def test_fault_code_keyword_hit(retriever):
    docs = [h.chunk.doc_id for h in retriever.search("E-214 spindle over-temperature", k=5,
                                                     config="bm25_only")]
    assert "MANUAL-CNC-07" in docs


def test_recency_boost_prefers_newer_revision(retriever):
    hits = retriever.search("CNC-07 spindle clamp bolts torque", k=10,
                            config="hybrid_rrf_recency")
    docs = [h.chunk.doc_id for h in hits]
    assert "INC-2026-001" in docs
    assert docs.index("INC-2026-001") < (docs.index("MANUAL-CNC-07") if "MANUAL-CNC-07" in docs else 99)


def test_supersede_penalty_is_section_targeted(retriever):
    sec4 = next(
        i for i, c in enumerate(retriever.chunks)
        if c.doc_id == "MANUAL-CNC-07" and c.section == "§4"
    )
    sec1 = next(
        i for i, c in enumerate(retriever.chunks)
        if c.doc_id == "MANUAL-CNC-07" and c.section == "§1"
    )
    assert retriever._boost(sec4)[0] is True
    assert retriever._boost(sec1)[0] is False


def test_bm25_only_ranking_is_populated(retriever):
    hits = retriever.search("coolant concentration refractometer", k=3, config="bm25_only")
    assert len(hits) == 3
    assert hits[0].bm25_rank == 1


def test_invalid_config_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("anything", config="does_not_exist")


def test_hits_carry_citations(retriever):
    for hit in retriever.search("coolant concentration", k=3):
        assert "(" in hit.citation and "rev" in hit.citation
