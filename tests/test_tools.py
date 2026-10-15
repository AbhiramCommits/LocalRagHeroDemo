"""Tool tests: get_spec_value, check_superseded, lookup_asset, search_docs."""

from __future__ import annotations


def test_get_spec_value_parses_unit(tools):
    obs, hits, cites = tools.get_spec_value("CNC-12", "turret clamp bolts torque")
    assert "38" in obs
    assert "N·m" in obs
    assert hits and "MANUAL-CNC-12" in cites[0]


def test_get_spec_value_unknown_spec_returns_empty(tools):
    obs, hits, cites = tools.get_spec_value("CNC-12", "nonexistent widget specification")
    assert hits == []
    assert "no numeric spec" in obs


def test_check_superseded_true(tools):
    obs, hits, cites = tools.check_superseded("MANUAL-CNC-07")
    assert "superseded" in obs.lower()
    assert hits and "INC-2026-001" in cites[0]


def test_check_superseded_false(tools):
    obs, hits, cites = tools.check_superseded("MANUAL-ASM-01")
    assert "not superseded" in obs.lower()
    assert hits == []


def test_check_superseded_truth_table(tools):
    superseded = {"MANUAL-CNC-07", "MANUAL-PRS-02", "MANUAL-LASER-02",
                  "MANUAL-WELD-06", "MANUAL-CONV-05", "QC-WI-005"}
    for doc_id in superseded:
        _, hits, _ = tools.check_superseded(doc_id)
        assert hits, f"{doc_id} should be superseded"
    for doc_id in ("MANUAL-ASM-01", "MANUAL-ROB-03"):
        _, hits, _ = tools.check_superseded(doc_id)
        assert not hits, f"{doc_id} should not be superseded"


def test_lookup_asset_returns_inventory(tools):
    obs, hits, cites = tools.lookup_asset("PRS-02")
    assert any("MANUAL-PRS-02" in c for c in cites)
    assert hits


def test_lookup_asset_unknown_returns_empty(tools):
    obs, hits, cites = tools.lookup_asset("ZZZ-99")
    assert hits == []


def test_search_docs_returns_citations(tools):
    obs, hits, cites = tools.search_docs("coolant concentration", k=3)
    assert len(hits) == 3
    assert len(cites) == 3


def test_langchain_tools_exposed(tools):
    names = {t.name for t in tools.langchain_tools()}
    assert names == {"search_docs", "lookup_asset", "get_spec_value", "check_superseded"}
