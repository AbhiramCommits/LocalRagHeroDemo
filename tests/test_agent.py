"""Agent loop tests (extractive backend, offline)."""

from __future__ import annotations


def test_agent_respects_max_tool_calls(agent, settings):
    result = agent.run("What is the current torque spec for the CNC-07 spindle clamp bolts?")
    assert len(result.tool_calls) <= settings.max_tool_calls


def test_agent_refusal_returns_sentinel(agent):
    result = agent.run("What is the serial number of the CNC-07 spindle motor?")
    assert result.refused is True
    assert result.answer == "INSUFFICIENT_EVIDENCE"
    assert result.citations == []


def test_agent_transcript_records_every_tool_call(agent):
    result = agent.run("What is the current torque spec for the CNC-07 spindle clamp bolts?")
    tools_used = {call.tool for call in result.tool_calls}
    assert {"search_docs", "check_superseded"} <= tools_used
    assert any(step.get("tool") for step in result.transcript)


def test_agent_supersede_answer_is_cited(agent):
    result = agent.run("What is the current torque spec for the CNC-07 spindle clamp bolts?")
    assert "52" in result.answer
    assert any("INC-2026-001" in c for c in result.citations)


def test_agent_grounding_reported(agent):
    result = agent.run("What is the current torque spec for the CNC-07 spindle clamp bolts?")
    assert result.grounding.get("grounded") is True


def test_agent_max_tool_calls_is_four(agent, settings):
    assert settings.max_tool_calls == 4


def test_naive_rag_uses_no_tools(retriever, settings):
    from localrag.agent import NaiveRAG
    from localrag.backends import ExtractiveBackend

    naive = NaiveRAG(retriever, ExtractiveBackend(settings), settings)
    result = naive.run("What does fault code E-318 indicate on the PRS-02 press?")
    assert result.mode == "naive"
    assert result.tool_calls == []
