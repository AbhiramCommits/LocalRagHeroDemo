"""CLI tests via Typer's CliRunner (offline; no model download)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from localrag.cli import app

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "localrag" in result.stdout


def test_cli_doctor_exit_zero():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "host profile" in result.stdout


def test_cli_corpus_build(tmp_path):
    result = runner.invoke(
        app,
        ["corpus", "build", "--out", str(tmp_path / "corpus"), "--gold", str(tmp_path / "q.jsonl")],
    )
    assert result.exit_code == 0
    assert len(list((tmp_path / "corpus").glob("*.md"))) >= 40
    assert len((tmp_path / "q.jsonl").read_text().strip().splitlines()) == 35


def test_cli_search(built_index_dir, monkeypatch):
    monkeypatch.setenv("LOCALRAG_ALLOW_DOWNLOAD", "false")
    result = runner.invoke(app, ["search", "coolant concentration", "--index", str(built_index_dir), "--k", "3"])
    assert result.exit_code == 0


def test_cli_ask_extractive(built_index_dir, monkeypatch):
    monkeypatch.setenv("LOCALRAG_ALLOW_DOWNLOAD", "false")
    result = runner.invoke(
        app,
        ["ask", "What does fault code E-318 indicate on the PRS-02 press?",
         "--backend", "extractive", "--index", str(built_index_dir)],
    )
    assert result.exit_code == 0
    assert "MANUAL-PRS-02" in result.stdout


def test_cli_index_info(built_index_dir):
    result = runner.invoke(app, ["index", "info", "--index", str(built_index_dir)])
    assert result.exit_code == 0
    assert "chunk_count" in result.stdout


def test_cli_bench_table(tmp_path):
    report = {
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "workload": {"n_prompts": 12, "repetitions": 3, "warmup_discarded": 1, "max_new_tokens": 16},
        "host": {"cpu": "test", "os": "test", "wsl": False, "torch": {}, "nvidia_smi": None},
        "index": {"chunk_count": 1, "build_seconds": 0.0, "embedding_model": "m",
                  "embedding_backend": "hashing_fallback", "device": "cpu", "ann_backend": "numpy"},
        "configs": {
            "extractive_cpu": {"status": "not_measured_on_this_host", "reason": "test"},
        },
    }
    src = tmp_path / "b.json"
    src.write_text(json.dumps(report), encoding="utf-8")
    result = runner.invoke(app, ["bench", "table", "--in", str(src), "--out", str(tmp_path / "b.md")])
    assert result.exit_code == 0
    assert (tmp_path / "b.md").exists()


def test_cli_backends_list():
    result = runner.invoke(app, ["backends", "list"])
    assert result.exit_code == 0
    assert "extractive" in result.stdout
