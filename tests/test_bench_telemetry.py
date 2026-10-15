"""Benchmark harness and telemetry tests (must degrade gracefully, never raise)."""

from __future__ import annotations

from localrag.telemetry import GpuMonitor, host_profile


def test_gpu_monitor_absent_degrades_without_raising():
    with GpuMonitor() as monitor:
        pass
    summary = monitor.summary()
    assert summary.status in ("measured", "not_measured_on_this_host")
    assert isinstance(summary.as_dict(), dict)


def test_host_profile_reports_real_fields():
    profile = host_profile()
    assert profile.python_version
    assert isinstance(profile.is_wsl, bool)
    assert isinstance(profile.gpus, list)


def test_bench_runs_extractive_config(tmp_path, settings):
    from localrag.bench import run_bench

    report = run_bench(
        settings,
        out=tmp_path / "bench.json",
        configs=("extractive_cpu",),
        repetitions=1,
        warmup=0,
        max_tokens=16,
    )
    result = report["configs"]["extractive_cpu"]
    assert result["status"] == "measured"
    assert result["n_prompts"] == 12
    assert result["tokens_per_s"]["mean"] is not None
    assert (tmp_path / "bench.json").exists()


def test_bench_skips_unsupported_configs(tmp_path, settings):
    from localrag.bench import run_bench

    report = run_bench(
        settings,
        out=tmp_path / "bench.json",
        configs=("hf_cuda_fp16", "hf_cuda_fp16_batch4", "endpoint_vllm_or_nim"),
        repetitions=1,
        warmup=0,
        max_tokens=16,
    )
    for config in ("hf_cuda_fp16", "hf_cuda_fp16_batch4", "endpoint_vllm_or_nim"):
        assert report["configs"][config]["status"] == "not_measured_on_this_host"
        assert report["configs"][config]["reason"]


def test_bench_markdown_renders_skips(tmp_path, settings):
    from localrag.bench import run_bench, write_bench_markdown

    report = run_bench(
        settings, out=tmp_path / "b.json", configs=("extractive_cpu", "hf_cuda_fp16"),
        repetitions=1, warmup=0, max_tokens=16,
    )
    path = write_bench_markdown(report, tmp_path / "b.md")
    text = path.read_text(encoding="utf-8")
    assert "extractive_cpu" in text
    assert "not_measured_on_this_host" in text
