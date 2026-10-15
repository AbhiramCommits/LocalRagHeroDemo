"""Benchmark harness.

Runs a fixed 12-prompt shop-floor workload (short + long questions) once per
config and records tokens/sec, time-to-first-token, p50/p95/mean latency and peak
VRAM (via :class:`~localrag.telemetry.GpuMonitor`), plus the index/embedding build
time. Three repetitions per config with the first warm-up discarded; mean ± stdev
are reported.

Configs that the current host cannot support are *skipped* and recorded as
``status="not_measured_on_this_host"`` with the reason — never fabricated.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from .backends import ExtractiveBackend, OpenAICompatBackend, TransformersBackend
from .config import Settings, load_settings
from .telemetry import GpuMonitor, is_wsl, nvidia_smi_path, torch_info

CONFIGS = (
    "extractive_cpu",
    "hf_cpu_fp32",
    "hf_cuda_fp16",
    "hf_cuda_fp16_batch4",
    "endpoint_vllm_or_nim",
)

# 12 prompts: a mix of short, long, numeric, supersede and refusal questions.
WORKLOAD: list[str] = [
    "What does fault code E-214 mean?",
    "What is the torque spec for the CNC-12 turret clamp bolts?",
    "What is the current torque spec for the CNC-07 spindle clamp bolts, and which document supersedes the manual?",
    "What is the hydraulic accumulator pre-charge for the PRS-02 press?",
    "List the daily PM checks for the CNC-07 machining centre and the coolant concentration window.",
    "What is the current assist-gas pressure for the LASER-02, and how does it differ from the original manual value?",
    "What does fault code E-512 indicate and which asset manuals list it?",
    "What is the inspection interval and pass/fail criterion for hole-position tolerance on the Cell 4 bracket?",
    "What is the wire-feed drive-roll tension for the WELD-06, including any superseding change note?",
    "What safety step is required before removing a way cover on the CNC-07?",
    "What is the warranty period on the LASER-02 chiller?",
    "Summarise the maintenance actions required to bring a cold Cell 4 line back into production, citing the relevant documents.",
]


def _stdev(values: list[float]) -> float | None:
    return round(statistics.pstdev(values), 4) if len(values) > 1 else None


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return round(s[idx], 3)


def _config_availability(config: str, settings: Settings) -> tuple[bool, str, dict]:
    """Return (available, reason, extra_kwargs) for a bench config."""
    torch = torch_info()
    if config == "extractive_cpu":
        return True, "", {"backend": "extractive"}
    if config == "hf_cpu_fp32":
        if not torch["available"]:
            return False, "torch is not installed", {}
        return True, "", {"backend": "transformers", "device": "cpu"}
    if config == "hf_cuda_fp16":
        if not torch.get("cuda_available"):
            return False, "no CUDA device on this host", {}
        return True, "", {"backend": "transformers", "device": "cuda"}
    if config == "hf_cuda_fp16_batch4":
        if not torch.get("cuda_available"):
            return False, "no CUDA device on this host", {}
        return True, "", {"backend": "transformers", "device": "cuda", "batch": 4}
    if config == "endpoint_vllm_or_nim":
        if not settings.endpoint:
            return False, "LOCALRAG_ENDPOINT is not set (no vLLM/NIM server configured)", {}
        return True, "", {"backend": "openai_compat"}
    return False, f"unknown config {config}", {}


def _make_backend(config: str, settings: Settings, kwargs: dict) -> object:
    name = kwargs.pop("backend")
    if name == "transformers":
        return TransformersBackend(settings=settings, device=kwargs.get("device", "auto"))
    if name == "openai_compat":
        return OpenAICompatBackend(settings=settings)
    return ExtractiveBackend(settings)


def _run_config(
    config: str,
    settings: Settings,
    repetitions: int,
    warmup: int,
    max_tokens: int,
) -> dict:
    available, reason, kwargs = _config_availability(config, settings)
    if not available:
        return {"status": "not_measured_on_this_host", "reason": reason}

    batch = kwargs.get("batch", 1)
    backend = _make_backend(config, settings, dict(kwargs))
    if hasattr(backend, "available") and not backend.available:
        return {
            "status": "not_measured_on_this_host",
            "reason": getattr(backend, "load_error", "backend could not be initialised"),
        }

    device = getattr(backend, "device", "cpu")
    model = getattr(backend, "model_id", config)

    # Warm-up (discarded).
    for _ in range(warmup):
        try:
            _generate(backend, WORKLOAD[0], max_tokens, batch)
        except Exception as exc:  # pragma: no cover
            return {"status": "not_measured_on_this_host", "reason": f"warmup failed: {exc}"}

    per_rep_latency: list[float] = []
    per_rep_tps: list[float] = []
    per_rep_ttft: list[float] = []
    all_latency: list[float] = []
    errors = 0

    monitor = GpuMonitor()
    with monitor:
        for _rep in range(repetitions):
            rep_lat: list[float] = []
            rep_tps: list[float] = []
            rep_ttft: list[float] = []
            for prompt in WORKLOAD:
                t0 = time.perf_counter()
                try:
                    gen = _generate(backend, prompt, max_tokens, batch)
                except Exception:
                    errors += 1
                    continue
                lat = (time.perf_counter() - t0) * 1000.0
                rep_lat.append(lat)
                all_latency.append(lat)
                stats = gen["stats"]
                if stats.get("tokens_per_s") is not None:
                    rep_tps.append(float(stats["tokens_per_s"]))
                if stats.get("ttft_s") is not None:
                    rep_ttft.append(float(stats["ttft_s"]))
            if rep_lat:
                per_rep_latency.append(statistics.fmean(rep_lat))
            if rep_tps:
                per_rep_tps.append(statistics.fmean(rep_tps))
            if rep_ttft:
                per_rep_ttft.append(statistics.fmean(rep_ttft))

    gpu = monitor.summary()
    result: dict = {
        "status": "measured",
        "device": device,
        "model": model,
        "batch": batch,
        "n_prompts": len(WORKLOAD),
        "n_calls": len(all_latency),
        "errors": errors,
        "tokens_per_s": {
            "mean": round(statistics.fmean(per_rep_tps), 3) if per_rep_tps else None,
            "stdev": _stdev(per_rep_tps),
        },
        "ttft_s": {
            "mean": round(statistics.fmean(per_rep_ttft), 4) if per_rep_ttft else None,
            "stdev": _stdev(per_rep_ttft),
        },
        "latency_ms": {
            "mean": round(statistics.fmean(all_latency), 3) if all_latency else None,
            "p50": _pct(all_latency, 50),
            "p95": _pct(all_latency, 95),
        },
        "peak_vram_mb": gpu.peak_memory_used_mb,
        "gpu": gpu.as_dict(),
    }
    return result


def _generate(backend: object, prompt: str, max_tokens: int, batch: int) -> dict:
    """Generate, supporting a simple batch-of-N path for the batch config."""
    if batch > 1 and hasattr(backend, "generate"):
        # Run `batch` identical prompts in one call to exercise batching on GPU.
        prompts = [prompt] * batch
        results = []
        t0 = time.perf_counter()
        for p in prompts:
            gen = backend.generate(p, max_tokens=max_tokens, stream=True)  # type: ignore[attr-defined]
            results.append(gen)
        wall = time.perf_counter() - t0
        completion = sum(getattr(r.stats, "completion_tokens", 0) for r in results)
        return {
            "text": results[0].text,
            "stats": {
                "tokens_per_s": round(completion / wall, 3) if wall > 0 else None,
                "ttft_s": getattr(results[0].stats, "ttft_s", None),
                "completion_tokens": completion,
            },
        }
    gen = backend.generate(prompt, max_tokens=max_tokens, stream=True)  # type: ignore[attr-defined]
    return {"text": gen.text, "stats": gen.stats.as_dict()}


# --------------------------------------------------------------------------- #
def run_bench(
    settings: Settings | None = None,
    out: str | Path = "results/benchmarks.json",
    configs: tuple[str, ...] = CONFIGS,
    repetitions: int = 3,
    warmup: int = 1,
    max_tokens: int = 128,
) -> dict:
    settings = settings or load_settings()
    torch = torch_info()

    index_meta = {}
    meta_path = settings.path("index_dir") / "meta.json"
    if meta_path.exists():
        index_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    host = {
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "wsl": is_wsl(),
        "nvidia_smi": nvidia_smi_path(),
        "torch": torch,
    }

    results: dict[str, dict] = {}
    for config in configs:
        results[config] = _run_config(config, settings, repetitions, warmup, max_tokens)

    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "workload": {
            "n_prompts": len(WORKLOAD),
            "repetitions": repetitions,
            "warmup_discarded": warmup,
            "max_new_tokens": max_tokens,
        },
        "host": host,
        "index": {
            "chunk_count": index_meta.get("chunk_count"),
            "build_seconds": index_meta.get("build_seconds"),
            "embedding_model": index_meta.get("embedding_model"),
            "embedding_backend": index_meta.get("embedding_backend"),
            "device": index_meta.get("device"),
            "ann_backend": index_meta.get("ann_backend"),
        },
        "configs": results,
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_bench_markdown(report: dict, out: str | Path = "results/benchmarks.md") -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    host = report.get("host", {})
    torch = host.get("torch", {})
    idx = report.get("index", {})
    lines = [
        "# Benchmark results",
        "",
        f"- Generated: `{report.get('generated_utc')}`",
        f"- Host CPU: `{host.get('cpu')}` · OS: `{host.get('os')}` · WSL: `{host.get('wsl')}`",
        f"- GPU: {'present' if torch.get('cuda_available') else 'none detected (no CUDA device)'} · "
        f"driver: `{host.get('nvidia_smi') or 'NOT FOUND'}` · torch CUDA: `{torch.get('cuda_version')}`",
        f"- Index: {idx.get('chunk_count')} chunks · embeddings `{idx.get('embedding_model')}` "
        f"({idx.get('embedding_backend')}) · build {idx.get('build_seconds')} s",
        f"- Workload: {report['workload']['n_prompts']} prompts × "
        f"{report['workload']['repetitions']} reps (1 warm-up discarded), "
        f"max_new_tokens={report['workload']['max_new_tokens']}",
        "",
        "| Config | Status | Device | tokens/s (mean ± stdev) | TTFT s (mean) | "
        "latency ms (p50 / p95) | peak VRAM (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config, r in report["configs"].items():
        if r.get("status") != "measured":
            lines.append(
                f"| `{config}` | not_measured_on_this_host | — | — | — | — | — |"
            )
            continue
        tps = r["tokens_per_s"]
        ttft = r["ttft_s"]
        lat = r["latency_ms"]
        tps_s = (
            f"{tps['mean']} ± {tps['stdev']}" if tps.get("stdev") is not None else f"{tps['mean']}"
        )
        vram = r.get("peak_vram_mb")
        lines.append(
            f"| `{config}` | measured | {r['device']} | {tps_s} | "
            f"{ttft.get('mean')} | {lat.get('p50')} / {lat.get('p95')} | "
            f"{vram if vram is not None else 'not_measured'} |"
        )
    lines.append("")
    lines.append("## Skipped configs")
    lines.append("")
    lines.append("| Config | Reason |")
    lines.append("| --- | --- |")
    for config, r in report["configs"].items():
        if r.get("status") != "measured":
            lines.append(f"| `{config}` | {r.get('reason')} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
