"""Host and GPU telemetry.

Two layers:

* **Host profiling** (this step) — probe the real machine: OS, Python, torch,
  CUDA, GPU inventory via ``nvidia-smi``/NVML, free VRAM and WSL detection.
  Every value is *probed*, never hard-coded.
* **NVML sampling monitor** (:class:`GpuMonitor`, added in the benchmark step) —
  samples utilization, power and peak VRAM in a background thread.

Everything degrades gracefully: on a machine without an NVIDIA GPU the GPU
fields are reported as unavailable rather than raising.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# Low-level probes                                                            #
# --------------------------------------------------------------------------- #
def is_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux."""
    try:
        text = Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False
    return "microsoft" in text or "wsl" in text


def nvidia_smi_path() -> str | None:
    return shutil.which("nvidia-smi")


def nvidia_smi_text(timeout: int = 10) -> str:
    """Capture the human-readable ``nvidia-smi`` output, or '' if unavailable."""
    path = nvidia_smi_path()
    if not path:
        return ""
    try:
        out = subprocess.run(
            [path], capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout or out.stderr or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _query_smi(fields: list[str]) -> list[list[str]]:
    """Run ``nvidia-smi --query-gpu`` and parse the CSV result."""
    path = nvidia_smi_path()
    if not path:
        return []
    query = ",".join(fields)
    try:
        out = subprocess.run(
            [path, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[list[str]] = []
    for line in (out.stdout or "").strip().splitlines():
        if line.strip():
            rows.append([part.strip() for part in line.split(",")])
    return rows


def gpu_inventory() -> list[dict[str, object]]:
    """Return a per-GPU inventory using ``nvidia-smi``. Empty when no GPU."""
    fields = [
        "index",
        "name",
        "driver_version",
        "memory.total",
        "memory.free",
        "utilization.gpu",
        "power.draw",
    ]
    rows = _query_smi(fields)

    # Try to enrich with the runtime CUDA version reported by NVML.
    cuda_version = nvml_cuda_version()
    gpus: list[dict[str, object]] = []
    for row in rows:
        if len(row) < len(fields):
            continue
        index, name, driver, mem_total, mem_free, util, power = row[:7]

        def _num(value: str) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        gpus.append(
            {
                "index": int(index) if index.isdigit() else index,
                "name": name,
                "driver_version": driver,
                "cuda_version": cuda_version,
                "memory_total_mb": _num(mem_total),
                "memory_free_mb": _num(mem_free),
                "utilization_pct": _num(util),
                "power_draw_w": _num(power),
                "source": "nvidia-smi",
            }
        )
    return gpus


# --------------------------------------------------------------------------- #
# NVML (pynvml) helpers                                                       #
# --------------------------------------------------------------------------- #
def nvml_cuda_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return str(pynvml.nvmlSystemGetCudaDriverVersion_v2())
        except AttributeError:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()
            return f"{raw // 1000}.{(raw % 1000) // 10}"
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # pragma: no cover - depends on host
        return None


def nvml_available() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# torch                                                                       #
# --------------------------------------------------------------------------- #
def torch_info() -> dict[str, object]:
    info: dict[str, object] = {"available": False, "version": None, "cuda_available": False,
                               "cuda_version": None, "device_count": 0, "mps_available": False}
    try:
        import torch

        info["available"] = True
        info["version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
        info["mps_available"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Aggregated host profile                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class HostProfile:
    os: str
    os_release: str
    platform: str
    python_version: str
    machine: str
    processor: str
    is_wsl: bool
    nvidia_smi: str | None
    nvml_available: bool
    gpus: list[dict[str, object]] = field(default_factory=list)
    torch: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def host_profile() -> HostProfile:
    """Probe the real host and return a structured profile."""
    return HostProfile(
        os=platform.system(),
        os_release=platform.release(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        machine=platform.machine(),
        processor=platform.processor() or platform.machine(),
        is_wsl=is_wsl(),
        nvidia_smi=nvidia_smi_path(),
        nvml_available=nvml_available(),
        gpus=gpu_inventory(),
        torch=torch_info(),
    )


def format_host_profile(profile: HostProfile | None = None) -> str:
    """Render the host profile as plain text (used by ``localrag doctor``)."""
    p = profile or host_profile()
    lines = [
        "LocalRAG Hero Demo — host profile",
        "=" * 34,
        f"OS                : {p.os} {p.os_release}",
        f"Platform          : {p.platform}",
        f"CPU               : {p.processor} ({p.machine})",
        f"Python            : {p.python_version}",
        f"Running under WSL : {p.is_wsl}",
        f"nvidia-smi        : {p.nvidia_smi or 'NOT FOUND'}",
        f"NVML available    : {p.nvml_available}",
        "-" * 34,
        f"torch available   : {p.torch.get('available')}",
        f"torch version     : {p.torch.get('version')}",
        f"CUDA available    : {p.torch.get('cuda_available')}",
        f"CUDA (torch)      : {p.torch.get('cuda_version')}",
        f"MPS available     : {p.torch.get('mps_available')}",
        "-" * 34,
    ]
    if p.gpus:
        lines.append(f"GPU count         : {len(p.gpus)}")
        for gpu in p.gpus:
            lines.append(
                f"  [{gpu['index']}] {gpu['name']} | driver {gpu['driver_version']} | "
                f"CUDA {gpu['cuda_version']} | free {gpu['memory_free_mb']} MiB / "
                f"{gpu['memory_total_mb']} MiB | util {gpu['utilization_pct']}% | "
                f"{gpu['power_draw_w']} W"
            )
    else:
        lines.append("GPU count         : 0 (no NVIDIA GPU detected on this host)")
        lines.append("                    -> GPU configs will be recorded as not_measured_on_this_host")
    lines.append("-" * 34)
    return "\n".join(lines)


def host_profile_json(profile: HostProfile | None = None) -> str:
    return json.dumps((profile or host_profile()).to_dict(), indent=2)


# --------------------------------------------------------------------------- #
# NVML sampling monitor                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class GpuSummary:
    available: bool
    status: str  # "measured" | "not_measured_on_this_host"
    device_name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    samples: int = 0
    peak_memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    peak_utilization_pct: float | None = None
    mean_utilization_pct: float | None = None
    mean_power_w: float | None = None
    peak_power_w: float | None = None
    reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class GpuMonitor:
    """Sample GPU utilization, power and peak VRAM in a background thread.

    Use as a context manager::

        with GpuMonitor() as m:
            ...workload...
        print(m.summary())

    When NVML / an NVIDIA GPU is unavailable the monitor degrades to a no-op and
    ``summary()`` reports ``status="not_measured_on_this_host"`` — it never raises.
    """

    def __init__(self, index: int = 0, interval: float = 0.1) -> None:
        self.index = index
        self.interval = interval
        self._thread = None
        self._stop = None
        self._handle = None
        self._pynvml = None
        self._samples: list[dict] = []
        self._device_name: str | None = None
        self._driver: str | None = None
        self._cuda: str | None = None
        self._mem_total: float | None = None
        self._reason: str | None = None

    # ------------------------------------------------------------------ #
    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
            self._device_name = _decode(pynvml.nvmlDeviceGetName(self._handle))
            self._driver = _decode(pynvml.nvmlSystemGetDriverVersion())
            self._cuda = _cuda_from_nvml(pynvml)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._mem_total = round(mem.total / (1024 * 1024), 1)
            return True
        except Exception as exc:
            self._reason = f"NVML unavailable: {type(exc).__name__}: {exc}"
            return False

    def __enter__(self) -> GpuMonitor:
        if not self._init_nvml():
            return self
        import threading

        self._stop = threading.Event()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    try:
                        power = self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                    except Exception:
                        power = None
                    self._samples.append(
                        {
                            "memory_used_mb": mem.used / (1024 * 1024),
                            "utilization_pct": float(util.gpu),
                            "power_w": power,
                        }
                    )
                except Exception:
                    pass
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def summary(self) -> GpuSummary:
        if not self._samples:
            return GpuSummary(
                available=False,
                status="not_measured_on_this_host",
                device_name=self._device_name,
                driver_version=self._driver,
                cuda_version=self._cuda,
                memory_total_mb=self._mem_total,
                reason=self._reason or "no NVML samples collected on this host",
            )
        utils = [s["utilization_pct"] for s in self._samples]
        powers = [s["power_w"] for s in self._samples if s["power_w"] is not None]
        return GpuSummary(
            available=True,
            status="measured",
            device_name=self._device_name,
            driver_version=self._driver,
            cuda_version=self._cuda,
            samples=len(self._samples),
            peak_memory_used_mb=round(max(s["memory_used_mb"] for s in self._samples), 1),
            memory_total_mb=self._mem_total,
            peak_utilization_pct=round(max(utils), 1),
            mean_utilization_pct=round(sum(utils) / len(utils), 1),
            mean_power_w=round(sum(powers) / len(powers), 1) if powers else None,
            peak_power_w=round(max(powers), 1) if powers else None,
        )


def _decode(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _cuda_from_nvml(pynvml: object) -> str | None:
    try:
        raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()  # type: ignore[attr-defined]
        return f"{raw // 1000}.{(raw % 1000) // 10}"
    except Exception:
        try:
            raw = pynvml.nvmlSystemGetCudaDriverVersion()  # type: ignore[attr-defined]
            return f"{raw // 1000}.{(raw % 1000) // 10}"
        except Exception:
            return None


if __name__ == "__main__":  # pragma: no cover
    print(format_host_profile())
