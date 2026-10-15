# Reproduction playbook — LocalRAG Hero Demo

A numbered path from a bare machine to a running agentic shop-floor assistant,
with the expected output after each step and a troubleshooting table of the real
failure modes this project hit or guarded against.

Use `scripts/bringup_check.sh` as step 0 in every environment; it tells you which
of the steps below you can skip.

---

## 1. Driver and OS baseline

**Linux (native).** Install an NVIDIA driver ≥ 525 (CUDA ≥ 12.0) and reboot.

```bash
nvidia-smi                     # must list your GPU, driver and CUDA version
```

Expected: a table with your GPU name, `Driver Version: 5xx.xx` and
`CUDA Version: 12.x`. If `nvidia-smi` is missing, you can still run the CPU
configs — the demo degrades honestly and records GPU rows as
`not_measured_on_this_host`.

## 2. Container runtime (optional but recommended)

```bash
# Docker Engine
curl -fsSL https://get.docker.com | sudo sh
# NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU-in-container
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Expected: `nvidia-smi` output from *inside* the container.

## 3. Windows + WSL2

Follow [`scripts/wsl_setup.md`](../scripts/wsl_setup.md). The one rule: install
the NVIDIA driver on **Windows**, never inside the distro. Expected:

```bash
nvidia-smi        # inside Ubuntu on WSL2
ls /usr/lib/wsl/lib
```

## 4. Python environment

```bash
conda env create -f environment.yml      # python 3.11, env name "localrag"
conda activate localrag
pip install -e ".[all]"                  # or: pip install -e ".[rag,gpu,dev]"
```

Expected: `localrag --version` prints `localrag 0.1.0`.

## 5. Preflight

```bash
bash scripts/bringup_check.sh | tee results/bringup_report.txt
localrag doctor | tee results/host_profile.txt
```

Expected: a PASS/FAIL/SKIP summary and a host profile. GPU-specific SKIPs are
normal on a CPU-only host.

## 6. Corpus + index

```bash
localrag corpus build        # 40 documents + 35 gold questions
localrag index build         # section-aware chunks -> FAISS + BM25
localrag index info
```

Expected: `corpus 40 docs`, `gold 35 questions`, then an index line like
`index 204 chunks · dim 384 · embeddings=sentence_transformers · ann=faiss`.
If the embedding model cannot be downloaded, the build falls back to the
offline hashing embedder and says so (`embeddings=hashing_fallback`).

## 7. Demo

```bash
make demo                    # corpus -> index -> 5 scripted questions (Rich output)
# or a single question with a full trace:
localrag ask "What is the current torque spec for the CNC-07 spindle clamp bolts?" --show-trace
```

Expected: a cited answer such as `52 N·m … INC-2026-001 §2`, a table of tool
calls, and (for the refusal question) `INSUFFICIENT_EVIDENCE` with no citations.

## 8. Evaluation and benchmarks

```bash
make eval                    # results/retrieval_eval.json + agent_eval.json (+ .md)
make bench                   # results/benchmarks.json + benchmarks.md
```

Expected: JSON + Markdown with real numbers; unsupported configs are recorded as
`not_measured_on_this_host` with a reason.

## 9. Container run (optional)

```bash
docker build -t localrag-hero:cpu --build-arg BASE=python:3.11-slim .
docker run --rm localrag-hero:cpu version
docker compose run --rm localrag demo                  # GPU default
```

## 10. Tests and lint

```bash
make lint
make test
```

Expected: `ruff` clean; the test suite green with coverage reported.

---

## Container build on this host (recorded outcome)

Build host: Apple Silicon (arm64), macOS, Docker Desktop. Because the host is
arm64, the amd64 CUDA base runs under emulation, which makes the build slow.

**GPU variant — `make docker-build` (default `nvidia/cuda:12.4.1-runtime-ubuntu22.04`): succeeded on the second attempt.**

- **Attempt 1 failed** (exit 1) at the `pip install` step with the exact error
  `python3: No module named pip` — the deadsnakes Python 3.11 installed onto the
  CUDA base had no `pip`.
- **Fix:** the Dockerfile now runs `python3.11 -m ensurepip --upgrade` after
  installing the interpreter.
- **Attempt 2 succeeded:** `GPU_BUILD_EXIT=0`, image `localrag-hero:gpu` (14.1 GB).
  Verified (run without `--gpus all` because this host has no NVIDIA GPU):

```
$ docker run --rm localrag-hero:gpu version
localrag 0.1.0
$ docker run --rm localrag-hero:gpu index info
{ "chunk_count": 204, "dim": 384, "ann_backend": "faiss", ... }   # index pre-created at build time
```

Log: `results/docker_build_gpu.log`.

**CPU variant — `docker build --build-arg BASE=python:3.11-slim`: succeeded.**

```
CPU_BUILD_EXIT=0
localrag-hero:cpu   10.5 GB
$ docker run --rm localrag-hero:cpu version
localrag 0.1.0
$ docker run --rm localrag-hero:cpu ask "What does fault code E-318 indicate on the PRS-02 press?" --backend extractive
  answer:     | E-318 | Hydraulic pressure loss — system pressure below 1800 PSI. |
  citations:  MANUAL-PRS-02 §3 (rev D, 2025-09-12)
```

Log: `results/docker_build_cpu.log`.

> `pip install -e ".[all]"` pulls the CUDA build of torch on Linux, which is why
> both images are ~10–14 GB. For a leaner CPU image, install a CPU-only torch
> wheel or trim the `all` extra.

---

## Troubleshooting table

| # | Symptom | Likely cause | Fix |
| --- | --- | --- | --- |
| 1 | `nvidia-smi: command not found` | No NVIDIA driver, or wrong OS layer (WSL host) | Install the driver on the host OS; on WSL run `wsl --update` and reboot. Never install a Linux driver inside WSL. |
| 2 | `docker run --gpus all` → *could not select device driver* | `nvidia-container-toolkit` missing or Docker not restarted | `sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. |
| 3 | CUDA out of memory during generation | Model + KV cache + context exceed VRAM | Use a smaller model, lower `max_new_tokens`, reduce `top_k`, or point the agent at a vLLM/NIM endpoint (`LOCALRAG_ENDPOINT`). |
| 4 | Hugging Face model download blocked (air-gapped / proxy) | No egress to `huggingface.co` | Pre-seed the cache (`HF_HOME`), set `HF_HUB_OFFLINE=1` after seeding, or run the offline `extractive` backend — the index then falls back to `hashing_fallback`. |
| 5 | `ImportError: libfaiss` / `faiss` fails to import | No `faiss-cpu` wheel for the platform | Nothing to do — `localrag` automatically uses its pure-numpy brute-force index (`ann=numpy`). |
| 6 | `nvidia-smi` works in Windows but not in WSL | A Linux driver was installed inside the distro, shadowing the injected one | `sudo apt-get purge 'nvidia-*'`, then `wsl --shutdown` and reopen. |
| 7 | Port 8000 already in use for vLLM/NIM | Another service holds 8000 | Change the published port (`-p 8001:8000`) and set `LOCALRAG_ENDPOINT=http://localhost:8001/v1`. |
| 8 | Segfault when a FAISS search runs after a torch model loads (macOS/arm64) | Duplicate OpenMP runtime between `faiss-cpu` and `torch` | Handled in `localrag/__init__.py`: it sets `KMP_DUPLICATE_LIB_OK` and `OMP_NUM_THREADS=1` before either library loads. Override `OMP_NUM_THREADS` to tune CPU throughput. |
| 9 | `RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'` on CPU | fp16 requested on a CPU-only host | Use `hf_cpu_fp32`; the backend already selects fp32 on CPU. |
| 10 | Index rebuild takes long / re-downloads | `.cache/embeddings.npz` was deleted or `HF_HOME` moved | Keep `.cache/` and set a stable `HF_HOME`; embeddings are content-hash cached. |
