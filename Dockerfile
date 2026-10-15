# ---------------------------------------------------------------------------
# LocalRAG Hero Demo — container image.
#
# GPU (default):
#   docker build -t localrag-hero:gpu .
#   docker run --rm --gpus all localrag-hero:gpu demo
#
# CPU (no NVIDIA GPU, no nvidia-container-toolkit needed):
#   docker build -t localrag-hero:cpu --build-arg BASE=python:3.11-slim .
#   docker run --rm localrag-hero:cpu demo
# ---------------------------------------------------------------------------
ARG BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

# Base packages + a Python >= 3.11 (the CUDA base ships 3.10; the slim base ships 3.11).
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl git build-essential; \
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then \
        apt-get install -y --no-install-recommends software-properties-common; \
        add-apt-repository -y ppa:deadsnakes/ppa; \
        apt-get update; \
        apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev; \
        ln -sf /usr/bin/python3.11 /usr/local/bin/python3; \
        python3.11 -m ensurepip --upgrade; \
    fi; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt environment.yml LICENSE README.md ./
COPY config ./config
COPY localrag ./localrag
COPY scripts ./scripts
COPY data ./data

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install -e ".[all]"

# Pre-create the hybrid index at build time when the corpus is present.
# Falls back to the offline hashing embedder if the model cannot be downloaded.
RUN if [ -n "$(ls -A data/corpus 2>/dev/null || true)" ]; then \
        python3 -m localrag.cli corpus build && python3 -m localrag.cli index build; \
    else \
        echo "no corpus in build context; skipping index prebuild"; \
    fi

ENTRYPOINT ["localrag"]
CMD ["--help"]
