# LocalRAG Hero Demo — one-command workflows.
#
# Activate the environment first (conda activate localrag) or point PYTHON at it:
#   make PYTHON=/opt/anaconda3/envs/localrag/bin/python demo
PYTHON  ?= python
LOCALRAG = $(PYTHON) -m localrag.cli
COMPOSE  ?= docker compose

.DEFAULT_GOAL := help
.PHONY: help setup corpus index demo ask bench eval test lint docker-build docker-demo smoke all clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install the package and all extras (editable).
	$(PYTHON) -m pip install -e ".[all]"

corpus:  ## Generate the synthetic corpus + gold set.
	$(LOCALRAG) corpus build

index:  ## Build the hybrid (FAISS + BM25) index.
	$(LOCALRAG) index build

demo:  ## End-to-end demo: corpus -> index -> scripted shop-floor questions.
	$(LOCALRAG) demo

bench:  ## Run the benchmark harness and render the Markdown table.
	$(LOCALRAG) bench run --out results/benchmarks.json
	$(LOCALRAG) bench table

eval:  ## Run retrieval + agent evaluation.
	$(LOCALRAG) eval retrieval --out results/retrieval_eval.json
	$(LOCALRAG) eval agent --out results/agent_eval.json

test:  ## Run the test suite (deselect slow/model tests).
	$(PYTHON) -m pytest -m "not slow" --cov=localrag --cov-report=term-missing

lint:  ## Lint with ruff.
	$(PYTHON) -m ruff check .

docker-build:  ## Build the GPU Docker image.
	docker build -t localrag-hero:gpu .

docker-demo:  ## Run the demo through docker compose.
	$(COMPOSE) run --rm localrag demo

smoke:  ## Fresh-clone smoke test (venv + install + cite check).
	bash scripts/smoke_test.sh

all: lint test corpus index demo  ## Lint, test, then run the demo end to end.

clean:  ## Remove build artefacts (keeps results/ and data/).
	rm -rf artifacts .cache .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
