# LocalRAG Hero Demo — Agentic Shop-Floor Assistant on a Single GPU Workstation

A fully local, GPU-accelerated agentic RAG system for a manufacturing
maintenance/QA use case. See the full write-up and measured results in
[`README.md`](README.md) (published in the documentation step) and
[`docs/PLAYBOOK.md`](docs/PLAYBOOK.md).

> This is an interim file created during the build; the complete README with
> measured results is written in the documentation step.

## Quick start

```bash
conda env create -f environment.yml && conda activate localrag
pip install -e ".[all]"
make corpus index demo
```
