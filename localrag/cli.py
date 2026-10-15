"""LocalRAG Hero Demo command-line interface (Typer)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from . import __version__
from .config import ROOT, load_settings
from .corpus import build_corpus
from .goldset import build_gold
from .telemetry import format_host_profile, host_profile

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LocalRAG Hero Demo — agentic shop-floor assistant on a single GPU workstation.",
)
corpus_app = typer.Typer(no_args_is_help=True, help="Build and inspect the shop-floor corpus.")
app.add_typer(corpus_app, name="corpus")

console = Console()


# --------------------------------------------------------------------------- #
@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"localrag {__version__}")


@app.command()
def doctor(
    out: Path | None = typer.Option(None, "--out", help="Also write the raw profile to this file."),
) -> None:
    """Probe the real host: Python, OS, torch, CUDA, GPUs, free VRAM and WSL."""
    profile = host_profile()
    console.print(Panel(format_host_profile(profile), title="localrag doctor", border_style="cyan"))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(format_host_profile(profile) + "\n", encoding="utf-8")
        console.print(f"wrote [bold]{out}[/bold]")


@corpus_app.command("build")
def corpus_build(
    out_dir: Path | None = typer.Option(None, "--out", help="Corpus output directory."),
    gold: Path | None = typer.Option(None, "--gold", help="Gold questions output path."),
) -> None:
    """Generate the synthetic shop-floor corpus and the gold question set."""
    settings = load_settings()
    corpus_dir = out_dir or settings.path("corpus_dir")
    gold_path = gold or settings.path("gold_path")

    files = build_corpus(corpus_dir)
    questions = build_gold(gold_path)

    manuals = sum(1 for f in files if f.name.startswith("MANUAL-"))
    specs = sum(1 for f in files if f.name.startswith("SPEC-"))
    wis = sum(1 for f in files if f.name.startswith("QC-WI-"))
    incs = sum(1 for f in files if f.name.startswith("INC-"))
    console.print(f"[green]corpus[/green] {len(files)} docs -> {corpus_dir}")
    console.print(
        f"  manuals={manuals} specs={specs} work_instructions={wis} incidents={incs}"
    )
    console.print(f"[green]gold[/green] {len(questions)} questions -> {gold_path}")


if __name__ == "__main__":  # pragma: no cover
    app()


# ROOT is exported for later sub-commands that resolve paths.
__all__ = ["app", "ROOT"]
