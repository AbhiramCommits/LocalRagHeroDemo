"""LocalRAG Hero Demo command-line interface (Typer)."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .chunking import build_chunks
from .config import ROOT, load_settings
from .corpus import build_corpus
from .evaluate import eval_retrieval, write_retrieval_markdown
from .goldset import build_gold, load_gold
from .index import build_index
from .retrieve import CONFIGS, load_retriever
from .telemetry import format_host_profile, host_profile

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LocalRAG Hero Demo — agentic shop-floor assistant on a single GPU workstation.",
)
corpus_app = typer.Typer(no_args_is_help=True, help="Build and inspect the shop-floor corpus.")
index_app = typer.Typer(no_args_is_help=True, help="Build and inspect the retrieval index.")
eval_app = typer.Typer(no_args_is_help=True, help="Run retrieval and agent evaluations.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")

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


# --------------------------------------------------------------------------- #
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
    console.print(f"  manuals={manuals} specs={specs} work_instructions={wis} incidents={incs}")
    console.print(f"[green]gold[/green] {len(questions)} questions -> {gold_path}")


# --------------------------------------------------------------------------- #
@index_app.command("build")
def index_build(
    out: Path | None = typer.Option(None, "--out", help="Index output directory."),
    corpus: Path | None = typer.Option(None, "--corpus", help="Corpus directory."),
) -> None:
    """Chunk the corpus, embed it and build the hybrid vector index."""
    from .embeddings import Embedder

    settings = load_settings()
    corpus_dir = corpus or settings.path("corpus_dir")
    out_dir = out or settings.path("index_dir")
    if not Path(corpus_dir).exists():
        console.print(f"[red]corpus dir not found:[/red] {corpus_dir} — run `localrag corpus build`")
        raise typer.Exit(code=1)

    with console.status("chunking corpus..."):
        chunks = build_chunks(corpus_dir, settings.chunk_size, settings.chunk_overlap)
    with console.status(f"embedding {len(chunks)} chunks..."):
        embedder = Embedder(
            model_name=settings.embedding_model,
            device=settings.device,
            cache_dir=settings.path("cache_dir"),
        )
        meta = build_index(chunks, embedder, out_dir, corpus_dir=str(corpus_dir))

    console.print(f"[green]index[/green] {meta.chunk_count} chunks -> {out_dir}")
    console.print(
        f"  ann={meta.ann_backend} dim={meta.dim} embeddings={meta.embedding_backend} "
        f"device={meta.device} build={meta.build_seconds}s"
    )
    if meta.load_error:
        console.print(f"  [yellow]embedding fallback reason:[/yellow] {meta.load_error}")


@index_app.command("info")
def index_info(
    index_dir: Path | None = typer.Option(None, "--index", help="Index directory."),
) -> None:
    """Show the persisted index metadata."""
    import json

    settings = load_settings()
    path = (index_dir or settings.path("index_dir")) / "meta.json"
    if not path.exists():
        console.print(f"[red]no index metadata at[/red] {path}")
        raise typer.Exit(code=1)
    console.print_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query."),
    k: int = typer.Option(5, "--k", help="Number of hits."),
    config: str = typer.Option("hybrid_rrf_recency", "--config", help=f"One of {CONFIGS}."),
    index_dir: Path | None = typer.Option(None, "--index", help="Index directory."),
) -> None:
    """Run a hybrid retrieval query and print the ranked, citable hits."""
    settings = load_settings()
    retriever = load_retriever(str(index_dir or settings.path("index_dir")), settings)
    hits = retriever.search(query, k=k, config=config)

    table = Table(title=f"search: {query}", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("citation", style="cyan")
    table.add_column("score", justify="right")
    table.add_column("section")
    table.add_column("snippet", overflow="fold")
    for rank, hit in enumerate(hits, start=1):
        snippet = " ".join(hit.chunk.text.split())[:110]
        marker = " [red](superseded)[/red]" if hit.superseded else ""
        table.add_row(str(rank), hit.citation + marker, f"{hit.score:.4f}", hit.chunk.section_title, snippet)
    console.print(table)


# --------------------------------------------------------------------------- #
@eval_app.command("retrieval")
def eval_retrieval_cmd(
    out: Path = typer.Option(
        Path("results/retrieval_eval.json"), "--out", help="JSON output path."
    ),
    index_dir: Path | None = typer.Option(None, "--index", help="Index directory."),
    gold: Path | None = typer.Option(None, "--gold", help="Gold questions path."),
) -> None:
    """Evaluate recall@k and MRR for dense / BM25 / hybrid retrieval."""
    settings = load_settings()
    retriever = load_retriever(str(index_dir or settings.path("index_dir")), settings)
    questions = load_gold(gold or settings.path("gold_path"))
    previous = None
    if Path(out).exists():
        import json

        try:
            previous = json.loads(Path(out).read_text(encoding="utf-8"))
        except Exception:
            previous = None
    report = eval_retrieval(retriever, questions, out=out, settings=settings)
    md_path = write_retrieval_markdown(report, Path(out).with_suffix(".md"), previous=previous)

    table = Table(title="retrieval evaluation")
    table.add_column("config", style="cyan")
    for k in report["k_values"]:
        table.add_column(f"R@{k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("mean ms", justify="right")
    for config, m in report["configs"].items():
        table.add_row(
            config,
            *[f"{m[f'recall@{k}']:.3f}" for k in report["k_values"]],
            f"{m['mrr']:.3f}",
            str(m["mean_latency_ms"]),
        )
    console.print(table)
    console.print(f"[green]wrote[/green] {out} and {md_path}")


if __name__ == "__main__":  # pragma: no cover
    app()

__all__ = ["app", "ROOT"]
