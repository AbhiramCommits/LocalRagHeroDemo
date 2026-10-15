"""LocalRAG Hero Demo command-line interface (Typer)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .agent import Agent, NaiveRAG
from .backends import backend_status, get_backend
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
backends_app = typer.Typer(no_args_is_help=True, help="Inspect available LLM backends.")
bench_app = typer.Typer(no_args_is_help=True, help="Run the benchmark harness.")
app.add_typer(corpus_app, name="corpus")
app.add_typer(index_app, name="index")
app.add_typer(eval_app, name="eval")
app.add_typer(backends_app, name="backends")
app.add_typer(bench_app, name="bench")

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
        table.add_row(str(rank), hit.citation + marker, f"{hit.score:.4f}",
                      hit.chunk.section_title, snippet)
    console.print(table)


# --------------------------------------------------------------------------- #
@backends_app.command("list")
def backends_list() -> None:
    """Show which LLM backends are usable on this host."""
    table = Table(title="LLM backends")
    table.add_column("backend", style="cyan")
    table.add_column("available")
    table.add_column("detail")
    for row in backend_status():
        avail = "[green]yes[/green]" if row["available"] else "[yellow]no[/yellow]"
        table.add_row(row["backend"], avail, str(row["detail"]))
    console.print(table)


# --------------------------------------------------------------------------- #
def _run_ask(question: str, backend_name: str | None, show_trace: bool, index_dir: Path | None) -> None:
    settings = load_settings()
    retriever = load_retriever(str(index_dir or settings.path("index_dir")), settings)
    backend = get_backend(backend_name or settings.backend, settings)
    agent = Agent(retriever, backend, settings)
    result = agent.run(question, show_trace=show_trace)

    console.print(Panel(result.answer, title=f"answer · {result.backend} · {result.device}", border_style="green"))
    if result.citations:
        console.print("citations: " + ", ".join(f"[cyan]{c}[/cyan]" for c in result.citations))
    else:
        console.print("[yellow]no citations (refusal or insufficient evidence)[/yellow]")
    console.print(
        f"[dim]refused={result.refused} tool_calls={len(result.tool_calls)} "
        f"context_chunks={result.n_context_chunks} latency={result.latency_s:.3f}s[/dim]"
    )
    if show_trace:
        table = Table(title="tool calls")
        table.add_column("#", justify="right")
        table.add_column("tool", style="cyan")
        table.add_column("args", overflow="fold")
        table.add_column("chunks", justify="right")
        table.add_column("seconds", justify="right")
        for i, call in enumerate(result.tool_calls, start=1):
            table.add_row(str(i), call.tool, str(call.args), str(call.n_chunks), f"{call.seconds:.4f}")
        console.print(table)
        for stat in result.gen_stats:
            console.print(
                f"[dim]gen: {stat.get('prompt_tokens')} prompt + {stat.get('completion_tokens')} "
                f"completion tokens, ttft={stat.get('ttft_s')}s, {stat.get('tokens_per_s')} tok/s[/dim]"
            )
        console.print(f"[dim]grounding: {result.grounding}[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Technician question."),
    show_trace: bool = typer.Option(False, "--show-trace", help="Show tool calls, timings and grounding."),
    backend: str | None = typer.Option(None, "--backend", help="auto|transformers|openai_compat|extractive"),
    index_dir: Path | None = typer.Option(None, "--index", help="Index directory."),
) -> None:
    """Ask the shop-floor assistant a question."""
    _run_ask(question, backend, show_trace, index_dir)


DEMO_QUESTIONS = [
    ("lookup", "What does fault code E-318 indicate on the PRS-02 press?"),
    ("numeric", "What is the torque spec for the CNC-12 turret clamp bolts?"),
    ("supersede", "What is the current torque spec for the CNC-07 spindle clamp bolts?"),
    ("supersede", "What is the current assist-gas pressure for the LASER-02?"),
    ("refusal", "What is the serial number of the CNC-07 spindle motor?"),
]


@app.command()
def demo(
    backend: str | None = typer.Option(None, "--backend", help="auto|transformers|openai_compat|extractive"),
    index_dir: Path | None = typer.Option(None, "--index", help="Index directory."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force corpus + index rebuild."),
) -> None:
    """End-to-end demo: build corpus + index if needed, then answer 5 scripted questions."""
    from .embeddings import Embedder

    settings = load_settings()
    corpus_dir = settings.path("corpus_dir")
    idx_dir = index_dir or settings.path("index_dir")

    console.rule("[bold cyan]LocalRAG Hero Demo — Northline Precision Cell 4")
    if rebuild or not any(Path(corpus_dir).glob("*.md")):
        files = build_corpus(corpus_dir)
        build_gold(settings.path("gold_path"))
        console.print(f"[green]corpus[/green] built {len(files)} documents")
    else:
        console.print(f"[green]corpus[/green] {len(list(Path(corpus_dir).glob('*.md')))} documents (existing)")

    if rebuild or not (Path(idx_dir) / "meta.json").exists():
        chunks = build_chunks(corpus_dir, settings.chunk_size, settings.chunk_overlap)
        embedder = Embedder(model_name=settings.embedding_model, device=settings.device,
                            cache_dir=settings.path("cache_dir"))
        meta = build_index(chunks, embedder, idx_dir, corpus_dir=str(corpus_dir))
        console.print(
            f"[green]index[/green] {meta.chunk_count} chunks · {meta.embedding_backend} · {meta.ann_backend}"
        )
    else:
        console.print("[green]index[/green] (existing)")

    retriever = load_retriever(str(idx_dir), settings)
    chosen = backend or settings.backend
    instance = get_backend(chosen, settings)
    agent = Agent(retriever, instance, settings)
    console.print(f"backend: [cyan]{agent.backend_name}[/cyan] on [cyan]{agent.device or 'cpu'}[/cyan]")
    console.print()

    for i, (category, question) in enumerate(DEMO_QUESTIONS, start=1):
        result = agent.run(question)
        tools = ", ".join(c.tool for c in result.tool_calls) or "none"
        cite = ", ".join(result.citations) if result.citations else "[yellow]no citation (refusal)[/yellow]"
        console.print(
            Panel(
                result.answer,
                title=f"Q{i} · {category} · {question}",
                subtitle=f"tools: {tools} · {result.latency_s:.2f}s",
                border_style="green" if not result.refused else "yellow",
            )
        )
        console.print(f"  citations: {cite}\n")



# --------------------------------------------------------------------------- #
@eval_app.command("retrieval")
def eval_retrieval_cmd(
    out: Path = typer.Option(Path("results/retrieval_eval.json"), "--out", help="JSON output path."),
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


@eval_app.command("agent")
def eval_agent_cmd(
    out: Path = typer.Option(Path("results/agent_eval.json"), "--out", help="JSON output path."),
    backend: list[str] = typer.Option(None, "--backend", help="Backends to evaluate (repeatable)."),
    index_dir: Path | None = typer.Option(None, "--index", help="Index directory."),
    gold: Path | None = typer.Option(None, "--gold", help="Gold questions path."),
) -> None:
    """Evaluate agentic vs naive RAG for the chosen backends."""
    import json

    from .evaluate import eval_agent, write_agent_markdown

    settings = load_settings()
    retriever = load_retriever(str(index_dir or settings.path("index_dir")), settings)
    questions = load_gold(gold or settings.path("gold_path"))
    backend_names = list(backend) if backend else ["transformers", "extractive"]

    report: dict = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "backends": {},
    }
    for name in backend_names:
        console.rule(f"backend: {name}")
        instance = get_backend(name, settings)
        if hasattr(instance, "available") and not instance.available:
            reason = getattr(instance, "load_error", "unavailable")
            console.print(f"[yellow]skipping[/yellow] {name}: {reason}")
            report["backends"][name] = {"status": "not_available_on_this_host", "reason": reason}
            continue
        agent = Agent(retriever, instance, settings)
        naive = NaiveRAG(retriever, instance, settings)
        block = eval_agent(agent, naive, questions, settings, progress=True)
        report["backends"][block["backend"]] = block
        console.print(f"[green]{name}[/green] done")

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = write_agent_markdown(report, path.with_suffix(".md"))
    console.print(f"[green]wrote[/green] {out} and {md_path}")


# --------------------------------------------------------------------------- #
@bench_app.command("run")
def bench_run(
    out: Path = typer.Option(Path("results/benchmarks.json"), "--out", help="JSON output path."),
    config: list[str] = typer.Option(None, "--config", help="Configs to run (repeatable)."),
    repetitions: int = typer.Option(3, "--reps", help="Repetitions per config."),
) -> None:
    """Run the 12-prompt benchmark workload across the supported configs."""
    from .bench import CONFIGS, run_bench, write_bench_markdown

    settings = load_settings()
    configs = tuple(config) if config else CONFIGS
    with console.status("running benchmarks (this can take a while)..."):
        report = run_bench(settings, out=out, configs=configs, repetitions=repetitions)
    md = write_bench_markdown(report, Path(out).with_suffix(".md"))
    table = Table(title="benchmarks")
    table.add_column("config", style="cyan")
    table.add_column("status")
    table.add_column("device")
    table.add_column("tok/s")
    table.add_column("p95 ms", justify="right")
    for name, r in report["configs"].items():
        if r.get("status") != "measured":
            table.add_row(name, "[yellow]not_measured_on_this_host[/yellow]", "—", "—", "—")
        else:
            table.add_row(
                name,
                "[green]measured[/green]",
                str(r["device"]),
                str((r["tokens_per_s"] or {}).get("mean")),
                str((r["latency_ms"] or {}).get("p95")),
            )
    console.print(table)
    console.print(f"[green]wrote[/green] {out} and {md}")


@bench_app.command("table")
def bench_table(
    source: Path = typer.Option(Path("results/benchmarks.json"), "--in", help="Benchmark JSON."),
    out: Path = typer.Option(Path("results/benchmarks.md"), "--out", help="Markdown output."),
) -> None:
    """Render the benchmark Markdown table from a results JSON."""
    import json

    from .bench import write_bench_markdown

    if not source.exists():
        console.print(f"[red]no benchmark results at[/red] {source}")
        raise typer.Exit(code=1)
    report = json.loads(source.read_text(encoding="utf-8"))
    md = write_bench_markdown(report, out)
    console.print(f"[green]wrote[/green] {md}")


if __name__ == "__main__":  # pragma: no cover
    app()

__all__ = ["app", "ROOT"]
