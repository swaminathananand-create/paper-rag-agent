#!/usr/bin/env python3
"""Command-line interface for the Academic Paper RAG Summarizer & Literature Review Agent.

Usage:
    python cli.py summarize <file_or_url>
    python cli.py batch <file_with_urls_or_paths>
    python cli.py review <file_or_url> [<file_or_url> ...]
"""

import sys
import time
import io
from pathlib import Path

# Force UTF-8 output to avoid cp1252 encoding errors on Windows for Rich symbols
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich import print as rprint
from rich.rule import Rule

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from paper_rag.config import Config
from paper_rag.summarizer import PaperSummarizer
from paper_rag.literature_review import LiteratureReviewAgent
from paper_rag.schemas import PaperSummary, LiteratureReviewOutline

console = Console()


def print_header():
    """Prints the application banner."""
    console.print(Panel.fit(
        "[bold cyan]Academic Paper RAG Agent[/bold cyan]\n"
        "[dim]Retrieval-Augmented Generation for Structured Paper Analysis[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))
    api_status = "[green][OK] API Key Loaded - Live Gemini Mode[/green]" if Config.has_api_key() \
        else "[yellow][!]  No API Key - Running in Offline BM25 Extractive Mode[/yellow]"
    console.print(f"  {api_status}")
    console.print(f"  [dim]Model: {Config.DEFAULT_MODEL} | Embedding: {Config.EMBEDDING_MODEL} | Top-K: {Config.TOP_K}[/dim]")
    console.print()


def display_summary(summary: PaperSummary):
    """Pretty-prints a PaperSummary with rich formatting."""
    console.print(Rule(f"[bold yellow]{summary.metadata.title}[/bold yellow]"))

    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="bold cyan", no_wrap=True)
    info_table.add_column()
    info_table.add_row("Authors", ", ".join(summary.metadata.authors))
    info_table.add_row("Year", str(summary.metadata.year or "Unknown"))
    info_table.add_row("Pages", str(summary.metadata.page_count))
    info_table.add_row("Source", summary.metadata.source)
    console.print(info_table)
    console.print()

    sections = [
        ("1. Abstract & Context", summary.abstract, "blue"),
        ("2. Main Argument & Thesis", summary.main_argument, "green"),
        ("3. Methodology", summary.methodology, "magenta"),
        ("5. Implications & Significance", summary.implications, "yellow"),
    ]

    for title, content, color in sections:
        console.print(Panel(content, title=f"[bold {color}]{title}[/bold {color}]", border_style=color, padding=(0, 1)))

    # Key findings
    findings_table = Table(show_header=False, box=None, padding=(0, 1))
    findings_table.add_column(style="bold green")
    findings_table.add_column()
    for i, finding in enumerate(summary.key_findings, 1):
        findings_table.add_row(f"[{i}]", finding)
    console.print(Panel(findings_table, title="[bold green]4. Key Findings & Results[/bold green]", border_style="green", padding=(0, 1)))

    if summary.limitations:
        console.print(Panel(
            f"[dim]{summary.limitations}[/dim]",
            title="[bold dim]Limitations[/bold dim]",
            border_style="dim",
            padding=(0, 1),
        ))

    # RAG evidence
    if summary.evidence:
        ev_table = Table(show_header=True, header_style="bold dim", box=None)
        ev_table.add_column("Aspect", style="cyan", no_wrap=True)
        ev_table.add_column("Page", style="yellow", width=5)
        ev_table.add_column("Retrieved Snippet", overflow="fold")
        for ev in summary.evidence[:5]:
            snippet = ev.snippet[:180].replace("\n", " ").strip()
            ev_table.add_row(ev.aspect, str(ev.page or "-"), f"[dim italic]\"{snippet}...\"[/dim italic]")
        console.print(Panel(ev_table, title="[bold dim]RAG Retrieved Evidence[/bold dim]", border_style="dim", padding=(0, 1)))


def display_review(review: LiteratureReviewOutline):
    """Pretty-prints a LiteratureReviewOutline with rich formatting."""
    console.print(Rule(f"[bold magenta]{review.title}[/bold magenta]"))
    console.print(Panel(review.executive_summary, title="[bold]Executive Summary[/bold]", border_style="cyan", padding=(0, 1)))

    for i, theme in enumerate(review.themes, 1):
        theme_content = []
        theme_content.append(f"[bold]Scope[/bold]: {theme.description}\n")
        theme_content.append(f"[bold]Papers[/bold]: {', '.join(f'[italic]{p}[/italic]' for p in theme.contributing_papers)}\n")
        theme_content.append(f"[bold]Synthesis[/bold]: {theme.synthesized_insights}\n")
        theme_content.append(f"[bold]Methods[/bold]: {theme.methodological_patterns}\n")
        if theme.consensus_points:
            theme_content.append("[bold]Consensus[/bold]:\n" + "\n".join(f"  * {c}" for c in theme.consensus_points))
        if theme.divergences_or_debates:
            theme_content.append("[bold]Debates[/bold]:\n" + "\n".join(f"  ! {d}" for d in theme.divergences_or_debates))
        console.print(Panel("\n".join(theme_content), title=f"[bold yellow]Theme {i}: {theme.theme_name}[/bold yellow]", border_style="yellow", padding=(0, 1)))

    console.print(Panel(review.comparative_analysis, title="[bold]Comparative Analysis[/bold]", border_style="blue", padding=(0, 1)))

    gaps_text = "\n".join(f"  * {g}" for g in review.identified_research_gaps)
    console.print(Panel(gaps_text, title="[bold red]Research Gaps and Future Directions[/bold red]", border_style="red", padding=(0, 1)))


def cmd_summarize(source: str):
    """Summarizes a single paper from a path or URL."""
    Config.ensure_directories()
    print_header()
    console.print(f"[bold]Source:[/bold] {source}\n")

    with Progress(SpinnerColumn(), TextColumn("[bold cyan]{task.description}"), TimeElapsedColumn(), console=console) as progress:
        t1 = progress.add_task("Loading & ingesting document...", total=None)
        summarizer = PaperSummarizer()

        progress.update(t1, description="Chunking document into semantic passages...")
        time.sleep(0.1)

        progress.update(t1, description="Indexing chunks & running targeted RAG retrieval...")
        time.sleep(0.1)

        progress.update(t1, description="Generating structured summary...")
        summary = summarizer.summarize(source)
        progress.update(t1, description="Done!", completed=True)

    console.print()
    display_summary(summary)

    md_path = Config.OUTPUT_DIR / f"{summary.paper_id}_summary.md"
    json_path = Config.OUTPUT_DIR / f"{summary.paper_id}_summary.json"
    console.print()
    console.print(f"[green]Reports saved:[/green]")
    console.print(f"  [link={md_path}]{md_path}[/link]")
    console.print(f"  [link={json_path}]{json_path}[/link]")


def cmd_batch(list_file: str):
    """Summarizes multiple papers from a file containing one URL/path per line."""
    Config.ensure_directories()
    print_header()

    list_path = Path(list_file)
    if not list_path.exists():
        console.print(f"[red]Error: File not found: {list_file}[/red]")
        sys.exit(1)

    sources = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not sources:
        console.print("[red]Error: No sources found in file.[/red]")
        sys.exit(1)

    console.print(f"[bold]Processing {len(sources)} paper(s) from:[/bold] {list_file}\n")
    summarizer = PaperSummarizer()
    summaries = []

    for i, source in enumerate(sources, 1):
        console.print(f"[cyan][{i}/{len(sources)}][/cyan] {source}")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Processing...", total=None)
            try:
                summary = summarizer.summarize(source)
                summaries.append(summary)
                progress.update(task, description=f"[green]✓ Done: {summary.metadata.title[:60]}[/green]")
            except Exception as e:
                console.print(f"  [red]✗ Failed: {e}[/red]")
        console.print()

    console.print(Rule("[bold green]Batch Summary Complete[/bold green]"))
    results_table = Table(show_header=True, header_style="bold cyan")
    results_table.add_column("#", width=3)
    results_table.add_column("Title")
    results_table.add_column("Pages", width=6)
    results_table.add_column("Output File")

    for i, s in enumerate(summaries, 1):
        out_file = str(Config.OUTPUT_DIR / f"{s.paper_id}_summary.md")
        results_table.add_row(str(i), s.metadata.title[:55], str(s.metadata.page_count), out_file)

    console.print(results_table)
    console.print(f"\n[green]{len(summaries)} paper(s) summarized. Reports in:[/green] {Config.OUTPUT_DIR}")


def cmd_review(sources: list[str], title: str = None):
    """Generates a multi-paper thematic literature review from given paper sources."""
    Config.ensure_directories()
    print_header()

    if len(sources) < 2:
        console.print("[yellow][!] At least 2 papers are recommended for a meaningful literature review.[/yellow]")

    console.print(f"[bold]Generating literature review for {len(sources)} papers...[/bold]\n")

    agent = LiteratureReviewAgent()
    summaries = []

    for i, source in enumerate(sources, 1):
        console.print(f"[cyan][{i}/{len(sources)}][/cyan] Summarizing: {source}")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
            task = progress.add_task("Summarizing...", total=None)
            try:
                summary = agent.summarizer.summarize(source)
                summaries.append(summary)
                progress.update(task, description=f"[green][OK] {summary.metadata.title[:55]}[/green]")
            except Exception as e:
                console.print(f"  [red][FAIL] Failed: {e}[/red]")

    console.print()
    console.print("[bold cyan]Synthesizing thematic literature review...[/bold cyan]")
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Clustering themes and generating outline...", total=None)
        review = agent.generate_review(summaries, review_title=title)
        progress.update(task, description="[green][OK] Literature review complete[/green]")

    console.print()
    display_review(review)

    md_path = Config.OUTPUT_DIR / "literature_review_outline.md"
    json_path = Config.OUTPUT_DIR / "literature_review_outline.json"
    console.print()
    console.print("[green]Literature review saved:[/green]")
    console.print(f"  [link={md_path}]{md_path}[/link]")
    console.print(f"  [link={json_path}]{json_path}[/link]")


def print_usage():
    """Prints help text."""
    console.print(Panel(
        "[bold]Usage:[/bold]\n\n"
        "  [cyan]python cli.py summarize <file_or_url>[/cyan]\n"
        "    Summarize a single paper (local PDF, .txt, or URL)\n\n"
        "  [cyan]python cli.py batch <list_file>[/cyan]\n"
        "    Summarize all papers listed in a file (one URL/path per line)\n\n"
        "  [cyan]python cli.py review <source1> [source2 ...][/cyan]\n"
        "    Generate a thematic literature review across multiple papers\n\n"
        "[bold]Examples:[/bold]\n\n"
        "  python cli.py summarize sample_papers/attention_is_all_you_need.txt\n"
        "  python cli.py batch sample_papers/paper_urls.txt\n"
        "  python cli.py review sample_papers/*.txt\n"
        "  python cli.py review https://arxiv.org/abs/1706.03762 https://arxiv.org/abs/2005.11401\n",
        title="[bold cyan]📄 Academic Paper RAG Agent — Help[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] in ("--help", "-h", "help"):
        print_usage()
        sys.exit(0)

    command = args[0].lower()

    if command == "summarize":
        if len(args) < 2:
            console.print("[red]Error: Please provide a paper path or URL.[/red]")
            sys.exit(1)
        cmd_summarize(args[1])

    elif command == "batch":
        if len(args) < 2:
            console.print("[red]Error: Please provide a file path containing paper URLs/paths.[/red]")
            sys.exit(1)
        cmd_batch(args[1])

    elif command == "review":
        if len(args) < 2:
            console.print("[red]Error: Please provide at least one paper source.[/red]")
            sys.exit(1)
        # Optional --title flag support
        title = None
        sources = []
        i = 1
        while i < len(args):
            if args[i] == "--title" and i + 1 < len(args):
                title = args[i + 1]
                i += 2
            else:
                sources.append(args[i])
                i += 1
        cmd_review(sources, title=title)

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        print_usage()
        sys.exit(1)
