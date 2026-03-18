"""
Semantic Search with Endee Vector Database
==========================================
Main entrypoint — supports two modes:

    python main.py index   : Load & index the sample corpus into Endee.
    python main.py search  : Interactive search prompt against indexed corpus.
    python main.py demo    : End-to-end demo (index + search).
"""

import sys
import logging
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from src.search_engine import SemanticSearchEngine
from src.data_loader import load_sample_documents, load_from_json, load_from_csv
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_engine(cfg: Config) -> SemanticSearchEngine:
    return SemanticSearchEngine(
        index_name=cfg.INDEX_NAME,
        endee_url=cfg.ENDEE_URL,
        auth_token=cfg.AUTH_TOKEN,
        model_name=cfg.EMBEDDING_MODEL,
    )


def print_results(results, query: str):
    """Pretty-print search results using Rich."""
    console.print(f"\n[bold cyan]Query:[/bold cyan] [italic]{query}[/italic]\n")

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(
        title=f"Top {len(results)} Semantic Search Results",
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="bold green", width=7)
    table.add_column("Title", style="bold white", width=35)
    table.add_column("Category", style="yellow", width=20)
    table.add_column("Snippet", style="dim white", width=60)

    for i, r in enumerate(results, 1):
        snippet = r.text[:120] + "…" if len(r.text) > 120 else r.text
        table.add_row(
            str(i),
            f"{r.score:.4f}",
            r.title,
            r.category,
            snippet,
        )

    console.print(table)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_index(args, cfg: Config):
    """Index documents into Endee."""
    engine = build_engine(cfg)

    if args.json:
        docs = load_from_json(args.json)
    elif args.csv:
        docs = load_from_csv(args.csv)
    else:
        docs = load_sample_documents()

    console.print(f"\n[bold]Indexing [green]{len(docs)}[/green] documents into Endee...[/bold]\n")
    count = engine.index_documents(docs)
    console.print(f"\n[bold green]✓ Successfully indexed {count} documents.[/bold green]\n")


def cmd_search(args, cfg: Config):
    """Interactive search loop."""
    engine = build_engine(cfg)

    console.print(
        Panel.fit(
            "[bold cyan]Semantic Search Engine[/bold cyan]\n"
            "Powered by [bold]Endee Vector Database[/bold] + [bold]all-MiniLM-L6-v2[/bold]\n"
            "Type [bold]'q'[/bold] or [bold]'exit'[/bold] to quit.",
            title="🔍 Endee Semantic Search",
            border_style="cyan",
        )
    )

    while True:
        try:
            query = console.input("\n[bold yellow]Enter query > [/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if query.lower() in ("q", "quit", "exit", ""):
            console.print("[dim]Goodbye![/dim]")
            break

        results = engine.search(
            query=query,
            top_k=cfg.DEFAULT_TOP_K,
            category_filter=args.category or None,
        )
        print_results(results, query)


def cmd_demo(args, cfg: Config):
    """Full end-to-end demo: index sample corpus then run preset queries."""
    engine = build_engine(cfg)

    # Index
    docs = load_sample_documents()
    console.print(f"\n[bold]Step 1/2 — Indexing {len(docs)} sample documents...[/bold]")
    engine.index_documents(docs)
    console.print("[green]✓ Documents indexed.[/green]\n")

    # Search
    demo_queries = [
        "How does machine learning work?",
        "What is a vector database?",
        "Explain the Transformer architecture",
        "Best tools for data science in Python",
        "How does semantic search differ from keyword search?",
    ]

    console.print("[bold]Step 2/2 — Running demo queries...[/bold]\n")
    for q in demo_queries:
        results = engine.search(q, top_k=3)
        print_results(results, q)
        console.print("─" * 80)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Semantic Search powered by Endee Vector Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # index subcommand
    p_index = sub.add_parser("index", help="Index documents into Endee")
    p_index.add_argument("--json", help="Path to a JSON document file")
    p_index.add_argument("--csv", help="Path to a CSV document file")

    # search subcommand
    p_search = sub.add_parser("search", help="Interactive search against Endee")
    p_search.add_argument("--category", help="Filter results by category")

    # demo subcommand
    sub.add_parser("demo", help="End-to-end demo (index + search)")

    args = parser.parse_args()
    cfg = Config()

    if args.command == "index":
        cmd_index(args, cfg)
    elif args.command == "search":
        cmd_search(args, cfg)
    elif args.command == "demo":
        cmd_demo(args, cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
