"""
ML Benchmark: TF-IDF Baseline vs Neural Semantic Search
=========================================================
Runs both retrieval systems on the same query set and compares:
  - Precision@K, Recall@K, NDCG@K, MRR, MAP

This script demonstrates the core ML contribution of the project:
showing *why* neural embeddings outperform keyword-based TF-IDF
on semantic / paraphrase queries.

Usage:
    python benchmark.py
"""

import logging
from rich.console import Console
from rich.panel import Panel

from data_loader import load_sample_documents
from tfidf_baseline import TFIDFSearchEngine
from query_expansion import QueryExpander
from evaluation import SearchEvaluator, QueryResult, MetricReport
from search_engine import SemanticSearchEngine
from config import Config

logging.basicConfig(level=logging.WARNING)   # Suppress info during benchmark
console = Console()

# ── Ground Truth ──────────────────────────────────────────────────────────────
# Manual relevance judgements for the sample corpus.
# Format: { query: [list of relevant document titles] }
GROUND_TRUTH = {
    "How does machine learning work?": [
        "Introduction to Machine Learning",
        "Deep Learning and Neural Networks",
    ],
    "What is a vector database used for?": [
        "Vector Databases Explained",
        "Cosine Similarity in Semantic Search",
    ],
    "Explain neural network architecture": [
        "Deep Learning and Neural Networks",
        "Transformers and Attention Mechanisms",
        "Introduction to Machine Learning",
    ],
    "Python libraries for data analysis": [
        "Python for Data Science",
    ],
    "How does RAG reduce hallucination?": [
        "Retrieval-Augmented Generation (RAG)",
        "Vector Databases Explained",
    ],
    "What is attention mechanism in AI?": [
        "Transformers and Attention Mechanisms",
        "Deep Learning and Neural Networks",
    ],
    "Container orchestration with Docker": [
        "Docker and Containerisation",
    ],
    "How to design a REST API?": [
        "REST API Design Best Practices",
    ],
}

QUERIES = list(GROUND_TRUTH.keys())


def title_to_id(title: str, documents: list) -> str:
    """Map document title back to its assigned doc_id."""
    for doc in documents:
        if doc.title == title:
            return doc.doc_id
    return title   # fallback


def run_tfidf_benchmark(documents, expander: QueryExpander, evaluator: SearchEvaluator, top_k: int = 5) -> MetricReport:
    """Run TF-IDF baseline with query expansion."""
    console.print("[bold]Running TF-IDF + Query Expansion baseline...[/bold]")

    # Fit TF-IDF on corpus
    engine = TFIDFSearchEngine()
    corpus = [{"doc_id": d.doc_id, "text": d.text, "title": d.title,
                "category": d.category, "url": d.url} for d in documents]
    engine.fit(corpus)

    query_results = []
    for query in QUERIES:
        # Expand query before TF-IDF search
        expanded_query = expander.expand(query)
        results = engine.search(expanded_query, top_k=top_k)
        retrieved_ids = [r["doc_id"] for r in results]

        relevant_titles = GROUND_TRUTH[query]
        relevant_ids = {title_to_id(t, documents) for t in relevant_titles}

        query_results.append(QueryResult(
            query=query,
            relevant_ids=relevant_ids,
            retrieved_ids=retrieved_ids,
        ))

    return evaluator.evaluate("TF-IDF + QueryExpansion", query_results)


def run_neural_benchmark(documents, expander: QueryExpander, evaluator: SearchEvaluator, top_k: int = 5) -> MetricReport:
    """Run neural semantic search (Endee + sentence-transformers)."""
    console.print("[bold]Running Neural Semantic Search (Endee + all-MiniLM-L6-v2)...[/bold]")

    cfg = Config()
    try:
        engine = SemanticSearchEngine(
            index_name="benchmark_index",
            endee_url=cfg.ENDEE_URL,
            auth_token=cfg.AUTH_TOKEN,
        )
        engine.index_documents(documents)

        query_results = []
        for query in QUERIES:
            results = engine.search(query, top_k=top_k)
            retrieved_ids = [r.doc_id for r in results]

            relevant_titles = GROUND_TRUTH[query]
            relevant_ids = {title_to_id(t, documents) for t in relevant_titles}

            query_results.append(QueryResult(
                query=query,
                relevant_ids=relevant_ids,
                retrieved_ids=retrieved_ids,
            ))

        # Clean up benchmark index
        engine.delete_index()
        return evaluator.evaluate("Neural (Endee)", query_results)

    except Exception as e:
        console.print(f"[red]Endee not reachable ({e}). Showing TF-IDF only.[/red]")
        return None


def main():
    console.print(Panel.fit(
        "[bold cyan]ML Benchmark[/bold cyan]\n"
        "TF-IDF + Query Expansion  vs  Neural Semantic Search\n"
        "Metrics: Precision@K · Recall@K · NDCG@K · MRR · MAP",
        title="📊 Evaluation Suite",
        border_style="cyan",
    ))

    documents = load_sample_documents()
    expander = QueryExpander()
    evaluator = SearchEvaluator(ks=[1, 3, 5])
    top_k = 5

    # Run both systems
    tfidf_report = run_tfidf_benchmark(documents, expander, evaluator, top_k)
    neural_report = run_neural_benchmark(documents, expander, evaluator, top_k)

    # Print individual reports
    console.print(tfidf_report.summary())
    if neural_report:
        console.print(neural_report.summary())

        # Side-by-side comparison
        console.print("\n[bold magenta]Side-by-Side Comparison (K=5)[/bold magenta]")
        console.print(evaluator.compare([tfidf_report, neural_report], k=5))

        # Insight
        ndcg_neural = neural_report.ndcg_at_k.get(5, 0)
        ndcg_tfidf = tfidf_report.ndcg_at_k.get(5, 0)
        delta = ndcg_neural - ndcg_tfidf
        console.print(
            f"\n[green]✓ Neural search improves NDCG@5 by "
            f"{delta:+.4f} over TF-IDF baseline.[/green]\n"
            if delta > 0 else
            f"\n[yellow]TF-IDF performed similarly or better on this small corpus.[/yellow]\n"
        )
    else:
        console.print("\n[yellow]Neural benchmark skipped (Endee not running).[/yellow]")
        console.print("[dim]Start Endee with: docker compose up -d[/dim]")


if __name__ == "__main__":
    main()
