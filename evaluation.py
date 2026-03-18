"""
Evaluation Metrics for Information Retrieval
Computes standard IR metrics to compare TF-IDF baseline vs Neural semantic search.

Metrics implemented:
  - Precision@K   : fraction of top-K results that are relevant
  - Recall@K      : fraction of all relevant docs found in top-K
  - NDCG@K        : Normalised Discounted Cumulative Gain (rank-aware)
  - MRR           : Mean Reciprocal Rank (position of first relevant result)
  - Average Precision (AP) : area under precision-recall curve
"""

import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Ground truth + retrieved results for a single query."""
    query: str
    relevant_ids: set          # Ground truth relevant document IDs
    retrieved_ids: list[str]   # Ordered list of retrieved doc IDs


@dataclass
class MetricReport:
    """Evaluation report for a retrieval system."""
    system_name: str
    precision_at_k: dict = field(default_factory=dict)   # {k: score}
    recall_at_k: dict = field(default_factory=dict)
    ndcg_at_k: dict = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0

    def summary(self) -> str:
        lines = [f"\n{'='*55}", f"  System: {self.system_name}", f"{'='*55}"]
        for k in sorted(self.precision_at_k):
            lines.append(f"  Precision@{k:<3} : {self.precision_at_k[k]:.4f}")
        for k in sorted(self.recall_at_k):
            lines.append(f"  Recall@{k:<5} : {self.recall_at_k[k]:.4f}")
        for k in sorted(self.ndcg_at_k):
            lines.append(f"  NDCG@{k:<7} : {self.ndcg_at_k[k]:.4f}")
        lines.append(f"  MRR        : {self.mrr:.4f}")
        lines.append(f"  MAP        : {self.map_score:.4f}")
        lines.append(f"{'='*55}")
        return "\n".join(lines)


# ── Metric Functions ──────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    """Fraction of top-K retrieved docs that are relevant."""
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    """Fraction of all relevant docs that appear in top-K results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def dcg_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    """Discounted Cumulative Gain at K."""
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            score += 1.0 / math.log2(i + 1)
    return score


def ndcg_at_k(retrieved: list[str], relevant: set, k: int) -> float:
    """Normalised DCG at K. Ideal DCG assumes all relevant docs at top positions."""
    actual_dcg = dcg_at_k(retrieved, relevant, k)
    # Ideal: place all relevant docs first
    ideal_retrieved = list(relevant) + [""] * max(0, k - len(relevant))
    ideal_dcg = dcg_at_k(ideal_retrieved, relevant, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set) -> float:
    """Reciprocal rank of the first relevant document."""
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set) -> float:
    """Average Precision: area under the precision-recall curve."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class SearchEvaluator:
    """
    Evaluates a retrieval system against labelled ground truth.

    Usage:
        evaluator = SearchEvaluator(ks=[1, 3, 5, 10])
        report = evaluator.evaluate(system_name="Neural", query_results=[...])
        print(report.summary())
    """

    def __init__(self, ks: list[int] = None):
        self.ks = ks or [1, 3, 5, 10]

    def evaluate(self, system_name: str, query_results: list[QueryResult]) -> MetricReport:
        """
        Compute all metrics averaged over the provided queries.

        Args:
            system_name:    Label for this retrieval system.
            query_results:  List of QueryResult objects (one per query).

        Returns:
            MetricReport with averaged metrics.
        """
        report = MetricReport(system_name=system_name)

        p_sums = {k: 0.0 for k in self.ks}
        r_sums = {k: 0.0 for k in self.ks}
        n_sums = {k: 0.0 for k in self.ks}
        mrr_sum = 0.0
        ap_sum = 0.0
        n = len(query_results)

        for qr in query_results:
            for k in self.ks:
                p_sums[k] += precision_at_k(qr.retrieved_ids, qr.relevant_ids, k)
                r_sums[k] += recall_at_k(qr.retrieved_ids, qr.relevant_ids, k)
                n_sums[k] += ndcg_at_k(qr.retrieved_ids, qr.relevant_ids, k)
            mrr_sum += reciprocal_rank(qr.retrieved_ids, qr.relevant_ids)
            ap_sum += average_precision(qr.retrieved_ids, qr.relevant_ids)

        if n > 0:
            report.precision_at_k = {k: p_sums[k] / n for k in self.ks}
            report.recall_at_k = {k: r_sums[k] / n for k in self.ks}
            report.ndcg_at_k = {k: n_sums[k] / n for k in self.ks}
            report.mrr = mrr_sum / n
            report.map_score = ap_sum / n

        return report

    def compare(self, reports: list[MetricReport], k: int = 5) -> str:
        """
        Print a side-by-side comparison table of multiple systems.

        Args:
            reports: List of MetricReport objects (one per system).
            k:       K value to use for comparison.
        """
        lines = [
            f"\n{'─'*65}",
            f"  {'Metric':<18} " + "  ".join(f"{r.system_name:<14}" for r in reports),
            f"{'─'*65}",
        ]
        metrics = [
            (f"Precision@{k}", lambda r: r.precision_at_k.get(k, 0)),
            (f"Recall@{k}", lambda r: r.recall_at_k.get(k, 0)),
            (f"NDCG@{k}", lambda r: r.ndcg_at_k.get(k, 0)),
            ("MRR", lambda r: r.mrr),
            ("MAP", lambda r: r.map_score),
        ]
        for name, fn in metrics:
            row = f"  {name:<18} " + "  ".join(f"{fn(r):<14.4f}" for r in reports)
            lines.append(row)
        lines.append(f"{'─'*65}")
        return "\n".join(lines)
