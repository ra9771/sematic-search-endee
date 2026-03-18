"""
Cross-Encoder Re-Ranker
A second-stage ML model that re-scores (query, document) pairs for higher precision.

Pipeline:
  Stage 1 (Endee bi-encoder) → fast ANN retrieval, top-k candidates
  Stage 2 (Cross-encoder)    → slow but accurate re-ranking of candidates

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Fine-tuned on MS-MARCO passage ranking (Microsoft)
  - Outputs a relevance score for each (query, passage) pair
  - ~67 MB, runs fast on CPU
"""

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Re-ranks retrieval results using a cross-encoder model.
    More accurate than bi-encoder similarity because the model
    sees both query and document together (full attention).
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        logger.info(f"Loading cross-encoder re-ranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Cross-encoder loaded.")

    def rerank(self, query: str, results: list[dict], text_key: str = "text") -> list[dict]:
        """
        Re-rank a list of retrieved results.

        Args:
            query:    Original user query.
            results:  List of result dicts (must contain text_key field).
            text_key: Key in each result dict holding the document text.

        Returns:
            Results re-sorted by cross-encoder relevance score (descending).
        """
        if not results:
            return results

        # Build (query, document) pairs
        pairs = [(query, r[text_key]) for r in results]

        # Score all pairs in one forward pass
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach cross-encoder score and re-sort
        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)
            result["original_score"] = result.get("score", 0.0)
            result["score"] = float(score)   # overwrite with reranked score

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        logger.debug(f"Re-ranked {len(reranked)} results for query: '{query}'")
        return reranked
