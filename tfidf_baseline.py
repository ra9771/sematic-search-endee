"""
TF-IDF Baseline Model
Classic ML baseline using TF-IDF vectorization + cosine similarity.
Used to benchmark against the neural (Endee + sentence-transformers) approach.
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TFIDFSearchEngine:
    """
    Keyword-based search using TF-IDF + cosine similarity.
    Serves as the ML baseline to compare against neural semantic search.
    """

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,       # unigrams + bigrams
            stop_words="english",
            sublinear_tf=True,             # log(1 + tf) damping
        )
        self.doc_matrix = None
        self.documents = []
        self.is_fitted = False

    def fit(self, documents: list[dict]):
        """
        Fit TF-IDF on a corpus and store document vectors.

        Args:
            documents: List of dicts with 'text', 'title', 'category', 'url' keys.
        """
        self.documents = documents
        texts = [d["text"] for d in documents]
        self.doc_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        logger.info(f"TF-IDF fitted on {len(texts)} docs | vocab size: {len(self.vectorizer.vocabulary_)}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search using TF-IDF cosine similarity.

        Returns:
            List of result dicts with score and document metadata.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "doc_id": self.documents[idx].get("doc_id", str(idx)),
                    "score": float(scores[idx]),
                    "title": self.documents[idx].get("title", ""),
                    "text": self.documents[idx].get("text", ""),
                    "category": self.documents[idx].get("category", ""),
                    "url": self.documents[idx].get("url", ""),
                })
        return results

    def save(self, path: str):
        """Persist the fitted model to disk."""
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "doc_matrix": self.doc_matrix, "documents": self.documents}, f)
        logger.info(f"TF-IDF model saved to {path}")

    def load(self, path: str):
        """Load a previously saved model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.doc_matrix = data["doc_matrix"]
        self.documents = data["documents"]
        self.is_fitted = True
        logger.info(f"TF-IDF model loaded from {path}")
