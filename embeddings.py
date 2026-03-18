"""
Embedding Generator
Converts raw text into dense vector embeddings using sentence-transformers.
Model: all-MiniLM-L6-v2  (384-dim, fast, high quality for semantic search)
"""

import logging
from typing import Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model — lightweight yet accurate for semantic search tasks
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingGenerator:
    """Generates sentence embeddings using a local HuggingFace model."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = EMBEDDING_DIM
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed(self, text: Union[str, list[str]]) -> list:
        """
        Generate embeddings for one or more texts.

        Args:
            text: A single string or a list of strings.

        Returns:
            A single embedding (list of floats) or a list of embeddings.
        """
        single = isinstance(text, str)
        texts = [text] if single else text

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # Normalise for cosine similarity
            show_progress_bar=False,
        )

        result = [emb.tolist() for emb in embeddings]
        return result[0] if single else result

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Embed a large corpus in batches, with a progress bar.

        Args:
            texts:      List of documents to embed.
            batch_size: Number of texts per forward pass.

        Returns:
            List of float lists, one per input text.
        """
        logger.info(f"Embedding {len(texts)} documents (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [emb.tolist() for emb in embeddings]
