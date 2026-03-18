"""
Semantic Search Engine
Core module that ties together embedding generation and Endee vector search.
"""

import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

from src.endee_client import EndeeClient
from src.embeddings import EmbeddingGenerator, EMBEDDING_DIM

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document to be indexed."""
    text: str
    title: str = ""
    url: str = ""
    category: str = ""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_payload(self) -> dict:
        return {
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "category": self.category,
        }


@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    doc_id: str
    score: float
    title: str
    text: str
    url: str
    category: str

    def __str__(self) -> str:
        snippet = self.text[:200] + "..." if len(self.text) > 200 else self.text
        return (
            f"[Score: {self.score:.4f}] {self.title}\n"
            f"  Category : {self.category}\n"
            f"  URL      : {self.url or 'N/A'}\n"
            f"  Snippet  : {snippet}\n"
        )


class SemanticSearchEngine:
    """
    End-to-end semantic search engine backed by the Endee vector database.

    Workflow:
        1. index_documents() — embeds and stores documents in Endee.
        2. search()          — embeds a query and retrieves the top-k matches.
    """

    def __init__(
        self,
        index_name: str = "semantic_search",
        endee_url: str = "http://localhost:8080",
        auth_token: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_name = index_name
        self.client = EndeeClient(base_url=endee_url, auth_token=auth_token)
        self.embedder = EmbeddingGenerator(model_name=model_name)
        self._ensure_index()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _ensure_index(self):
        """Create the Endee index if it does not already exist."""
        existing = [idx.get("name") for idx in self.client.list_indexes()]
        if self.index_name not in existing:
            logger.info(f"Creating Endee index '{self.index_name}' (dim={EMBEDDING_DIM}, metric=cosine)")
            self.client.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
            )
        else:
            logger.info(f"Index '{self.index_name}' already exists — skipping creation.")

    def delete_index(self):
        """Permanently remove the index and all its vectors."""
        self.client.delete_index(self.index_name)
        logger.info(f"Deleted index '{self.index_name}'.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_documents(self, documents: list[Document], batch_size: int = 64) -> int:
        """
        Embed and upsert a list of documents into Endee.

        Args:
            documents:  List of Document objects.
            batch_size: Embedding batch size (controls GPU/CPU memory usage).

        Returns:
            Number of documents indexed.
        """
        if not documents:
            logger.warning("No documents provided.")
            return 0

        # Embed all document texts
        texts = [doc.text for doc in documents]
        embeddings = self.embedder.embed_batch(texts, batch_size=batch_size)

        # Build vector dicts for Endee upsert API
        vectors = [
            {
                "id": doc.doc_id,
                "values": embedding,
                "payload": doc.to_payload(),
            }
            for doc, embedding in zip(documents, embeddings)
        ]

        # Upsert in chunks of 500 (safe for Endee's batch endpoint)
        chunk_size = 500
        total = 0
        for i in range(0, len(vectors), chunk_size):
            chunk = vectors[i : i + chunk_size]
            self.client.upsert_vectors(self.index_name, chunk)
            total += len(chunk)
            logger.info(f"Upserted {total}/{len(vectors)} vectors.")

        return total

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Run a semantic search against the indexed corpus.

        Args:
            query:           Natural language query string.
            top_k:           Number of top results to return.
            category_filter: Optional metadata filter on the 'category' field.

        Returns:
            List of SearchResult objects ranked by cosine similarity.
        """
        query_embedding = self.embedder.embed(query)

        filters = None
        if category_filter:
            filters = {"must": [{"key": "category", "match": {"value": category_filter}}]}

        raw_results = self.client.search(
            index_name=self.index_name,
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        results = []
        for r in raw_results:
            payload = r.get("payload", {})
            results.append(
                SearchResult(
                    doc_id=r.get("id", ""),
                    score=r.get("score", 0.0),
                    title=payload.get("title", ""),
                    text=payload.get("text", ""),
                    url=payload.get("url", ""),
                    category=payload.get("category", ""),
                )
            )

        return results

    # ── Info ──────────────────────────────────────────────────────────────────

    def index_info(self) -> dict:
        return self.client.index_info(self.index_name)

    def health_check(self) -> dict:
        return self.client.health()
