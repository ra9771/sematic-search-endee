"""
Endee Vector Database Client
Wraps the Endee HTTP API for index management and vector operations.
"""

import requests
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EndeeClient:
    """Client for interacting with the Endee vector database HTTP API."""

    def __init__(self, base_url: str = "http://localhost:8080", auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = auth_token

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _get(self, path: str) -> dict:
        resp = requests.get(self._url(path), headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(self._url(path), headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> dict:
        resp = requests.delete(self._url(path), headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    # ── Index Management ──────────────────────────────────────────────────────

    def list_indexes(self) -> list:
        """Return all existing indexes."""
        return self._get("/api/v1/index/list").get("indexes", [])

    def create_index(self, name: str, dimension: int, metric: str = "cosine") -> dict:
        """
        Create a new vector index.

        Args:
            name: Index name (alphanumeric + underscores).
            dimension: Embedding vector dimension (e.g. 384 for all-MiniLM-L6-v2).
            metric: Distance metric — 'cosine', 'l2', or 'dot'.
        """
        payload = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
        }
        return self._post("/api/v1/index/create", payload)

    def delete_index(self, name: str) -> dict:
        """Delete an existing index."""
        return self._delete(f"/api/v1/index/{name}")

    def index_info(self, name: str) -> dict:
        """Get metadata about an index."""
        return self._get(f"/api/v1/index/{name}/info")

    # ── Vector Operations ─────────────────────────────────────────────────────

    def upsert_vectors(self, index_name: str, vectors: list[dict]) -> dict:
        """
        Upsert vectors into an index.

        Each vector dict should contain:
            id       (str)   : Unique identifier
            values   (list)  : Float embedding list
            payload  (dict)  : Arbitrary metadata (title, text, url, etc.)
        """
        payload = {"vectors": vectors}
        return self._post(f"/api/v1/index/{index_name}/upsert", payload)

    def search(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for the top-k nearest vectors.

        Args:
            index_name:    Target index.
            query_vector:  Embedding of the query.
            top_k:         Number of results to return.
            filters:       Optional payload filter dict (see Endee filter docs).
        """
        payload: dict = {"vector": query_vector, "top_k": top_k}
        if filters:
            payload["filter"] = filters
        result = self._post(f"/api/v1/index/{index_name}/search", payload)
        return result.get("results", [])

    def delete_vectors(self, index_name: str, ids: list[str]) -> dict:
        """Delete specific vectors by ID."""
        payload = {"ids": ids}
        return self._post(f"/api/v1/index/{index_name}/delete", payload)

    def health(self) -> dict:
        """Check server health."""
        return self._get("/api/v1/health")
