"""
Configuration
All tuneable parameters in one place. Values can be overridden via environment variables.
"""

import os


class Config:
    # Endee server
    ENDEE_URL: str = os.getenv("ENDEE_URL", "http://localhost:8080")
    AUTH_TOKEN: str | None = os.getenv("ENDEE_AUTH_TOKEN", None)

    # Index settings
    INDEX_NAME: str = os.getenv("INDEX_NAME", "semantic_search_docs")

    # Embedding model (HuggingFace model ID)
    # all-MiniLM-L6-v2 : 384-dim, fast, excellent quality/speed trade-off
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Search defaults
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
