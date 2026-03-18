"""
Data Loader
Utilities to load documents from various sources (JSON, CSV, plain text)
and from a built-in sample corpus for demo purposes.
"""

import csv
import json
import logging
from pathlib import Path

from search_engine import Document

logger = logging.getLogger(__name__)


# ── Built-in Sample Corpus ────────────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    Document(
        title="Introduction to Machine Learning",
        text=(
            "Machine learning is a subset of artificial intelligence that enables systems to learn "
            "and improve from experience without being explicitly programmed. It focuses on developing "
            "algorithms that can access data and use it to learn for themselves. Common approaches "
            "include supervised learning, unsupervised learning, and reinforcement learning."
        ),
        category="AI/ML",
        url="https://example.com/intro-ml",
    ),
    Document(
        title="Deep Learning and Neural Networks",
        text=(
            "Deep learning uses multi-layered neural networks to model complex patterns in data. "
            "Inspired by the human brain, these architectures excel at tasks like image recognition, "
            "natural language processing, and speech synthesis. Popular frameworks include TensorFlow, "
            "PyTorch, and JAX."
        ),
        category="AI/ML",
        url="https://example.com/deep-learning",
    ),
    Document(
        title="Vector Databases Explained",
        text=(
            "A vector database stores high-dimensional embeddings and enables efficient similarity search. "
            "Unlike traditional relational databases that match exact values, vector databases find "
            "semantically similar items using distance metrics such as cosine similarity or Euclidean "
            "distance. They power modern AI applications like semantic search, RAG, and recommendations."
        ),
        category="Databases",
        url="https://example.com/vector-databases",
    ),
    Document(
        title="Natural Language Processing Overview",
        text=(
            "Natural language processing (NLP) is a field of AI that gives computers the ability to "
            "understand, interpret, and generate human language. Key tasks include tokenization, "
            "named entity recognition, sentiment analysis, machine translation, and summarisation. "
            "Transformer models like BERT and GPT revolutionised NLP benchmarks."
        ),
        category="AI/ML",
        url="https://example.com/nlp-overview",
    ),
    Document(
        title="Python for Data Science",
        text=(
            "Python has become the dominant language for data science due to its readable syntax and "
            "rich ecosystem. Libraries such as NumPy, Pandas, Matplotlib, Scikit-learn, and Jupyter "
            "notebooks are indispensable tools for exploratory data analysis, feature engineering, "
            "and model training."
        ),
        category="Programming",
        url="https://example.com/python-data-science",
    ),
    Document(
        title="Retrieval-Augmented Generation (RAG)",
        text=(
            "RAG combines information retrieval with large language models to produce accurate, "
            "grounded answers. A retriever fetches relevant documents from a knowledge base, and the "
            "LLM synthesises them into a coherent response. This reduces hallucination and keeps "
            "answers up-to-date without expensive fine-tuning."
        ),
        category="AI/ML",
        url="https://example.com/rag",
    ),
    Document(
        title="Transformers and Attention Mechanisms",
        text=(
            "The Transformer architecture, introduced in 'Attention Is All You Need' (2017), relies "
            "on self-attention to capture long-range dependencies in sequences. It replaced recurrent "
            "models in most NLP tasks and later spread to vision (ViT), audio, and multimodal domains."
        ),
        category="AI/ML",
        url="https://example.com/transformers",
    ),
    Document(
        title="Docker and Containerisation",
        text=(
            "Docker packages applications and their dependencies into lightweight, portable containers. "
            "Containers share the host OS kernel, making them faster and more resource-efficient than "
            "virtual machines. Docker Compose orchestrates multi-container applications, while "
            "Kubernetes manages containers at scale in production."
        ),
        category="DevOps",
        url="https://example.com/docker",
    ),
    Document(
        title="REST API Design Best Practices",
        text=(
            "A well-designed REST API uses meaningful resource URLs, appropriate HTTP verbs (GET, POST, "
            "PUT, DELETE), consistent error codes, pagination for large collections, and versioning to "
            "ensure backward compatibility. Authentication is commonly handled via JWT tokens or OAuth 2.0."
        ),
        category="Software Engineering",
        url="https://example.com/rest-api",
    ),
    Document(
        title="Cosine Similarity in Semantic Search",
        text=(
            "Cosine similarity measures the angle between two vectors in a high-dimensional space. "
            "A score of 1 means the vectors are identical in direction, 0 means they are orthogonal. "
            "It is the standard distance metric for semantic search because embedding magnitudes are "
            "irrelevant — only the direction encodes meaning."
        ),
        category="Mathematics",
        url="https://example.com/cosine-similarity",
    ),
]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_sample_documents() -> list[Document]:
    """Return the built-in sample corpus (10 documents)."""
    logger.info(f"Loaded {len(SAMPLE_DOCUMENTS)} sample documents.")
    return SAMPLE_DOCUMENTS


def load_from_json(filepath: str) -> list[Document]:
    """
    Load documents from a JSON file.

    Expected format:
        [
          {"title": "...", "text": "...", "url": "...", "category": "..."},
          ...
        ]
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    docs = [
        Document(
            title=r.get("title", ""),
            text=r.get("text", r.get("content", "")),
            url=r.get("url", ""),
            category=r.get("category", "general"),
        )
        for r in records
        if r.get("text") or r.get("content")
    ]
    logger.info(f"Loaded {len(docs)} documents from {filepath}.")
    return docs


def load_from_csv(filepath: str, text_col: str = "text", title_col: str = "title") -> list[Document]:
    """
    Load documents from a CSV file.

    Args:
        filepath:  Path to the CSV file.
        text_col:  Column name containing document text.
        title_col: Column name containing document title.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_col, "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    title=row.get(title_col, ""),
                    text=text,
                    url=row.get("url", ""),
                    category=row.get("category", "general"),
                )
            )

    logger.info(f"Loaded {len(docs)} documents from {filepath}.")
    return docs
