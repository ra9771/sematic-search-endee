"""
Unit Tests for Semantic Search Engine
Mocks both the Endee client and embedding generator for fast, offline testing.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.search_engine import SemanticSearchEngine, Document, SearchResult
from src.data_loader import load_sample_documents, load_from_json
from src.embeddings import EmbeddingGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_engine():
    """Create a SemanticSearchEngine with mocked Endee client and embedder."""
    with patch("src.search_engine.EndeeClient") as MockClient, \
         patch("src.search_engine.EmbeddingGenerator") as MockEmbedder:

        # Mock client
        client = MockClient.return_value
        client.list_indexes.return_value = []
        client.create_index.return_value = {"status": "ok"}
        client.upsert_vectors.return_value = {"status": "ok"}
        client.search.return_value = [
            {
                "id": "abc123",
                "score": 0.95,
                "payload": {
                    "title": "Test Document",
                    "text": "This is a test document about machine learning.",
                    "url": "https://example.com",
                    "category": "AI/ML",
                },
            }
        ]

        # Mock embedder
        embedder = MockEmbedder.return_value
        embedder.dimension = 384
        embedder.embed.return_value = [0.1] * 384
        embedder.embed_batch.return_value = [[0.1] * 384]

        engine = SemanticSearchEngine(
            index_name="test_index",
            endee_url="http://localhost:8080",
        )
        engine.client = client
        engine.embedder = embedder

        yield engine, client, embedder


# ── Endee Client Tests ────────────────────────────────────────────────────────

class TestEndeeClient:
    def test_index_creation_called(self, mock_engine):
        _, client, _ = mock_engine
        client.create_index.assert_called_once()

    def test_upsert_called_on_indexing(self, mock_engine):
        engine, client, embedder = mock_engine
        doc = Document(title="Test", text="Hello world", category="test")
        embedder.embed_batch.return_value = [[0.1] * 384]

        engine.index_documents([doc])
        client.upsert_vectors.assert_called_once()

    def test_search_called_with_correct_args(self, mock_engine):
        engine, client, embedder = mock_engine
        embedder.embed.return_value = [0.2] * 384

        engine.search("machine learning", top_k=3)
        client.search.assert_called_once_with(
            index_name="test_index",
            query_vector=[0.2] * 384,
            top_k=3,
            filters=None,
        )


# ── Search Engine Tests ───────────────────────────────────────────────────────

class TestSemanticSearchEngine:
    def test_search_returns_search_results(self, mock_engine):
        engine, _, _ = mock_engine
        results = engine.search("what is machine learning?")
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_search_result_fields(self, mock_engine):
        engine, _, _ = mock_engine
        result = engine.search("test")[0]
        assert result.score == 0.95
        assert result.title == "Test Document"
        assert result.category == "AI/ML"

    def test_category_filter_passed_to_client(self, mock_engine):
        engine, client, _ = mock_engine
        engine.search("ml", top_k=5, category_filter="AI/ML")
        call_kwargs = client.search.call_args.kwargs
        assert call_kwargs["filters"] is not None
        assert call_kwargs["filters"]["must"][0]["match"]["value"] == "AI/ML"

    def test_no_filter_when_category_none(self, mock_engine):
        engine, client, _ = mock_engine
        engine.search("test", top_k=5, category_filter=None)
        call_kwargs = client.search.call_args.kwargs
        assert call_kwargs["filters"] is None

    def test_index_documents_batch(self, mock_engine):
        engine, client, embedder = mock_engine
        docs = [Document(title=f"Doc {i}", text=f"Text {i}") for i in range(10)]
        embedder.embed_batch.return_value = [[0.1] * 384] * 10

        count = engine.index_documents(docs)
        assert count == 10
        client.upsert_vectors.assert_called()

    def test_empty_document_list(self, mock_engine):
        engine, client, _ = mock_engine
        count = engine.index_documents([])
        assert count == 0
        client.upsert_vectors.assert_not_called()


# ── Data Loader Tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_sample_documents_loaded(self):
        docs = load_sample_documents()
        assert len(docs) == 10
        assert all(isinstance(d, Document) for d in docs)

    def test_sample_documents_have_text(self):
        docs = load_sample_documents()
        assert all(d.text for d in docs)

    def test_load_from_json(self, tmp_path):
        import json
        data = [
            {"title": "A", "text": "First document.", "category": "test"},
            {"title": "B", "text": "Second document.", "url": "https://b.com"},
        ]
        f = tmp_path / "docs.json"
        f.write_text(json.dumps(data))

        docs = load_from_json(str(f))
        assert len(docs) == 2
        assert docs[0].title == "A"
        assert docs[1].url == "https://b.com"

    def test_load_from_json_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_from_json("/nonexistent/path/docs.json")

    def test_document_payload(self):
        doc = Document(
            title="T", text="Some text", url="https://x.com", category="cat"
        )
        payload = doc.to_payload()
        assert payload["title"] == "T"
        assert payload["category"] == "cat"


# ── Embedding Tests ───────────────────────────────────────────────────────────

class TestEmbeddingGenerator:
    @pytest.fixture(autouse=True)
    def mock_st(self):
        """Mock sentence_transformers so tests run without downloading models."""
        with patch("src.embeddings.SentenceTransformer") as MockST:
            import numpy as np
            instance = MockST.return_value
            instance.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
            yield MockST

    def test_embed_single_returns_list(self):
        gen = EmbeddingGenerator()
        result = gen.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == 384

    def test_embed_batch_returns_list_of_lists(self):
        gen = EmbeddingGenerator()
        result = gen.embed_batch(["text one", "text two"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, list) for r in result)
