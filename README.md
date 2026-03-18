# 🔍 Semantic Search Engine — Powered by Endee Vector Database

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Endee](https://img.shields.io/badge/Vector%20DB-Endee-purple.svg)](https://endee.io)
[![Model](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-orange.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

A production-ready **Semantic Search Engine** built on the [Endee Vector Database](https://endee.io). Unlike keyword-based search, this system understands the *meaning* behind queries — returning results based on conceptual similarity rather than exact word matches.

---

## 📌 Overview

Traditional search systems fail when the user phrases a query differently from how documents are written. Semantic search solves this by:

1. **Embedding** every document into a high-dimensional vector space using a pretrained language model.
2. **Storing** those embeddings in Endee — a high-performance vector database optimised for fast ANN (Approximate Nearest Neighbour) retrieval.
3. **Embedding** the user's query at search time and retrieving the most similar document vectors via cosine similarity.

This project demonstrates a full end-to-end pipeline from raw text ingestion to interactive query — all in Python.

---

## 🏗️ System Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                          User Query                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────┐
               │     Query Expansion       │  ← WordNet synonyms (NLTK)
               │  "ML" → "ML machine AI"   │     improves recall
               └──────────────┬────────────┘
                              │ expanded query
                              ▼
               ┌───────────────────────────┐
               │   Bi-Encoder Embedding    │  ← all-MiniLM-L6-v2
               │  (sentence-transformers)  │     384-dim, normalised
               └──────────────┬────────────┘
                              │ query vector
                              ▼
               ┌───────────────────────────┐
               │    Endee Vector DB        │  ← HTTP API on :8080
               │  ANN / cosine similarity  │     HNSW index
               └──────────────┬────────────┘
                              │ top-K candidates
                              ▼
               ┌───────────────────────────┐
               │  Cross-Encoder Re-Ranker  │  ← ms-marco-MiniLM-L-6-v2
               │  scores (query, doc) pair │     higher precision
               └──────────────┬────────────┘
                              │ re-ranked results
                              ▼
               ┌───────────────────────────┐
               │  Final Ranked Results     │
               │  + Metadata Payload       │
               └───────────────────────────┘
```

### ML Pipeline — 4 Stages

| Stage | Module | ML Technique |
|---|---|---|
| Query Expansion | `src/query_expansion.py` | WordNet synonym graph (NLP) |
| Bi-Encoder Retrieval | `src/embeddings.py` + Endee | Transformer embeddings + ANN |
| Cross-Encoder Re-ranking | `src/reranker.py` | Fine-tuned MS-MARCO cross-encoder |
| Evaluation | `src/evaluation.py` | Precision@K, Recall@K, NDCG, MRR, MAP |

### All Modules

| Module | Responsibility |
|---|---|
| `src/endee_client.py` | Thin HTTP client wrapping every Endee REST endpoint |
| `src/embeddings.py` | Loads `all-MiniLM-L6-v2` and batches text → vectors |
| `src/search_engine.py` | Orchestrates indexing + search; exposes `Document` / `SearchResult` dataclasses |
| `src/tfidf_baseline.py` | TF-IDF + cosine similarity baseline (scikit-learn) |
| `src/query_expansion.py` | Expands queries with WordNet synonyms to improve recall |
| `src/reranker.py` | Cross-encoder re-ranks top-K candidates for higher precision |
| `src/evaluation.py` | Precision@K, Recall@K, NDCG@K, MRR, MAP metrics |
| `src/data_loader.py` | Loads docs from JSON, CSV, or the built-in 10-doc sample corpus |
| `config.py` | Centralised env-variable–backed configuration |
| `main.py` | CLI with `index`, `search`, and `demo` subcommands |
| `benchmark.py` | Compares TF-IDF vs Neural search with full evaluation metrics |
| `tests/` | Offline unit tests with full mocking (no Endee server needed) |

---

## ⚙️ Setup

### Prerequisites

- Python 3.11+
- Docker (to run Endee)

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/semantic-search-endee.git
cd semantic-search-endee
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Endee Vector Database

```bash
docker compose up -d
```

This pulls `endeeio/endee-server:latest` and exposes it on `http://localhost:8080`. Data is persisted in a Docker volume (`endee-data`).

Verify it is running:

```bash
curl http://localhost:8080/api/v1/health
```

---

## 🚀 Usage

### Demo mode (quickest start)

```bash
python main.py demo
```

Indexes 10 sample documents covering ML, NLP, vector databases, DevOps, and software engineering, then runs 5 preset queries against them.

### Index documents

```bash
# Index built-in sample corpus (default)
python main.py index

# Index from a JSON file
python main.py index --json data/my_docs.json

# Index from a CSV file
python main.py index --csv data/my_docs.csv
```

**JSON format:**
```json
[
  {"title": "Article Title", "text": "Full document text...", "category": "Tech", "url": "https://..."},
  ...
]
```

**CSV format:** Must have a `text` column. Optional: `title`, `category`, `url`.

### Run the ML Benchmark (TF-IDF vs Neural)

```bash
python benchmark.py
```

This evaluates both systems on 8 semantic queries and prints **Precision@K, Recall@K, NDCG@K, MRR, and MAP** side by side — demonstrating why neural embeddings outperform keyword matching.

```bash
python main.py search

# Filter results by category
python main.py search --category "AI/ML"
```

---

## 🔬 How Semantic Search Works

### Step 1 — Embedding (offline, during indexing)

Each document text is passed through `all-MiniLM-L6-v2`, producing a **384-dimensional float vector**. Vectors are L2-normalised so that cosine similarity equals the dot product — enabling fast inner-product search in Endee.

### Step 2 — Upsert into Endee

Vectors are sent to Endee's `POST /api/v1/index/{name}/upsert` endpoint along with a metadata **payload** (title, URL, category). Endee builds an HNSW index internally for sub-millisecond ANN queries.

### Step 3 — Query (online, at search time)

1. The query string is embedded with the **same model** (critical — query and documents must share the embedding space).
2. Endee's `POST /api/v1/index/{name}/search` returns the top-k vectors by cosine similarity.
3. Each result includes the similarity **score** and the stored **payload**, which is formatted and displayed.

### Why Cosine Similarity?

Cosine similarity measures the angle between vectors regardless of their magnitude. A score of **1.0** = identical direction (perfect match), **0.0** = orthogonal (unrelated). This is robust to document length differences.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

All tests mock the Endee client and embedding model — **no Docker required** to run the test suite.

```
tests/test_search_engine.py::TestEndeeClient::test_index_creation_called     PASSED
tests/test_search_engine.py::TestEndeeClient::test_upsert_called_on_indexing PASSED
tests/test_search_engine.py::TestSemanticSearchEngine::test_search_...       PASSED
...
```

---

## 📁 Project Structure

```
semantic-search-endee/
├── src/
│   ├── __init__.py
│   ├── endee_client.py      # Endee HTTP API client
│   ├── embeddings.py        # Sentence embedding generator
│   ├── search_engine.py     # Core search engine
│   └── data_loader.py       # Document loaders (JSON / CSV / sample)
├── tests/
│   └── test_search_engine.py
├── data/                    # Drop custom JSON/CSV docs here
├── docs/
├── config.py                # Centralised configuration
├── main.py                  # CLI entrypoint
├── requirements.txt
├── docker-compose.yml       # Endee server setup
└── README.md
```

---

## 🔧 Configuration

All settings can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `ENDEE_URL` | `http://localhost:8080` | Endee server base URL |
| `ENDEE_AUTH_TOKEN` | _(empty)_ | Auth token (leave empty for open mode) |
| `INDEX_NAME` | `semantic_search_docs` | Name of the Endee index |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `DEFAULT_TOP_K` | `5` | Default number of search results |

---

## 📊 Model Choice: all-MiniLM-L6-v2

| Property | Value |
|---|---|
| Embedding dimension | 384 |
| Model size | ~80 MB |
| Avg. encoding speed | ~14,000 sentences/s (CPU) |
| SBERT benchmark (STS) | 68.1 (strong for its size) |
| Licence | Apache 2.0 |

This model offers an excellent **quality-to-speed trade-off**, making it ideal for real-time semantic search without GPU requirements.

---

## 🔮 Potential Extensions

- **Hybrid Search** — combine dense (semantic) + sparse (BM25/TF-IDF) retrieval using Endee's sparse vector support.
- **RAG Pipeline** — plug retrieved results into an LLM (Claude, GPT-4, Llama) for question-answering.
- **FastAPI Web Service** — expose the search engine as a REST API endpoint.
- **Streaming Ingestion** — connect a Kafka consumer to index documents in real time.
- **Evaluation** — benchmark recall@k using BEIR or MS-MARCO datasets.

---

## 🤝 Acknowledgements

- [Endee Vector Database](https://endee.io) — high-performance open-source vector DB.
- [Sentence Transformers](https://www.sbert.net) — `all-MiniLM-L6-v2` embedding model.
- [Rich](https://github.com/Textualize/rich) — beautiful terminal output.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
