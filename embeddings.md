# Pattern: Semantic Search & Retrieval-Augmented Generation (RAG)

## Use Case
Implementing semantic search over large, unstructured document sets to provide relevant context to an LLM for reasoning, analysis, or question answering.

## Technical Stack (One Reference Implementation)

Many valid combinations exist. The stack below was chosen for ease of local setup
and a free-tier embedding API. Swap any layer independently.

- **Embedding Provider**: Google Gemini (`models/gemini-embedding-001`) — alternatives: OpenAI `text-embedding-3-small`, local `all-MiniLM-L6-v2` (see memory-layer.md)
- **Vector Database**: ChromaDB (persistent, file-backed) — alternatives: pgvector, Pinecone, Weaviate, sqlite-vec
- **Orchestration**: LangChain — alternatives: direct SDK calls (removes the abstraction layer)

## Strategic Lessons & Best Practices

### 1. Model Selection: Stable vs. Preview
- **Recommendation**: Prioritize stable models (e.g., `v1`) for production-grade reliability. 
- **Learning**: High-dimensional preview models (e.g., `gemini-embedding-2`) may offer superior semantic density but can suffer from instability (`503/504` errors) and unpredictable rate limits during high-demand periods.

### 2. Handling API Client Discrepancies
- **Batching Issues**: Standard library methods for `embed_documents` (batch processing) occasionally return fewer vectors than input strings due to silent failures in upstream SDKs.
- **Fallback Pattern**: If batching results in `IndexError` or mismatched counts, implement a sequential wrapper (looping over `embed_query`) to ensure 1:1 mapping between chunks and embeddings.

### 3. Chunking Strategy

Semantic search effectiveness is bounded by the quality of the chunks. Choose a strategy based on your document type.

#### Structural Chunking (default)

Split at explicit document boundaries using regex. Fast, deterministic, and easy to debug.

```python
import re

def chunk_by_structure(text: str) -> list[str]:
    # Split on Markdown headers, numbered sections, or any repeating delimiter
    chunks = re.split(r'\n(?=#{1,3} |\d+\.\s)', text)
    return [c.strip() for c in chunks if c.strip()]
```

**Use when:**
- Documents have reliable formatting: legal contracts (Article 1, Section 2.a), API docs, technical specs, Markdown wikis.
- You need reproducible, auditable splits — the same document always produces the same chunks.
- Speed matters: a 1,000-page corpus can be chunked in milliseconds.

**Avoid when:** source material has inconsistent or no structure — transcripts, emails, web-scraped HTML, support tickets, chat logs.

---

#### Semantic Chunking

Embed every sentence, then split where the cosine similarity between adjacent sentences drops sharply — i.e., where the topic changes. Requires a second embedding pass just to chunk, before the retrieval embeddings are generated.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def chunk_by_semantics(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.3,   # lower = more splits; tune per corpus
    min_chunk_words: int = 40,
) -> list[str]:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []

    embeddings = model.encode(sentences, normalize_embeddings=True)

    # Cosine distance between consecutive sentences (1 - dot product of unit vectors)
    distances = [
        1 - float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    chunks, current = [], [sentences[0]]
    for i, dist in enumerate(distances):
        if dist > threshold and len(" ".join(current).split()) >= min_chunk_words:
            chunks.append(". ".join(current) + ".")
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])

    if current:
        chunks.append(". ".join(current) + ".")
    return chunks
```

**Use when:**
- Documents are free-form prose with no consistent formatting: interview transcripts, customer support tickets, scraped articles, meeting notes, emails.
- A single document covers multiple distinct topics and you want retrieval to surface the right subtopic, not the whole document.
- OCR output (see `ocr.md`) where headers were not preserved during scanning.

**Avoid when:**
- Documents are short (< 500 words) — splitting adds noise without benefit.
- Chunks must map back to source page numbers or section headers for citations — semantic boundaries don't align with document structure.
- You're on a tight latency or cost budget — the chunking pass doubles your embedding calls.

**Tuning `threshold`:** start at 0.3 and inspect a sample. If chunks are too large (topics bleeding together), lower it. If chunks are too small (single sentences), raise it or increase `min_chunk_words`.

---

#### Metadata Alignment (both strategies)

Enrich every chunk with metadata regardless of chunking method. This is critical for generating accurate citations in the final LLM output.

```python
{"source": "contract_v2.pdf", "section": "Article 3", "page": 12, "chunk_index": 4}
```

### 4. Retrieval Strategy

Once chunks are indexed, choose how to query them. Hybrid retrieval is recommended for most applications.

#### Vector-Only Retrieval

Query the vector database by cosine similarity. Simple, no additional dependencies.

```python
results = collection.query(query_texts=[query], n_results=10)
```

**Use when:**
- Queries are conceptual or exploratory: "what does this contract say about liability?" — no obvious keywords to match.
- Your corpus is dense prose where meaning matters more than terminology.
- You need a simple baseline to establish before adding complexity.

**Avoid when:** users query with specific terms, names, or identifiers (product SKUs, error codes, legal citations) — vector search can miss exact-match results if the embedding space doesn't cluster them tightly.

---

#### Hybrid Retrieval (recommended)

Run keyword search (BM25) and vector search in parallel, then merge the ranked lists using Reciprocal Rank Fusion (RRF). Results that rank well in both systems score highest.

```python
from rank_bm25 import BM25Okapi

def reciprocal_rank_fusion(
    keyword_ids: list[str],
    vector_ids: list[str],
    k: int = 60,  # standard RRF constant; rarely needs tuning
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(keyword_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    chunks: list[str],
    chunk_ids: list[str],
    bm25: BM25Okapi,
    vector_collection,
    n_results: int = 10,
) -> list[str]:
    # 1. BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    keyword_ids = [
        chunk_ids[i]
        for i in sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_results]
    ]

    # 2. Vector similarity search
    vec_results = vector_collection.query(query_texts=[query], n_results=n_results)
    vector_ids = vec_results["ids"][0]

    # 3. Merge via RRF
    merged = reciprocal_rank_fusion(keyword_ids, vector_ids)
    return [doc_id for doc_id, _ in merged[:n_results]]
```

RRF is robust to score-scale differences between BM25 and cosine similarity — no normalization needed. The `k=60` constant is the standard default; lower values amplify top-rank results more aggressively.

**Use when:**
- Queries mix conceptual intent with specific terms: "how does the refund policy handle chargebacks?" — benefits from both keyword match on "chargeback" and semantic match on "refund policy."
- Users search with names, error codes, or identifiers that may not embed well.
- You want better recall overall — hybrid consistently outperforms either method alone across corpus types.

**Avoid when:**
- You cannot add a BM25 index to your stack (e.g., managed vector DB with no keyword search tier).
- Latency is critical and you cannot run both searches in parallel — hybrid adds one extra query per request.

**SQLite implementation:** `memory-layer.md` has a complete working implementation using FTS5 (SQLite's built-in BM25) + sqlite-vec, with no external services required.

---

### 5. Environment-Aware Pathing
- **The Absolute Path Mandate**: When deploying vector search as a service (e.g., via MCP, Docker, or Cloud Functions), relative paths to local databases (like `.chroma_db`) will cause "Read-only file system" or "Database not found" errors.
- **Solution**: Always resolve paths relative to the application's root directory at runtime using `os.path.abspath`.

## When to Implement This Pattern
Implement Semantic Search/Embeddings when:
- **Semantic Search**: You need to find information based on *meaning* or *intent* rather than exact keywords.
- **Scaling Knowledge**: The source material exceeds the LLM's context window.
- **Dynamic Context**: You need to retrieve precise subsets of data based on user intent rather than keyword matches.
- **Citation Requirements**: The application must prove its reasoning by pointing to specific source material.
- **Cost Efficiency**: You want to reduce token consumption by only sending relevant data to the LLM.
