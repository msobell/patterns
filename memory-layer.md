# Memory Layer — Implementation Guide

Hybrid local knowledge base for saving and retrieving notes across sessions.
Combines keyword search (FTS5/BM25) and semantic search (vector embeddings)
fused via Reciprocal Rank Fusion. No external APIs — everything runs locally.

For retrieval strategy context (vector-only vs. hybrid, when to use each, RRF
tradeoffs), see `embeddings.md` § Retrieval Strategy. This file is the SQLite
implementation of that pattern.

## Stack

- **SQLite** — main DB (via SQLAlchemy ORM)
- **FTS5** — SQLite built-in full-text search, trigram tokenizer
- **sqlite-vec** — SQLite extension for vector storage and cosine similarity
- **sentence-transformers** — local embedding model (`all-MiniLM-L6-v2`, 384 dims)

## Dependencies

```toml
# pyproject.toml
sqlite-vec = "*"
sentence-transformers = "*"
numpy = "*"
```

Or with pip:

```bash
pip install sqlite-vec sentence-transformers numpy
```

---

## File Structure

The paths below use `app/` as an example root — adapt to your project layout.

```
<project_root>/
  models/
    memory.py          # Memory + KnowledgeEdge ORM
  memory/
    db_setup.py        # ensure_virtual_tables() — creates FTS5 + vec0 virtual tables
    embeddings.py      # generate_embedding() — local all-MiniLM-L6-v2
    search.py          # hybrid_search() — BM25 + cosine fused via RRF
  mcp/
    memory_tools.py    # save_memory, query_memory, get_related_entities (optional)
```

---

## 1. ORM Models (`app/models/memory.py`)

The `Memory` table is both the note store and the knowledge graph node table.
Entity nodes (player names, exercise names, etc.) are just `Memory` rows with
`metadata_json={"type": "entity"}`.

```python
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    out_edges: Mapped[list[KnowledgeEdge]] = relationship(
        "KnowledgeEdge", foreign_keys="KnowledgeEdge.source_id",
        back_populates="source", cascade="all, delete-orphan",
    )
    in_edges: Mapped[list[KnowledgeEdge]] = relationship(
        "KnowledgeEdge", foreign_keys="KnowledgeEdge.target_id",
        back_populates="target", cascade="all, delete-orphan",
    )


class KnowledgeEdge(Base):
    __tablename__ = "knowledge_graph"

    source_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), primary_key=True)
    target_id: Mapped[int] = mapped_column(Integer, ForeignKey("memories.id"), primary_key=True)
    relationship_type: Mapped[str] = mapped_column(String, primary_key=True)

    source: Mapped[Memory] = relationship("Memory", foreign_keys=[source_id], back_populates="out_edges")
    target: Mapped[Memory] = relationship("Memory", foreign_keys=[target_id], back_populates="in_edges")
```

Register in `create_all_tables()`:

```python
import app.models.memory  # noqa: F401
```

**Do not** add the FTS5/vec0 virtual tables to `create_all_tables()` — they require
the sqlite-vec extension to be loaded first. See step 2.

---

## 2. Virtual Tables (`app/memory/db_setup.py`)

Virtual tables must be created **after** the sqlite-vec extension is loaded.
Call `ensure_virtual_tables()` before the first memory operation, not at startup.

```python
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def ensure_virtual_tables(session: Session):
    """Create FTS5 and vec0 virtual tables if they don't exist."""

    # BM25 keyword search — trigram tokenizer for substring matching
    session.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            content,
            id UNINDEXED,
            tokenize="trigram"
        )
    """))

    # Vector similarity search — 384 dims matches all-MiniLM-L6-v2
    session.execute(text("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
            id INTEGER PRIMARY KEY,
            embedding float[384]
        )
    """))

    session.commit()
    logger.info("Memory Layer virtual tables ensured.")
```

### Loading sqlite-vec

Load the extension in the SQLAlchemy engine connect event so it's available
for every connection:

```python
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

    try:
        import sqlite_vec
        dbapi_conn.enable_load_extension(True)
        sqlite_vec.load(dbapi_conn)
        dbapi_conn.enable_load_extension(False)
    except ImportError:
        logging.warning("sqlite-vec not installed, vector search disabled.")
    except Exception as e:
        logging.error(f"Failed to load sqlite-vec: {e}")
```

---

## 3. Embeddings (`app/memory/embeddings.py`)

Lazy-load the model on first use. `all-MiniLM-L6-v2` is small (~80MB),
fast, and produces good general-purpose 384-dim embeddings.

```python
import logging
from typing import List

logger = logging.getLogger(__name__)
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading all-MiniLM-L6-v2...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(text: str) -> List[float]:
    """Return a 384-dimensional embedding for the given text."""
    return _get_model().encode(text).tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Batch generate embeddings."""
    return _get_model().encode(texts).tolist()
```

---

## 4. Hybrid Search (`app/memory/search.py`)

Runs BM25 and cosine similarity in parallel, then merges with Reciprocal Rank
Fusion (RRF). RRF is robust to score-scale differences between the two systems —
no normalization needed.

```python
import logging
from typing import Dict, List, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.memory.embeddings import get_embedding
from app.models.memory import Memory

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    fts_ids: List[int],
    vec_ids: List[int],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Merge two ranked lists. k=60 is the standard RRF constant."""
    scores: Dict[int, float] = {}
    for rank, doc_id in enumerate(fts_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(vec_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(session: Session, query: str, n_results: int = 10) -> List[Memory]:
    """BM25 + cosine similarity, fused via RRF."""

    # 1. FTS5 — BM25 keyword search
    fts_rows = session.execute(
        text("SELECT id FROM memories_fts WHERE content MATCH :q LIMIT :n"),
        {"q": query, "n": n_results},
    ).fetchall()
    fts_ids = [row[0] for row in fts_rows]

    # 2. Vector — cosine similarity
    vec_ids = []
    try:
        import sqlite_vec
        query_blob = sqlite_vec.serialize_float32(get_embedding(query))
        vec_rows = session.execute(
            text("""
                SELECT id FROM memories_vec
                ORDER BY vec_distance_cosine(embedding, :qv)
                LIMIT :n
            """),
            {"qv": query_blob, "n": n_results},
        ).fetchall()
        vec_ids = [row[0] for row in vec_rows]
    except Exception as e:
        logger.warning(f"Vector search failed, falling back to FTS only: {e}")

    # 3. RRF merge
    merged = reciprocal_rank_fusion(fts_ids, vec_ids)
    top_ids = [doc_id for doc_id, _ in merged[:n_results]]

    if not top_ids:
        return []

    memories = session.query(Memory).filter(Memory.id.in_(top_ids)).all()
    memory_map = {m.id: m for m in memories}
    return [memory_map[mid] for mid in top_ids if mid in memory_map]
```

---

## 5. MCP Tools (`app/mcp/memory_tools.py`)

Three tools exposed via the MCP server. Adapt the `_get_session()` import
to however your app provides a DB session.

```python
from __future__ import annotations
import json
import logging
from typing import List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.memory.db_setup import ensure_virtual_tables
from app.memory.embeddings import get_embedding
from app.memory.search import hybrid_search
from app.models.memory import KnowledgeEdge, Memory

logger = logging.getLogger(__name__)


def save_memory(content: str, entities: List[str], metadata: Optional[dict] = None) -> str:
    """
    Save a note and link it to named entities.

    Writes to memories table, indexes in FTS5 and vec0,
    upserts entity nodes, and creates MENTIONS edges.
    """
    session = _get_session()
    try:
        ensure_virtual_tables(session)

        # 1. Main record
        memory = Memory(
            content=content,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        session.add(memory)
        session.flush()  # need memory.id

        # 2. FTS index
        session.execute(
            text("INSERT INTO memories_fts(content, id) VALUES(:c, :id)"),
            {"c": content, "id": memory.id},
        )

        # 3. Vector index
        import sqlite_vec
        blob = sqlite_vec.serialize_float32(get_embedding(content))
        session.execute(
            text("INSERT INTO memories_vec(id, embedding) VALUES(:id, :emb)"),
            {"id": memory.id, "emb": blob},
        )

        # 4. Entity nodes + edges
        for name in entities:
            entity = session.query(Memory).filter(Memory.content == name).first()
            if not entity:
                entity = Memory(content=name, metadata_json=json.dumps({"type": "entity"}))
                session.add(entity)
                session.flush()
                session.execute(
                    text("INSERT INTO memories_fts(content, id) VALUES(:c, :id)"),
                    {"c": name, "id": entity.id},
                )
            session.add(KnowledgeEdge(
                source_id=memory.id,
                target_id=entity.id,
                relationship_type="MENTIONS",
            ))

        session.commit()
        return f"Saved memory {memory.id}, linked to {len(entities)} entities."
    except Exception as e:
        session.rollback()
        return f"Error: {e}"
    finally:
        session.close()


def query_memory(query: str, n_results: int = 5) -> str:
    """Hybrid search across saved memories."""
    session = _get_session()
    try:
        ensure_virtual_tables(session)
        results = hybrid_search(session, query, n_results)
        if not results:
            return "No matching memories found."
        lines = ["### Results:"]
        for i, m in enumerate(results):
            lines.append(f"{i+1}. [ID:{m.id}] {m.content}")
            if i < 3:
                entities = [e.target.content for e in m.out_edges if e.relationship_type == "MENTIONS"]
                if entities:
                    lines.append(f"   Entities: {', '.join(entities)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"
    finally:
        session.close()


def get_related_entities(entity_name: str) -> str:
    """Walk the knowledge graph for all memories/entities linked to a name."""
    session = _get_session()
    try:
        entity = session.query(Memory).filter(Memory.content == entity_name).first()
        if not entity:
            results = hybrid_search(session, entity_name, n_results=1)
            if not results:
                return f"'{entity_name}' not found."
            entity = results[0]
            lines = [f"Closest match: '{entity.content}'"]
        else:
            lines = [f"Related to '{entity_name}':"]

        mentions = [e.source.content for e in entity.in_edges if e.relationship_type == "MENTIONS"]
        if mentions:
            lines.append("\n**Memories mentioning this:**")
            lines.extend(f"- {m}" for m in mentions)

        related = [e.target.content for e in entity.out_edges if e.relationship_type == "MENTIONS"]
        if related:
            lines.append("\n**Also linked to:**")
            lines.extend(f"- {r}" for r in related)

        return "\n".join(lines) if len(lines) > 1 else f"No relations found for '{entity_name}'."
    except Exception as e:
        return f"Error: {e}"
    finally:
        session.close()


def _get_session() -> Session:
    """
    Replace this with however your app provides a SQLAlchemy Session.
    Common patterns:
      - FastAPI dependency injection: inject Session directly into the tool function
      - Global engine: return SessionLocal() where SessionLocal = sessionmaker(bind=engine)
      - Flask-SQLAlchemy: return db.session
    """
    raise NotImplementedError("Wire _get_session() to your app's session factory.")
```

---

## 6. MCP Server Registration

In your `mcp_server.py`, register the three tools:

```python
from app.mcp.memory_tools import get_related_entities, query_memory, save_memory

mcp.tool()(save_memory)
mcp.tool()(query_memory)
mcp.tool()(get_related_entities)
```

---

## Known Limitations & Extension Points

**Entity deduplication is exact string match.** `"Bench Press"` and `"bench press"`
create two nodes. Normalize to lowercase (or use `normalize_name()`) before
looking up entities if your domain has inconsistent casing.

**Only one edge type (`MENTIONS`).** You could extend `KnowledgeEdge` with more
specific types: `TARGETS`, `CONTRADICTS`, `SUPERSEDES`, etc. The primary key
includes `relationship_type` so multiple edge types between the same two nodes
are allowed.

**No update or delete tools.** Add them if needed:
- Delete: `DELETE FROM memories WHERE id=?`, plus FTS/vec cleanup
- Update: delete + re-insert (FTS5 and vec0 don't support in-place updates cleanly)

**FTS5 trigram tokenizer** works well for substring/partial-word matching but is
slower than the default tokenizer for large corpora. Switch to `porter` or
`unicode61` if you only need whole-word matching at scale.

**Vector search cold start.** The `all-MiniLM-L6-v2` model (~80MB) takes 2–5s
to load on first use. The lazy singleton in `embeddings.py` amortizes this across
the process lifetime.

**Testing.** Virtual tables won't exist in in-memory SQLite fixtures unless you
call `ensure_virtual_tables(session)` and have `sqlite-vec` installed in the test
environment. Consider mocking `hybrid_search` in unit tests that don't need it.
