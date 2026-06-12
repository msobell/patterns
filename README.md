# Patterns Library

A reusable reference library of implementation patterns. Each file is a self-contained guide that can be applied to any project.

| File | Pattern |
|------|---------|
| [`beads-issue-tracking.md`](beads-issue-tracking.md) | Graph-based CLI issue tracker (`bd`) for AI coding agents — session-start commands, claim/close workflow, replaces markdown TODO lists as the single source of truth |
| [`embeddings.md`](embeddings.md) | Semantic search and RAG — chunking strategies (structural vs. semantic), vector-only vs. hybrid retrieval (BM25 + vector + RRF), embedding model selection |
| [`memory-layer.md`](memory-layer.md) | Persistent hybrid memory — SQLite + FTS5/BM25 + vector similarity fused via RRF |
| [`multi-agent-rag-review.md`](multi-agent-rag-review.md) | Multi-stage RAG pipeline for reviewing a request that bundles multiple items — Parser → Retriever → Evaluator (per topic) → Synthesizer, with per-item citations and a coverage-check audit trail |
| [`ocr.md`](ocr.md) | PDF OCR using a vision LLM — per-page text extraction with quality scoring and resume-safe indexing; any capable vision LLM (Claude, Gemini, GPT-4o) works |
| [`venv-direnv.md`](venv-direnv.md) | Python virtual environment + direnv — automatic venv activation and secret injection on `cd` |

See [CLAUDE.md](CLAUDE.md) for usage guidance, the relationships between patterns, and conventions for adding new ones.
