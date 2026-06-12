# Patterns Library

This directory is a reusable reference library for common implementation patterns. Each file is a self-contained guide that can be applied to any project.

## Contents

| File | Pattern |
|------|---------|
| `beads-issue-tracking.md` | Graph-based CLI issue tracker (`bd`) for AI coding agents — session-start commands, claim/close workflow, replaces markdown TODO lists as the single source of truth |
| `embeddings.md` | Semantic search and RAG — chunking strategies (structural vs. semantic), vector-only vs. hybrid retrieval (BM25 + vector + RRF), embedding model selection |
| `memory-layer.md` | Persistent hybrid memory — SQLite + FTS5/BM25 + vector similarity fused via RRF; see `embeddings.md` § Retrieval Strategy for the conceptual background |
| `multi-agent-rag-review.md` | Multi-stage RAG pipeline for reviewing a request that bundles multiple items — Parser → Retriever → Evaluator (per topic) → Synthesizer, with per-item citations and a coverage-check audit trail |
| `ocr.md` | PDF OCR using a vision LLM — per-page text extraction with quality scoring and resume-safe indexing; any capable vision LLM (Claude, Gemini, GPT-4o) works |
| `venv-direnv.md` | Python virtual environment + direnv — automatic venv activation and secret injection on `cd` |

## How to Use

When starting a new project or feature, check this directory first. If a pattern applies:

1. Read the relevant file to understand the approach and its tradeoffs.
2. Adapt the code to your project's structure — file paths, import roots, and session/client wiring are illustrative, not prescriptive.
3. Follow the "When to Implement" or tradeoff guidance before committing to a stack.

When you solve a problem that isn't covered here and is likely to recur, add a new pattern file following the same structure: use case, stack, key lessons, code, and known limitations. Title it `# Pattern: <Name>`, and update the Contents table here, the Relationships Between Files section (if the new pattern connects to existing ones), and the table in `README.md`.

## Relationships Between Files

- `embeddings.md` is the conceptual home for RAG — chunking and retrieval strategy both live there.
- `memory-layer.md` is the SQLite implementation of hybrid RAG. Read `embeddings.md` first for the why, then `memory-layer.md` for the how.
- `multi-agent-rag-review.md` is a pipeline pattern built on top of single-pass RAG. Read `embeddings.md` first if you're new to RAG; reach for the multi-agent pattern only when a single request bundles multiple items and per-item citation traceability matters.
- `ocr.md` pairs naturally with `embeddings.md` — OCR output is a common input to a RAG pipeline, and semantic chunking (covered in `embeddings.md`) is often needed when OCR doesn't preserve document headers.

## Conventions

- Each pattern is stack-specific enough to be immediately useful but generic enough to apply across projects.
- Prompts, thresholds, and configuration values that require tuning are annotated with comments explaining what to change and why.
- Project-specific wiring (session factories, app entry points, file paths) is always marked as a placeholder to replace.
- Don't overclaim performance for specific providers. Where multiple LLMs or services are interchangeable, say so and note what should drive the choice (cost, rate limits, existing API setup).
