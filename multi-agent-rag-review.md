# Multi-Agent RAG Review — Implementation Guide

Pipeline pattern for reviewing a free-text request against a document corpus,
where the request bundles multiple items and you need traceable per-item
citations. Solves the "single-pass RAG collapses into one fuzzy decision" problem
by decomposing the request and giving each item its own focused retrieval +
evaluation pass.

Originally built for HOA architectural review (resident requests vs. design
guidelines + covenants), but the shape is generic: legal contract review against
a master agreement, code review against a style guide, policy review against a
compliance handbook, etc.

## When to use

Use this pattern when **all** of these hold:

- A single request typically bundles multiple distinct items or sub-questions.
- The corpus has a **hierarchy of specificity** — specific guidelines you want
  to cite, plus generic procedural sections that semantically match everything
  and will be picked by lazy retrieval.
- You need traceable citations per item, not one rolled-up decision.

If your task is a single-shot lookup or Q&A over a flat corpus, this pattern is
overkill — use single-pass RAG.

## The shape

```
┌──────────┐
│  Parser  │  →  items[], global_concerns[], optional metadata
└──────────┘
     │
     ▼
┌────────────────────────────┐
│ Retriever (deterministic)  │  For each item / concern:
│                            │   1. vector search filtered to specific corpus
│                            │   2. if empty → fall back to generic corpus
└────────────────────────────┘
     │
     ▼
┌──────────────┐
│  Evaluator   │  Per topic: pick the most specific applicable rule(s),
│  (per topic) │  copy verbatim quote, write concrete ask, decide outcome
└──────────────┘
     │
     ▼
┌──────────────┐
│ Synthesizer  │  Dedupe citations, roll up overall decision,
│              │  emit failure_notes for any uncovered topic
└──────────────┘
     │
     ▼
┌──────────────┐
│   Renderer   │  Pure formatter over structured citations.
│   (optional) │  No raw chunks. No re-deciding citations.
└──────────────┘
```

---

## Stage 1 — Parser

**Role:** decompose the request into discrete topics. One LLM call.

**Model:** small/cheap (Haiku-class). This is structured extraction, not reasoning.

**Input:** raw request text.

**Output:**
```python
{
  "items": [str, ...],            # distinct things being requested
  "global_concerns": [str, ...],  # cross-cutting regulatory dimensions
  "metadata": {...}               # optional: names, dates, IDs extracted with confidence
}
```

The `items` vs. `global_concerns` split is the non-obvious move. **Items** are
what the requester asked for (e.g. "basketball court", "concrete walkway").
**Global concerns** are dimensions that apply across items but weren't explicitly
named (e.g. "drainage", "easements", "impervious surface limit"). The
Synthesizer treats them identically downstream, but separating them at the
Parser stage makes the prompt clearer and the parsed output auditable.

```python
PARSER_SYSTEM = """Extract three things from the request:

1. items — distinct physical [things] the requester is asking for. Short noun
   phrases. One per distinct item. Do not split a single item into sub-parts.

2. global_concerns — request-wide regulatory dimensions likely to apply,
   given what was described. Examples: [list a non-exhaustive sample].
   Err on the side of inclusion.

3. metadata — [name / ID / date] if it appears with HIGH confidence.
   Return null if not confident. Do not guess.

Return JSON only: {"items": [...], "global_concerns": [...], "metadata": {...}}"""
```

**Tell the parser to err toward inclusion on global concerns.** Missing a
concern silently produces a missing citation downstream. The Synthesizer's
coverage check will surface it as a failure note, but you'd rather catch it
here.

---

## Stage 2 — Retriever

**Role:** per-topic vector search with source filtering. **No LLM.**

This is the load-bearing design choice. Making the Retriever a deterministic
function (not another agent) means:
- Failure modes are inspectable — you can see exactly which chunks were retrieved.
- Cost is zero LLM calls per topic.
- Source-filtering is enforceable, not "please prefer X".

```python
SPECIFIC_SOURCE = "design_guidelines.txt"   # the source you WANT cited
GENERIC_SOURCE = "covenants.txt"            # only when nothing else covers it

def retrieve(vectorstore, topic: str, k: int = 8) -> dict:
    primary = vectorstore.similarity_search(
        topic, k=k, filter={"source": SPECIFIC_SOURCE}
    )
    fallback = []
    if not primary:
        fallback = vectorstore.similarity_search(
            topic, k=k, filter={"source": GENERIC_SOURCE}
        )
    return {"topic": topic, "primary_chunks": primary, "fallback_chunks": fallback}
```

Without source filtering, generic procedural sections ("must obtain approval
before construction") match *every* topic semantically and dominate the top-k.
The Evaluator then picks them because they're the only thing it sees. Filtering
at retrieval is more reliable than asking the Evaluator to "prefer specific over
generic" in prose.

**Fallback rule:** only return generic-source chunks when the specific source
returned zero hits for this topic. This handles topics genuinely covered only in
the generic doc (e.g. impervious surface limits often live in covenants, not
design guidelines).

---

## Stage 3 — Evaluator

**Role:** per topic, pick the most specific applicable rule, copy a verbatim
quote, write a concrete ask, assign an outcome. One LLM call per topic.

**Model:** capable (Sonnet-class). This is where citation quality lives.

**Input:**
```python
{
  "topic": str,
  "request_text": str,        # full request as context
  "chunks": list[Document],   # primary or fallback, whichever the Retriever returned
}
```

**Output:**
```python
{
  "topic": str,
  "outcome": "compliant" | "violates" | "needs_info" | "not_addressed",
  "citations": [
    {
      "source": str, "article": str, "section": str,  # verbatim from chunk metadata
      "quote": str,           # 1-2 sentence verbatim excerpt from chunk content
      "topic_label": str,     # short label for downstream rendering
      "ask": str,             # concrete one-sentence ask
    },
    ...
  ],
  "notes": str | None,
}
```

Key prompt directives:

- **"Be selective"** — empty citations is a valid answer. Tell the model
  explicitly that `not_addressed` + `citations: []` is correct when only generic
  procedural sections are available.
- **"Asks must be concrete"** — a number, a document, a confirmation. Never
  vague ("provide additional details"). This is what makes the rendered output
  actionable.
- **Verbatim quote** — the Evaluator must copy from the chunk content, not
  paraphrase. The verbatimness is what makes downstream rendering trustworthy.

```python
def evaluate(client, topic, request_text, chunks):
    # Wrap with one retry on JSON parse failure. Two attempts total.
    last_error = None
    for _ in range(2):
        response = client.messages.create(model=SONNET, system=EVALUATOR_SYSTEM, ...)
        try:
            return parse_json_response(response.content[0].text)
        except (JSONDecodeError, ValueError) as e:
            last_error = e
    raise RuntimeError(f"Evaluator failed for '{topic}': {last_error}")
```

**Run sequentially or in parallel.** The pattern doesn't require concurrency;
sequential is fine when latency is acceptable and easier to debug. For real-time
UX, use `asyncio.gather` or a `ThreadPoolExecutor`.

---

## Stage 4 — Synthesizer

**Role:** combine per-topic results into the final decision. Mostly pure code,
optionally one small LLM call for reasoning prose.

This stage is deterministic on purpose. The roll-up rules are too important to
delegate to a model.

### Dedupe citations

Across topics, the same rule may be cited multiple times (e.g. drainage applies
to walkway + patio + court). Dedupe by `(source, section)` and keep the first
occurrence:

```python
def dedupe_citations(citations):
    seen = set()
    out = []
    for c in citations:
        key = (c.get("source"), c.get("section"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
```

### Decision rollup

Precedence-based, no LLM:

```python
def rollup(outcomes):
    if "violates" in outcomes:    return "Denial"
    if "needs_info" in outcomes:  return "Conditional Approval"
    if outcomes and all(o == "compliant" for o in outcomes):
        return "Approval"
    return "Conditional Approval"  # default uncertain bucket
```

### Coverage check + failure_notes

The single most useful diagnostic this pattern produces. For every parsed item
and concern, did *something* come back? If not, surface it:

```python
def coverage_notes(parsed, results, retrieval_failures, evaluator_failures):
    notes = []
    for topic in retrieval_failures:
        notes.append(f"No excerpt retrieved for '{topic}'. Manual review required.")
    for msg in evaluator_failures:
        notes.append(msg)
    for r in results:
        if not r.get("citations"):
            notes.append(
                f"No specific rule cited for '{r['topic']}' "
                f"(outcome: {r['outcome']}). Manual review required."
            )
        if r.get("citations") and all(c["source"] == GENERIC_SOURCE for c in r["citations"]):
            notes.append(
                f"Only generic-source citations for '{r['topic']}'; "
                "no specific source covers this topic."
            )
    return notes
```

**Don't rely on a binary `hitl_required` flag.** It's uninformative. The
actionable signal is *which topic failed and how*. `failure_notes` is the
channel a human actually reads when the output looks off.

### Reasoning prose (optional)

A small Haiku call summarizing the per-topic outcomes into 2-3 sentences for a
`reasoning` field. Falls back to a template if the call fails:

```python
def generate_reasoning(client, decision, results, failure_notes):
    try:
        return client.messages.create(model=HAIKU, system=REASONING_SYSTEM,
            messages=[{"role": "user", "content": json.dumps({
                "decision": decision,
                "topics": [{"topic": r["topic"], "outcome": r["outcome"]} for r in results],
                "failure_notes": failure_notes,
            })}],
            max_tokens=256,
        ).content[0].text.strip()
    except Exception:
        return f"Decision: {decision}. Reviewed {len(results)} topics."
```

---

## Stage 5 — Renderer (optional)

**Role:** format the synthesizer's structured output into whatever the consumer
needs (email, report, JSON, UI).

**Critical rule:** the Renderer receives ONLY the structured citations, never
the raw chunks. Otherwise the model is tempted to re-decide citations from the
raw context, defeating the entire pipeline.

```
Renderer input:   {decision, citations: [...], reasoning, failure_notes}
Renderer input:   NEVER {raw_chunks, ...}
```

If the Renderer is an LLM call, instruct it to format only — "use ONLY the
citations provided. Do not invent or drop citations. Quote rule text verbatim in
double quotes."

---

## File structure

The paths below use `app/` as an example root.

```
<project_root>/
  agents/
    __init__.py      # model constants, JSON parse helper
    parser.py        # parse_request(client, text) -> dict
    retriever.py     # retrieve(vectorstore, topic, k) -> dict
    evaluator.py     # evaluate(client, topic, request, chunks) -> dict
    synthesizer.py   # synthesize(client, parsed, results, ...) -> dict
  reviewer.py        # thin orchestrator that wires the four stages
```

Keep each agent module under ~100 lines. If one starts growing, it usually
means you're conflating stages — e.g. doing retrieval inside the Evaluator, or
decision rollup inside the Renderer.

---

## Orchestrator

The whole pipeline is ~25 lines of glue code:

```python
def review_request(self, request_text: str) -> dict:
    parsed = parse_request(self.client, request_text)
    topics = list(parsed["items"]) + list(parsed["global_concerns"])

    evaluator_results = []
    retrieval_failures = []
    evaluator_failures = []

    for topic in topics:
        retrieval = retrieve(self.vectorstore, topic)
        chunks = retrieval["primary_chunks"] or retrieval["fallback_chunks"]
        if not chunks:
            retrieval_failures.append(topic)
            continue
        try:
            evaluator_results.append(evaluate(self.client, topic, request_text, chunks))
        except Exception as e:
            evaluator_failures.append(f"Evaluator failed for '{topic}': {e}")

    return synthesize(
        self.client, parsed,
        evaluator_results, retrieval_failures, evaluator_failures,
        request_text,
    )
```

---

## Non-obvious lessons

**Source filtering at retrieval beats source preferences in prompts.** Asking an
LLM to "prefer the specific document over the generic one" sounds reasonable
and works ~70% of the time. Filtering at retrieval works 100% of the time and
costs nothing.

**Two-pass citation selection is a bug, not a feature.** If the review stage
already produces structured citations, the rendering stage must not re-decide.
Pass only the structured citations to the renderer — no raw chunks.

**Coverage check pays for itself the first time it fires.** It's ~20 lines of
code and catches every "the model just didn't notice this topic" failure mode.

**`hitl_required` is uninformative; `failure_notes` is the real signal.** A
flag tells you something went wrong. A note tells you what.

**Parser items vs. global_concerns is worth the split.** Conceptually they're
both "topics to retrieve and evaluate", but separating them keeps the Parser
prompt clear and lets the Synthesizer's output preserve "what was requested"
vs. "what applies."

**Determinism where you can.** Retriever, dedupe, rollup, coverage check — all
pure code. LLM calls only where reasoning is actually needed: parsing,
per-topic evaluation, optional reasoning prose, optional rendering.

---

## Cost & latency

For an N-topic request:

- Parser: 1 small call (Haiku)
- Retriever: 0 LLM calls
- Evaluator: N calls (Sonnet)
- Synthesizer: 0–1 small calls (Haiku, for reasoning prose)
- Renderer: 0 or 1 call depending on output format

Sequential wall-clock: roughly `(N + 2) × per-call latency`. Parallelizable
across the Evaluator stage if needed — they share no state.

For a 10-topic request: ~12 LLM calls vs. 1–2 for single-pass RAG. The cost
buys you per-item traceability, deterministic source preference, and a
coverage-check audit trail. Worth it when output quality matters more than
throughput.

---

## Known limitations

**Parser can split or merge items inconsistently.** "Two retaining walls and a
flagstone path" might parse as 2 items or 3. The Synthesizer treats each as
independent, so over-splitting wastes calls and under-splitting drops items.
Adjust prompt examples to match your domain's natural granularity.

**Source filter requires consistent metadata.** Every indexed chunk needs a
reliable `source` field. If your ingest doesn't normalize source names,
retrieval filtering silently returns empty results. Add an `ingest_status` /
`server_status` tool that lists the distinct sources in the index.

**No cross-topic reasoning.** Each evaluator call sees only its topic's chunks.
Rules that depend on combining facts across topics ("if you have both X and Y,
then Z applies") need either a special "combined" topic emitted by the Parser
or a separate cross-cutting stage between Evaluator and Synthesizer.

**JSON parse retry is fragile.** Two attempts with the same prompt won't help
if the model is consistently misformatting. Consider switching to a tool-use /
structured-output API for the Evaluator if retries fail often in production.
