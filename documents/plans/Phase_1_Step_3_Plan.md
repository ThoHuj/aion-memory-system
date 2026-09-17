# Phase 1, Step 3: Flat Retrieval API

**Status:** Draft  
**Epic:** [VG-5](https://jira.atlassian.com/browse/VG-5) — Phase 1: Minimum Viable Memory  
**Story:** [VG-4](https://jira.atlassian.com/browse/VG-4) — Flat Retrieval API & External Agent Integration  
**Depends on:** [Phase_1_Step_1_Plan.md](./Phase_1_Step_1_Plan.md), [Phase_1_Step_2_Plan.md](./Phase_1_Step_2_Plan.md)  
**Replaces:** Development_Plan.md § Phase 1 → Step 3 (1.3.1–1.3.6)  
**Glossary:** See [Phase_1_Step_1_Plan.md §0](./Phase_1_Step_1_Plan.md#0-phase-naming-glossary) for architecture vs dev-plan phase naming.

---

## 1. Objective

Complete the **read path** for Phase 1 in this repository: expose flat vector retrieval over REST so an **external consumer agent** (separate repo) can call the memory API before LLM generation.

**This repo delivers:**

- pgvector similarity query + `POST /api/v1/retrieve/flat`
- Retrieval tuning via env config
- API-level E2E verification (curl / pytest)
- Integration guide documenting the REST contract for external agents

**External agent repo delivers (out of scope here):**

- Mandatory pre-fetch hook calling `/retrieve/flat`
- Suffix-based prompt injection (static system prompt, KV-cache friendly)
- LLM orchestration

Step 3 completes **Phase 1 for this codebase**. No graph traversal, hybrid search, or provenance tooling yet.

**Ingest scope:** Phase 1 does **not** auto-ingest user chat — seed data via `POST /api/v1/ingest/flat`. Auto-ingest is deferred to post-Phase 1 or Dev Phase 2.

---

## 2. Architecture Alignment

| Architecture requirement | Step 3 approach (this repo) | External agent repo |
|--------------------------|----------------------------|---------------------|
| Deterministic pre-fetch (§3) | Stable retrieve REST contract | Hook calls API on every turn |
| Decoupled engine (§3) | Memory API only; no agent code | Agent integrates via HTTP |
| Cache-preserving injection (§4 Phase 3) | Document suffix pattern in integration guide | Implements prompt assembly |
| pgvector semantic search (§2) | Cosine `<=>` query against HNSW index | Consumes ranked chunks |
| MCP future-proofing (§3) | Versioned `POST /api/v1/retrieve/flat` | Wraps as MCP tool later |
| Separate repositories (§3) | **This repo = memory service** | **Agent repo = orchestration** |

**Intentional simplification:** Architecture Phase 2 retrieval (entity extraction, RRF, temporal pruning, MMR) is deferred. Step 3 uses **query embed → cosine top-K → JSON response**.

---

## 3. Prerequisites (Steps 1 & 2 Complete)

Assumes:

- Step 1: API + Postgres running; CORS configured; `/api/v1` prefix established
- Step 2: `documents` table populated via `POST /api/v1/ingest/flat`; embedding client reusable; HNSW index on `embedding`

---

## 4. Target Repository Layout (End of Step 3)

```
Aion/
├── api/
│   └── app/
│       ├── clients/
│       │   └── embeddings.py         # reuse embed_text() from Step 2
│       ├── db/
│       │   ├── models.py
│       │   └── repository.py         # + retrieve_similar()
│       ├── schemas/
│       │   ├── ingest.py
│       │   └── retrieve.py           # FlatRetrieveRequest/Response
│       ├── services/
│       │   ├── __init__.py
│       │   └── flat_retrieve.py      # embed query → search → rank
│       └── routes/
│           ├── ingest.py
│           └── retrieve.py           # POST /retrieve/flat
├── docker/
│   └── .env.example                  # + RETRIEVE_* vars
├── documents/
│   └── integration/
│       └── external_agent.md         # REST contract + pre-fetch guidance
└── documents/plans/
```

---

## 5. Subtasks

### 3.1 Vector similarity query

**Goal:** Retrieve the top-K closest documents to a query embedding using the Step 2 HNSW index.

**Decision:** Use **cosine distance** (`<=>` operator) to match the Step 2 index (`vector_cosine_ops`).

**File:** `api/app/db/repository.py`

```python
async def retrieve_similar(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Returns (document, distance) pairs ordered by ascending distance.
    distance = embedding <=> query_embedding  (0 = identical)
    """
```

**Implementation notes:**

- Parameterized query via SQLAlchemy / raw SQL with bound `:embedding` — no string interpolation
- `LIMIT top_k` enforced server-side
- Optional `max_distance` filter (env `RETRIEVE_MAX_DISTANCE`) to drop weak matches

**Acceptance criteria:**

- [ ] Query uses `<=>` (cosine distance), not `<->` (L2), matching the HNSW index
- [ ] Returns results ordered best-first
- [ ] Empty table returns empty list (no error)
- [ ] `top_k` clamped to `[1, 20]` at the API layer

---

### 3.2 Flat retrieval service

**Goal:** Orchestrate embed → search in the API service layer (no LangGraph needed for this linear read path).

**File:** `api/app/services/flat_retrieve.py`

```python
async def retrieve_flat(query: str, top_k: int) -> FlatRetrieveResponse:
    embedding = await embed_text(query)
    rows = await retrieve_similar(session, embedding, top_k)
    return format_chunks(rows)  # map distance → similarity score
```

**Score mapping for response:**

- Expose `score = 1 - distance` (cosine similarity proxy) so callers see higher = better
- Include `document_id` and `content` per chunk

**Score formula assumption:** Valid when embeddings are **L2-normalized** (true for OpenAI/OpenRouter models such as `text-embedding-3-small`). If the embedding model changes to one that does not normalize vectors, revisit this mapping.

**Acceptance criteria:**

- [ ] Reuses Step 2 `embed_text()` — no duplicate embedding client
- [ ] Service testable with mocked embedding + fixture DB
- [ ] Embedding failure propagates as `EmbeddingError` (→ HTTP 502)

---

### 3.3 API retrieval route

**Goal:** Expose flat retrieval over versioned REST.

**Schemas:** `api/app/schemas/retrieve.py`

```python
class FlatRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)

class RetrievedChunk(BaseModel):
    id: UUID
    content: str
    score: float = Field(..., ge=0.0, le=1.0)

class FlatRetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
```

**Route:** `POST /api/v1/retrieve/flat`

- Call `retrieve_flat()`
- Return `200 OK` + `FlatRetrieveResponse`
- Empty chunks list is valid (no matches), not an error

**Acceptance criteria:**

- [ ] `curl -X POST .../retrieve/flat -d '{"query":"project codename"}'` returns ranked chunks
- [ ] OpenAPI docs show schemas alongside ingest routes
- [ ] Route registered in `main.py` under `/api/v1` router

---

### 3.4 Configuration additions (API side)

**Goal:** Tune retrieval without code changes.

**Add to `docker/.env.example` / `api/app/config.py`:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `RETRIEVE_DEFAULT_TOP_K` | Default when client omits top_k | `5` |
| `RETRIEVE_MAX_DISTANCE` | Drop chunks with distance above threshold | `0.5` (optional filter) |

**Acceptance criteria:**

- [ ] Defaults applied when request body omits `top_k`
- [ ] Distance filter documented; disabling = set empty or `1.0`

---

### 3.5 External agent integration guide

**Goal:** Document how a separate consumer-agent repository integrates with this API.

**File:** `documents/integration/external_agent.md`

**Must cover:**

| Topic | Content |
|-------|---------|
| Base URL | e.g. `http://localhost:8000` when API container is running |
| Pre-fetch pattern | Call `POST /api/v1/retrieve/flat` on **every** user turn before LLM generation |
| Request/response | JSON schemas with examples |
| Prompt injection | Append chunks as user-message suffix; keep system prompt static (KV cache) |
| Suggested memory block format | `[score=0.92] {content}` per chunk |
| Error handling | Empty chunks on no match; graceful degradation if API unreachable |
| Ingest | Seed memory via `POST /api/v1/ingest/flat` (not auto-ingest from chat in Phase 1) |

**Acceptance criteria:**

- [ ] Guide is sufficient for an agent developer without reading source code
- [ ] Includes curl examples for ingest + retrieve
- [ ] References OpenAPI `/docs` for authoritative schemas

---

### 3.6 API end-to-end verification

**Goal:** Prove Phase 1 read path works in this repository.

**Setup script** (manual or `scripts/phase1_retrieve_demo.sh`):

```bash
# 1. Start stack
docker compose -f docker/docker-compose.yml up -d

# 2. Ingest seed memories
curl -X POST http://localhost:8000/api/v1/ingest/flat \
  -H "Content-Type: application/json" \
  -d '{"content": "The project codename is Aion."}'

curl -X POST http://localhost:8000/api/v1/ingest/flat \
  -H "Content-Type: application/json" \
  -d '{"content": "Phase 1 uses flat vector retrieval before graph traversal."}'

# 3. Test retrieval
curl -X POST http://localhost:8000/api/v1/retrieve/flat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the project codename?"}'
# → chunks containing "Aion" with highest score

# 4. Test empty/no-match query
curl -X POST http://localhost:8000/api/v1/retrieve/flat \
  -H "Content-Type: application/json" \
  -d '{"query": "quantum physics"}'
# → empty chunks list (200 OK)
```

**Acceptance criteria:**

- [ ] Retrieval returns "Aion" chunk for codename query
- [ ] Unrelated query returns empty chunks (not an error)
- [ ] OpenAPI docs list both ingest and retrieve routes

---

## 6. Definition of Done

Step 3 — and **Phase 1 for this repo** — is complete when:

1. `POST /api/v1/retrieve/flat` returns ranked chunks from pgvector cosine search
2. Retrieval config is env-driven (`RETRIEVE_*`)
3. `documents/integration/external_agent.md` documents the REST contract and pre-fetch pattern
4. API E2E demo (ingest → retrieve via curl) passes
5. No consumer-agent code exists in this repository
6. VG-4 acceptance criteria in Jira can be checked off

---

## 7. Out of Scope (Explicit Deferrals)

| Item | Owner / deferred to |
|------|---------------------|
| Consumer agent CLI / REPL | External agent repository |
| Pre-fetch hook implementation | External agent repository |
| Prompt formatting / LLM client | External agent repository |
| Entity extraction from query | Dev Phase 3, Step 2 (RRF pipeline) |
| pg_trgm lexical search | Dev Phase 3 |
| Graph traversal / temporal edges | Dev Phase 2 |
| MMR halting | Dev Phase 3, Step 3 |
| On-demand provenance | Dev Phase 4 |
| Auto-ingest user chat turns | Post-Phase 1 or Dev Phase 2 |
| LangGraph retrieval workflow | Dev Phase 3+ |

---

## 8. Improvements Over Original Plan (1.3.1–1.3.6)

| Gap in original | Addressed by |
|-----------------|--------------|
| Consumer agent assumed in same repo | §1, §7 — agent is external; this repo is API-only |
| Distance operator unspecified | §3.1 cosine `<=>` aligned with Step 2 HNSW index |
| Route path unspecified | §3.3 `/api/v1/retrieve/flat` |
| No integration contract for external agents | §3.5 `external_agent.md` |
| E2E assumed in-repo agent | §3.6 curl-based API demo |
| Score formula assumes normalized vectors | §3.2 documented assumption |

---

## 9. Jira Subtasks

Parent story: [VG-4](https://jira.atlassian.com/browse/VG-4) — Phase 1 Step 3 — Flat Retrieval API

| # | Jira | Subtask | Implementation step |
|---|------|---------|---------------------|
| 1 | [VG-23](https://jira.atlassian.com/browse/VG-23) | Implement cosine similarity query + repository | §3.1 Vector similarity query |
| 2 | [VG-21](https://jira.atlassian.com/browse/VG-21) | Add flat retrieval service | §3.2 Flat retrieval service |
| 3 | [VG-25](https://jira.atlassian.com/browse/VG-25) | Expose POST /api/v1/retrieve/flat | §3.3 API retrieval route |
| 4 | [VG-28](https://jira.atlassian.com/browse/VG-28) | Extend API config for retrieval tuning | §3.4 Configuration additions (API side) |
| 5 | [VG-30](https://jira.atlassian.com/browse/VG-30) | Document REST integration for external agents | §3.5 External agent integration guide |
| 6 | [VG-27](https://jira.atlassian.com/browse/VG-27) | Run Phase 1 API E2E demo | §3.6 API end-to-end verification |

**Cancelled** (moved to external agent repository): [VG-22](https://jira.atlassian.com/browse/VG-22), [VG-24](https://jira.atlassian.com/browse/VG-24), [VG-26](https://jira.atlassian.com/browse/VG-26), [VG-29](https://jira.atlassian.com/browse/VG-29)

---

## 10. Phase 1 Completion Summary (This Repo)

When Steps 1–3 are done, **this repository** provides:

```
┌──────────────────┐     ingest/flat      ┌──────────────────┐
│ External caller  │ ──────────────────► │   Memory API     │
│ (curl, agent,    │                     │   (Docker)       │
│  tests)          │ ◄── retrieve/flat ──│                  │
└──────────────────┘                     │  documents +     │
                                         │  pgvector        │
                                         └──────────────────┘
        ▲
        │ HTTP (separate repository)
        │
┌───────┴──────────┐
│  Consumer Agent  │  pre-fetch hook + prompt injection + LLM
│  (external repo) │
└──────────────────┘
```

The memory service exposes a stable REST surface; the external agent implements the architecture's deterministic pre-fetch pattern.
