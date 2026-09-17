# Phase 1, Step 2: Flat Ingestion Pipeline

**Status:** Draft  
**Epic:** [VG-5](https://jira.atlassian.com/browse/VG-5) — Phase 1: Minimum Viable Memory  
**Story:** [VG-3](https://jira.atlassian.com/browse/VG-3) — Configure LangGraph state loop & generate flat vector embedding  
**Depends on:** [Phase_1_Step_1_Plan.md](./Phase_1_Step_1_Plan.md) (infrastructure complete)  
**Replaces:** Development_Plan.md § Phase 1 → Step 2 (1.2.1–1.2.7)  
**Glossary:** See [Phase_1_Step_1_Plan.md §0](./Phase_1_Step_1_Plan.md#0-phase-naming-glossary) for architecture vs dev-plan phase naming.

---

## 1. Objective

Implement the **minimum viable ingestion path** for Phase 1: accept raw text, embed it, and persist it in a flat `documents` table backed by pgvector.

Deliverables:

- `documents` schema (SQL init + SQLAlchemy model)
- Embedding client (OpenRouter-compatible HTTP API)
- LangGraph workflow orchestrating embed → persist
- `POST /api/v1/ingest/flat` REST endpoint with Pydantic request/response schemas

Step 2 delivers **write-path only**. No retrieval, consumer agent, or structured entity extraction.

---

## 2. Architecture Alignment

| Architecture requirement | Step 2 approach | Notes |
|--------------------------|-----------------|-------|
| LangGraph inside microservice (§2) | Linear ingestion graph in `/api/app/graph/` | Single-purpose workflow; expands in Phase 2 |
| PostgreSQL + pgvector (§2) | `documents.embedding VECTOR(n)` with optional HNSW index | Flat store, not temporal edges yet |
| Pydantic validation (§2) | Request/response schemas for API; graph state TypedDict | Entity/temporal schemas deferred to Phase 2 |
| OpenRouter / fast models (§2) | OpenRouter HTTP client for **embeddings** | LLM extraction not needed for flat ingest |
| Decoupled engine (§3) | Ingestion lives entirely in `/api`; external agent repo consumes via REST | Step 3 retrieve API reads what Step 2 writes |
| MCP future-proofing (§3) | Versioned REST under `/api/v1/ingest/flat` | Stable contract for later MCP wrapper |
| Background ingestion (§4 Phase 1) | **Synchronous** for MVP | Async/queue deferred; endpoint returns after persist |

**Intentional simplification:** The architecture's full ingestion pipeline (Pydantic entity extraction, temporal edges, provenance, multimodal) is Phase 2+. Step 2 stores **raw text + embedding** only — the smallest slice that unblocks flat retrieval in Step 3.

---

## 3. Prerequisites (Step 1 Complete)

Assumes Step 1 delivered:

- `docker compose up --build` runs API + Postgres with pgvector verified
- `/health` and `/ready` operational
- `api/app/config.py` with `DATABASE_URL`, `CORS_ORIGINS`
- `db/init/01_extensions.sql` enables `vector` extension

---

## 4. Target Repository Layout (End of Step 2)

```
Aion/
├── api/
│   ├── requirements.txt          # + langgraph, httpx, pgvector
│   ├── main.py                   # mounts /api/v1 router
│   └── app/
│       ├── config.py             # + OPENROUTER_*, EMBEDDING_* settings
│       ├── db/
│       │   ├── connection.py
│       │   └── models.py         # Document ORM model
│       ├── clients/
│       │   ├── __init__.py
│       │   ├── openrouter.py     # shared httpx async client
│       │   └── embeddings.py     # embed_text(content) -> list[float]
│       ├── graph/
│       │   ├── __init__.py
│       │   ├── state.py          # FlatIngestState
│       │   └── flat_ingest.py    # LangGraph: validate → embed → persist
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── ingest.py         # FlatIngestRequest, FlatIngestResponse
│       └── routes/
│           ├── __init__.py
│           └── ingest.py         # POST /ingest/flat
├── db/
│   └── init/
│       ├── 01_extensions.sql
│       └── 02_documents.sql      # documents table + index
├── docker/
│   └── .env.example              # + embedding/OpenRouter keys
└── documents/plans/
```

---

## 5. Subtasks

### 2.1 Flat store schema

**Goal:** Persistent table for raw text and its vector embedding.

**SQL:** `db/init/02_documents.sql`

```sql
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    embedding   vector(1536) NOT NULL,   -- dimension matches EMBEDDING_DIMENSION
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
    ON documents USING hnsw (embedding vector_cosine_ops);
```

**ORM:** `api/app/db/models.py`

- SQLAlchemy 2.0 declarative model `Document` mapping the table
- Use `pgvector.sqlalchemy.Vector` for the embedding column
- `EMBEDDING_DIMENSION` env var must match SQL dimension (default `1536`)

**Acceptance criteria:**

- [ ] Fresh DB boot creates `documents` table via init script
- [ ] ORM model round-trips insert/select on a test row
- [ ] Dimension mismatch between env, SQL, and model fails at startup or migration time (documented)
- [ ] HNSW index exists (supports Step 3 cosine similarity queries)

**Note:** Changing `EMBEDDING_DIMENSION` after data exists requires re-embedding; document in `.env.example`.

**Existing Postgres volumes (Step 1 → Step 2 upgrade):** Docker init scripts in `/docker-entrypoint-initdb.d/` run **only on first boot** of a new volume. If Step 1 already created `postgres_data`, adding `02_documents.sql` will **not** apply automatically.

Choose one:

```bash
# Dev reset (destroys all DB data)
docker compose -f docker/docker-compose.yml down -v
docker compose -f docker/docker-compose.yml up --build -d

# Or apply manually against the running DB
docker compose -f docker/docker-compose.yml exec postgres \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f /docker-entrypoint-initdb.d/02_documents.sql
```

Document the chosen path in the Step 2 PR/commit notes.

---

### 2.2 Environment and configuration

**Goal:** Externalize all embedding/LLM provider settings.

**Add to `docker/.env.example`:**

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | API auth for OpenRouter | `sk-or-...` |
| `OPENROUTER_BASE_URL` | Override base URL | `https://openrouter.ai/api/v1` |
| `EMBEDDING_MODEL` | Model identifier | `openai/text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Vector size (must match schema) | `1536` |
| `EMBEDDING_TIMEOUT_SECONDS` | HTTP timeout | `30` |

**Extend `api/app/config.py`:**

- Validate `OPENROUTER_API_KEY` is set when ingestion routes are loaded
- Expose `embedding_dimension: int` used by clients and schema checks

**Acceptance criteria:**

- [ ] API refuses to start (or `/ready` fails) if `OPENROUTER_API_KEY` missing in dev
- [ ] All new settings documented in `docker/.env.example`
- [ ] No API keys in source code or logs

---

### 2.3 HTTP client and embedding integration

**Goal:** Convert text to a float vector via an OpenRouter-compatible embeddings API.

**Decision:** Use OpenRouter's OpenAI-compatible `/embeddings` endpoint. Avoid local sentence-transformers in Phase 1 to keep Docker images slim and deployment self-consistent with the architecture's cloud-model strategy.

**Files:**

- `api/app/clients/openrouter.py` — shared `httpx.AsyncClient` with auth header, base URL, timeout, retry-on-429 (max 2 retries)
- `api/app/clients/embeddings.py` — `async def embed_text(text: str) -> list[float]`
  - Truncate/log warning if text exceeds model context (document limit, e.g. 8191 tokens ≈ rough char guard)
  - Validate returned vector length == `EMBEDDING_DIMENSION`
  - Raise typed `EmbeddingError` on failure

**Acceptance criteria:**

- [ ] Unit-testable client with injectable httpx transport (mocked response)
- [ ] `embed_text("hello world")` returns vector of correct dimension against live API (manual/integration test)
- [ ] Errors surface as structured exceptions, not raw stack traces to callers

**Deferred:** Fast LLM chat/completions client (needed for Phase 2 Pydantic extraction, not flat ingest)

---

### 2.4 LangGraph state and workflow

**Goal:** Orchestrate ingestion as an explicit, testable state machine inside the microservice.

**State:** `api/app/graph/state.py`

```python
class FlatIngestState(TypedDict):
    content: str
    embedding: list[float] | None
    document_id: str | None
    status: Literal["pending", "embedded", "stored", "failed"]
    error: str | None
```

**Graph:** `api/app/graph/flat_ingest.py`

Linear workflow:

```
validate_input → embed → persist → END
                    ↘ (on error) → mark_failed → END
```

| Node | Responsibility |
|------|----------------|
| `validate_input` | Reject empty/whitespace-only content; set status |
| `embed` | Call `embed_text`; store vector in state |
| `persist` | Insert `Document` via async SQLAlchemy session; set `document_id` |
| `mark_failed` | Set status `failed`, capture error message |

**Acceptance criteria:**

- [ ] Graph compiles and runs end-to-end with mocked embedding client
- [ ] Failed embedding does not insert partial rows
- [ ] Successful run leaves exactly one row in `documents`
- [ ] Graph is invokable independently of FastAPI (testable in isolation)

---

### 2.5 Database insertion logic

**Goal:** Safe, parameterized persistence — no string-concatenated SQL.

**Tasks:**

- `api/app/db/repository.py` (or methods on a service layer):
  - `async def insert_document(session, content: str, embedding: list[float]) -> UUID`
- Use SQLAlchemy ORM insert with bound parameters
- Commit inside the LangGraph `persist` node; rollback on exception

**Acceptance criteria:**

- [ ] No raw f-string SQL with user content
- [ ] Duplicate submissions create separate rows (no dedup in MVP)
- [ ] `created_at` populated automatically

---

### 2.6 API route and schemas

**Goal:** Expose ingestion over versioned REST with strict I/O contracts.

**Schemas:** `api/app/schemas/ingest.py`

```python
class FlatIngestRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100_000)


class FlatIngestResponse(BaseModel):
    id: UUID
    status: Literal["stored"]
```

**Route:** `POST /api/v1/ingest/flat`

- Invoke LangGraph workflow with request body
- Return `201 Created` + `FlatIngestResponse` on success
- Return `422` for validation errors, `502` for embedding provider failure, `500` for unexpected DB errors
- OpenAPI docs auto-generated (FastAPI default)

**Wire in `main.py`:**

- Mount router at prefix `/api/v1`
- Keep `/health` and `/ready` at root (Step 1 convention)

**Acceptance criteria:**

- [ ] `curl -X POST http://localhost:8000/api/v1/ingest/flat -H "Content-Type: application/json" -d '{"content":"Test memory."}'` → 201 with UUID
- [ ] Empty content → 422
- [ ] OpenAPI `/docs` shows request/response models
- [ ] Response JSON is stable enough to wrap as MCP tool later

---

### 2.7 Dependencies

**Goal:** Add only what Step 2 needs.

**Append to `api/requirements.txt`:**

```
langgraph>=0.2.0
langchain-core>=0.3.0
httpx>=0.27.0
pgvector>=0.3.0
```

**Acceptance criteria:**

- [ ] Docker image rebuild succeeds
- [ ] No torch/transformers/sentence-transformers in deps
- [ ] Pin major versions; avoid unpinned `latest`

---

### 2.8 Verification and smoke test

**Goal:** Prove the write path works end-to-end before Step 3 builds retrieval.

**Manual checklist:**

```bash
# Stack running with valid OPENROUTER_API_KEY in docker/.env
docker compose -f docker/docker-compose.yml up --build -d

# Ingest
curl -sf -X POST http://localhost:8000/api/v1/ingest/flat \
  -H "Content-Type: application/json" \
  -d '{"content": "The project codename is Aion."}'
# → {"id":"<uuid>","status":"stored"}

# Verify row in DB
docker compose -f docker/docker-compose.yml exec postgres \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT id, left(content, 40), vector_dims(embedding) FROM documents LIMIT 5;"

# Validation failure
curl -sf -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/api/v1/ingest/flat \
  -H "Content-Type: application/json" -d '{"content":""}'
# → 422
```

**Optional automated test:** `api/tests/test_flat_ingest.py` with mocked httpx embedding response + test DB (pytest-asyncio).

**Acceptance criteria:**

- [ ] Smoke test passes with live embedding API
- [ ] Row visible in Postgres with correct `vector_dims`
- [ ] Invalid input rejected without side effects

---

## 6. Definition of Done

Step 2 is complete when:

1. `documents` table exists with HNSW index and matching ORM model
2. LangGraph flat ingestion workflow runs embed → persist reliably
3. `POST /api/v1/ingest/flat` accepts text and returns a stored document ID
4. Embedding config is env-driven; no secrets in repo
5. No retrieval routes, consumer agent code, or entity extraction exist yet
6. VG-3 acceptance criteria in Jira can be checked off

---

## 7. Out of Scope (Explicit Deferrals)

| Item | Deferred to |
|------|-------------|
| `POST /retrieve/flat` and similarity search | Step 3 |
| Consumer agent and pre-fetch hook | Step 3 |
| Pydantic entity/temporal extraction | Phase 2, Step 1 |
| `nodes`, `edges`, `sources` tables | Phase 2 |
| Alembic migration tooling | Phase 2, Step 2 |
| Async background job queue (Celery, etc.) | Post-MVP |
| Batch ingestion, file upload | Phase 4 |
| Embedding cache / deduplication | Post-MVP |
| OpenRouter chat/completions (fast LLM) | Phase 2 |
| Local HuggingFace sentence-transformers | Post-MVP (optional alt) |

---

## 8. Improvements Over Original Plan (1.2.1–1.2.7)

| Gap in original | Addressed by |
|-----------------|--------------|
| LLM client included but unused in flat ingest | §2.3 embedding-only client; chat LLM deferred to Phase 2 |
| Embedding provider ambiguous (OpenAI vs local HF) | §2.3 OpenRouter-compatible embeddings (slim Docker, consistent stack) |
| Vector dimension unspecified | §2.1 `EMBEDDING_DIMENSION` + schema alignment |
| No index for retrieval performance | §2.1 HNSW index at schema creation |
| LangGraph state underspecified | §2.4 explicit `FlatIngestState` with status/error |
| No API request/response contracts | §2.6 Pydantic schemas + HTTP status codes |
| Route path unspecified | §2.6 `/api/v1/ingest/flat` (MCP-ready versioning) |
| SQL vs SQLAlchemy unclear | §2.1 both: init SQL + ORM model |
| No error handling specification | §2.4 failed node + §2.6 422/502/500 mapping |
| No verification checklist | §2.8 smoke test |
| `requirements.txt` deps mixed with Step 1 | §2.7 incremental deps only |
| Sync vs async ingestion unclear | §2 Architecture table: synchronous MVP |
| Schema init on existing Step 1 volume | §2.1 upgrade path (`down -v` or manual `psql -f`) |

---

## 9. Jira Subtasks

Parent story: [VG-3](https://jira.atlassian.com/browse/VG-3) — Phase 1 Step 2 — Flat Ingestion Pipeline

| # | Jira | Subtask | Implementation step |
|---|------|---------|---------------------|
| 1 | [VG-13](https://jira.atlassian.com/browse/VG-13) | Create documents schema and ORM model | §2.1 Flat store schema |
| 2 | [VG-17](https://jira.atlassian.com/browse/VG-17) | Extend config and .env.example for embeddings | §2.2 Environment and configuration |
| 3 | [VG-14](https://jira.atlassian.com/browse/VG-14) | Implement OpenRouter embedding client | §2.3 HTTP client and embedding integration |
| 4 | [VG-15](https://jira.atlassian.com/browse/VG-15) | Build LangGraph flat ingest workflow | §2.4 LangGraph state and workflow |
| 5 | [VG-16](https://jira.atlassian.com/browse/VG-16) | Add document repository insert logic | §2.5 Database insertion logic |
| 6 | [VG-18](https://jira.atlassian.com/browse/VG-18) | Expose POST /api/v1/ingest/flat | §2.6 API route and schemas |
| 7 | [VG-19](https://jira.atlassian.com/browse/VG-19) | Update requirements and rebuild Docker | §2.7 Dependencies |
| 8 | [VG-20](https://jira.atlassian.com/browse/VG-20) | Run ingestion smoke test | §2.8 Verification and smoke test |
