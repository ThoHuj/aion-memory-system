# Phase 1, Step 1: Infrastructure Foundation

**Status:** Draft  
**Epic:** [VG-5](https://jira.atlassian.com/browse/VG-5) — Phase 1: Minimum Viable Memory  
**Story:** [VG-2](https://jira.atlassian.com/browse/VG-2) — Initialize Docker configuration for PostgreSQL (pgvector) & FastAPI  
**Replaces:** Development_Plan.md § Phase 1 → Step 1 (1.1.1–1.1.7)

---

## 0. Phase Naming Glossary

The architecture plan and development plan both use "Phase 1/2/3/4" for **different things**. Use this map during implementation:

| Term | Meaning |
|------|---------|
| **Architecture §4 Phase 1** | Runtime workflow: ingestion & extraction (background pipeline) |
| **Architecture §4 Phase 2** | Runtime workflow: hybrid retrieval & graph traversal |
| **Architecture §4 Phase 3** | Runtime workflow: generation & lazy-loading context |
| **Dev plan Phase 1** | Delivery milestone: MVP flat memory — [Step 1](./Phase_1_Step_1_Plan.md), [Step 2](./Phase_1_Step_2_Plan.md), [Step 3](./Phase_1_Step_3_Plan.md) |
| **Dev plan Phase 2** | Delivery milestone: bi-temporal graph & structured extraction |
| **Dev plan Phase 3** | Delivery milestone: hybrid discovery (pg_trgm, RRF, MMR) |
| **Dev plan Phase 4** | Delivery milestone: multimodal ingestion & provenance |

**Cross-reference:**

```
Architecture workflow          Dev plan milestone
─────────────────────         ──────────────────
Ingestion & extraction    →   Dev Phase 1 (flat) + Phase 2 (structured)
Retrieval (hybrid/graph)  →   Dev Phase 1 Step 3 (flat) + Phase 3 (RRF/MMR)
Generation & provenance   →   External agent repo (suffix) + Dev Phase 4 (sources)
```

---

## 1. Objective

Stand up a **runnable, decoupled memory microservice** that matches the architecture plan's deployment model:

- FastAPI service in Docker, isolated from the consumer agent
- PostgreSQL with **pgvector enabled and verified**
- Persistent database storage across container restarts
- Health and readiness endpoints the orchestrator can depend on
- Configuration via environment variables, not hardcoded secrets

Step 1 delivers **infrastructure only**. No ingestion, retrieval, LangGraph workflows, or consumer-agent logic.

---



## 2. Architecture Alignment


| Architecture requirement       | How Step 1 satisfies it                                                     |
| ------------------------------ | --------------------------------------------------------------------------- |
| Decoupled memory engine (§3)   | `/api` service only; consumer agent is a **separate repository** |
| Docker + FastAPI (§2)          | Containerized FastAPI with hot-reload dev config                            |
| PostgreSQL + pgvector (§2)     | pgvector image + explicit `CREATE EXTENSION vector` on startup              |
| Service owns DB layer (§3)     | `/db` holds init scripts; API connects via env vars                         |
| MCP future-proofing (§3)       | REST endpoints under `/health` and `/ready`; versioned API prefix reserved  |
| Self-contained deployment (§2) | No cloud dependencies; local volumes for Postgres data                      |


---



## 3. Current Repository State

Already present at repo root (from initial scaffold):

```
Aion/
├── .gitignore          # ignores venv/, .env
├── env.example         # empty — needs population
├── requirements.txt    # empty — relocate to /api
├── settings.json       # empty {} — keep for local IDE/tooling; not used by Docker services
├── venv/               # local dev venv (gitignored)
└── documents/plans/
```

**Decision:** Move service-specific artifacts into `/api`. Keep root `settings.json` for editor/local tooling. Root `env.example` becomes a pointer or symlink target; the canonical template lives at `docker/.env.example`.

---



## 4. Target Repository Layout (End of Step 1)

```
Aion/
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 # FastAPI app factory + /health + /ready
│   └── app/
│       ├── __init__.py
│       ├── config.py           # pydantic-settings: DB URL, CORS origins, log level
│       └── db/
│           ├── __init__.py
│           └── connection.py   # async engine/session; used by /ready
├── db/
│   └── init/
│       └── 01_extensions.sql   # CREATE EXTENSION IF NOT EXISTS vector;
├── docker/
│   ├── docker-compose.yml
│   ├── docker-compose.override.example.yml   # optional local overrides
│   └── .env.example
├── documents/plans/
└── .gitignore
```

---



## 5. Subtasks



### 1.1 Repository scaffolding

**Goal:** Establish the decoupled layout the architecture assumes.

**Tasks:**

- Create `/api`, `/api/app`, `/api/app/db`, `/db/init`, `/docker`
- Add package `__init__.py` files where needed
- Remove or relocate root `requirements.txt` content into `api/requirements.txt`

**Acceptance criteria:**

- [ ] Directory tree matches §4 above
- [ ] No application logic beyond skeleton files
- [ ] Root `requirements.txt` either removed or replaced with a one-line comment pointing to `api/requirements.txt`

---



### 1.2 Environment and configuration

**Goal:** All secrets and service URLs come from environment variables.

**Tasks:**

- Create `docker/.env.example` with documented keys:

  | Variable                | Purpose                                       | Example                                                 |
  | ----------------------- | --------------------------------------------- | ------------------------------------------------------- |
  | `POSTGRES_USER`         | DB role                                       | `aion`                                                  |
  | `POSTGRES_PASSWORD`     | DB password                                   | `changeme`                                              |
  | `POSTGRES_DB`           | Database name                                 | `aion`                                                  |
  | `DATABASE_URL`          | Async SQLAlchemy URL (API)                    | `postgresql+asyncpg://aion:changeme@postgres:5432/aion` |
  | `CORS_ORIGINS`          | Comma-separated allowed origins for external clients (browser or cross-origin agent services) | `http://localhost:3000,http://localhost:8001` |
  | `LOG_LEVEL`             | API log level                                 | `info`                                                  |
  | `API_HOST` / `API_PORT` | Bind address (documented; compose sets these) | `0.0.0.0` / `8000`                                      |

- Implement `api/app/config.py` using `pydantic-settings` (`BaseSettings`)
- Add `docker/.env.example` copy instructions to a short comment in root `env.example`

**Acceptance criteria:**

- [ ] No secrets committed; `.env` remains gitignored
- [ ] API fails fast at startup if `DATABASE_URL` is missing
- [ ] CORS origins configurable without code changes

**Deferred to Step 2:** `OPENROUTER_API_KEY`, embedding model keys

---



### 1.3 API dependencies

**Goal:** Pin minimal dependencies for the infrastructure slice only.

**File:** `api/requirements.txt`

```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.6.0
sqlalchemy[asyncio]>=2.0.36
asyncpg>=0.30.0
```

**Explicitly excluded from Step 1** (added in Step 2): `langgraph`, `httpx`, embedding/LLM SDKs

**Acceptance criteria:**

- [ ] `pip install -r api/requirements.txt` succeeds in a clean venv
- [ ] No unused heavy deps (LangGraph, torch, etc.) at this stage

---



### 1.4 FastAPI skeleton

**Goal:** Minimal service with observability endpoints and CORS for external clients (separate agent repository).

**Tasks:**

- `api/main.py`: create app, attach CORS middleware, include router
- `GET /health` — liveness probe; returns `{"status": "ok"}` (no DB call)
- `GET /ready` — readiness probe; executes `SELECT 1` via async SQLAlchemy session
- Reserve API prefix `/api/v1/` in comments/structure for Step 2 routes (no routes yet)
- Log startup: config loaded (redact password from DATABASE_URL)

**Acceptance criteria:**

- [ ] `curl http://localhost:8000/health` → 200
- [ ] `curl http://localhost:8000/ready` → 200 when Postgres is up; 503 when Postgres is down
- [ ] CORS headers present for configured origins
- [ ] App starts with `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` inside container

---



### 1.5 Database image and extension bootstrap

**Goal:** PostgreSQL with pgvector **installed and enabled**, not just present in the image.

**Tasks:**

- Use `pgvector/pgvector:pg16` as the Postgres service image
- Add `db/init/01_extensions.sql`:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```
- Mount `db/init/` to `/docker-entrypoint-initdb.d/` so extensions are created on first boot
- Document that **init scripts run only on first volume creation** — re-applying requires a fresh volume (`docker compose down -v`) or manual SQL (see Step 2 §2.1 for schema additions)

**Acceptance criteria:**

- [ ] Fresh `docker compose up` runs init script without error
- [ ] `\dx` in psql shows `vector` extension installed
- [ ] `SELECT '[1,2,3]'::vector;` succeeds

---



### 1.6 Docker Compose

**Goal:** One command starts Postgres + API on a shared network with persistent storage.

**File:** `docker/docker-compose.yml`

**Services:**


| Service    | Image / build            | Ports                         | Notes                                                                    |
| ---------- | ------------------------ | ----------------------------- | ------------------------------------------------------------------------ |
| `postgres` | `pgvector/pgvector:pg16` | `5432:5432` (optional expose) | Named volume `postgres_data`; healthcheck via `pg_isready`               |
| `api`      | build `../api`           | `8000:8000`                   | Depends on `postgres` with `condition: service_healthy`; env from `.env` |


**Tasks:**

- Write `api/Dockerfile` (python:3.12-slim, install deps, copy app, CMD uvicorn)
- Map `./db/init` → `/docker-entrypoint-initdb.d`
- Add Postgres healthcheck; API waits for healthy Postgres
- Bind-mount `api/` for dev hot-reload (via override file or compose dev profile)

**Acceptance criteria:**

- [ ] `docker compose -f docker/docker-compose.yml up --build` starts both services
- [ ] Postgres data persists after `docker compose down` + `up` (without `-v`)
- [ ] API container can resolve hostname `postgres` on the internal network

---



### 1.7 Verification and smoke test

**Goal:** Prove the stack is production-shaped enough for Step 2 to build on.

**Manual checklist:**

```bash
# From repo root
cp docker/.env.example docker/.env   # fill if needed
docker compose -f docker/docker-compose.yml up --build -d

curl -sf http://localhost:8000/health
curl -sf http://localhost:8000/ready

docker compose -f docker/docker-compose.yml exec postgres \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extname FROM pg_extension WHERE extname = 'vector';"

docker compose -f docker/docker-compose.yml down
docker compose -f docker/docker-compose.yml up -d
curl -sf http://localhost:8000/ready   # persistence check
```

**Acceptance criteria:**

- [ ] All commands above pass on a clean machine with Docker installed
- [ ] `/ready` returns non-200 when Postgres is stopped (proves probe is real)
- [ ] No manual SQL needed after first boot to enable pgvector

---



## 6. Definition of Done

Step 1 is complete when:

1. Repository layout matches §4
2. `docker compose up --build` yields a healthy API + Postgres with pgvector verified
3. `/health` and `/ready` behave as specified
4. Configuration is env-driven via `docker/.env.example`
5. No ingestion, retrieval, LangGraph, or consumer-agent code exists yet
6. Changes are committed; VG-2 acceptance criteria in Jira can be checked off

---



## 7. Out of Scope (Explicit Deferrals)


| Item                                     | Deferred to     |
| ---------------------------------------- | --------------- |
| LangGraph, embeddings, OpenRouter client | Step 2          |
| `documents` table / flat store schema    | Step 2          |
| Alembic migrations                       | Phase 2, Step 2 |
| `pg_trgm` extension                      | Phase 3         |
| Consumer agent / pre-fetch hook          | External agent repo (not this codebase) |
| Local file storage volume                | Phase 4         |
| MCP tool wrappers                        | Post-MVP        |


---



## 8. Improvements Over Original Plan (1.1.1–1.1.7)


| Gap in original                              | Addressed by                            |
| -------------------------------------------- | --------------------------------------- |
| pgvector assumed but never verified          | §1.5 init SQL + §1.7 smoke test         |
| `/db` folder purpose undefined               | §1.5 `db/init/` for extension bootstrap |
| Root vs service `requirements.txt` ambiguous | §3 decision + §1.3 scoped deps          |
| LangGraph in Step 1 deps (unused)            | Removed until Step 2                    |
| Only `/health`, no readiness distinction     | §1.4 `/ready` with DB probe             |
| `.env.example` location unclear              | §1.2 canonical `docker/.env.example`    |
| No CORS origin configuration detail          | §1.2 `CORS_ORIGINS` env var             |
| Compose location unspecified                 | §1.6 under `docker/`                    |
| Consumer agent in repo                       | Explicitly excluded — separate repository |
| No definition of done                        | §6                                      |


---



## 9. Jira Subtasks

Parent story: [VG-2](https://jira.atlassian.com/browse/VG-2) — Phase 1 Step 1 — Infrastructure Foundation

| # | Jira | Subtask | Implementation step |
|---|------|---------|---------------------|
| 1 | [VG-9](https://jira.atlassian.com/browse/VG-9) | Scaffold repository layout | §1.1 Repository scaffolding |
| 2 | [VG-10](https://jira.atlassian.com/browse/VG-10) | Configure env and pydantic-settings | §1.2 Environment and configuration |
| 3 | [VG-6](https://jira.atlassian.com/browse/VG-6) | Add API requirements and Dockerfile | §1.3 API dependencies + §1.6 Docker Compose |
| 4 | [VG-7](https://jira.atlassian.com/browse/VG-7) | Implement /health and /ready | §1.4 FastAPI skeleton |
| 5 | [VG-8](https://jira.atlassian.com/browse/VG-8) | Bootstrap pgvector via init SQL | §1.5 Database image and extension bootstrap |
| 6 | [VG-12](https://jira.atlassian.com/browse/VG-12) | Write docker-compose.yml | §1.6 Docker Compose |
| 7 | [VG-11](https://jira.atlassian.com/browse/VG-11) | Run smoke test checklist | §1.7 Verification and smoke test |


