# Aion memory

Long-term memory for conversational agents, as a Python library.

After each user–assistant turn, the host agent passes the messages to the library. The library **extracts** entities and relational facts from them, **stores** them in a per-user knowledge graph (PostgreSQL + pgvector), and **resolves** duplicates and contradictions. On later turns it **retrieves** the facts relevant to a query, so the host can put them in front of its model. Every fact keeps a link to the messages it came from, so a host can show where a memory came from.

The library is imported by host agents, which supply the Postgres server. It has no agent, CLI or entry point of its own. `aion-agent-system` is the reference host.

---

## Usage

```python
from aion_memory import create_memory_system

memory_system = create_memory_system(
    api_key=openrouter_api_key,
    extraction_model="<an OpenRouter chat model with structured output>",
    embedding_model="<an OpenRouter embedding model>",
    database_url="postgresql://user:password@host:5432/database",
)
try:
    # After a turn: extract and store what was learned.
    memory_system.add(
        [
            {"role": "user", "content": "I moved to Utrecht last month."},
            {"role": "assistant", "content": "Nice, how do you like it?"},
        ],
        user_id="user_01",
        session_id="session_01",
    )

    # Before the next turn: fetch the facts relevant to the user's message.
    result = memory_system.retrieve("Where do I live?", user_id="user_01", limit=15)
    for fact_data in result.retrieved_facts:
        fact = fact_data.fact
        print(fact.fact_id, fact.source_entity, fact.relation, fact.destination_entity)

    # On demand: the original messages a fact was extracted from.
    sources = memory_system.get(fact.fact_id, user_id="user_01").origin.messages
finally:
    memory_system.close()
```

The library reads no environment variables. The host decides where its settings come from.

### Install

In a uv project next to this repo:

```toml
[project]
dependencies = ["aion-memory"]

[tool.uv.sources]
aion-memory = { path = "../aion-memory-system", editable = true }
```

---

## Public API

Everything a host needs is exported from `aion_memory`. Other modules are internal.

`create_memory_system(api_key, extraction_model, embedding_model, database_url, create_schema=True)` builds a `MemorySystem`. It opens a small pool of database connections of its own, so calls from several threads do not share one connection.

`MemorySystem`:

| Method | Purpose |
|--------|---------|
| `add(messages, user_id, session_id)` | Run the full extraction pipeline; return a `MemoryAddResult` with the entities, new facts and invalidations |
| `retrieve(query, user_id, limit)` | A `RetrievalResult` with up to `limit` facts, BM25-ranked after graph retrieval |
| `get(fact_id, user_id)` | A `FactData`: the stored fact plus the messages it was extracted from. Raises `KeyError` for an unknown fact or another user's fact |
| `export_graph(user_id)` | The user's whole graph as a `GraphExport`: every entity and fact, ended facts included |
| `close()` | Release the database connections; call once at shutdown |

`render_graph_html(graph)` turns a `GraphExport` into a standalone HTML page that draws the graph. It touches no database or files; the host decides where the page goes. The page loads a pinned vis-network from unpkg, so viewing it needs internet access.

`schema_sql()` returns the setup SQL, for hosts that set up the database themselves.

`update` and `delete` are not implemented yet.

---

## Database requirements

- **PostgreSQL with pgvector.** A plain `postgres` image does not work. The library is developed against PostgreSQL 17 (the `pgvector/pgvector:pg17` image in `docker-compose.yml`), and the tests also pass on PostgreSQL 16. Other versions are untested.
- **Setup.** By default `create_memory_system` enables the extension and creates the tables if they are missing. Pass `create_schema=False` to leave the database untouched and run the SQL from `schema_sql()` through your own pipeline instead. If pgvector is not available, the library stops with an error that says what is needed.
- **Where the tables go.** `add_log`, `entities` and `facts` are created in the connection's default schema. They are not prefixed or namespaced yet, so avoid a database where those names are taken.
- **Privileges for setup, once.**
  - Enabling pgvector (`CREATE EXTENSION vector`, which the library does unless you pass `create_schema=False`) usually needs a superuser, because the extension was not marked trusted in the build used for testing. On a managed server, ask for it to be enabled once per database.
  - With pgvector enabled, a role that may create tables in the schema can run the rest of the setup without further rights.
- **Privileges for everyday use.** `SELECT`, `INSERT` and `UPDATE` on the three tables are enough. To run with those rights only, set the tables up yourself with `schema_sql()` and pass `create_schema=False`.

### Models

The library calls [OpenRouter](https://openrouter.ai/) for extraction and embeddings. It needs:

- an extraction model that supports structured JSON output, and
- an embedding model.

Structured extraction and conflict resolution need capable models; weak models may yield empty or inconsistent graphs.

---

## Architecture

### Data flow

```mermaid
flowchart TB
  Host[Host agent]

  subgraph memsys["MemorySystem"]
    Add[add]
    Retrieve[retrieve]
    Get[get]
  end

  subgraph addp["Add pipeline"]
    CM[ContextManager]
    EE[EntityExtractor]
    RE[RelationExtractor]
    EM[EmbeddingManager]
    ER[EntityResolver]
    CR[ConflictResolver]
  end

  subgraph retp["Retrieve pipeline"]
    GR[GraphRetriever]
  end

  FS[FactsStore]

  Host --> Retrieve
  Retrieve --> GR
  GR --> FS
  Host --> Add
  Add --> CM
  CM --> EE
  EE --> RE
  RE --> EM
  EM --> ER
  ER --> CR
  CR --> FS
  Host --> Get
  Get --> FS
```

### Memory model

1. **Entities**: typed nodes (person, place, concept, etc.), with **embedding vectors** for similarity search.
2. **Facts**: directed edges `source -- relation -- destination`, with two time dimensions:
   - **`valid_from` / `valid_to`**: when the fact was believed true in the **system** (belief interval).
   - **`world_valid_from` / `world_valid_to`**: when the fact was true in the **real world**, as inferred from dialogue (e.g. "I moved last month").
3. **`add_log`**: each `add()` call records the **exact message list** used for extraction (`add_id`), so `get(fact_id, user_id)` can return a fact's sources.

Retrieved facts are compact summaries. The full dialogue behind a fact is fetched **on demand** through `get()`, for example from a tool the host gives its model.

### Add pipeline

Implemented in `MemorySystem.add()` (`src/aion_memory/memory_system.py`):

1. **Context**: `ContextManager` formats the new messages together with recent messages of the same session from earlier `add_log` rows.
2. **Entity extraction**: LLM structured output (`EntityExtractor`).
3. **Relation extraction**: the LLM extracts fact edges among the entities (`RelationExtractor`).
4. **Embeddings**: OpenRouter embeddings for entity names (`EmbeddingManager`).
5. **Entity resolution**: new entities are matched to existing ones by vector similarity plus LLM disambiguation (`EntityResolver`). A stable **user entity** is keyed by `user_id`.
6. **Conflict resolution**: duplicate new facts are dropped, and the LLM proposes **invalidations** of stored facts that the new information contradicts (`ConflictResolver`).
7. **Persist**: `store_graph_update()` writes the entities, facts, invalidations and the add log in one transaction.

### Retrieve pipeline

`GraphRetriever.retrieve_facts()`:

1. Extract **seed entities** from the query (LLM).
2. Seed the graph from the **user entity** and **vector-similar** stored entities.
3. **Multi-hop expansion** (2 hops by default) over valid fact edges.
4. **Rank** candidate facts with **BM25** over `source relation destination` text and return the top `limit`.

### Database schema

Defined in `src/aion_memory/facts_store/schema.sql`:

- **`add_log`**: provenance: `add_id`, `user_id`, `session_id`, `messages` (JSONB), `created_at`
- **`entities`**: `entity_id`, `user_id`, `name`, `entity_type`, `embedding` (vector)
- **`facts`**: edges between entities, validity columns, `add_id` foreign key

Invalidated facts are closed with **`valid_to`** / **`world_valid_to`** rather than deleted.

### Design choices

- **Knowledge graph instead of a flat memory list.** Relations support multi-hop retrieval and explicit handling of contradictions between edges that share endpoints.
- **LLM-heavy extraction.** Staged extraction modelled on [Mem0](https://github.com/mem0ai/mem0) (a conceptual reference, not a dependency); quality depends on the model and on the prompts in `src/aion_memory/*/prompts.py`.
- **Hybrid retrieval.** Graph traversal from embedding-matched seeds plus BM25 re-ranking balances recall and relevance to the query.
- **Provenance as a first-class feature.** `add_log` and `get()` let a host answer from the original words instead of the compressed fact line.
- **Storage behind an internal interface.** The components depend on `FactsStore`; Postgres is the only shipped backend, and an in-memory store is used for fast tests. Hosts implement nothing.

---

## Project layout

```
aion-memory-system/
├── pyproject.toml          # Package metadata and dependencies (uv)
├── docker-compose.yml      # Postgres + pgvector on localhost:5433, for development and tests
├── .env.example            # Copy to .env for the tests (not committed)
├── src/aion_memory/        # The library (import as `aion_memory`)
│   ├── __init__.py         # The public API
│   ├── factory.py          # create_memory_system
│   ├── memory_system.py    # MemorySystem: add, retrieve, get, export_graph, close
│   ├── facts_store/        # Internal storage interface, Postgres store, SQL schema
│   ├── entity_extraction/
│   ├── relation_extraction/
│   ├── entity_resolution/
│   ├── conflict_resolution/
│   ├── graph_retriever.py
│   ├── context_manager.py
│   ├── embedding_manager.py
│   ├── graph_models.py
│   ├── graph_view.py       # render_graph_html
│   └── open_router_client.py
├── tests/                  # pytest suite: in-memory store, plus Postgres when available
└── docs/
    └── library-readiness.md  # What is still needed to be a well-behaved library
```

---

## Development

```bash
uv sync
uv run pytest
```

The store and pipeline tests run on an in-memory store and use scripted stand-ins for the models, so they need no API key. When `DATABASE_URL` points to a reachable Postgres with pgvector, they also run on Postgres, in a throwaway schema that is dropped afterwards. Without a database the Postgres variants are skipped.

To run them against Postgres locally:

```bash
cp .env.example .env    # set POSTGRES_PASSWORD and the matching DATABASE_URL
docker compose up -d
uv run pytest
```

`.env` is gitignored. Do not commit passwords.

---

## Known limitations

- **Cost and latency.** Each `add()` and `retrieve()` makes several LLM and embedding calls, and the API is synchronous.
- **Tests do not call real models.** Extraction quality is checked by running a host with real models.
- **Only OpenRouter.** The model client is fixed; other providers are not supported yet.
- See [docs/library-readiness.md](docs/library-readiness.md) for what is still open before a first release.
