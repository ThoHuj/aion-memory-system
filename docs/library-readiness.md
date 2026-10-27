# Library readiness

Aion is meant to be imported by external agent systems. Those hosts provide the Postgres server (with pgvector); this library only connects to it. This file tracks what is still needed to make that a well-behaved, professional library.

It is a reminder list, not a task list. Nothing here is assigned work. When an item is picked up, turn it into a scoped ticket and remove it from this file.

Items marked **Decided** reflect decisions already made. They are recorded with their reasoning in the orchestration repo's `docs/architecture.md`, and the steps that follow from them are in its `docs/memory-integration-plan.md`. Items marked **Open** or **Not decided** still need an owner decision.

Items are grouped by when they become expensive to change.

## 1. Decide before others depend on the API

These shape the public surface. Changing them after other code depends on the library is a breaking change.

- [x] **Connection ownership.** **Decided:** hosts pass a `database_url` only. The library manages its own connections, with a pool internally, and offers a clean shutdown.
  - Done: `PostgresFactsStore.connect` opens a pool (1 to 5 connections), each operation borrows a connection, and `MemorySystem.close()` closes the pool. An unreachable database fails right away with psycopg's own error.
  - Possibly later: also accept a host-managed pool. Adding that does not break existing hosts. When it is added, close only what the library opened itself, and do not change connection-level settings (`autocommit`, `row_factory`) on a borrowed connection.
  - *Why:* a minimal integration is one string, and the pool option stays available without committing to it now.
- [ ] **Sync vs. async.** **Decided:** sync first. Async methods may be added alongside later. The pool question is covered by connection ownership.
  - Expected hosts for now are CLIs and headless agents. Running real sessions shows where latency matters (`add()` makes several model calls) before anything is optimized.
  - Adding async later does not break hosts, but it can mean reworking the internals. Revisit when a host needs it.
- [x] **Schema setup.** **Decided:** tables are created automatically on start by default, with a switch to turn that off for hosts that run setup in their own pipeline.
  - Done: `create_memory_system(..., create_schema=False)` leaves the database untouched, and `schema_sql()` returns the setup SQL for hosts that run it themselves.
  - Done: a missing pgvector, whether not installed or not allowed to be enabled, stops setup with a clear message. The check also runs, read-only, when setup is off. Other database errors are passed on unchanged.
  - Note: enabling pgvector usually needs a superuser. It was not marked trusted in the build used for testing.
  - *Why:* first use stays a single step, and hosts that care about their database permissions keep control.
- [ ] **Versioned migrations.** **Not decided:** numbered, idempotent migrations tracked in a version table inside the library's own schema, so the schema can evolve safely.
  - *Why:* without them, a later change to a table cannot be applied to databases that already have the old one.
- [ ] **Interfaces for storage and model clients.**
  - **Decided (storage), done:** the facts store sits behind an internal interface (`FactsStore`) that the components depend on. Postgres (`PostgresFactsStore`) is the only shipped backend. The interface is private: hosts implement nothing, and it is not exported. Setup (`create_schema`) is not part of the interface.
  - **Open (model clients):** `create_memory_system` currently hard-wires `OpenRouterClient` for extraction and embeddings. Whether and how hosts can use other providers is not decided.
  - *Why:* the internal interface keeps the components testable without a database, and it makes a second backend possible later without touching hosts.

## 2. Before the first release

- [ ] **Be a good tenant in a shared database.** Use a dedicated Postgres schema (e.g. `aion_memory`) or a configurable table prefix, and keep per-user / per-agent scoping in every query.
  - *Why:* the library will share a database with the host's own tables and with other agents.
- [x] **Scope everything by user.** **Decided:** the library is multi-user ready, so every store lookup is scoped by `user_id`.
  - Done: `get_fact`, `get_add_log_record`, `get_most_recent_session_messages` and `MemorySystem.get` take a `user_id`. Another user's data is reported as unknown (`KeyError`).
  - Also done: the invalidation `UPDATE` in `store_graph_update` only closes facts of the user the update belongs to.
  - *Why:* a host serving several users must not be able to read across them by accident.
- [ ] **Removing a user's data.** **Not designed:** the library keeps its own copy of the messages it is given (decided), so a host that must erase a user needs a way to remove that user's `add_log` rows, entities and facts. `delete` is currently only a commented-out stub in `memory_system.py`.
  - *Why:* erasing a user has to cover the library's copy as well as the host's history.
- [ ] **Vector index.** `entities.embedding` is declared as `VECTOR` without a dimension, so pgvector cannot build an index on it and every similarity search scans the user's entities. Fine at small scale; revisit if it becomes slow.
- [ ] **Configuration is explicit.** The package reads no environment variables and loads no `.env`; configuration comes through arguments or a typed config object. Env handling belongs to the host; nothing in this repo reads the environment.
  - *Why:* hidden environment reads make a library hard to embed and hard to test.
- [ ] **Own exception hierarchy.** Raise library-specific exceptions rather than leaking raw `psycopg` or HTTP errors.
  - *Why:* hosts need stable errors to catch without depending on the library's internals.
- [ ] **Logging etiquette.** Use `logging.getLogger(__name__)` and never configure handlers.
  - *Why:* the host decides where logs go.
- [ ] **Tests against real Postgres.** Run storage tests against the `pgvector/pgvector` image (testcontainers locally, a service container in CI), ideally across more than one Postgres version.
  - Done locally: the test run starts a throwaway pgvector container with testcontainers, and the store and pipeline tests run on it as well as on the in-memory store. Without Docker they skip. The repo no longer has a compose file or `.env`.
  - Still open: running them in CI, and more than one Postgres version.
  - *Why:* pgvector queries and transactions are where bugs hide, and mocks do not catch them.
- [x] **Document the requirements.** README states the Postgres version, the pgvector requirement, and the privileges needed for setup. A plain `postgres` image does not work.
  - Done: the "Database requirements" section in the README. The privilege claims are pinned by tests. Only PostgreSQL 16 and 17 were tried; other versions are open (see the real-Postgres tests item).
  - *Why:* this is the first thing a new host runs into.
- [x] **No demo code in the library repo.** The mock agent, its CLI and the other `dev/` tools were removed; `aion-agent-system` is the reference host.
  - Done: `python-dotenv` is no longer a dependency, and the chat methods only the mock agent used are gone from the OpenRouter client.

## 3. Packaging and polish

- [x] **Small public API.** Export only what hosts need from `__init__.py` with `__all__`; keep everything else private. The internal store interface and the Postgres store stay out of the exports.
  - Done: the package exports `create_memory_system`, `MemorySystem`, its result types and `schema_sql`. A test pins the list.
- [ ] **Typing and versioning.** Ship `py.typed`, follow semantic versioning, and keep a changelog.
- [ ] **Loose dependency bounds.** Use wide ranges for library dependencies (e.g. `psycopg>=3.1,<4`) rather than pinning; the lockfile is for development only.
- [x] **In-memory store implementation.** **Decided:** an in-memory implementation of the internal store interface, used for fast tests only and not offered to hosts. Run the same contract tests against it and the Postgres store.
  - Done: it lives in `tests/in_memory_facts_store.py`, and the store-level and pipeline tests run on both stores.

## Alternative to keep in mind

The other common architecture is running the memory system as its own service (own server, own database, an API in front) instead of an embedded library. This project is deliberately the embedded-library route, with the host supplying Postgres, so the items above apply.
