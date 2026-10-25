from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, LiteralString

import psycopg
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from aion_memory.graph_models import (
    AddLogRecord,
    GraphUpdate,
    StoredEntity,
    StoredFact,
)

if TYPE_CHECKING:
    from aion_memory.context_manager import ContextResult

FACT_SELECT = (
    "SELECT facts.*, "
    "source_entities.name AS source_entity, "
    "destination_entities.name AS destination_entity "
    "FROM facts "
    "JOIN entities AS source_entities "
    "ON source_entities.entity_id = facts.source_entity_id "
    "JOIN entities AS destination_entities "
    "ON destination_entities.entity_id = facts.destination_entity_id"
)

CONNECT_TIMEOUT_SECONDS = 10
POOL_MAX_SIZE = 5

PGVECTOR_MISSING_MESSAGE = (
    "pgvector is not available in this database. Aion needs Postgres with the "
    "pgvector extension installed (for example the pgvector/pgvector image) and "
    "enabled in the database with CREATE EXTENSION vector, which needs a role "
    "with enough privileges, often a superuser."
)


def schema_sql() -> str:
    """Return the SQL that enables pgvector and creates the tables; safe to run repeatedly."""
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def _vector_literal(embedding: list[float]) -> str:
    """Format an embedding as a pgvector text literal."""
    return "[" + ",".join(map(str, embedding)) + "]"


def _optional_isoformat(value: datetime | None) -> str | None:
    """Convert an optional datetime to an ISO string."""
    return value.isoformat() if value is not None else None


def _optional_datetime(value: str | None) -> datetime | None:
    """Convert an optional ISO string to a datetime."""
    return datetime.fromisoformat(value) if value else None


def _to_stored_fact(row: dict[str, Any]) -> StoredFact:
    """Convert a fact row to a StoredFact."""
    return StoredFact(
        fact_id=row["fact_id"],
        user_id=row["user_id"],
        source_entity_id=row["source_entity_id"],
        source_entity=row["source_entity"],
        relation=row["relation"],
        destination_entity_id=row["destination_entity_id"],
        destination_entity=row["destination_entity"],
        valid_from=row["valid_from"].isoformat(),
        valid_to=_optional_isoformat(row["valid_to"]),
        world_valid_from=row["world_valid_from"].isoformat(),
        world_valid_to=_optional_isoformat(row["world_valid_to"]),
        add_id=row["add_id"],
    )


def _to_stored_entity(row: dict[str, Any]) -> StoredEntity:
    """Convert an entity row to a StoredEntity."""
    return StoredEntity(
        entity_id=row["entity_id"],
        name=row["name"],
        entity_type=row["entity_type"],
    )


class PostgresFactsStore:
    """Persists the knowledge graph and add log in Postgres with pgvector."""

    def __init__(self, pool: ConnectionPool[psycopg.Connection[DictRow]]) -> None:
        self._pool = pool

    @classmethod
    def connect(cls, database_url: str) -> PostgresFactsStore:
        """Open a pool of connections to database_url; close() releases them."""
        options: dict[str, Any] = {
            "autocommit": True,
            "row_factory": dict_row,
            "connect_timeout": CONNECT_TIMEOUT_SECONDS,
        }
        # Connect once up front so an unreachable database fails with its own error.
        psycopg.connect(database_url, **options).close()
        pool: ConnectionPool[psycopg.Connection[DictRow]] = ConnectionPool(
            database_url,
            min_size=1,
            max_size=POOL_MAX_SIZE,
            kwargs=options,
            open=False,
        )
        try:
            pool.open(wait=True, timeout=CONNECT_TIMEOUT_SECONDS)
        except BaseException:
            pool.close()
            raise
        return cls(pool)

    def close(self) -> None:
        """Close the pool; the store cannot be used afterwards."""
        self._pool.close()

    def create_schema(self) -> None:
        """Apply schema.sql idempotently; raises RuntimeError if pgvector cannot be enabled."""
        try:
            with self._pool.connection() as connection:
                connection.execute(schema_sql())  # type: ignore[arg-type]
        except (
            psycopg.errors.FeatureNotSupported,
            psycopg.errors.InsufficientPrivilege,
        ) as error:
            if self._has_pgvector():
                raise
            raise RuntimeError(PGVECTOR_MISSING_MESSAGE) from error

    def require_pgvector(self) -> None:
        """Raise RuntimeError unless pgvector is enabled in the database; changes nothing."""
        if not self._has_pgvector():
            raise RuntimeError(PGVECTOR_MISSING_MESSAGE)

    def _has_pgvector(self) -> bool:
        """Return whether the vector extension is enabled in the database."""
        row = self._fetch_one(
            "SELECT 1 AS present FROM pg_extension WHERE extname = 'vector'", ()
        )
        return row is not None

    def _fetch_one(
        self, query: LiteralString, params: tuple[Any, ...]
    ) -> DictRow | None:
        """Run a query on a pooled connection and return the first row, if any."""
        with self._pool.connection() as connection:
            return connection.execute(query, params).fetchone()

    def _fetch_all(
        self, query: LiteralString, params: tuple[Any, ...]
    ) -> list[DictRow]:
        """Run a query on a pooled connection and return all rows."""
        with self._pool.connection() as connection:
            return connection.execute(query, params).fetchall()

    def find_entity_by_name(self, user_id: str, name: str) -> StoredEntity | None:
        """Return the oldest entity of user_id with the given name, if any."""
        row = self._fetch_one(
            "SELECT entity_id, name, entity_type FROM entities "
            "WHERE user_id = %s AND name = %s "
            "ORDER BY created_at ASC LIMIT 1",
            (user_id, name),
        )
        return _to_stored_entity(row) if row is not None else None

    def find_similar_entities(
        self,
        user_id: str,
        embedding: list[float],
        minimum_similarity: float,
        limit: int,
    ) -> list[StoredEntity]:
        """Return entities by cosine similarity at or above the minimum, best first."""
        literal = _vector_literal(embedding)
        rows = self._fetch_all(
            "SELECT entity_id, name, entity_type FROM entities "
            "WHERE user_id = %s "
            "AND 1 - (embedding <=> %s::vector) >= %s "
            "ORDER BY embedding <=> %s::vector ASC "
            "LIMIT %s",
            (user_id, literal, minimum_similarity, literal, limit),
        )
        return [_to_stored_entity(row) for row in rows]

    def get_valid_facts_for_entities(
        self,
        user_id: str,
        entity_ids: list[str],
        limit: int,
    ) -> list[StoredFact]:
        """Return system-valid facts touching any entity id, newest first."""
        if not entity_ids:
            return []
        rows = self._fetch_all(
            FACT_SELECT + " WHERE facts.user_id = %s AND facts.valid_to IS NULL "
            "AND (facts.source_entity_id = ANY(%s) "
            "OR facts.destination_entity_id = ANY(%s)) "
            "ORDER BY facts.valid_from DESC LIMIT %s",
            (user_id, entity_ids, entity_ids, limit),
        )
        return [_to_stored_fact(row) for row in rows]

    def find_valid_fact_id(
        self,
        user_id: str,
        source_entity_id: str,
        relation: str,
        destination_entity_id: str,
    ) -> str | None:
        """Return the id of a valid fact with the same edge, if any."""
        row = self._fetch_one(
            "SELECT fact_id FROM facts "
            "WHERE user_id = %s AND source_entity_id = %s AND relation = %s "
            "AND destination_entity_id = %s AND valid_to IS NULL "
            "ORDER BY valid_from DESC LIMIT 1",
            (user_id, source_entity_id, relation, destination_entity_id),
        )
        return row["fact_id"] if row is not None else None

    def get_fact(self, user_id: str, fact_id: str) -> StoredFact:
        """Return a fact of user_id of any validity; raises KeyError if unknown or owned by another user."""
        row = self._fetch_one(
            FACT_SELECT + " WHERE facts.user_id = %s AND facts.fact_id = %s",
            (user_id, fact_id),
        )
        if row is None:
            raise KeyError(fact_id)
        return _to_stored_fact(row)

    def get_add_log_record(self, user_id: str, add_id: str) -> AddLogRecord:
        """Return the add log record for add_id of user_id; raises KeyError if unknown or owned by another user."""
        row = self._fetch_one(
            "SELECT add_id, user_id, session_id, messages, created_at "
            "FROM add_log WHERE user_id = %s AND add_id = %s",
            (user_id, add_id),
        )
        if row is None:
            raise KeyError(add_id)
        return AddLogRecord(
            add_id=row["add_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            messages=row["messages"],
            created_at=row["created_at"].isoformat(),
        )

    def get_most_recent_session_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Return up to limit most recent messages of the session of user_id, oldest first."""
        if limit <= 0:
            return []
        rows = self._fetch_all(
            "SELECT messages FROM add_log WHERE user_id = %s AND session_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (user_id, session_id, limit),
        )
        messages: list[dict[str, str]] = []
        for row in reversed(rows):
            messages.extend(row["messages"])
        return messages[-limit:]

    def store_graph_update(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
    ) -> None:
        """Persist the add log, new entities, invalidations, and facts in one transaction."""
        user_id = context_result.metadata.user_id
        added_at = datetime.fromisoformat(context_result.metadata.added_at)
        add_id = str(uuid.uuid4())
        entity_ids_by_name = graph_update.entity_ids_by_name()
        fact_endpoint_names = {
            name
            for fact in graph_update.facts
            for name in (fact.source_entity, fact.destination_entity)
        }
        with self._pool.connection() as connection, connection.transaction():
            connection.execute(
                "INSERT INTO add_log (add_id, user_id, session_id, messages, created_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    add_id,
                    user_id,
                    context_result.metadata.session_id,
                    Jsonb(context_result.messages),
                    added_at,
                ),
            )
            for entity in graph_update.entities:
                if not entity.is_new or entity.name not in fact_endpoint_names:
                    continue
                connection.execute(
                    "INSERT INTO entities "
                    "(entity_id, user_id, name, entity_type, embedding, created_at) "
                    "VALUES (%s, %s, %s, %s, %s::vector, %s)",
                    (
                        entity.entity_id,
                        user_id,
                        entity.name,
                        entity.entity_type,
                        _vector_literal(entity.embedding),
                        added_at,
                    ),
                )
            for invalidation in graph_update.invalidations:
                connection.execute(
                    "UPDATE facts SET valid_to = %s, "
                    "world_valid_to = COALESCE(world_valid_to, %s) "
                    "WHERE user_id = %s AND fact_id = %s AND valid_to IS NULL",
                    (
                        added_at,
                        _optional_datetime(invalidation.world_valid_to),
                        user_id,
                        invalidation.fact_id,
                    ),
                )
            for fact in graph_update.facts:
                fact.fact_id = str(uuid.uuid4())
                connection.execute(
                    "INSERT INTO facts "
                    "(fact_id, user_id, source_entity_id, relation, "
                    "destination_entity_id, valid_from, valid_to, "
                    "world_valid_from, world_valid_to, add_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s, NULL, %s, %s, %s)",
                    (
                        fact.fact_id,
                        user_id,
                        entity_ids_by_name[fact.source_entity],
                        fact.relation,
                        entity_ids_by_name[fact.destination_entity],
                        added_at,
                        datetime.fromisoformat(fact.world_valid_from),
                        _optional_datetime(fact.world_valid_to),
                        add_id,
                    ),
                )
