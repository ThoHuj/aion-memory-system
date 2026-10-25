"""Setting up the database: automatic by default, optional, and clear about pgvector."""

import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from aion_memory import create_memory_system, schema_sql
from aion_memory.facts_store import PostgresFactsStore
from aion_memory.graph_models import FactInvalidation
from tests.helpers import add_fact, entity, fact, store_update

TABLES = {"add_log", "entities", "facts"}


@dataclass
class RoleDatabase:
    """A fresh database with a role that has no privileges in it."""

    role: str
    role_url: str
    admin_url: str


def table_names(database_url: str) -> set[str]:
    """Return the names of the tables in the first schema of the connection's search path."""
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = current_schema()"
        ).fetchall()
    return {row[0] for row in rows}


def create_memory(database_url: str, **options: bool):
    """Create a memory system with placeholder model settings; none is called."""
    return create_memory_system(
        "key", "chat-model", "embedding-model", database_url, **options
    )


def role_database(
    admin_connection: psycopg.Connection, database_url: str, enable_pgvector: bool
) -> Iterator[RoleDatabase]:
    """Yield a fresh database and a role without privileges in it, then remove both."""
    suffix = uuid.uuid4().hex[:12]
    database_name = f"test_db_{suffix}"
    role_name = f"test_role_{suffix}"
    try:
        admin_connection.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name))
        )
        admin_connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'password'").format(
                sql.Identifier(role_name)
            )
        )
    except psycopg.errors.InsufficientPrivilege:
        pytest.skip("The database user may not create databases and roles")
    try:
        admin_url = make_conninfo(database_url, dbname=database_name)
        with psycopg.connect(admin_url, autocommit=True) as database_admin:
            if enable_pgvector:
                database_admin.execute("CREATE EXTENSION vector")
            database_admin.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        yield RoleDatabase(
            role=role_name,
            role_url=make_conninfo(admin_url, user=role_name, password="password"),
            admin_url=admin_url,
        )
    finally:
        admin_connection.execute(
            sql.SQL("DROP DATABASE {} WITH (FORCE)").format(
                sql.Identifier(database_name)
            )
        )
        admin_connection.execute(
            sql.SQL("DROP ROLE {}").format(sql.Identifier(role_name))
        )


@pytest.fixture
def database_without_pgvector(
    database_url: str, admin_connection: psycopg.Connection
) -> Iterator[RoleDatabase]:
    """A database where pgvector is not enabled, and a role that cannot enable it."""
    yield from role_database(admin_connection, database_url, False)


@pytest.fixture
def database_with_pgvector(
    database_url: str, admin_connection: psycopg.Connection
) -> Iterator[RoleDatabase]:
    """A database where pgvector is enabled, and a role with no other privileges."""
    yield from role_database(admin_connection, database_url, True)


def grant(database: RoleDatabase, statement: str) -> None:
    """Run a GRANT for the role as the database's administrator."""
    with psycopg.connect(database.admin_url, autocommit=True) as connection:
        connection.execute(
            sql.SQL(statement).format(role=sql.Identifier(database.role))
        )


def test_tables_are_created_by_default(test_schema_url: str) -> None:
    """A fresh pgvector database works on the first run, and setup can run again."""
    for _ in range(2):
        memory_system = create_memory(test_schema_url)
        memory_system.close()

    assert table_names(test_schema_url) == TABLES


def test_nothing_is_created_when_setup_is_off(test_schema_url: str) -> None:
    """With the switch off the library leaves the database untouched."""
    memory_system = create_memory(test_schema_url, create_schema=False)
    memory_system.close()

    assert table_names(test_schema_url) == set()


def test_the_raw_sql_sets_up_a_database_by_hand(test_schema_url: str) -> None:
    """A host that runs schema_sql() itself can use the library with setup off."""
    with psycopg.connect(test_schema_url, autocommit=True) as connection:
        connection.execute(schema_sql())  # type: ignore[arg-type]

    memory_system = create_memory(test_schema_url, create_schema=False)
    try:
        with pytest.raises(KeyError):
            memory_system.get("unknown-fact", "alice")
    finally:
        memory_system.close()


def test_missing_pgvector_gives_a_clear_message(
    database_without_pgvector: RoleDatabase,
) -> None:
    """When pgvector cannot be enabled, setup says what is needed instead of a raw database error."""
    with pytest.raises(RuntimeError, match="pgvector is not available"):
        create_memory(database_without_pgvector.role_url)


def test_missing_pgvector_is_found_even_with_setup_off(
    database_without_pgvector: RoleDatabase,
) -> None:
    """The pgvector check is read-only, so it also runs when the host sets up the database."""
    with pytest.raises(RuntimeError, match="pgvector is not available"):
        create_memory(database_without_pgvector.role_url, create_schema=False)


def test_other_permission_errors_are_not_reported_as_missing_pgvector(
    database_with_pgvector: RoleDatabase,
) -> None:
    """A role that cannot create tables gets the database's own error, not the pgvector message."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        create_memory(database_with_pgvector.role_url)


def test_setup_needs_no_superuser_once_pgvector_is_enabled(
    database_with_pgvector: RoleDatabase,
) -> None:
    """With pgvector already enabled, a role that may create tables can run the setup."""
    grant(database_with_pgvector, "GRANT CREATE ON SCHEMA public TO {role}")

    memory_system = create_memory(database_with_pgvector.role_url)
    memory_system.close()

    assert table_names(database_with_pgvector.role_url) == TABLES


def test_select_insert_and_update_are_enough_with_setup_off(
    database_with_pgvector: RoleDatabase,
) -> None:
    """A role with only those rights on the tables can add facts, read them and invalidate them."""
    with psycopg.connect(database_with_pgvector.admin_url, autocommit=True) as admin:
        admin.execute(schema_sql())  # type: ignore[arg-type]
    grant(
        database_with_pgvector,
        "GRANT SELECT, INSERT, UPDATE ON add_log, entities, facts TO {role}",
    )
    owns = fact("alice", "owns", "rex")

    memory_system = create_memory(database_with_pgvector.role_url, create_schema=False)
    memory_system.close()
    store = PostgresFactsStore.connect(database_with_pgvector.role_url)
    try:
        store.require_pgvector()
        add_fact(store, "bob", "session", "hello")
        store_update(
            store,
            "alice",
            [entity("alice", [0, 0, 1]), entity("rex", [1, 0, 0])],
            [owns],
        )
        store_update(
            store,
            "alice",
            [],
            [],
            minutes=1,
            invalidations=[FactInvalidation(fact_id=owns.fact_id, world_valid_to=None)],
        )
        stored = store.get_fact("alice", owns.fact_id)
    finally:
        store.close()

    assert stored.valid_to is not None
