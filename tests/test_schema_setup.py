"""Setting up the database: automatic by default, optional, and clear about pgvector."""

import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from aion_memory import create_memory_system, schema_sql

TABLES = {"add_log", "entities", "facts"}


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


def limited_role_database(
    admin_connection: psycopg.Connection, database_url: str, enable_pgvector: bool
) -> Iterator[str]:
    """Yield a URL for a role without privileges in a fresh database, then remove both."""
    suffix = uuid.uuid4().hex[:12]
    database = sql.Identifier(f"test_db_{suffix}")
    role = sql.Identifier(f"test_role_{suffix}")
    try:
        admin_connection.execute(sql.SQL("CREATE DATABASE {}").format(database))
        admin_connection.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'password'").format(role)
        )
    except psycopg.errors.InsufficientPrivilege:
        pytest.skip("The database user may not create databases and roles")
    try:
        admin_url = make_conninfo(database_url, dbname=f"test_db_{suffix}")
        with psycopg.connect(admin_url, autocommit=True) as database_admin:
            if enable_pgvector:
                database_admin.execute("CREATE EXTENSION vector")
            database_admin.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        yield make_conninfo(
            database_url,
            dbname=f"test_db_{suffix}",
            user=f"test_role_{suffix}",
            password="password",
        )
    finally:
        admin_connection.execute(
            sql.SQL("DROP DATABASE {} WITH (FORCE)").format(database)
        )
        admin_connection.execute(sql.SQL("DROP ROLE {}").format(role))


@pytest.fixture
def role_without_pgvector(
    database_url: str, admin_connection: psycopg.Connection
) -> Iterator[str]:
    """A URL for a role that cannot enable pgvector in a database where it is not enabled."""
    yield from limited_role_database(admin_connection, database_url, False)


@pytest.fixture
def role_with_pgvector_but_no_table_rights(
    database_url: str, admin_connection: psycopg.Connection
) -> Iterator[str]:
    """A URL for a role that cannot create tables in a database where pgvector is enabled."""
    yield from limited_role_database(admin_connection, database_url, True)


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


def test_missing_pgvector_gives_a_clear_message(role_without_pgvector: str) -> None:
    """When pgvector cannot be enabled, setup says what is needed instead of a raw database error."""
    with pytest.raises(RuntimeError, match="pgvector is not available"):
        create_memory(role_without_pgvector)


def test_missing_pgvector_is_found_even_with_setup_off(
    role_without_pgvector: str,
) -> None:
    """The pgvector check is read-only, so it also runs when the host sets up the database."""
    with pytest.raises(RuntimeError, match="pgvector is not available"):
        create_memory(role_without_pgvector, create_schema=False)


def test_other_permission_errors_are_not_reported_as_missing_pgvector(
    role_with_pgvector_but_no_table_rights: str,
) -> None:
    """A role that cannot create tables gets the database's own error, not the pgvector message."""
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        create_memory(role_with_pgvector_but_no_table_rights)
