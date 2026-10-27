import uuid
from collections.abc import Iterator

import psycopg
import pytest
from docker.errors import DockerException
from psycopg import sql
from psycopg.conninfo import make_conninfo
from testcontainers.community.postgres import PostgresContainer

from aion_memory.facts_store import FactsStore, PostgresFactsStore
from tests.in_memory_facts_store import InMemoryFactsStore

POSTGRES_IMAGE = "pgvector/pgvector:0.8.7-pg17"


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A throwaway Postgres with pgvector in Docker for the whole test run; skipped without Docker."""
    try:
        container = PostgresContainer(POSTGRES_IMAGE, driver=None).start()
    except DockerException as error:
        pytest.skip(f"Docker is not available: {error}")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture
def admin_connection(database_url: str) -> Iterator[psycopg.Connection]:
    """A connection to the configured database for setting up and removing test data."""
    with psycopg.connect(database_url, autocommit=True) as connection:
        yield connection


@pytest.fixture
def test_schema_url(
    database_url: str, admin_connection: psycopg.Connection
) -> Iterator[str]:
    """A database URL whose connections use a fresh, empty schema that is dropped afterwards."""
    schema_name = f"test_{uuid.uuid4().hex}"
    # The extension lives in the public schema so dropping the test schema keeps it.
    admin_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin_connection.execute(
        sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
    )
    try:
        yield make_conninfo(
            database_url, options=f"-c search_path={schema_name},public"
        )
    finally:
        admin_connection.execute(
            sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
        )


@pytest.fixture
def postgres_facts_store(test_schema_url: str) -> Iterator[PostgresFactsStore]:
    """A PostgresFactsStore with its tables created in a throwaway schema."""
    store = PostgresFactsStore.connect(test_schema_url)
    store.create_schema()
    try:
        yield store
    finally:
        store.close()


@pytest.fixture(params=["in_memory", "postgres"])
def facts_store(request: pytest.FixtureRequest) -> FactsStore:
    """Each store implementation in turn, so store-level tests run on both."""
    if request.param == "in_memory":
        return InMemoryFactsStore()
    return request.getfixturevalue("postgres_facts_store")
