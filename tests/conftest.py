import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg import sql

from aion_memory.facts_store import FactsStore, PostgresFactsStore
from tests.in_memory_facts_store import InMemoryFactsStore

load_dotenv()


@pytest.fixture
def postgres_facts_store() -> Iterator[PostgresFactsStore]:
    """A PostgresFactsStore on a throwaway schema; skipped when no database is reachable."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    try:
        admin_connection = psycopg.connect(
            database_url, autocommit=True, connect_timeout=3
        )
    except psycopg.OperationalError:
        pytest.skip("The database is not reachable")

    schema_name = f"test_{uuid.uuid4().hex}"
    # The extension lives in the public schema so dropping the test schema keeps it.
    admin_connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    admin_connection.execute(
        sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name))
    )
    store = PostgresFactsStore.connect(
        database_url,
        options=f"-c search_path={schema_name},public",
    )
    store.create_schema()
    try:
        yield store
    finally:
        store.close()
        admin_connection.execute(
            sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
        )
        admin_connection.close()


@pytest.fixture(params=["in_memory", "postgres"])
def facts_store(request: pytest.FixtureRequest) -> FactsStore:
    """Each store implementation in turn, so store-level tests run on both."""
    if request.param == "in_memory":
        return InMemoryFactsStore()
    return request.getfixturevalue("postgres_facts_store")
