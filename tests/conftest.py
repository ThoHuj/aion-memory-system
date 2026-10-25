import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row

from aion_memory.facts_store import FactsStore, PostgresFactsStore
from tests.in_memory_facts_store import InMemoryFactsStore

load_dotenv()


def _postgres_facts_store() -> Iterator[FactsStore]:
    """A PostgresFactsStore on a throwaway schema; skipped when no database is reachable."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL is not set")
    try:
        connection = psycopg.connect(
            database_url,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=3,
        )
    except psycopg.OperationalError:
        pytest.skip("The database is not reachable")

    schema = sql.Identifier(f"test_{uuid.uuid4().hex}")
    # The extension lives in the public schema so dropping the test schema keeps it.
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    connection.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
    connection.execute(sql.SQL("SET search_path TO {}, public").format(schema))
    store = PostgresFactsStore(connection)
    store.create_schema()
    try:
        yield store
    finally:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
        connection.close()


@pytest.fixture(params=["in_memory", "postgres"])
def facts_store(request: pytest.FixtureRequest) -> Iterator[FactsStore]:
    """Each store implementation in turn, so store-level tests run on both."""
    if request.param == "in_memory":
        yield InMemoryFactsStore()
    else:
        yield from _postgres_facts_store()
