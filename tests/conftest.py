import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row

from aion_memory.facts_store import FactsStore

load_dotenv()


@pytest.fixture
def facts_store() -> Iterator[FactsStore]:
    """A FactsStore on a throwaway schema; skipped when no database is reachable."""
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
    store = FactsStore(connection)
    store.create_schema()
    try:
        yield store
    finally:
        connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(schema))
        connection.close()
