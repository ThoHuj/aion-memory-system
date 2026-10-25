"""Connection handling of the Postgres store: failing fast, concurrent use, and closing."""

import threading
from typing import Any

import psycopg
import pytest
from psycopg_pool import PoolClosed

from aion_memory.context_manager import ContextManager
from aion_memory.facts_store import PostgresFactsStore
from aion_memory.memory_system import MemorySystem
from tests.helpers import add_fact


def test_connect_fails_with_the_databases_own_error() -> None:
    """An unreachable database raises psycopg's error right away, not a pool timeout."""
    with pytest.raises(psycopg.OperationalError):
        PostgresFactsStore.connect("postgresql://aion@127.0.0.1:1/none")


def test_concurrent_updates_each_use_their_own_connection(
    postgres_facts_store: PostgresFactsStore,
) -> None:
    """Updates and reads from many threads at once all succeed and keep users apart."""
    user_ids = [f"user-{number}" for number in range(12)]
    errors: list[psycopg.Error] = []

    def add_and_read(user_id: str) -> None:
        try:
            add_fact(postgres_facts_store, user_id, f"session-{user_id}", "hello")
            postgres_facts_store.find_entity_by_name(user_id, user_id)
        except psycopg.Error as error:
            errors.append(error)

    threads = [threading.Thread(target=add_and_read, args=(user,)) for user in user_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    for user_id in user_ids:
        assert postgres_facts_store.find_entity_by_name(user_id, user_id) is not None


def test_closing_the_memory_system_closes_the_pool(
    postgres_facts_store: PostgresFactsStore,
) -> None:
    """After close the store refuses new work, and closing again is harmless."""
    unused: Any = None
    memory_system = MemorySystem(
        facts_store=postgres_facts_store,
        context_manager=ContextManager(postgres_facts_store),
        entity_extractor=unused,
        relation_extractor=unused,
        embedding_manager=unused,
        entity_resolver=unused,
        conflict_resolver=unused,
        graph_retriever=unused,
    )

    memory_system.close()
    memory_system.close()

    with pytest.raises(PoolClosed):
        postgres_facts_store.find_entity_by_name("alice", "rex")
