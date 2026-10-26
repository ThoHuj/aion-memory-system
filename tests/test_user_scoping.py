from typing import Any

import pytest

from aion_memory.context_manager import ContextManager
from aion_memory.facts_store import FactsStore
from aion_memory.memory_system import MemorySystem
from tests.helpers import add_fact

SHARED_SESSION = "shared-session"


def test_session_messages_are_scoped_to_the_user(facts_store: FactsStore) -> None:
    """Two users sharing a session id each see only their own recent messages."""
    add_fact(facts_store, "alice", SHARED_SESSION, "I have a dog")
    add_fact(facts_store, "bob", SHARED_SESSION, "I have a cat")

    alice_messages = facts_store.get_most_recent_session_messages(
        "alice", SHARED_SESSION
    )

    assert [message["content"] for message in alice_messages] == ["I have a dog"]


def test_context_excludes_another_users_session_messages(
    facts_store: FactsStore,
) -> None:
    """The extraction context of one user leaves out the other user's messages."""
    add_fact(facts_store, "alice", SHARED_SESSION, "I have a dog")
    add_fact(facts_store, "bob", SHARED_SESSION, "I have a cat")

    context_result = ContextManager(facts_store).produce_context_result(
        [{"role": "user", "content": "hello"}],
        "bob",
        SHARED_SESSION,
    )

    assert "cat" in context_result.recent_session_messages_text
    assert "dog" not in context_result.recent_session_messages_text


def test_fact_is_not_returned_to_another_user(facts_store: FactsStore) -> None:
    """A fact can be read by its owner and is unknown to any other user."""
    fact_id = add_fact(facts_store, "alice", SHARED_SESSION, "I have a dog")

    assert facts_store.get_fact("alice", fact_id).fact_id == fact_id
    with pytest.raises(KeyError):
        facts_store.get_fact("bob", fact_id)


def test_add_log_record_is_not_returned_to_another_user(
    facts_store: FactsStore,
) -> None:
    """An add log record can be read by its owner and is unknown to any other user."""
    fact_id = add_fact(facts_store, "alice", SHARED_SESSION, "I have a dog")
    add_id = facts_store.get_fact("alice", fact_id).add_id

    assert facts_store.get_add_log_record("alice", add_id).add_id == add_id
    with pytest.raises(KeyError):
        facts_store.get_add_log_record("bob", add_id)


def test_memory_system_get_is_scoped_to_the_user(facts_store: FactsStore) -> None:
    """MemorySystem.get returns a fact and its origin to the owner only."""
    unused: Any = None
    memory_system = MemorySystem(
        facts_store=facts_store,
        context_manager=ContextManager(facts_store),
        entity_extractor=unused,
        relation_extractor=unused,
        embedding_manager=unused,
        entity_resolver=unused,
        conflict_resolver=unused,
        graph_retriever=unused,
    )
    fact_id = add_fact(facts_store, "alice", SHARED_SESSION, "I have a dog")

    fact_data = memory_system.get(fact_id, "alice")

    assert fact_data.origin.messages == [{"role": "user", "content": "I have a dog"}]
    with pytest.raises(KeyError):
        memory_system.get(fact_id, "bob")


def test_memory_system_export_graph_is_scoped_to_the_user(
    facts_store: FactsStore,
) -> None:
    """MemorySystem.export_graph returns only the graph of the user asked for."""
    unused: Any = None
    memory_system = MemorySystem(
        facts_store=facts_store,
        context_manager=unused,
        entity_extractor=unused,
        relation_extractor=unused,
        embedding_manager=unused,
        entity_resolver=unused,
        conflict_resolver=unused,
        graph_retriever=unused,
    )
    add_fact(facts_store, "alice", SHARED_SESSION, "I have a dog")
    add_fact(facts_store, "bob", SHARED_SESSION, "I have a cat")

    graph = memory_system.export_graph("bob")

    assert {found.name for found in graph.entities} == {"bob", "pet"}
    assert [found.source_entity for found in graph.facts] == ["bob"]
