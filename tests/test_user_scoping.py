import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from aion_memory.context_manager import ContextManager
from aion_memory.facts_store import FactsStore
from aion_memory.graph_models import ExtractedEntity, ExtractedFact, GraphUpdate
from aion_memory.memory_system import MemorySystem

SHARED_SESSION = "shared-session"


def add_fact(store: FactsStore, user_id: str, session_id: str, message: str) -> str:
    """Store one fact for user_id from one message and return the fact id."""
    context_result = ContextManager(store).produce_context_result(
        [{"role": "user", "content": message}],
        user_id,
        session_id,
    )
    graph_update = GraphUpdate(
        entities=[
            ExtractedEntity(
                name=user_id,
                entity_type="person",
                embedding=[1.0, 0.0, 0.0],
                entity_id=str(uuid.uuid4()),
            ),
            ExtractedEntity(
                name="pet",
                entity_type="animal",
                embedding=[0.0, 1.0, 0.0],
                entity_id=str(uuid.uuid4()),
            ),
        ],
        facts=[
            ExtractedFact(
                source_entity=user_id,
                relation="owns",
                destination_entity="pet",
                world_valid_from=datetime.now(UTC).isoformat(),
                world_valid_to=None,
            )
        ],
    )
    store.store_graph_update(graph_update, context_result)
    return graph_update.facts[0].fact_id


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
