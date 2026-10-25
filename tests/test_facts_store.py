"""Behavior every FactsStore implementation must share."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from aion_memory.context_manager import ContextMetadata, ContextResult
from aion_memory.facts_store import FactsStore
from aion_memory.graph_models import (
    ExtractedEntity,
    ExtractedFact,
    FactInvalidation,
    GraphUpdate,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def entity(
    name: str,
    embedding: list[float],
    entity_id: str | None = None,
    is_new: bool = True,
) -> ExtractedEntity:
    """Build an extracted entity with a fresh id unless one is given."""
    return ExtractedEntity(
        name=name,
        entity_type="thing",
        embedding=embedding,
        entity_id=entity_id or str(uuid.uuid4()),
        is_new=is_new,
    )


def fact(source: str, relation: str, destination: str) -> ExtractedFact:
    """Build an extracted fact between two entity names."""
    return ExtractedFact(
        source_entity=source,
        relation=relation,
        destination_entity=destination,
        world_valid_from=START.isoformat(),
        world_valid_to=None,
    )


def store_update(
    store: FactsStore,
    user_id: str,
    entities: list[ExtractedEntity],
    facts: list[ExtractedFact],
    *,
    minutes: int = 0,
    session_id: str = "session",
    messages: list[dict[str, str]] | None = None,
    invalidations: list[FactInvalidation] | None = None,
) -> None:
    """Store one graph update at START plus the given minutes."""
    context_result = ContextResult(
        messages=messages or [{"role": "user", "content": "hello"}],
        metadata=ContextMetadata(
            user_id=user_id,
            session_id=session_id,
            added_at=(START + timedelta(minutes=minutes)).isoformat(),
        ),
        new_messages_text="",
        recent_session_messages_text="",
    )
    graph_update = GraphUpdate(
        entities=entities,
        facts=facts,
        invalidations=invalidations or [],
    )
    store.store_graph_update(graph_update, context_result)


def test_find_entity_by_name_returns_the_oldest_entity_of_the_user(
    facts_store: FactsStore,
) -> None:
    """Of several entities with one name, the oldest is returned, and only for its owner."""
    first_rex_id = str(uuid.uuid4())
    alice_id = str(uuid.uuid4())
    store_update(
        facts_store,
        "alice",
        [entity("alice", [0, 0, 1], alice_id), entity("rex", [1, 0, 0], first_rex_id)],
        [fact("alice", "owns", "rex")],
        minutes=0,
    )
    store_update(
        facts_store,
        "alice",
        [entity("alice", [0, 0, 1], alice_id, is_new=False), entity("rex", [1, 0, 0])],
        [fact("alice", "likes", "rex")],
        minutes=5,
    )

    found = facts_store.find_entity_by_name("alice", "rex")

    assert found is not None
    assert found.entity_id == first_rex_id
    assert facts_store.find_entity_by_name("alice", "unknown") is None
    assert facts_store.find_entity_by_name("bob", "rex") is None


def test_find_similar_entities_orders_by_similarity_within_threshold_and_limit(
    facts_store: FactsStore,
) -> None:
    """Entities come back best first, at or above the minimum, up to the limit."""
    store_update(
        facts_store,
        "alice",
        [
            entity("alice", [0, 0, 1]),
            entity("near", [0.9, 0.1, 0]),
            entity("same", [1, 0, 0]),
            entity("far", [0, 1, 0]),
        ],
        [
            fact("alice", "knows", "near"),
            fact("alice", "knows", "same"),
            fact("alice", "knows", "far"),
        ],
    )

    def names(minimum: float, limit: int, user_id: str = "alice") -> list[str]:
        similar = facts_store.find_similar_entities(user_id, [1, 0, 0], minimum, limit)
        return [found.name for found in similar]

    assert names(0.5, 10) == ["same", "near"]
    assert names(0.5, 1) == ["same"]
    assert names(0.999, 10) == ["same"]
    assert names(0.5, 10, user_id="bob") == []


def test_get_valid_facts_for_entities_returns_valid_facts_newest_first(
    facts_store: FactsStore,
) -> None:
    """Invalidated facts are left out, the rest come back newest first up to the limit."""
    alice = entity("alice", [0, 0, 1])
    old_fact = fact("alice", "owns", "rex")
    new_fact = fact("alice", "owns", "tom")
    store_update(
        facts_store,
        "alice",
        [alice, entity("rex", [1, 0, 0])],
        [old_fact],
        minutes=0,
    )
    store_update(
        facts_store,
        "alice",
        [
            entity("alice", [0, 0, 1], alice.entity_id, is_new=False),
            entity("tom", [0, 1, 0]),
        ],
        [new_fact],
        minutes=10,
    )

    def relations(limit: int = 10, user_id: str = "alice") -> list[str]:
        facts = facts_store.get_valid_facts_for_entities(
            user_id, [alice.entity_id], limit
        )
        return [found.destination_entity for found in facts]

    assert relations() == ["tom", "rex"]
    assert relations(limit=1) == ["tom"]
    assert relations(user_id="bob") == []
    assert facts_store.get_valid_facts_for_entities("alice", [], 10) == []

    store_update(
        facts_store,
        "alice",
        [],
        [],
        minutes=20,
        invalidations=[FactInvalidation(fact_id=old_fact.fact_id, world_valid_to=None)],
    )

    assert relations() == ["tom"]


def test_find_valid_fact_id_matches_only_a_valid_edge_of_the_user(
    facts_store: FactsStore,
) -> None:
    """The id is found for the exact edge while valid, and not for other users or relations."""
    alice = entity("alice", [0, 0, 1])
    rex = entity("rex", [1, 0, 0])
    owns = fact("alice", "owns", "rex")
    store_update(facts_store, "alice", [alice, rex], [owns])

    def find(relation: str = "owns", user_id: str = "alice") -> str | None:
        return facts_store.find_valid_fact_id(
            user_id, alice.entity_id, relation, rex.entity_id
        )

    assert find() == owns.fact_id
    assert find(relation="likes") is None
    assert find(user_id="bob") is None

    store_update(
        facts_store,
        "alice",
        [],
        [],
        minutes=1,
        invalidations=[FactInvalidation(fact_id=owns.fact_id, world_valid_to=None)],
    )

    assert find() is None


def test_invalidation_closes_the_fact_but_keeps_it_readable(
    facts_store: FactsStore,
) -> None:
    """An invalidated fact gets its end times and can still be fetched by id."""
    owns = fact("alice", "owns", "rex")
    store_update(
        facts_store,
        "alice",
        [entity("alice", [0, 0, 1]), entity("rex", [1, 0, 0])],
        [owns],
        minutes=0,
    )
    stopped_at = START + timedelta(hours=2)

    store_update(
        facts_store,
        "alice",
        [],
        [],
        minutes=30,
        invalidations=[
            FactInvalidation(
                fact_id=owns.fact_id, world_valid_to=stopped_at.isoformat()
            )
        ],
    )

    stored = facts_store.get_fact("alice", owns.fact_id)
    assert stored.valid_to is not None
    assert datetime.fromisoformat(stored.valid_to) == START + timedelta(minutes=30)
    assert stored.world_valid_to is not None
    assert datetime.fromisoformat(stored.world_valid_to) == stopped_at


def test_only_entities_used_by_a_fact_are_stored(facts_store: FactsStore) -> None:
    """An entity that no fact refers to is not persisted."""
    store_update(
        facts_store,
        "alice",
        [
            entity("alice", [0, 0, 1]),
            entity("rex", [1, 0, 0]),
            entity("unused", [0, 1, 0]),
        ],
        [fact("alice", "owns", "rex")],
    )

    assert facts_store.find_entity_by_name("alice", "rex") is not None
    assert facts_store.find_entity_by_name("alice", "unused") is None


def test_stored_fact_links_to_the_add_log_record(facts_store: FactsStore) -> None:
    """A stored fact gets an id and points at the record of the messages it came from."""
    owns = fact("alice", "owns", "rex")
    messages = [
        {"role": "user", "content": "I have a dog named Rex"},
        {"role": "assistant", "content": "Nice!"},
    ]
    store_update(
        facts_store,
        "alice",
        [entity("alice", [0, 0, 1]), entity("rex", [1, 0, 0])],
        [owns],
        session_id="chat-1",
        messages=messages,
    )

    stored = facts_store.get_fact("alice", owns.fact_id)
    record = facts_store.get_add_log_record("alice", stored.add_id)

    assert owns.fact_id
    assert (stored.source_entity, stored.relation, stored.destination_entity) == (
        "alice",
        "owns",
        "rex",
    )
    assert record.session_id == "chat-1"
    assert record.messages == messages


def test_recent_session_messages_are_the_latest_oldest_first(
    facts_store: FactsStore,
) -> None:
    """The newest add calls of the session are returned in order, trimmed to the limit."""
    for minute, text in enumerate(["first", "second", "third"]):
        store_update(
            facts_store,
            "alice",
            [],
            [],
            minutes=minute,
            messages=[{"role": "user", "content": text}],
        )
    store_update(
        facts_store,
        "alice",
        [],
        [],
        minutes=10,
        session_id="other",
        messages=[{"role": "user", "content": "elsewhere"}],
    )

    def recent(limit: int, session_id: str = "session") -> list[str]:
        messages = facts_store.get_most_recent_session_messages(
            "alice", session_id, limit
        )
        return [message["content"] for message in messages]

    assert recent(10) == ["first", "second", "third"]
    assert recent(2) == ["second", "third"]
    assert recent(0) == []
    assert recent(10, session_id="other") == ["elsewhere"]
    assert recent(10, session_id="unknown") == []


def test_unknown_ids_raise_key_error(facts_store: FactsStore) -> None:
    """Looking up a fact or add log record that does not exist raises KeyError."""
    with pytest.raises(KeyError):
        facts_store.get_fact("alice", "missing")
    with pytest.raises(KeyError):
        facts_store.get_add_log_record("alice", "missing")


def test_a_store_can_be_closed_twice(facts_store: FactsStore) -> None:
    """Closing releases the store's resources, and closing again does no harm."""
    facts_store.close()
    facts_store.close()
