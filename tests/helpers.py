"""Helpers shared by several test modules."""

import uuid
from datetime import UTC, datetime, timedelta

from aion_memory.context_manager import ContextManager, ContextMetadata, ContextResult
from aion_memory.facts_store import FactsStore
from aion_memory.graph_models import (
    ExtractedEntity,
    ExtractedFact,
    FactInvalidation,
    GraphUpdate,
)


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
