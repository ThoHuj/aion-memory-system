"""Helpers shared by several test modules."""

import uuid
from datetime import UTC, datetime

from aion_memory.context_manager import ContextManager
from aion_memory.facts_store import FactsStore
from aion_memory.graph_models import ExtractedEntity, ExtractedFact, GraphUpdate


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
