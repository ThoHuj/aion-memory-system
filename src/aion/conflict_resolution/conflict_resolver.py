from pydantic import BaseModel, Field

from aion.conflict_resolution.prompts import (
    CONFLICT_RESOLUTION_INVALIDATIONS_FIELD_DESCRIPTION,
    CONFLICT_RESOLUTION_WORLD_VALID_TO_FIELD_DESCRIPTION,
    format_conflict_resolution_system_prompt,
    format_conflict_resolution_user_prompt,
)
from aion.context_manager import ContextResult
from aion.facts_store import FactsStore
from aion.graph_models import (
    ExtractedFact,
    FactInvalidation,
    GraphUpdate,
    format_fact_line,
    parse_world_time,
)
from aion.open_router_client import OpenRouterClient


class _InvalidationItem(BaseModel):
    """One stored fact the LLM wants to invalidate."""

    fact_id: str
    world_valid_to: str | None = Field(
        description=CONFLICT_RESOLUTION_WORLD_VALID_TO_FIELD_DESCRIPTION,
    )


class _ConflictResolutionPayload(BaseModel):
    """Structured LLM output for fact invalidation."""

    invalidations: list[_InvalidationItem] = Field(
        description=CONFLICT_RESOLUTION_INVALIDATIONS_FIELD_DESCRIPTION,
    )


class ConflictResolver:
    """Drop duplicate facts and invalidate stored facts that new facts contradict."""

    def __init__(
        self,
        extraction_client: OpenRouterClient,
        facts_store: FactsStore,
        existing_facts_limit: int = 100,
    ) -> None:
        self._extraction_client = extraction_client
        self._facts_store = facts_store
        self._existing_facts_limit = existing_facts_limit

    def resolve_conflicts(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
    ) -> None:
        """Remove duplicate facts, then set graph_update.invalidations via the LLM."""
        user_id = context_result.metadata.user_id
        duplicate_fact_ids = self._remove_duplicate_facts(graph_update, user_id)

        matched_entity_ids = [
            entity.entity_id for entity in graph_update.entities if not entity.is_new
        ]
        existing_facts = self._facts_store.get_valid_facts_for_entities(
            user_id,
            matched_entity_ids,
            self._existing_facts_limit,
        )
        if not existing_facts:
            return

        existing_memories = "\n".join(
            f"{fact.fact_id}: {format_fact_line(fact)}" for fact in existing_facts
        )
        payload = self._extraction_client.complete_structured(
            [
                {
                    "role": "system",
                    "content": format_conflict_resolution_system_prompt(
                        user_id,
                        context_result.metadata.added_at,
                    ),
                },
                {
                    "role": "user",
                    "content": format_conflict_resolution_user_prompt(
                        existing_memories,
                        context_result.new_messages_text,
                        context_result.recent_session_messages_text,
                    ),
                },
            ],
            _ConflictResolutionPayload,
        )

        candidate_fact_ids = {fact.fact_id for fact in existing_facts}
        invalidations: list[FactInvalidation] = []
        seen_fact_ids: set[str] = set()
        for item in payload.invalidations:
            if (
                item.fact_id not in candidate_fact_ids
                or item.fact_id in duplicate_fact_ids
                or item.fact_id in seen_fact_ids
            ):
                continue
            seen_fact_ids.add(item.fact_id)
            invalidations.append(
                FactInvalidation(
                    fact_id=item.fact_id,
                    world_valid_to=parse_world_time(item.world_valid_to),
                )
            )
        graph_update.invalidations = invalidations

    def _remove_duplicate_facts(
        self,
        graph_update: GraphUpdate,
        user_id: str,
    ) -> set[str]:
        """Drop facts already stored or repeated in the batch; return stored ids."""
        entities_by_name = {entity.name: entity for entity in graph_update.entities}
        duplicate_fact_ids: set[str] = set()
        seen_keys: set[tuple[str, str, str]] = set()
        unique_facts: list[ExtractedFact] = []

        for fact in graph_update.facts:
            source = entities_by_name[fact.source_entity]
            destination = entities_by_name[fact.destination_entity]
            key = (source.entity_id, fact.relation, destination.entity_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if not source.is_new and not destination.is_new:
                stored_fact_id = self._facts_store.find_valid_fact_id(
                    user_id,
                    source.entity_id,
                    fact.relation,
                    destination.entity_id,
                )
                if stored_fact_id is not None:
                    duplicate_fact_ids.add(stored_fact_id)
                    continue
            unique_facts.append(fact)

        graph_update.facts = unique_facts
        return duplicate_fact_ids
