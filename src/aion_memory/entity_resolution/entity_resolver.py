import uuid

from pydantic import BaseModel, Field

from aion_memory.context_manager import ContextResult
from aion_memory.entity_resolution.prompts import (
    ENTITY_RESOLUTION_EXISTING_ENTITY_ID_FIELD_DESCRIPTION,
    ENTITY_RESOLUTION_MATCHES_FIELD_DESCRIPTION,
    ENTITY_RESOLUTION_SYSTEM_PROMPT,
    format_entity_resolution_user_prompt,
)
from aion_memory.facts_store import FactsStore
from aion_memory.graph_models import (
    ExtractedEntity,
    GraphUpdate,
    StoredEntity,
    format_fact_line,
    normalize_graph_name,
)
from aion_memory.open_router_client import OpenRouterClient


class _EntityMatch(BaseModel):
    """One resolution decision for a new entity."""

    entity_name: str
    existing_entity_id: str | None = Field(
        description=ENTITY_RESOLUTION_EXISTING_ENTITY_ID_FIELD_DESCRIPTION,
    )


class _EntityResolutionPayload(BaseModel):
    """Structured LLM output for entity resolution."""

    matches: list[_EntityMatch] = Field(
        description=ENTITY_RESOLUTION_MATCHES_FIELD_DESCRIPTION,
    )


class EntityResolver:
    """Resolve extracted entities to stored entities, or assign new ids."""

    def __init__(
        self,
        extraction_client: OpenRouterClient,
        facts_store: FactsStore,
        candidate_limit: int = 5,
        minimum_candidate_similarity: float = 0.5,
        candidate_facts_limit: int = 10,
    ) -> None:
        self._extraction_client = extraction_client
        self._facts_store = facts_store
        self._candidate_limit = candidate_limit
        self._minimum_candidate_similarity = minimum_candidate_similarity
        self._candidate_facts_limit = candidate_facts_limit

    def resolve_entities(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
    ) -> None:
        """Set entity_id and is_new on every entity of the graph update."""
        user_id = context_result.metadata.user_id
        user_entity_name = normalize_graph_name(user_id)
        candidates_by_name: dict[str, list[StoredEntity]] = {}

        for entity in graph_update.entities:
            if entity.name == user_entity_name:
                self._resolve_user_entity(entity, user_id)
                continue
            candidates = self._facts_store.find_similar_entities(
                user_id,
                entity.embedding,
                self._minimum_candidate_similarity,
                self._candidate_limit,
            )
            if candidates:
                candidates_by_name[entity.name] = candidates

        chosen_ids: dict[str, str] = {}
        if candidates_by_name:
            chosen_ids = self._choose_candidates(
                graph_update,
                context_result,
                candidates_by_name,
            )

        for entity in graph_update.entities:
            if entity.name == user_entity_name:
                continue
            chosen_id = chosen_ids.get(entity.name)
            if chosen_id is not None:
                entity.entity_id = chosen_id
                entity.is_new = False
            else:
                entity.entity_id = str(uuid.uuid4())
                entity.is_new = True

    def _resolve_user_entity(self, entity: ExtractedEntity, user_id: str) -> None:
        """Match the user's own entity by exact name; create a new id otherwise."""
        stored_entity = self._facts_store.find_entity_by_name(user_id, entity.name)
        if stored_entity is None:
            entity.entity_id = str(uuid.uuid4())
            entity.is_new = True
        else:
            entity.entity_id = stored_entity.entity_id
            entity.is_new = False

    def _choose_candidates(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
        candidates_by_name: dict[str, list[StoredEntity]],
    ) -> dict[str, str]:
        """Ask the LLM which candidates match; return only valid candidate ids."""
        user_id = context_result.metadata.user_id
        entity_blocks = [
            self._format_entity_block(entity, candidates_by_name[entity.name], user_id)
            for entity in graph_update.entities
            if entity.name in candidates_by_name
        ]
        user_content = format_entity_resolution_user_prompt(
            context_result.new_messages_text,
            context_result.recent_session_messages_text,
            "\n\n".join(entity_blocks),
        )
        payload = self._extraction_client.complete_structured(
            [
                {"role": "system", "content": ENTITY_RESOLUTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            _EntityResolutionPayload,
        )

        chosen_ids: dict[str, str] = {}
        for match in payload.matches:
            entity_name = normalize_graph_name(match.entity_name)
            candidates = candidates_by_name.get(entity_name, [])
            candidate_ids = {candidate.entity_id for candidate in candidates}
            if match.existing_entity_id in candidate_ids:
                chosen_ids[entity_name] = match.existing_entity_id
        return chosen_ids

    def _format_entity_block(
        self,
        entity: ExtractedEntity,
        candidates: list[StoredEntity],
        user_id: str,
    ) -> str:
        """Format one new entity with its candidates and their stored facts."""
        lines = [f"new entity: {entity.name} ({entity.entity_type})"]
        for candidate in candidates:
            lines.append(
                f"- id {candidate.entity_id}: {candidate.name} ({candidate.entity_type})"
            )
            candidate_facts = self._facts_store.get_valid_facts_for_entities(
                user_id,
                [candidate.entity_id],
                self._candidate_facts_limit,
            )
            lines.extend(f"  {format_fact_line(fact)}" for fact in candidate_facts)
        return "\n".join(lines)
