from pydantic import BaseModel, Field

from classes.context_manager import ContextResult
from classes.graph_models import (
    ExtractedEntity,
    ExtractedFact,
    GraphUpdate,
    normalize_graph_name,
    parse_world_time,
)
from classes.open_router_client import OpenRouterClient
from classes.relation_extraction.prompts import (
    RELATION_EXTRACTION_DESTINATION_FIELD_DESCRIPTION,
    RELATION_EXTRACTION_RELATIONS_FIELD_DESCRIPTION,
    RELATION_EXTRACTION_RELATIONSHIP_FIELD_DESCRIPTION,
    RELATION_EXTRACTION_SOURCE_FIELD_DESCRIPTION,
    RELATION_EXTRACTION_WORLD_VALID_FROM_FIELD_DESCRIPTION,
    RELATION_EXTRACTION_WORLD_VALID_TO_FIELD_DESCRIPTION,
    format_relation_extraction_system_prompt,
    format_relation_extraction_user_prompt,
)


class _RelationItem(BaseModel):
    """One relationship between two entities, with optional world times."""

    source: str = Field(description=RELATION_EXTRACTION_SOURCE_FIELD_DESCRIPTION)
    relationship: str = Field(
        description=RELATION_EXTRACTION_RELATIONSHIP_FIELD_DESCRIPTION,
    )
    destination: str = Field(
        description=RELATION_EXTRACTION_DESTINATION_FIELD_DESCRIPTION,
    )
    world_valid_from: str | None = Field(
        description=RELATION_EXTRACTION_WORLD_VALID_FROM_FIELD_DESCRIPTION,
    )
    world_valid_to: str | None = Field(
        description=RELATION_EXTRACTION_WORLD_VALID_TO_FIELD_DESCRIPTION,
    )


class _RelationExtractionPayload(BaseModel):
    """Structured LLM output for relation extraction."""

    relations: list[_RelationItem] = Field(
        default_factory=list,
        description=RELATION_EXTRACTION_RELATIONS_FIELD_DESCRIPTION,
    )


class RelationExtractor:
    """Extract fact edges among entities via an LLM (Mem0 call 2)."""

    def __init__(self, extraction_client: OpenRouterClient) -> None:
        self._extraction_client = extraction_client

    def attach_facts(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
    ) -> None:
        """Extract facts among the update's entities; add missing endpoints as unknown."""
        if not graph_update.entities:
            return
        added_at = context_result.metadata.added_at
        extraction_messages = [
            {
                "role": "system",
                "content": format_relation_extraction_system_prompt(
                    context_result.metadata.user_id,
                    added_at,
                ),
            },
            {
                "role": "user",
                "content": format_relation_extraction_user_prompt(
                    [entity.name for entity in graph_update.entities],
                    context_result.new_messages_text,
                    context_result.recent_session_messages_text,
                ),
            },
        ]
        payload = self._extraction_client.complete_structured(
            extraction_messages,
            _RelationExtractionPayload,
        )
        known_names = {entity.name for entity in graph_update.entities}
        for item in payload.relations:
            source = normalize_graph_name(item.source)
            relation = normalize_graph_name(item.relationship)
            destination = normalize_graph_name(item.destination)
            if not source or not relation or not destination:
                continue
            for endpoint in (source, destination):
                if endpoint not in known_names:
                    known_names.add(endpoint)
                    graph_update.entities.append(
                        ExtractedEntity(name=endpoint, entity_type="unknown"),
                    )
            graph_update.facts.append(
                ExtractedFact(
                    source_entity=source,
                    relation=relation,
                    destination_entity=destination,
                    world_valid_from=parse_world_time(item.world_valid_from) or added_at,
                    world_valid_to=parse_world_time(item.world_valid_to),
                ),
            )
