from pydantic import BaseModel, Field

from aion.context_manager import ContextResult
from aion.entity_extraction.prompts import (
    ENTITY_EXTRACTION_ENTITIES_FIELD_DESCRIPTION,
    ENTITY_EXTRACTION_ENTITY_FIELD_DESCRIPTION,
    ENTITY_EXTRACTION_ENTITY_TYPE_FIELD_DESCRIPTION,
    format_entity_extraction_system_prompt,
    format_entity_extraction_user_prompt,
)
from aion.graph_models import ExtractedEntity, GraphUpdate, normalize_graph_name
from aion.open_router_client import OpenRouterClient


class _EntityItem(BaseModel):
    """One entity with its type."""

    entity: str = Field(description=ENTITY_EXTRACTION_ENTITY_FIELD_DESCRIPTION)
    entity_type: str = Field(description=ENTITY_EXTRACTION_ENTITY_TYPE_FIELD_DESCRIPTION)


class _EntityExtractionPayload(BaseModel):
    """Structured LLM output for entity extraction."""

    entities: list[_EntityItem] = Field(
        default_factory=list,
        description=ENTITY_EXTRACTION_ENTITIES_FIELD_DESCRIPTION,
    )


class EntityExtractor:
    """Extract typed entities from text via an LLM (Mem0 call 1)."""

    def __init__(self, extraction_client: OpenRouterClient) -> None:
        self._extraction_client = extraction_client

    def extract_entities(
        self,
        text: str,
        user_id: str,
        recent_session_messages: str = "(none)",
    ) -> list[ExtractedEntity]:
        """Return normalized, unique entities extracted from the text."""
        extraction_messages = [
            {
                "role": "system",
                "content": format_entity_extraction_system_prompt(user_id),
            },
            {
                "role": "user",
                "content": format_entity_extraction_user_prompt(
                    text,
                    recent_session_messages,
                ),
            },
        ]
        payload = self._extraction_client.complete_structured(
            extraction_messages,
            _EntityExtractionPayload,
        )
        entities: list[ExtractedEntity] = []
        seen_names: set[str] = set()
        for item in payload.entities:
            name = normalize_graph_name(item.entity)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            entities.append(
                ExtractedEntity(
                    name=name,
                    entity_type=normalize_graph_name(item.entity_type),
                )
            )
        return entities

    def attach_entities(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
    ) -> None:
        """Extract entities from the new messages onto the graph update."""
        graph_update.entities = self.extract_entities(
            context_result.new_messages_text,
            context_result.metadata.user_id,
            context_result.recent_session_messages_text,
        )
