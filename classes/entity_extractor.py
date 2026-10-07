from pydantic import BaseModel, Field

from classes.facts_extraction_result import NewFacts
from classes.open_router_client import OpenRouterClient


class _EntityExtractionPayload(BaseModel):
    """Structured LLM output for entity extraction."""

    entities: list[str] = Field(
        default_factory=list,
        description="Named entities mentioned in the fact (people, places, organizations, etc.).",
    )


_ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
You extract named entities from a single semantic fact. Return people, organizations, \
places, products, and other distinct proper nouns or stable identifiers explicitly \
mentioned or clearly implied. Do not invent entities not supported by the fact.\
"""

_ENTITY_EXTRACTION_USER_PREAMBLE = """\
Extract entities from the fact in <fact> below.\
"""


class EntityExtractor:
    """Extract entities from facts via an LLM."""

    def __init__(self, extraction_client: OpenRouterClient) -> None:
        self._extraction_client = extraction_client

    def attach_entities(self, new_facts: NewFacts) -> None:
        """Extract entities for each fact and store them on extracted_fact.entities."""
        for extracted_fact in new_facts.facts:
            extraction_messages = self._build_entity_extraction_messages(
                extracted_fact.content,
            )
            payload = self._extraction_client.complete_structured(
                extraction_messages,
                _EntityExtractionPayload,
            )
            extracted_fact.entities = [
                entity.strip() for entity in payload.entities if entity.strip()
            ]

    def _build_entity_extraction_messages(
        self,
        fact: str,
    ) -> list[dict[str, str]]:
        """Build system and user messages for entity extraction on one fact."""
        user_content = (
            f"{_ENTITY_EXTRACTION_USER_PREAMBLE}\n\n"
            f"<fact>\n{fact}\n</fact>"
        )
        return [
            {"role": "system", "content": _ENTITY_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
