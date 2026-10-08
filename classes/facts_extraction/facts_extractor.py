from pydantic import BaseModel, Field

from classes.context_manager import ContextResult
from classes.facts_extraction.prompts import (
    FACTS_EXTRACTION_FACTS_FIELD_DESCRIPTION,
    FACTS_EXTRACTION_SYSTEM_PROMPT,
    format_facts_extraction_user_prompt,
)
from classes.open_router_client import OpenRouterClient


class _FactsExtractionPayload(BaseModel):
    """Structured LLM output for fact extraction."""

    facts: list[str] = Field(
        default_factory=list,
        description=FACTS_EXTRACTION_FACTS_FIELD_DESCRIPTION,
    )


class FactsExtractor:
    """Extract semantic facts from conversation context via an LLM."""

    def __init__(self, extraction_client: OpenRouterClient) -> None:
        self._extraction_client = extraction_client

    def _build_extraction_messages(
        self,
        context_result: ContextResult,
    ) -> list[dict[str, str]]:
        """Build system and user messages for the extraction LLM call."""
        lookup = context_result.context_lookup
        related_facts = lookup.vector_search_result.facts
        related_facts_text = (
            "\n".join(fact.content for fact in related_facts)
            if related_facts
            else "(none)"
        )
        recent_session_messages = self._messages_to_text(lookup.recent_session_messages)
        new_messages = self._messages_to_text(context_result.messages)
        user_content = format_facts_extraction_user_prompt(
            semantic_search=related_facts_text,
            recent_session_messages=recent_session_messages,
            new_messages=new_messages,
        )
        return [
            {"role": "system", "content": FACTS_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def extract_facts(
        self,
        context_result: ContextResult,
    ) -> list[str]:
        """Return fact strings extracted from a context result."""
        extraction_messages = self._build_extraction_messages(context_result)
        payload = self._extraction_client.complete_structured(
            extraction_messages,
            _FactsExtractionPayload,
        )
        return payload.facts

    @staticmethod
    def _messages_to_text(messages: list[dict[str, str]]) -> str:
        """Format messages as role-labeled lines."""
        if not messages:
            return "(none)"
        return "\n".join(
            f"{message.get('role', 'unknown').strip().lower()}: "
            f"{message.get('content', '').strip()}"
            for message in messages
        )
