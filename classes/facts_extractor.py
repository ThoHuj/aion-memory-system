from pydantic import BaseModel, Field

from classes.context_manager import ContextResult
from classes.open_router_client import OpenRouterClient


class _FactsExtractionPayload(BaseModel):
    """Structured LLM output for fact extraction."""

    facts: list[str] = Field(
        default_factory=list,
        description="Semantic facts extracted from the new messages.",
    )


_FACTS_EXTRACTION_SYSTEM_PROMPT = """\
You extract stable semantic facts about the user and their world from conversation \
context. Each fact should be a concise standalone statement. Do not invent \
information that is not supported by the conversation.\
"""

_EXTRACTION_USER_PREAMBLE = """\
Extract semantic facts for this turn. Ground new facts only in <new_messages>; \
treat <semantic_search> and <recent_session_messages> as background.\
"""


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
        recent_session_messages = self._messages_to_text(
            lookup.recent_session_messages
        )
        new_messages = self._messages_to_text(context_result.messages)
        user_content = (
            f"{_EXTRACTION_USER_PREAMBLE}\n\n"
            "<semantic_search>\n"
            f"{related_facts_text}\n"
            "</semantic_search>\n\n"
            "<recent_session_messages>\n"
            f"{recent_session_messages}\n"
            "</recent_session_messages>\n\n"
            "<new_messages>\n"
            f"{new_messages}\n"
            "</new_messages>"
        )
        return [
            {"role": "system", "content": _FACTS_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def extract_facts_from_context_result(
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
