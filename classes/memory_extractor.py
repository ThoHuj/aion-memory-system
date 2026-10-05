from pydantic import BaseModel, Field

from classes.memory_models import (
    ExtractedEpisodicMemory,
    ExtractedSemanticFact,
    ExtractionResult,
)
from classes.open_router_client import OpenRouterClient

_EXTRACTION_SYSTEM_PROMPT = """\
You extract agent memory from a short conversation window.

Produce:

1. episodic_memories — discrete episodes worth recalling later (what happened in \
this exchange, decisions, tasks, or ongoing situations the user described). \
Each episode is a third-person summary.

2. semantic_facts — stable distilled facts about the user or their world that \
should persist beyond this episode (preferences, constraints, relationships, \
standing obligations). One fact per item; concise standalone statements.

Rules:
- If the window is only greetings, small talk, or generic advice with no new \
user-specific information, return empty episodic_memories and semantic_facts.
- Do not invent information not supported by the conversation.
- Prefer one rich episodic memory over many thin duplicates.
"""


class _ExtractedContentPayload(BaseModel):
    content: str = Field(
        description="Extracted episodic or semantic text from the conversation."
    )


class _ExtractionResultPayload(BaseModel):
    episodic_memories: list[_ExtractedContentPayload] = Field(default_factory=list)
    semantic_facts: list[_ExtractedContentPayload] = Field(default_factory=list)


class MemoryExtractor:
    """Extract episodic memories and semantic facts from conversation via structured LLM output."""

    def __init__(self, extraction_client: OpenRouterClient) -> None:
        self._extraction_client = extraction_client

    def extract_memories_and_facts_from_messages(
        self, messages: list[dict[str, str]]
    ) -> ExtractionResult:
        """Build the extraction prompt, call the extraction model, and return structured results."""
        if not messages:
            return ExtractionResult()
        extraction_messages = self._build_extraction_messages(messages)
        payload = self._extraction_client.complete_structured(
            extraction_messages,
            _ExtractionResultPayload,
        )
        return ExtractionResult(
            episodic_memories=[
                ExtractedEpisodicMemory.from_content(item.content)
                for item in payload.episodic_memories
            ],
            semantic_facts=[
                ExtractedSemanticFact(content=item.content)
                for item in payload.semantic_facts
            ],
        )

    @staticmethod
    def _format_conversation_for_prompt(messages: list[dict[str, str]]) -> str:
        """Render message history as plain text for the extraction user prompt."""
        blocks: list[str] = []
        for message in messages:
            role = message.get("role", "unknown").strip().capitalize()
            content = message.get("content", "").strip()
            blocks.append(f"{role}:\n{content}")
        return "\n\n".join(blocks)

    def _build_extraction_messages(
        self, conversation_messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Build chat messages for the extraction model call."""
        conversation_text = self._format_conversation_for_prompt(conversation_messages)
        return [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract episodic memories and semantic facts from this "
                    f"conversation window:\n\n{conversation_text}"
                ),
            },
        ]
