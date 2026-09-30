from classes.models import (
    EpisodicMemoryCandidate,
    EpisodicMemoryExtractionResult,
)
from classes.open_router_client import OpenRouterClient

EPISODIC_MEMORY_EXTRACTION_PROMPT = """Extract episodic memories from this conversation.

An episodic memory is a past event involving a decision, outcome, or development — not a passing update on the user's current state. Store events, not facts about the user (those belong elsewhere). This is not a conversation log. Most exchanges yield none.

If no event qualifies, return an empty list.

If the conversation has no substantive user or assistant text (empty, whitespace-only, or missing dialogue), return an empty list. Do not invent or infer events that are not clearly stated in the messages.

For each memory:
- description: what happened
- keywords: search terms for retrieval
- extraction_reason: how retrieving this would change what the agent should do or prioritize"""


class MemoryExtractor:
    """Extract memory candidates from conversation messages via structured LLM output."""

    def __init__(self, open_router_client: OpenRouterClient) -> None:
        self._open_router_client = open_router_client

    def extract_episodic_candidates(
        self, messages: list[dict[str, str]]
    ) -> list[EpisodicMemoryCandidate]:
        """Extract episodic memory candidates for later deduplication and storage."""
        extraction_messages = [
            {"role": "system", "content": EPISODIC_MEMORY_EXTRACTION_PROMPT},
            *messages,
        ]
        result = self._open_router_client.complete_structured(
            extraction_messages,
            EpisodicMemoryExtractionResult,
        )
        return result.episodic_memories
