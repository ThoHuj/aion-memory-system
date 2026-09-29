from classes.episodic_memory import EpisodicMemory, EpisodicMemoryExtractionResult
from classes.memory_store import MemoryStore
from classes.open_router_client import OpenRouterClient

EPISODIC_MEMORY_EXTRACTION_PROMPT = """You extract episodic memories from a conversation.

Purpose: 
- The memories will be stored in a database and retrieved before each LLM conversation in the future to give the agent a long-term memory.

What to **not** extract: 
- Do not extract memories that an LLM doesn't need to store in a long-term memory database.

Review the conversation messages and identify episodic memories. For each one, provide:
- description: a concise summary of what happened
- keywords: search terms that would help retrieve this memory later
- extraction_reason: why this specific moment qualifies as an episodic memory worth storing"""


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        memory_store: MemoryStore,
        open_router_client: OpenRouterClient,
    ) -> None:
        self._memory_store = memory_store
        self._open_router_client = open_router_client

    def add(self, messages: list[dict[str, str]]) -> None:
        """Add a memory."""
        episodic_memories = self._extract_episodic_memories(messages)
        for episodic_memory in episodic_memories:
            self._memory_store.store_episodic_memory(
                description=episodic_memory.description,
                keywords=episodic_memory.keywords,
                extraction_reason=episodic_memory.extraction_reason,
            )

    def _extract_episodic_memories(
        self, messages: list[dict[str, str]]
    ) -> list[EpisodicMemory]:
        """Extract episodic memories from messages."""
        extraction_messages = [
            {"role": "system", "content": EPISODIC_MEMORY_EXTRACTION_PROMPT},
            *messages,
        ]
        result = self._open_router_client.complete_structured(
            extraction_messages,
            EpisodicMemoryExtractionResult,
        )
        return result.episodic_memories

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
