from classes.memory_extraction import extract_memories_and_facts_from_messages
from classes.memory_store import MemoryStore
from classes.open_router_client import OpenRouterClient


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        memory_store: MemoryStore,
        extraction_client: OpenRouterClient,
    ) -> None:
        self._memory_store = memory_store
        self._extraction_client = extraction_client

    def add(self, messages: list[dict[str, str]]) -> None:
        """Extract episodic memory creates/updates/merges from the conversation and persist them."""
        extraction_result = extract_memories_and_facts_from_messages(
            self._extraction_client, messages
        )
        self._memory_store.store_extraction_result(extraction_result)

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
