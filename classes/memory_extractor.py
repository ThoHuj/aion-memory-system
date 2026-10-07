from classes.open_router_client import OpenRouterClient


class MemoryExtractor:
    """Extract episodic memories and semantic facts from conversation via structured LLM output."""

    def __init__(self, extraction_client: OpenRouterClient) -> None:
        self._extraction_client = extraction_client

    def extract_memories_and_facts_from_messages(self) -> None:
        """Return structured extraction results (not yet implemented)."""
