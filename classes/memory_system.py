from classes.memory_extractor import MemoryExtractor
from classes.memory_store import MemoryStore


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        memory_store: MemoryStore,
        memory_extractor: MemoryExtractor,
    ) -> None:
        self._memory_store = memory_store
        self._memory_extractor = memory_extractor

    def add(self) -> None:
        """Extract memories and persist them."""

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
