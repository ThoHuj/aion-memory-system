from classes.memory_store import MemoryStore


class MemorySystem:
    """Memory system."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self._memory_store = memory_store

    def add(self, messages: list[dict[str, str]]) -> None:
        """Add a memory."""

    # def update(self, memory_id: str, messages: list[dict[str, str]]) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
