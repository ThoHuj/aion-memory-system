class MemorySystem:
    """Memory system."""

    def add(self, content: str) -> None:
        """Add a memory."""

    def update(self, memory_id: str, content: str) -> None:
        """Update a memory."""

    def delete(self, memory_id: str) -> None:
        """Delete a memory."""

    def get(self, query: str) -> list[str]:
        """Search memories."""
