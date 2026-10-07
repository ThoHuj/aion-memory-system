from classes.context_manager import ContextManager, ContextResult
from classes.facts_extractor import FactsExtractor
from classes.facts_store import FactsStore


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        facts_store: FactsStore,
        facts_extractor: FactsExtractor,
        context_manager: ContextManager,
    ) -> None:
        self._facts_store = facts_store
        self._facts_extractor = facts_extractor
        self._context_manager = context_manager

    def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: str,
    ) -> None:
        """Extract memories and persist them."""
        context_result: ContextResult = self._context_manager.produce_context_result(
            messages,
            user_id,
            session_id,
        )

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
