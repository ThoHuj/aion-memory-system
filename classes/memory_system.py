from classes.context_manager import ContextManager, ContextResult
from classes.embedding_manager import EmbeddingManager
from classes.facts_extraction_result import NewFacts
from classes.facts_extractor import FactsExtractor
from classes.facts_store import FactsStore


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        facts_store: FactsStore,
        facts_extractor: FactsExtractor,
        context_manager: ContextManager,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self._facts_store = facts_store
        self._facts_extractor = facts_extractor
        self._context_manager = context_manager
        self._embedding_manager = embedding_manager

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
        extracted_fact_strings = (
            self._facts_extractor.extract_facts_from_context_result(context_result)
        )
        new_facts: NewFacts = NewFacts.from_fact_strings(extracted_fact_strings)
        self._embedding_manager.attach_embeddings(new_facts)

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
