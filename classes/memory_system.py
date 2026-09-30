from classes.embedding_manager import EmbeddingManager
from classes.memory_extractor import MemoryExtractor
from classes.memory_store import MemoryStore


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        memory_store: MemoryStore,
        memory_extractor: MemoryExtractor,
        embedding_manager: EmbeddingManager,
    ) -> None:
        self._memory_store = memory_store
        self._memory_extractor = memory_extractor
        self._embedding_manager = embedding_manager

    def add(self, messages: list[dict[str, str]]) -> None:
        """Extract episodic memories from the conversation, omit near-duplicates, and store the rest."""
        episodic_memory_candidates = self._memory_extractor.extract_episodic_candidates(
            messages
        )
        if not episodic_memory_candidates:
            return

        stored_memories = self._memory_store.load_episodic_memories()
        candidate_embeddings = self._embedding_manager.embed_episodic_candidates(
            episodic_memory_candidates
        )
        stored_embeddings = self._embedding_manager.get_stored_memory_embeddings(
            stored_memories
        )
        unique_candidates = self._embedding_manager.filter_duplicate_candidates(
            episodic_memory_candidates,
            candidate_embeddings,
            stored_embeddings,
        )

        for candidate, embedding in unique_candidates:
            self._memory_store.store_episodic_memory(
                description=candidate.description,
                keywords=candidate.keywords,
                extraction_reason=candidate.extraction_reason,
                embedding=embedding,
            )

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
