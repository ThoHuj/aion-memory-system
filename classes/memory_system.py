from classes.embedding_manager import EmbeddingManager
from classes.memory_extractor import MemoryExtractor
from classes.memory_store import MemoryStore

_EXTRACTION_DIALOGUE_ROLES = frozenset({"user", "assistant"})

# Set False to skip embedding API calls and near-duplicate filtering in add().
_EMBEDDINGS_ENABLED = False


def _has_any_dialogue(messages: list[dict[str, str]]) -> bool:
    """True if at least one user or assistant message has non-whitespace content."""
    for message in messages:
        if message.get("role") not in _EXTRACTION_DIALOGUE_ROLES:
            continue
        content = message.get("content")
        if content is not None and content.strip():
            return True
    return False


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
        # 1. Check for substantive dialogue from user or assistant before extracting.
        if not _has_any_dialogue(messages):
            return

        # 2. Use the memory extractor to find new episodic memory candidates from the current messages
        episodic_memory_candidates = self._memory_extractor.extract_episodic_candidates(
            messages
        )
        if not episodic_memory_candidates:
            return

        if _EMBEDDINGS_ENABLED:
            # 3. Load stored episodic memories for deduplication and comparison
            stored_memories = self._memory_store.load_episodic_memories()

            # 4. Embed the new candidates and all currently stored episodic memories
            candidate_embeddings = self._embedding_manager.embed_episodic_candidates(
                episodic_memory_candidates
            )
            stored_embeddings = self._embedding_manager.get_stored_memory_embeddings(
                stored_memories
            )

            # 5. Filter out candidate memories that are near-duplicates of those already stored
            unique_candidates = self._embedding_manager.filter_duplicate_candidates(
                episodic_memory_candidates,
                candidate_embeddings,
                stored_embeddings,
            )
        else:
            unique_candidates = [
                (candidate, []) for candidate in episodic_memory_candidates
            ]

        # 6. Store only the unique (non-duplicate) candidate memories in the memory store
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
