from classes.memory_embedder import MemoryEmbedder
from classes.memory_models import ExtractedEpisodicMemory, ExtractionResult
from classes.memory_store import EpisodicMemoryRecord, MemoriesDocument, MemoryStore

EPISODIC_MEMORY_CONFLICT_SIMILARITY_THRESHOLD = 0.88


class ConflictResolver:
    """Reconcile newly extracted memories with what is already stored."""

    def __init__(
        self,
        memory_store: MemoryStore,
        memory_embedder: MemoryEmbedder,
        similarity_threshold: float = EPISODIC_MEMORY_CONFLICT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._memory_store = memory_store
        self._memory_embedder = memory_embedder
        self._similarity_threshold = similarity_threshold

    def resolve_extraction_conflicts(
        self,
        extraction_result: ExtractionResult,
    ) -> ExtractionResult:
        """Reconcile a new extraction result with memories already on disk."""
        stored_memories = self._memory_store.read_memories_document()
        embedded_episodic_memories = self._memory_embedder.embed_episodic_memory(
            extraction_result.episodic_memories,
        )

        episodic_memories_to_store: list[ExtractedEpisodicMemory] = []
        for candidate in embedded_episodic_memories:
            conflicting_stored_memories = (
                self._stored_episodic_memories_above_similarity_threshold(
                    candidate,
                    stored_memories,
                )
            )
            if not conflicting_stored_memories:
                episodic_memories_to_store.append(candidate)
                continue
            self._handle_episodic_memory_conflict(
                candidate,
                conflicting_stored_memories,
            )

        return extraction_result.model_copy(
            update={"episodic_memories": episodic_memories_to_store}
        )

    def _stored_episodic_memories_above_similarity_threshold(
        self,
        candidate: ExtractedEpisodicMemory,
        stored_memories: MemoriesDocument,
    ) -> list[EpisodicMemoryRecord]:
        """Return all stored episodic memories at or above the similarity threshold."""
        if not candidate.embedding:
            return []

        conflicting_stored_memories: list[EpisodicMemoryRecord] = []
        for stored_memory in stored_memories.episodic_memories:
            if not stored_memory.embedding:
                continue
            similarity = self._cosine_similarity(
                candidate.embedding,
                stored_memory.embedding,
            )
            if similarity >= self._similarity_threshold:
                conflicting_stored_memories.append(stored_memory)
        return conflicting_stored_memories

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        dot = sum(
            left_value * right_value for left_value, right_value in zip(left, right)
        )
        left_norm = sum(left_value * left_value for left_value in left) ** 0.5
        right_norm = sum(right_value * right_value for right_value in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _handle_episodic_memory_conflict(
        self,
        candidate: ExtractedEpisodicMemory,
        conflicting_stored_memories: list[EpisodicMemoryRecord],
    ) -> None:
        """Apply policy when a new episodic memory conflicts with existing ones."""
        _ = candidate, conflicting_stored_memories
