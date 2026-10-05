from classes.memory_embedder import MemoryEmbedder
from classes.memory_models import ExtractionResult
from classes.memory_store import MemoriesDocument


class ConflictResolver:
    """Reconcile newly extracted memories with what is already stored."""

    def __init__(self, memory_embedder: MemoryEmbedder) -> None:
        self._memory_embedder = memory_embedder

    def resolve_extraction_conflicts(
        self,
        extraction_result: ExtractionResult,
        stored_memories: MemoriesDocument,
    ) -> ExtractionResult:
        """Reconcile a new extraction result with memories already on disk."""
        embedded_episodic_memories = self._memory_embedder.embed_episodic_memory(
            extraction_result.episodic_memories,
        )
        return extraction_result.model_copy(
            update={"episodic_memories": embedded_episodic_memories}
        )
