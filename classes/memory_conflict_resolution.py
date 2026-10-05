from classes.memory_embedder import embed_episodic_memory
from classes.memory_models import ExtractionResult
from classes.memory_store import MemoriesDocument
from classes.open_router_client import OpenRouterClient


class ConflictResolver:
    """Reconcile newly extracted memories with what is already stored."""

    def __init__(self, embedding_client: OpenRouterClient) -> None:
        self._embedding_client = embedding_client

    def resolve_extraction_conflicts(
        self,
        extraction_result: ExtractionResult,
        stored_memories: MemoriesDocument,
    ) -> ExtractionResult:
        """Reconcile a new extraction result with memories already on disk."""
        embedded_episodic_memories = embed_episodic_memory(
            self._embedding_client,
            extraction_result.episodic_memories,
        )
        return extraction_result.model_copy(
            update={"episodic_memories": embedded_episodic_memories}
        )
