from classes.memory_extraction import ExtractionResult
from classes.memory_store import MemoriesDocument


def resolve_extraction_conflicts(
    extraction_result: ExtractionResult,
    stored_memories: MemoriesDocument,
) -> ExtractionResult:
    """Reconcile a new extraction result with memories already on disk."""
    return extraction_result
