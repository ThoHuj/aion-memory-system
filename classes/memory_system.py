from classes.conflict_resolver import ConflictResolutionResult, ConflictResolver
from classes.memory_extractor import MemoryExtractor
from classes.memory_models import ExtractionResult
from classes.memory_store import MemoryStore


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        memory_store: MemoryStore,
        memory_extractor: MemoryExtractor,
        conflict_resolver: ConflictResolver,
    ) -> None:
        self._memory_store = memory_store
        self._memory_extractor = memory_extractor
        self._conflict_resolver = conflict_resolver

    def add(self, messages: list[dict[str, str]]) -> None:
        """Extract episodic memory creates/updates/merges from the conversation and persist them."""
        extraction_result = (
            self._memory_extractor.extract_memories_and_facts_from_messages(messages)
        )
        resolution = self._conflict_resolver.resolve_extraction_conflicts(
            extraction_result,
        )
        self._store_pending_memories(resolution)

    def _store_pending_memories(self, resolution: ConflictResolutionResult) -> None:
        """Append new episodic and semantic memories that survived conflict resolution."""
        pending = ExtractionResult(
            episodic_memories=resolution.episodic_memories_to_create,
            semantic_facts=resolution.semantic_facts_to_create,
        )
        self._memory_store.store_extraction_result(pending)

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
