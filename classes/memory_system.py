from classes.embedding_manager import EmbeddingManager
from classes.memory_extractor import MemoryExtractor
from classes.memory_store import MemoryStore

_EXTRACTION_DIALOGUE_ROLES = frozenset({"user", "assistant"})


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

    def add(self, thread_id: str, messages: list[dict[str, str]]) -> None:
        """Extract episodic memory creates/updates from the conversation and persist them."""
        if not _has_any_dialogue(messages):
            return

        extraction_result = self._memory_extractor.extract_episodic_memories(
            thread_id, messages
        )
        if not extraction_result.creates and not extraction_result.updates:
            return

        thread_memories = self._memory_store.load_episodic_memories_for_thread(
            thread_id
        )
        thread_memory_uuids = {
            memory_uuid
            for memory in thread_memories
            if isinstance((memory_uuid := memory.get("uuid")), str)
        }

        for update in extraction_result.updates:
            if update.uuid not in thread_memory_uuids:
                raise ValueError(
                    f"Episodic memory update targets uuid not in thread {thread_id!r}: "
                    f"{update.uuid!r}"
                )
            self._memory_store.update_episodic_memory(
                memory_uuid=update.uuid,
                description=update.description,
                keywords=update.keywords,
            )

        for create in extraction_result.creates:
            self._memory_store.store_episodic_memory(
                thread_id=thread_id,
                description=create.description,
                keywords=create.keywords,
                extraction_reason=create.extraction_reason,
                embedding=[],
            )

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
