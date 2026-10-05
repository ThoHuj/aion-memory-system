from classes.memory_models import ExtractedEpisodicMemory
from classes.open_router_client import OpenRouterClient


class MemoryEmbedder:
    """Attach embedding vectors to extracted episodic memories."""

    def __init__(self, embedding_client: OpenRouterClient) -> None:
        self._embedding_client = embedding_client

    def embed_episodic_memory(
        self, episodic_memories: list[ExtractedEpisodicMemory]
    ) -> list[ExtractedEpisodicMemory]:
        """Attach embedding vectors to each extracted episodic memory."""
        if not episodic_memories:
            return []

        texts = [episodic_memory.content for episodic_memory in episodic_memories]
        vectors = self._embedding_client.embed(texts)
        return [
            episodic_memory.model_copy(update={"embedding": vector})
            for episodic_memory, vector in zip(episodic_memories, vectors)
        ]
