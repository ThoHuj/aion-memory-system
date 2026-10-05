from classes.memory_models import ExtractedEpisodicMemory
from classes.open_router_client import OpenRouterClient


def embed_episodic_memory(
    embedding_client: OpenRouterClient,
    episodic_memories: list[ExtractedEpisodicMemory],
) -> list[ExtractedEpisodicMemory]:
    """Attach embedding vectors to each extracted episodic memory."""
    if not episodic_memories:
        return []

    texts = [episodic_memory.content for episodic_memory in episodic_memories]
    vectors = embedding_client.embed(texts)
    return [
        episodic_memory.model_copy(update={"embedding": vector})
        for episodic_memory, vector in zip(episodic_memories, vectors)
    ]
