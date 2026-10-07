from classes.open_router_client import OpenRouterClient


class EmbeddingManager:
    """Attach embedding vectors to text."""

    def __init__(self, embedding_client: OpenRouterClient) -> None:
        self._embedding_client = embedding_client

    def embed_string(self, text: str) -> list[float]:
        """Return the embedding vector for a single string."""
        if not text.strip():
            return []
        vectors = self._embedding_client.embed([text])
        return vectors[0]

    def embed_facts(self, facts: list[str]) -> None:
        """Embed multiple facts in batch."""
