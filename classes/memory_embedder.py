from classes.open_router_client import OpenRouterClient


class MemoryEmbedder:
    """Attach embedding vectors to extracted episodic memories."""

    def __init__(self, embedding_client: OpenRouterClient) -> None:
        self._embedding_client = embedding_client

    def embed_contents(self, contents: list[str]) -> list[list[float]]:
        """Return embedding vectors for each content string."""
        if not contents:
            return []
        return self._embedding_client.embed(contents)
