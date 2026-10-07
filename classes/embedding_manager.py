from classes.open_router_client import OpenRouterClient


class EmbeddingManager:
    """Attach embedding vectors to facts."""

    def __init__(self, embedding_client: OpenRouterClient) -> None:
        self._embedding_client = embedding_client

    def embed_fact(self, fact: str) -> None:
        """Embed a single fact."""

    def embed_facts(self, facts: list[str]) -> None:
        """Embed multiple facts in batch."""
