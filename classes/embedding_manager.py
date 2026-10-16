from classes.graph_models import GraphUpdate
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

    def embed_strings(self, strings: list[str]) -> list[list[float]]:
        """Return embedding vectors for each string in one batch request."""
        if not strings:
            return []
        return self._embedding_client.embed(strings)

    def attach_embeddings(
        self,
        graph_update: GraphUpdate,
    ) -> None:
        """Batch-embed each entity name onto entity.embedding."""
        entity_names = [entity.name for entity in graph_update.entities]
        embeddings = self.embed_strings(entity_names)
        if len(embeddings) != len(graph_update.entities):
            raise RuntimeError("Embedding count does not match extracted entity count.")
        for entity, embedding in zip(
            graph_update.entities,
            embeddings,
            strict=True,
        ):
            entity.embedding = embedding
