from classes.facts_extraction_result import NewFacts
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
        new_facts: NewFacts,
    ) -> None:
        """Batch-embed each extracted fact's content onto the pipeline result."""
        fact_contents = [fact.content for fact in new_facts.facts]
        embeddings = self.embed_strings(fact_contents)
        if len(embeddings) != len(new_facts.facts):
            raise RuntimeError("Embedding count does not match extracted fact count.")
        for extracted_fact, embedding in zip(
            new_facts.facts,
            embeddings,
            strict=True,
        ):
            extracted_fact.embedding = embedding
