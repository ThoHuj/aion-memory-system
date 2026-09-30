import numpy as np

from classes.episodic_memory_models import EpisodicMemoryCandidate
from classes.open_router_client import OpenRouterClient

EPISODIC_MEMORY_SIMILARITY_THRESHOLD = 0.85


class EmbeddingManager:
    """Build episodic memory embeddings and filter near-duplicates by similarity."""

    def __init__(self, open_router_client: OpenRouterClient) -> None:
        self._open_router_client = open_router_client

    def embed_episodic_candidates(
        self, candidates: list[EpisodicMemoryCandidate]
    ) -> list[list[float]]:
        """Return embedding vectors for episodic memory candidates in one request."""
        texts = [
            self.text_for_embedding(candidate.description, candidate.keywords)
            for candidate in candidates
        ]
        return self._open_router_client.embed(texts)

    @staticmethod
    def text_for_embedding(description: str, keywords: list[str]) -> str:
        """Build the string passed to the embedding model."""
        if not keywords:
            return description
        return f"{description}\n{', '.join(keywords)}"

    def stored_embedding_matrix(
        self, stored_memories: list[dict[str, object]]
    ) -> np.ndarray:
        """Build a matrix of stored memory embeddings, embedding legacy rows if needed."""
        if not stored_memories:
            return np.empty((0, 0))

        vectors: list[list[float] | None] = [None] * len(stored_memories)
        descriptions_to_embed: list[str] = []
        indices_to_embed: list[int] = []

        for index, memory in enumerate(stored_memories):
            embedding = memory.get("embedding")
            if isinstance(embedding, list) and embedding:
                vectors[index] = embedding
                continue

            description = memory.get("description")
            if not isinstance(description, str):
                raise TypeError("Stored episodic memory missing description")
            raw_keywords = memory.get("keywords")
            keywords = (
                [keyword for keyword in raw_keywords if isinstance(keyword, str)]
                if isinstance(raw_keywords, list)
                else []
            )
            descriptions_to_embed.append(
                self.text_for_embedding(description, keywords)
            )
            indices_to_embed.append(index)

        if descriptions_to_embed:
            embedded_vectors = self._open_router_client.embed(descriptions_to_embed)
            for index, vector in zip(indices_to_embed, embedded_vectors):
                vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise RuntimeError("Failed to build embedding matrix for stored memories")

        return np.asarray(vectors)

    def filter_duplicate_candidates(
        self,
        candidates: list[EpisodicMemoryCandidate],
        candidate_embeddings: list[list[float]],
        stored_embedding_matrix: np.ndarray,
    ) -> list[tuple[EpisodicMemoryCandidate, list[float]]]:
        """Discard candidates whose max similarity to stored memories exceeds the threshold."""
        if stored_embedding_matrix.size == 0:
            return list(zip(candidates, candidate_embeddings))

        candidate_matrix = np.asarray(candidate_embeddings)
        stored_norms = np.linalg.norm(stored_embedding_matrix, axis=1)
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
        similarities = (
            candidate_matrix
            @ stored_embedding_matrix.T
            / (candidate_norms[:, np.newaxis] * stored_norms)
        )
        max_similarities = np.max(similarities, axis=1)

        return [
            (candidate, embedding)
            for candidate, embedding, similarity in zip(
                candidates, candidate_embeddings, max_similarities
            )
            if similarity < EPISODIC_MEMORY_SIMILARITY_THRESHOLD
        ]
