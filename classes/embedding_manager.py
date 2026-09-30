from classes.models import EpisodicMemoryCandidate
from classes.open_router_client import OpenRouterClient

EPISODIC_MEMORY_SIMILARITY_THRESHOLD = 0.88


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

    def get_stored_memory_embeddings(
        self, stored_memories: list[dict[str, object]]
    ) -> list[list[float]]:
        """Return embedding vectors for stored memories, embedding legacy rows if needed."""
        if not stored_memories:
            return []

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
            descriptions_to_embed.append(self.text_for_embedding(description, keywords))
            indices_to_embed.append(index)

        if descriptions_to_embed:
            embedded_vectors = self._open_router_client.embed(descriptions_to_embed)
            for index, vector in zip(indices_to_embed, embedded_vectors):
                vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise RuntimeError("Failed to build embeddings for stored memories")

        return [vector for vector in vectors if vector is not None]

    @staticmethod
    def calculate_cosine_similarity(left: list[float], right: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def max_similarity_to_stored(
        self, candidate_embedding: list[float], stored_embeddings: list[list[float]]
    ) -> float:
        """Highest cosine similarity between one candidate and any stored memory."""
        if not stored_embeddings:
            return 0.0
        return max(
            self.calculate_cosine_similarity(candidate_embedding, stored)
            for stored in stored_embeddings
        )

    def filter_duplicate_candidates(
        self,
        candidates: list[EpisodicMemoryCandidate],
        candidate_embeddings: list[list[float]],
        stored_embeddings: list[list[float]],
    ) -> list[tuple[EpisodicMemoryCandidate, list[float]]]:
        """Discard candidates whose max similarity to stored memories exceeds the threshold."""
        unique: list[tuple[EpisodicMemoryCandidate, list[float]]] = []
        for candidate, embedding in zip(candidates, candidate_embeddings):
            similarity = self.max_similarity_to_stored(embedding, stored_embeddings)
            if similarity < EPISODIC_MEMORY_SIMILARITY_THRESHOLD:
                unique.append((candidate, embedding))
        return unique
