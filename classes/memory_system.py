import numpy as np

from classes.episodic_memory import (
    EpisodicMemoryCandidate,
    EpisodicMemoryExtractionResult,
)
from classes.memory_store import MemoryStore
from classes.open_router_client import OpenRouterClient

EPISODIC_MEMORY_SIMILARITY_THRESHOLD = 0.90

EPISODIC_MEMORY_EXTRACTION_PROMPT = """Extract episodic memories from this conversation.

An episodic memory is a past event involving a decision, outcome, or development — not a passing update on the user's current state. Store events, not facts about the user (those belong elsewhere). This is not a conversation log. Most exchanges yield none.

If no event qualifies, return an empty list.

For each memory:
- description: what happened
- keywords: search terms for retrieval
- extraction_reason: how retrieving this would change what the agent should do or prioritize"""


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        memory_store: MemoryStore,
        open_router_client: OpenRouterClient,
    ) -> None:
        self._memory_store = memory_store
        self._open_router_client = open_router_client

    def add(self, messages: list[dict[str, str]]) -> None:
        """Add a memory."""
        episodic_memory_candidates = self._extract_episodic_memories(messages)
        if not episodic_memory_candidates:
            return

        stored_memories = self._memory_store.load_episodic_memories()
        candidate_embeddings = self._open_router_client.embed(
            [candidate.description for candidate in episodic_memory_candidates]
        )
        stored_embedding_matrix = self._stored_embedding_matrix(stored_memories)
        unique_candidates = self._filter_duplicate_candidates(
            episodic_memory_candidates,
            candidate_embeddings,
            stored_embedding_matrix,
        )

        for candidate, embedding in unique_candidates:
            self._memory_store.store_episodic_memory(
                description=candidate.description,
                keywords=candidate.keywords,
                extraction_reason=candidate.extraction_reason,
                embedding=embedding,
            )

    def _extract_episodic_memories(
        self, messages: list[dict[str, str]]
    ) -> list[EpisodicMemoryCandidate]:
        """Extract episodic memory candidates from messages for later deduplication and storage."""
        extraction_messages = [
            {"role": "system", "content": EPISODIC_MEMORY_EXTRACTION_PROMPT},
            *messages,
        ]
        result = self._open_router_client.complete_structured(
            extraction_messages,
            EpisodicMemoryExtractionResult,
        )
        return result.episodic_memories

    def _embed_episodic_memory_candidate(
        self, candidate: EpisodicMemoryCandidate
    ) -> list[float]:
        """Return an embedding vector for an episodic memory candidate."""
        return self._open_router_client.embed([candidate.description])[0]

    def _stored_embedding_matrix(
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
            descriptions_to_embed.append(description)
            indices_to_embed.append(index)

        if descriptions_to_embed:
            embedded_vectors = self._open_router_client.embed(descriptions_to_embed)
            for index, vector in zip(indices_to_embed, embedded_vectors):
                vectors[index] = vector

        if any(vector is None for vector in vectors):
            raise RuntimeError("Failed to build embedding matrix for stored memories")

        return np.asarray(vectors)

    def _filter_duplicate_candidates(
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
        similarities = candidate_matrix @ stored_embedding_matrix.T / (
            candidate_norms[:, np.newaxis] * stored_norms
        )
        max_similarities = np.max(similarities, axis=1)

        return [
            (candidate, embedding)
            for candidate, embedding, similarity in zip(
                candidates, candidate_embeddings, max_similarities
            )
            if similarity < EPISODIC_MEMORY_SIMILARITY_THRESHOLD
        ]

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    # def get(self, query: str) -> list[str]:
    #     """Search memories."""
