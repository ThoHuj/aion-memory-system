from pathlib import Path

from pydantic import BaseModel, Field


class RetrievedVectorStoreFact(BaseModel):
    """One fact returned from a vector store search."""

    fact_id: str
    content: str
    fact_hash: str


class VectorSearchResult(BaseModel):
    """Facts retrieved by semantic search in the vector store."""

    facts: list[RetrievedVectorStoreFact] = Field(default_factory=list)


class FactsStore:
    """Persists extracted memories as JSON."""

    def __init__(self, memories_path: Path = Path("data/memories.json")) -> None:
        self._memories_path = memories_path

    def search_nearest_facts(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> VectorSearchResult:
        """Return up to limit semantically nearest facts from the vector store."""
        return VectorSearchResult()

    def get_most_recent_session_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Return up to limit most recent messages for the given session."""
        return []

    def store_fact_in_vector_store(self, fact: str) -> None:
        """Store a fact in the vector store."""

    def add_entity(self, entity: str) -> None:
        """Add an entity to the store."""

    def link_fact_ids_to_entities(
        self,
        fact_ids: list[str],
        entity_ids: list[str],
    ) -> None:
        """Link stored fact identifiers to entity identifiers."""

    def log_stored_facts(self) -> None:
        """Log facts that have been stored."""
