from pathlib import Path


class FactsStore:
    """Persists extracted memories as JSON."""

    def __init__(self, memories_path: Path = Path("data/memories.json")) -> None:
        self._memories_path = memories_path

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
