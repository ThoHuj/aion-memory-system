class ConflictResolver:
    """Resolve conflicts between newly extracted facts and stored facts."""

    def hash_fact(self, fact: str) -> None:
        """Hash a fact for duplicate detection."""

    def remove_duplicated_facts(self, facts: list[str]) -> None:
        """Remove duplicated facts from a list of fact strings."""
