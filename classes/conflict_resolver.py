import hashlib

from classes.facts_extraction_result import NewFacts


class ConflictResolver:
    """Resolve conflicts between newly extracted facts and stored facts."""

    def hash_fact(self, fact: str) -> str:
        """Hash a fact string for duplicate detection."""
        return hashlib.sha256(fact.encode("utf-8")).hexdigest()

    def attach_fact_hashes(self, new_facts: NewFacts) -> None:
        """Set fact_hash on each extracted fact from its content."""
        for extracted_fact in new_facts.facts:
            extracted_fact.fact_hash = self.hash_fact(extracted_fact.content)

    def remove_duplicated_facts(self, new_facts: NewFacts) -> None:
        """Drop later facts that share a fact_hash with an earlier fact in new_facts."""
        seen_hashes: set[str] = set()
        unique_facts = []
        for extracted_fact in new_facts.facts:
            if extracted_fact.fact_hash in seen_hashes:
                continue
            seen_hashes.add(extracted_fact.fact_hash)
            unique_facts.append(extracted_fact)
        new_facts.facts = unique_facts
