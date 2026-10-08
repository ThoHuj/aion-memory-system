from classes.context_manager import ContextResult
from classes.facts_extraction_result import NewFacts


class ConflictResolver:
    """Resolve conflicts between newly extracted facts and stored facts."""

    def remove_duplicated_facts(
        self,
        new_facts: NewFacts,
        context_result: ContextResult,
    ) -> None:
        """Drop new facts that duplicate each other or a vector-search stored fact."""
        vector_search_facts = (
            context_result.context_lookup.vector_search_result.facts
        )
        seen_hashes = {
            stored_fact.fact_hash
            for stored_fact in vector_search_facts
            if stored_fact.fact_hash
        }
        unique_facts = []
        for extracted_fact in new_facts.facts:
            if extracted_fact.fact_hash in seen_hashes:
                continue
            seen_hashes.add(extracted_fact.fact_hash)
            unique_facts.append(extracted_fact)
        new_facts.facts = unique_facts
