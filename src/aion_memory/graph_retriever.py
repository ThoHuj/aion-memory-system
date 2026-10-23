import re

from rank_bm25 import BM25Okapi

from aion_memory.embedding_manager import EmbeddingManager
from aion_memory.entity_extraction import EntityExtractor
from aion_memory.facts_store import FactsStore
from aion_memory.graph_models import StoredFact, normalize_graph_name


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rank_facts_by_bm25(query: str, facts: list[StoredFact]) -> list[StoredFact]:
    """Return facts ordered by BM25 score against query, keeping input order on ties."""
    if not facts:
        return []
    corpus = [
        tokenize(f"{fact.source_entity} {fact.relation} {fact.destination_entity}")
        for fact in facts
    ]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    ranked_indices = sorted(range(len(facts)), key=lambda i: -scores[i])
    return [facts[i] for i in ranked_indices]


class GraphRetriever:
    """Finds facts relevant to a query by graph traversal from matched entities."""

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        embedding_manager: EmbeddingManager,
        facts_store: FactsStore,
        minimum_seed_similarity: float = 0.7,
        seeds_per_entity: int = 5,
        hops: int = 2,
        facts_per_hop_limit: int = 100,
    ) -> None:
        self._entity_extractor = entity_extractor
        self._embedding_manager = embedding_manager
        self._facts_store = facts_store
        self._minimum_seed_similarity = minimum_seed_similarity
        self._seeds_per_entity = seeds_per_entity
        self._hops = hops
        self._facts_per_hop_limit = facts_per_hop_limit

    def retrieve_facts(self, query: str, user_id: str, limit: int) -> list[StoredFact]:
        """Return up to limit valid facts within the hop range of the user entity and the query entities, BM25 ranked."""
        if limit <= 0 or not query.strip():
            return []
        entities = self._entity_extractor.extract_entities(query, user_id)
        embeddings = (
            self._embedding_manager.embed_strings([entity.name for entity in entities])
            if entities
            else []
        )
        seed_ids: list[str] = []
        user_entity = self._facts_store.find_entity_by_name(
            user_id, normalize_graph_name(user_id)
        )
        if user_entity is not None:
            seed_ids.append(user_entity.entity_id)
        for embedding in embeddings:
            similar_entities = self._facts_store.find_similar_entities(
                user_id,
                embedding,
                self._minimum_seed_similarity,
                self._seeds_per_entity,
            )
            for similar_entity in similar_entities:
                if similar_entity.entity_id not in seed_ids:
                    seed_ids.append(similar_entity.entity_id)
        visited_ids: set[str] = set(seed_ids)
        frontier_ids: list[str] = seed_ids
        facts_by_id: dict[str, StoredFact] = {}
        for _ in range(self._hops):
            if not frontier_ids:
                break
            hop_facts = self._facts_store.get_valid_facts_for_entities(
                user_id, frontier_ids, self._facts_per_hop_limit
            )
            next_frontier_ids: list[str] = []
            for fact in hop_facts:
                facts_by_id.setdefault(fact.fact_id, fact)
                for endpoint_id in (fact.source_entity_id, fact.destination_entity_id):
                    if endpoint_id not in visited_ids:
                        visited_ids.add(endpoint_id)
                        next_frontier_ids.append(endpoint_id)
            frontier_ids = next_frontier_ids
        return rank_facts_by_bm25(query, list(facts_by_id.values()))[:limit]
