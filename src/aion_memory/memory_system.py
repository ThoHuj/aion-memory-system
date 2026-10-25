from pydantic import BaseModel, Field

from aion_memory.conflict_resolution.conflict_resolver import ConflictResolver
from aion_memory.context_manager import ContextManager, ContextResult
from aion_memory.embedding_manager import EmbeddingManager
from aion_memory.entity_extraction import EntityExtractor
from aion_memory.entity_resolution.entity_resolver import EntityResolver
from aion_memory.facts_store import FactsStore
from aion_memory.graph_models import (
    AddLogRecord,
    ExtractedEntity,
    ExtractedFact,
    FactInvalidation,
    GraphUpdate,
    StoredFact,
)
from aion_memory.graph_retriever import GraphRetriever
from aion_memory.relation_extraction.relation_extractor import RelationExtractor


class FactData(BaseModel):
    """A stored fact together with the add call it originated from."""

    fact: StoredFact
    origin: AddLogRecord


class MemoryAddResult(BaseModel):
    """Outcome of one memory add."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    stored_facts: list[ExtractedFact] = Field(default_factory=list)
    invalidations: list[FactInvalidation] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Outcome of one memory retrieval."""

    retrieved_facts: list[FactData] = Field(default_factory=list)


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        facts_store: FactsStore,
        context_manager: ContextManager,
        entity_extractor: EntityExtractor,
        relation_extractor: RelationExtractor,
        embedding_manager: EmbeddingManager,
        entity_resolver: EntityResolver,
        conflict_resolver: ConflictResolver,
        graph_retriever: GraphRetriever,
    ) -> None:
        self._facts_store = facts_store
        self._context_manager = context_manager
        self._entity_extractor = entity_extractor
        self._relation_extractor = relation_extractor
        self._embedding_manager = embedding_manager
        self._entity_resolver = entity_resolver
        self._conflict_resolver = conflict_resolver
        self._graph_retriever = graph_retriever

    def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: str,
    ) -> MemoryAddResult:
        """Extract entities and facts from messages, persist them, and return the outcome."""
        context_result: ContextResult = self._context_manager.produce_context_result(
            messages,
            user_id,
            session_id,
        )
        graph_update = GraphUpdate()
        self._entity_extractor.attach_entities(graph_update, context_result)
        self._relation_extractor.attach_facts(graph_update, context_result)
        self._embedding_manager.attach_embeddings(graph_update)
        self._entity_resolver.resolve_entities(graph_update, context_result)
        self._conflict_resolver.resolve_conflicts(graph_update, context_result)
        self._facts_store.store_graph_update(graph_update, context_result)
        return MemoryAddResult(
            entities=graph_update.entities,
            stored_facts=graph_update.facts,
            invalidations=graph_update.invalidations,
        )

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    def retrieve(
        self,
        query: str,
        user_id: str,
        limit: int,
    ) -> RetrievalResult:
        """Return up to limit valid facts for user_id that best match query."""
        facts = self._graph_retriever.retrieve_facts(query, user_id, limit)
        return RetrievalResult(
            retrieved_facts=[self.get(fact.fact_id, user_id) for fact in facts]
        )

    def close(self) -> None:
        """Release the database connections. The memory system cannot be used afterwards."""
        self._facts_store.close()

    def get(self, fact_id: str, user_id: str) -> FactData:
        """Return a fact of user_id and its origin. Raises KeyError if the fact is unknown or owned by another user."""
        fact = self._facts_store.get_fact(user_id, fact_id)
        return FactData(
            fact=fact,
            origin=self._facts_store.get_add_log_record(user_id, fact.add_id),
        )
