from pydantic import BaseModel, Field

from classes.conflict_resolver import ConflictResolver
from classes.context_manager import ContextManager, ContextResult
from classes.embedding_manager import EmbeddingManager
from classes.entity_extraction import EntityExtractor
from classes.facts_extraction import FactsExtractor
from classes.facts_extraction.facts_extractor import (
    FactsExtractionContext,
    build_facts_extraction_context,
)
from classes.facts_extraction_result import NewFacts
from classes.facts_store import FactsStore


class StoredFactLogEntry(BaseModel):
    """One fact persisted from a memory add, with entity extraction output."""

    fact_id: str
    content: str
    entities: list[str] = Field(default_factory=list)


class MemoryAddResult(BaseModel):
    """Outcome of one memory add, including facts extraction inputs."""

    stored_facts: list[StoredFactLogEntry] = Field(default_factory=list)
    extraction_context: FactsExtractionContext


class FactSource(BaseModel):
    """The add call (and its messages) a fact was extracted from."""

    add_id: str
    session_id: str
    created_at: str
    messages: list[dict[str, str]] = Field(default_factory=list)


class FactData(BaseModel):
    """All stored data related to one fact."""

    fact_id: str
    user_id: str
    content: str
    fact_hash: str
    stored_at: str
    entities: list[str] = Field(default_factory=list)
    source: FactSource | None = None


class MemorySystem:
    """A high-level interface for extracting, storing, and retrieving agent memories."""

    def __init__(
        self,
        facts_store: FactsStore,
        facts_extractor: FactsExtractor,
        context_manager: ContextManager,
        embedding_manager: EmbeddingManager,
        conflict_resolver: ConflictResolver,
        entity_extractor: EntityExtractor,
    ) -> None:
        self._facts_store = facts_store
        self._facts_extractor = facts_extractor
        self._context_manager = context_manager
        self._embedding_manager = embedding_manager
        self._conflict_resolver = conflict_resolver
        self._entity_extractor = entity_extractor

    def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: str,
    ) -> MemoryAddResult:
        """Extract memories, persist them, and return stored facts plus extraction context."""
        context_result: ContextResult = self._context_manager.produce_context_result(
            messages,
            user_id,
            session_id,
        )
        extraction_context = build_facts_extraction_context(context_result)
        extracted_fact_strings = self._facts_extractor.extract_facts(context_result)
        new_facts: NewFacts = NewFacts.from_fact_strings(extracted_fact_strings)
        self._conflict_resolver.remove_duplicated_facts(new_facts, context_result)
        self._embedding_manager.attach_embeddings(new_facts)
        self._entity_extractor.attach_entities(new_facts)
        self._facts_store.store_new_facts(new_facts, context_result)
        return MemoryAddResult(
            stored_facts=[
                StoredFactLogEntry(
                    fact_id=extracted_fact.fact_id,
                    content=extracted_fact.content,
                    entities=list(extracted_fact.entities),
                )
                for extracted_fact in new_facts.facts
            ],
            extraction_context=extraction_context,
        )

    # def update(self, memory_id: str, content: str) -> None:
    #     """Update a memory."""

    # def delete(self, memory_id: str) -> None:
    #     """Delete a memory."""

    def get(self, fact_id: str) -> FactData:
        """Return all stored data for fact_id. Raises KeyError if the fact is unknown."""
        records = self._facts_store.get_fact_records(fact_id)
        source: FactSource | None = None
        add_log_entry = records.add_log_entry
        if add_log_entry is not None:
            source = FactSource(
                add_id=add_log_entry.add_id,
                session_id=add_log_entry.session_id,
                created_at=add_log_entry.created_at,
                messages=list(add_log_entry.messages),
            )
        stored_at = records.fact.stored_at
        if not stored_at and add_log_entry is not None:
            stored_at = add_log_entry.created_at
        return FactData(
            fact_id=records.fact.fact_id,
            user_id=records.fact.user_id,
            content=records.fact.content,
            fact_hash=records.fact.fact_hash,
            stored_at=stored_at,
            entities=records.entity_names,
            source=source,
        )
