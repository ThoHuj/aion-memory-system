from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, Field

from classes.embedding_manager import EmbeddingManager
from classes.facts_extraction_result import NewFacts

if TYPE_CHECKING:
    from classes.context_manager import ContextResult


class RetrievedVectorStoreFact(BaseModel):
    """One fact returned from a vector store search."""

    fact_id: str
    content: str
    fact_hash: str
    stored_at: str = ""


class VectorSearchResult(BaseModel):
    """Facts retrieved by semantic search in the vector store."""

    facts: list[RetrievedVectorStoreFact] = Field(default_factory=list)


def order_retrieved_facts_by_recency(
    vector_search_result: VectorSearchResult,
) -> VectorSearchResult:
    """Reorder semantically retrieved facts with the most recently stored first."""
    return VectorSearchResult(
        facts=sorted(
            vector_search_result.facts,
            key=lambda retrieved_fact: retrieved_fact.stored_at,
            reverse=True,
        )
    )


def format_retrieved_fact_line(retrieved_fact: RetrievedVectorStoreFact) -> str:
    """Format one retrieved fact with its storage timestamp for LLM context."""
    if retrieved_fact.stored_at:
        return f"[{retrieved_fact.stored_at}] {retrieved_fact.content}"
    return retrieved_fact.content


def semantic_search_text_from_retrieved_facts(
    retrieved_facts: list[RetrievedVectorStoreFact],
) -> str:
    """Build the semantic_search section for fact extraction prompts."""
    if not retrieved_facts:
        return "(none)"
    return "\n".join(
        format_retrieved_fact_line(retrieved_fact) for retrieved_fact in retrieved_facts
    )


class VectorStoreFactRecord(BaseModel):
    """One fact persisted in the vector store."""

    fact_id: str
    user_id: str = ""
    content: str
    fact_hash: str
    stored_at: str = ""
    embedding: list[float] = Field(default_factory=list)


class VectorStoreDocument(BaseModel):
    """On-disk vector store payload."""

    facts: list[VectorStoreFactRecord] = Field(default_factory=list)


class EntityRecord(BaseModel):
    """One unique entity with links to stored facts."""

    entity_id: str
    name: str
    fact_ids: list[str] = Field(default_factory=list)


class EntityStoreDocument(BaseModel):
    """On-disk entity store payload."""

    entities: list[EntityRecord] = Field(default_factory=list)


class AddLogRecord(BaseModel):
    """Log entry for the messages of one add call and the facts stored from them."""

    add_id: str
    user_id: str
    session_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    created_at: str


class AddLogDocument(BaseModel):
    """On-disk add log payload."""

    entries: list[AddLogRecord] = Field(default_factory=list)


StoreDocument = TypeVar(
    "StoreDocument",
    VectorStoreDocument,
    EntityStoreDocument,
    AddLogDocument,
)


class FactsStore:
    """Persists facts across vector, entity, and add log backends (JSON v1)."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        data_directory: Path = Path("data"),
    ) -> None:
        self._embedding_manager = embedding_manager
        self._data_directory = data_directory
        self._vector_store_path = data_directory / "vector_store.json"
        self._entity_store_path = data_directory / "entity_store.json"
        self._add_log_store_path = data_directory / "add_log_store.json"

    def store_new_facts(
        self,
        new_facts: NewFacts,
        context_result: ContextResult,
    ) -> None:
        """Persist new facts, then always log the add call's messages to the add log."""
        if new_facts.facts:
            self.store_facts_in_vector_store(
                new_facts,
                user_id=context_result.metadata.user_id,
            )
            self.store_entities_for_facts(new_facts)
        self.log_add_call(new_facts, context_result)

    def store_facts_in_vector_store(
        self,
        new_facts: NewFacts,
        user_id: str,
    ) -> None:
        """Append new facts to the vector store and set fact_id on each ExtractedFact."""
        document = self._read_vector_store_document()
        stored_at = datetime.now(UTC).isoformat()
        for extracted_fact in new_facts.facts:
            fact_id = str(uuid.uuid4())
            extracted_fact.fact_id = fact_id
            document.facts.append(
                VectorStoreFactRecord(
                    fact_id=fact_id,
                    user_id=user_id,
                    content=extracted_fact.content,
                    fact_hash=extracted_fact.fact_hash,
                    stored_at=stored_at,
                    embedding=list(extracted_fact.embedding),
                )
            )
        self._write_vector_store_document(document)

    def store_entities_for_facts(self, new_facts: NewFacts) -> None:
        """Ensure entities exist and link each entity to its fact ids."""
        document = self._read_entity_store_document()
        entities_by_normalized_name = {
            entity.name.casefold(): entity for entity in document.entities
        }
        for extracted_fact in new_facts.facts:
            if not extracted_fact.fact_id:
                continue
            for entity_name in extracted_fact.entities:
                normalized_name = entity_name.strip()
                if not normalized_name:
                    continue
                lookup_key = normalized_name.casefold()
                entity_record = entities_by_normalized_name.get(lookup_key)
                if entity_record is None:
                    entity_record = EntityRecord(
                        entity_id=str(uuid.uuid4()),
                        name=normalized_name,
                    )
                    document.entities.append(entity_record)
                    entities_by_normalized_name[lookup_key] = entity_record
                if extracted_fact.fact_id not in entity_record.fact_ids:
                    entity_record.fact_ids.append(extracted_fact.fact_id)
        self._write_entity_store_document(document)

    def log_add_call(
        self,
        new_facts: NewFacts,
        context_result: ContextResult,
    ) -> None:
        """Append an add log entry for one add call; fact_ids is empty if none stored."""
        fact_ids = [fact.fact_id for fact in new_facts.facts if fact.fact_id]
        document = self._read_add_log_document()
        document.entries.append(
            AddLogRecord(
                add_id=str(uuid.uuid4()),
                user_id=context_result.metadata.user_id,
                session_id=context_result.metadata.session_id,
                messages=list(context_result.messages),
                fact_ids=fact_ids,
                created_at=datetime.now(UTC).isoformat(),
            )
        )
        self._write_add_log_document(document)

    def search_nearest_facts(
        self,
        user_id: str,
        message: str,
        limit: int,
    ) -> VectorSearchResult:
        """Return up to limit nearest facts for user_id, semantically matching message."""
        if limit <= 0 or not message.strip():
            return VectorSearchResult()

        query_embedding = self._embedding_manager.embed_string(message.strip())
        if not query_embedding:
            return VectorSearchResult()

        document = self._read_vector_store_document()
        scored_facts: list[tuple[float, VectorStoreFactRecord]] = []
        for stored_fact in document.facts:
            if stored_fact.user_id != user_id:
                continue
            if not stored_fact.embedding:
                continue
            similarity = _cosine_similarity(query_embedding, stored_fact.embedding)
            scored_facts.append((similarity, stored_fact))

        scored_facts.sort(key=lambda item: item[0], reverse=True)
        stored_at_by_fact_id = self._stored_at_by_fact_id_from_add_log()
        return VectorSearchResult(
            facts=[
                self._to_retrieved_fact(stored_fact, stored_at_by_fact_id)
                for _, stored_fact in scored_facts[:limit]
            ]
        )

    def get_most_recent_session_messages(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Return up to limit most recent messages for the given session."""
        if limit <= 0:
            return []

        document = self._read_add_log_document()
        session_entries = [
            entry for entry in document.entries if entry.session_id == session_id
        ]
        session_entries.sort(key=lambda entry: entry.created_at, reverse=True)

        collected_messages: list[dict[str, str]] = []
        for entry in session_entries:
            for message in reversed(entry.messages):
                collected_messages.insert(0, message)
                if len(collected_messages) >= limit:
                    return collected_messages[-limit:]
        return collected_messages

    def _to_retrieved_fact(
        self,
        stored_fact: VectorStoreFactRecord,
        stored_at_by_fact_id: dict[str, str],
    ) -> RetrievedVectorStoreFact:
        stored_at = stored_fact.stored_at.strip()
        if not stored_at:
            stored_at = stored_at_by_fact_id.get(stored_fact.fact_id, "")
        return RetrievedVectorStoreFact(
            fact_id=stored_fact.fact_id,
            content=stored_fact.content,
            fact_hash=stored_fact.fact_hash,
            stored_at=stored_at,
        )

    def _stored_at_by_fact_id_from_add_log(self) -> dict[str, str]:
        document = self._read_add_log_document()
        stored_at_by_fact_id: dict[str, str] = {}
        for entry in document.entries:
            for fact_id in entry.fact_ids:
                previous = stored_at_by_fact_id.get(fact_id)
                if previous is None or entry.created_at > previous:
                    stored_at_by_fact_id[fact_id] = entry.created_at
        return stored_at_by_fact_id

    def _read_vector_store_document(self) -> VectorStoreDocument:
        return self._read_document(self._vector_store_path, VectorStoreDocument)

    def _write_vector_store_document(self, document: VectorStoreDocument) -> None:
        self._write_document(self._vector_store_path, document)

    def _read_entity_store_document(self) -> EntityStoreDocument:
        return self._read_document(self._entity_store_path, EntityStoreDocument)

    def _write_entity_store_document(self, document: EntityStoreDocument) -> None:
        self._write_document(self._entity_store_path, document)

    def _read_add_log_document(self) -> AddLogDocument:
        return self._read_document(self._add_log_store_path, AddLogDocument)

    def _write_add_log_document(self, document: AddLogDocument) -> None:
        self._write_document(self._add_log_store_path, document)

    def _read_document(
        self,
        path: Path,
        document_model: type[StoreDocument],
    ) -> StoreDocument:
        if not path.exists():
            return document_model()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return document_model.model_validate(payload)

    def _write_document(self, path: Path, document: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right)
    )
    left_norm = math.sqrt(sum(left_value * left_value for left_value in left))
    right_norm = math.sqrt(sum(right_value * right_value for right_value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)
