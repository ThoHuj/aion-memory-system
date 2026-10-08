from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from classes.facts_extraction_result import NewFacts

if TYPE_CHECKING:
    from classes.context_manager import ContextResult


class RetrievedVectorStoreFact(BaseModel):
    """One fact returned from a vector store search."""

    fact_id: str
    content: str
    fact_hash: str


class VectorSearchResult(BaseModel):
    """Facts retrieved by semantic search in the vector store."""

    facts: list[RetrievedVectorStoreFact] = Field(default_factory=list)


class VectorStoreFactRecord(BaseModel):
    """One fact persisted in the vector store."""

    fact_id: str
    content: str
    fact_hash: str
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


class FactHistoryRecord(BaseModel):
    """Log entry for facts added from one extraction window."""

    history_id: str
    user_id: str
    session_id: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    created_at: str


class FactHistoryDocument(BaseModel):
    """On-disk fact history payload."""

    entries: list[FactHistoryRecord] = Field(default_factory=list)


class FactsStore:
    """Persists facts across vector, entity, and history backends (JSON v1)."""

    def __init__(self, data_directory: Path = Path("data")) -> None:
        self._data_directory = data_directory
        self._vector_store_path = data_directory / "vector_store.json"
        self._entity_store_path = data_directory / "entity_store.json"
        self._history_store_path = data_directory / "history_store.json"

    def store_new_facts(
        self,
        new_facts: NewFacts,
        context_result: ContextResult,
    ) -> None:
        """Persist new facts to the vector store, entity store, and history log."""
        if not new_facts.facts:
            return
        self.store_facts_in_vector_store(new_facts)
        self.store_entities_for_facts(new_facts)
        self.log_fact_history(new_facts, context_result)

    def store_facts_in_vector_store(self, new_facts: NewFacts) -> None:
        """Append new facts to the vector store and set fact_id on each ExtractedFact."""
        document = self._read_vector_store_document()
        for extracted_fact in new_facts.facts:
            fact_id = str(uuid.uuid4())
            extracted_fact.fact_id = fact_id
            document.facts.append(
                VectorStoreFactRecord(
                    fact_id=fact_id,
                    content=extracted_fact.content,
                    fact_hash=extracted_fact.fact_hash,
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

    def log_fact_history(
        self,
        new_facts: NewFacts,
        context_result: ContextResult,
    ) -> None:
        """Append a history entry with the message window and stored fact ids."""
        fact_ids = [fact.fact_id for fact in new_facts.facts if fact.fact_id]
        if not fact_ids:
            return
        document = self._read_history_document()
        document.entries.append(
            FactHistoryRecord(
                history_id=str(uuid.uuid4()),
                user_id=context_result.metadata.user_id,
                session_id=context_result.metadata.session_id,
                messages=list(context_result.messages),
                fact_ids=fact_ids,
                created_at=datetime.now(UTC).isoformat(),
            )
        )
        self._write_history_document(document)

    def search_nearest_facts(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> VectorSearchResult:
        """Return up to limit semantically nearest facts from the vector store."""
        if not query_embedding or limit <= 0:
            return VectorSearchResult()

        document = self._read_vector_store_document()
        scored_facts: list[tuple[float, VectorStoreFactRecord]] = []
        for stored_fact in document.facts:
            if not stored_fact.embedding:
                continue
            similarity = _cosine_similarity(query_embedding, stored_fact.embedding)
            scored_facts.append((similarity, stored_fact))

        scored_facts.sort(key=lambda item: item[0], reverse=True)
        return VectorSearchResult(
            facts=[
                RetrievedVectorStoreFact(
                    fact_id=stored_fact.fact_id,
                    content=stored_fact.content,
                    fact_hash=stored_fact.fact_hash,
                )
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

        document = self._read_history_document()
        session_entries = [
            entry
            for entry in document.entries
            if entry.session_id == session_id
        ]
        session_entries.sort(key=lambda entry: entry.created_at, reverse=True)

        collected_messages: list[dict[str, str]] = []
        for entry in session_entries:
            for message in reversed(entry.messages):
                collected_messages.insert(0, message)
                if len(collected_messages) >= limit:
                    return collected_messages[-limit:]
        return collected_messages

    def _read_vector_store_document(self) -> VectorStoreDocument:
        return self._read_document(self._vector_store_path, VectorStoreDocument)

    def _write_vector_store_document(self, document: VectorStoreDocument) -> None:
        self._write_document(self._vector_store_path, document)

    def _read_entity_store_document(self) -> EntityStoreDocument:
        return self._read_document(self._entity_store_path, EntityStoreDocument)

    def _write_entity_store_document(self, document: EntityStoreDocument) -> None:
        self._write_document(self._entity_store_path, document)

    def _read_history_document(self) -> FactHistoryDocument:
        return self._read_document(self._history_store_path, FactHistoryDocument)

    def _write_history_document(self, document: FactHistoryDocument) -> None:
        self._write_document(self._history_store_path, document)

    def _read_document(
        self,
        path: Path,
        document_model: type[VectorStoreDocument]
        | type[EntityStoreDocument]
        | type[FactHistoryDocument],
    ) -> VectorStoreDocument | EntityStoreDocument | FactHistoryDocument:
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
