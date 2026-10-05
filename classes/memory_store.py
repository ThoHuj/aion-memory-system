import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field

from classes.memory_models import (
    ExtractedEpisodicMemory,
    ExtractedSemanticFact,
    ExtractionResult,
)


class EpisodicMemoryRecord(BaseModel):
    """Persisted episodic memory with storage metadata."""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    embedding: list[float] = Field(default_factory=list)
    created_at: str
    last_changed: str

    @classmethod
    def from_extracted_episodic_memory(
        cls, extracted_memory: ExtractedEpisodicMemory
    ) -> Self:
        """Create a record from an extracted episodic memory."""
        timestamp = datetime.now(UTC).isoformat()
        return cls(
            content=extracted_memory.content,
            embedding=list(extracted_memory.embedding),
            created_at=timestamp,
            last_changed=timestamp,
        )


class SemanticFactRecord(BaseModel):
    """Persisted semantic fact with storage metadata."""

    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    created_at: str
    last_changed: str

    @classmethod
    def from_extracted_semantic_fact(
        cls, extracted_fact: ExtractedSemanticFact
    ) -> Self:
        """Create a record from an extracted semantic fact."""
        timestamp = datetime.now(UTC).isoformat()
        return cls(
            content=extracted_fact.content,
            created_at=timestamp,
            last_changed=timestamp,
        )


class MemoriesDocument(BaseModel):
    """Episodic and semantic memories persisted together, stored separately."""

    episodic_memories: list[EpisodicMemoryRecord] = Field(default_factory=list)
    semantic_facts: list[SemanticFactRecord] = Field(default_factory=list)


class MemoryStore:
    """Persists extracted memories as JSON."""

    def __init__(self, memories_path: Path = Path("data/memories.json")) -> None:
        self._memories_path = memories_path

    def store_extraction_result(self, extraction_result: ExtractionResult) -> None:
        """Append episodic memories and semantic facts from an extraction result."""
        if (
            not extraction_result.episodic_memories
            and not extraction_result.semantic_facts
        ):
            return

        document = self.read_memories_document()
        document.episodic_memories.extend(
            EpisodicMemoryRecord.from_extracted_episodic_memory(extracted_memory)
            for extracted_memory in extraction_result.episodic_memories
        )
        document.semantic_facts.extend(
            SemanticFactRecord.from_extracted_semantic_fact(extracted_fact)
            for extracted_fact in extraction_result.semantic_facts
        )
        self._write_memories_document(document)

    def read_memories_document(self) -> MemoriesDocument:
        """Return the current persisted episodic and semantic memories."""
        if not self._memories_path.exists():
            return MemoriesDocument()

        elements = json.loads(self._memories_path.read_text(encoding="utf-8"))
        if isinstance(elements, list):
            return MemoriesDocument(
                episodic_memories=[
                    EpisodicMemoryRecord.model_validate(element)
                    for element in elements
                ],
            )
        return MemoriesDocument.model_validate(elements)

    def write_memories_document(self, document: MemoriesDocument) -> None:
        """Overwrite the memories JSON file from an in-memory document."""
        self._write_memories_document(document)

    def update_episodic_memory_in_document(
        self,
        document: MemoriesDocument,
        memory_uuid: str,
        content: str,
        embedding: list[float],
    ) -> bool:
        """Update one episodic memory in a document; return False if uuid was not found."""
        for stored_memory in document.episodic_memories:
            if stored_memory.uuid != memory_uuid:
                continue
            stored_memory.content = content
            stored_memory.embedding = embedding
            stored_memory.last_changed = datetime.now(UTC).isoformat()
            return True
        return False

    def _write_memories_document(self, document: MemoriesDocument) -> None:
        self._memories_path.parent.mkdir(parents=True, exist_ok=True)
        self._memories_path.write_text(
            json.dumps(document.model_dump(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
