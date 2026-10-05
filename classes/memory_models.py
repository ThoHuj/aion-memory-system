from typing import Self

from pydantic import BaseModel, Field


class ExtractedMemoryContent(BaseModel):
    """Content-only memory text from extraction (no embedding)."""

    content: str


class ExtractedEpisodicMemory(BaseModel):
    """Episodic memory in the extraction and conflict-resolution pipeline."""

    content: str
    embedding: list[float] = Field(default_factory=list)

    @classmethod
    def from_content(cls, content: str) -> Self:
        """Build a pipeline episodic memory from extracted content text."""
        return cls(content=content)


class ExtractedSemanticFact(ExtractedMemoryContent):
    """Semantic fact in the extraction and conflict-resolution pipeline."""


class ExtractionResult(BaseModel):
    """Structured output from memory and fact extraction."""

    episodic_memories: list[ExtractedEpisodicMemory] = Field(default_factory=list)
    semantic_facts: list[ExtractedSemanticFact] = Field(default_factory=list)
