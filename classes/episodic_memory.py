from pydantic import BaseModel, Field


class EpisodicMemory(BaseModel):
    """An episodic memory extracted from a conversation."""

    description: str
    keywords: list[str] = Field(default_factory=list)


class EpisodicMemoryExtractionResult(BaseModel):
    """Structured model response for episodic memory extraction."""

    episodic_memories: list[EpisodicMemory] = Field(default_factory=list)
