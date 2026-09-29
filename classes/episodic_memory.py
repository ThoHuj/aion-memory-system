from pydantic import BaseModel, Field


class EpisodicMemory(BaseModel):
    """An episodic memory extracted from a conversation."""

    description: str
    keywords: list[str] = Field(default_factory=list)
    extraction_reason: str = Field(
        description="Why this moment was chosen as an episodic memory worth storing."
    )


class EpisodicMemoryExtractionResult(BaseModel):
    """Structured model response for episodic memory extraction."""

    episodic_memories: list[EpisodicMemory] = Field(default_factory=list)
