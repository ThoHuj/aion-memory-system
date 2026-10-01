from pydantic import BaseModel, Field


class EpisodicMemoryCandidate(BaseModel):
    """An episodic memory extracted from a conversation, before embedding."""

    description: str = Field(
        description="What happened — an event, a decision, an outcome, or development."
    )
    keywords: list[str] = Field(default_factory=list)
    extraction_reason: str = Field(description="One word only.")


class EpisodicMemory(EpisodicMemoryCandidate):
    """An episodic memory with an embedding vector for semantic retrieval."""

    embedding: list[float] = Field(default_factory=list)


class EpisodicMemoryExtractionResult(BaseModel):
    """Structured model response for episodic memory extraction."""

    episodic_memories: list[EpisodicMemoryCandidate] = Field(
        default_factory=list,
        description="Events involving a decision, outcome, or development. Empty if none qualify.",
    )
