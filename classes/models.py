from pydantic import BaseModel, Field


class EpisodicMemoryCreate(BaseModel):
    """A new episodic memory to store after extraction."""

    description: str = Field(
        description="What happened — an event, a decision, an outcome, or development."
    )
    keywords: list[str] = Field(default_factory=list)
    extraction_reason: str = Field(description="One word only.")


class EpisodicMemoryUpdate(BaseModel):
    """Replace description and keywords for an existing episodic memory."""

    uuid: str
    description: str = Field(
        description="What happened — an event, a decision, an outcome, or development."
    )
    keywords: list[str] = Field(default_factory=list)


class EpisodicMemory(EpisodicMemoryCreate):
    """An episodic memory with an embedding vector for semantic retrieval."""

    embedding: list[float] = Field(default_factory=list)


class EpisodicMemoryExtractionResult(BaseModel):
    """Structured model response for episodic memory extraction."""

    creates: list[EpisodicMemoryCreate] = Field(
        default_factory=list,
        description="New episodic memories to store. Empty if none qualify.",
    )
    updates: list[EpisodicMemoryUpdate] = Field(
        default_factory=list,
        description="Existing memories to revise by uuid. Empty if none need changes.",
    )
