from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    """One fact moving through the add pipeline (extract, embed, hash, entities)."""

    content: str
    embedding: list[float] = Field(default_factory=list)
    fact_hash: str = ""
    entities: list[str] = Field(default_factory=list)


class NewFacts(BaseModel):
    """Facts and enrichments accumulated during MemorySystem.add."""

    facts: list[ExtractedFact] = Field(default_factory=list)

    @classmethod
    def from_fact_strings(cls, fact_strings: list[str]) -> "NewFacts":
        """Build a pipeline result from raw extracted fact text."""
        return cls(
            facts=[
                ExtractedFact(content=content.strip())
                for content in fact_strings
            ],
        )
