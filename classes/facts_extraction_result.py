import hashlib

from pydantic import BaseModel, Field


def hash_fact_content(content: str) -> str:
    """Return a stable hash for fact text used in duplicate detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ExtractedFact(BaseModel):
    """One fact moving through the add pipeline (extract, embed, hash, entities)."""

    content: str
    embedding: list[float] = Field(default_factory=list)
    fact_hash: str = ""
    fact_id: str = ""
    entities: list[str] = Field(default_factory=list)


class NewFacts(BaseModel):
    """Facts and enrichments accumulated during MemorySystem.add."""

    facts: list[ExtractedFact] = Field(default_factory=list)

    @classmethod
    def from_fact_strings(cls, fact_strings: list[str]) -> "NewFacts":
        """Build a pipeline result from raw extracted fact text."""
        facts: list[ExtractedFact] = []
        for raw_content in fact_strings:
            content = raw_content.strip()
            facts.append(
                ExtractedFact(
                    content=content,
                    fact_hash=hash_fact_content(content),
                )
            )
        return cls(facts=facts)
