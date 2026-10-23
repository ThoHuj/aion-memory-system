from datetime import UTC, datetime

from pydantic import BaseModel, Field


def normalize_graph_name(name: str) -> str:
    """Normalize an entity name, entity type, or relation (lowercase, underscores)."""
    return name.strip().lower().replace(" ", "_")


def parse_world_time(value: str | None) -> str | None:
    """Parse an ISO 8601 string to a UTC-aware ISO string; None if empty or invalid."""
    if not value:
        return None
    try:
        parsed_time = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed_time.tzinfo is None:
        parsed_time = parsed_time.replace(tzinfo=UTC)
    return parsed_time.isoformat()


class ExtractedEntity(BaseModel):
    """One entity extracted from messages; is_new means not matched to a stored entity."""

    name: str
    entity_type: str
    embedding: list[float] = Field(default_factory=list, exclude=True)
    entity_id: str = ""
    is_new: bool = True


class ExtractedFact(BaseModel):
    """One fact edge between two entities, referenced by entity name."""

    source_entity: str
    relation: str
    destination_entity: str
    world_valid_from: str
    world_valid_to: str | None
    fact_id: str = ""


class FactInvalidation(BaseModel):
    """A stored fact to invalidate, with the world time it stopped being true."""

    fact_id: str
    world_valid_to: str | None


class GraphUpdate(BaseModel):
    """Entities, facts, and invalidations produced by one add call."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)
    invalidations: list[FactInvalidation] = Field(default_factory=list)

    def entity_ids_by_name(self) -> dict[str, str]:
        """Map each entity name to its entity id."""
        return {entity.name: entity.entity_id for entity in self.entities}


class StoredEntity(BaseModel):
    """One entity persisted in the graph."""

    entity_id: str
    name: str
    entity_type: str


class StoredFact(BaseModel):
    """One fact edge persisted in the graph, with endpoint names resolved."""

    fact_id: str
    user_id: str
    source_entity_id: str
    source_entity: str
    relation: str
    destination_entity_id: str
    destination_entity: str
    valid_from: str
    valid_to: str | None
    world_valid_from: str
    world_valid_to: str | None
    add_id: str


class AddLogRecord(BaseModel):
    """Log entry for the source messages of one add call."""

    add_id: str
    user_id: str
    session_id: str
    messages: list[dict[str, str]]
    created_at: str


def format_fact_line(fact: StoredFact) -> str:
    """Format a fact as `source -- relation -- destination`."""
    return f"{fact.source_entity} -- {fact.relation} -- {fact.destination_entity}"
