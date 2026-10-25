"""An in-memory FactsStore for tests, behaving like the Postgres store."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from aion_memory.graph_models import (
    AddLogRecord,
    GraphUpdate,
    StoredEntity,
    StoredFact,
)

if TYPE_CHECKING:
    from aion_memory.context_manager import ContextResult


@dataclass
class _EntityRow:
    user_id: str
    entity: StoredEntity
    embedding: list[float]
    created_at: datetime


@dataclass
class _FactRow:
    fact_id: str
    user_id: str
    source_entity_id: str
    relation: str
    destination_entity_id: str
    valid_from: datetime
    valid_to: datetime | None
    world_valid_from: datetime
    world_valid_to: datetime | None
    add_id: str


@dataclass
class _AddLogRow:
    add_id: str
    user_id: str
    session_id: str
    messages: list[dict[str, str]]
    created_at: datetime


def _cosine_similarity(first: list[float], second: list[float]) -> float | None:
    """Return the cosine similarity, None for a zero vector; raises on a size mismatch."""
    if len(first) != len(second):
        raise ValueError(
            f"Vectors have different dimensions ({len(first)} and {len(second)})."
        )
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return None
    dot_product = sum(a * b for a, b in zip(first, second, strict=True))
    return dot_product / (first_norm * second_norm)


def _optional_datetime(value: str | None) -> datetime | None:
    """Convert an optional ISO string to a datetime."""
    return datetime.fromisoformat(value) if value else None


class InMemoryFactsStore:
    """Keeps the knowledge graph and add log in dicts, for fast tests without a database."""

    def __init__(self) -> None:
        self._entities: dict[str, _EntityRow] = {}
        self._facts: dict[str, _FactRow] = {}
        self._add_log: dict[str, _AddLogRow] = {}

    def find_entity_by_name(self, user_id: str, name: str) -> StoredEntity | None:
        """Return the oldest entity of user_id with the given name, if any."""
        rows = [
            row
            for row in self._entities.values()
            if row.user_id == user_id and row.entity.name == name
        ]
        if not rows:
            return None
        return min(rows, key=lambda row: row.created_at).entity

    def find_similar_entities(
        self,
        user_id: str,
        embedding: list[float],
        minimum_similarity: float,
        limit: int,
    ) -> list[StoredEntity]:
        """Return entities by cosine similarity at or above the minimum, best first."""
        scored: list[tuple[float, StoredEntity]] = []
        for row in self._entities.values():
            if row.user_id != user_id:
                continue
            similarity = _cosine_similarity(embedding, row.embedding)
            if similarity is not None and similarity >= minimum_similarity:
                scored.append((similarity, row.entity))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entity for _, entity in scored[:limit]]

    def get_valid_facts_for_entities(
        self,
        user_id: str,
        entity_ids: list[str],
        limit: int,
    ) -> list[StoredFact]:
        """Return system-valid facts touching any entity id, newest first."""
        wanted_ids = set(entity_ids)
        rows = [
            row
            for row in self._facts.values()
            if row.user_id == user_id
            and row.valid_to is None
            and (
                row.source_entity_id in wanted_ids
                or row.destination_entity_id in wanted_ids
            )
        ]
        rows.sort(key=lambda row: row.valid_from, reverse=True)
        return [self._to_stored_fact(row) for row in rows[:limit]]

    def find_valid_fact_id(
        self,
        user_id: str,
        source_entity_id: str,
        relation: str,
        destination_entity_id: str,
    ) -> str | None:
        """Return the id of a valid fact with the same edge, if any."""
        rows = [
            row
            for row in self._facts.values()
            if row.user_id == user_id
            and row.source_entity_id == source_entity_id
            and row.relation == relation
            and row.destination_entity_id == destination_entity_id
            and row.valid_to is None
        ]
        if not rows:
            return None
        return max(rows, key=lambda row: row.valid_from).fact_id

    def get_fact(self, user_id: str, fact_id: str) -> StoredFact:
        """Return a fact of user_id of any validity; raises KeyError if unknown or owned by another user."""
        row = self._facts.get(fact_id)
        if row is None or row.user_id != user_id:
            raise KeyError(fact_id)
        return self._to_stored_fact(row)

    def get_add_log_record(self, user_id: str, add_id: str) -> AddLogRecord:
        """Return the add log record for add_id of user_id; raises KeyError if unknown or owned by another user."""
        row = self._add_log.get(add_id)
        if row is None or row.user_id != user_id:
            raise KeyError(add_id)
        return AddLogRecord(
            add_id=row.add_id,
            user_id=row.user_id,
            session_id=row.session_id,
            messages=[dict(message) for message in row.messages],
            created_at=row.created_at.isoformat(),
        )

    def get_most_recent_session_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Return up to limit most recent messages of the session of user_id, oldest first."""
        if limit <= 0:
            return []
        rows = sorted(
            (
                row
                for row in self._add_log.values()
                if row.user_id == user_id and row.session_id == session_id
            ),
            key=lambda row: row.created_at,
            reverse=True,
        )[:limit]
        messages: list[dict[str, str]] = []
        for row in reversed(rows):
            messages.extend(dict(message) for message in row.messages)
        return messages[-limit:]

    def store_graph_update(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
    ) -> None:
        """Persist the add log, new entities, invalidations, and facts as one unit."""
        user_id = context_result.metadata.user_id
        added_at = datetime.fromisoformat(context_result.metadata.added_at)
        add_id = str(uuid.uuid4())
        entity_ids_by_name = graph_update.entity_ids_by_name()
        fact_endpoint_names = {
            name
            for fact in graph_update.facts
            for name in (fact.source_entity, fact.destination_entity)
        }

        new_entity_rows: dict[str, _EntityRow] = {}
        for entity in graph_update.entities:
            if not entity.is_new or entity.name not in fact_endpoint_names:
                continue
            if not entity.embedding:
                raise ValueError(f"Entity {entity.name!r} has no embedding.")
            new_entity_rows[entity.entity_id] = _EntityRow(
                user_id=user_id,
                entity=StoredEntity(
                    entity_id=entity.entity_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                ),
                embedding=list(entity.embedding),
                created_at=added_at,
            )

        new_fact_rows: list[_FactRow] = []
        for fact in graph_update.facts:
            source_entity_id = entity_ids_by_name[fact.source_entity]
            destination_entity_id = entity_ids_by_name[fact.destination_entity]
            for entity_id in (source_entity_id, destination_entity_id):
                if entity_id not in self._entities and entity_id not in new_entity_rows:
                    raise ValueError(f"Fact refers to unknown entity {entity_id!r}.")
            fact.fact_id = str(uuid.uuid4())
            new_fact_rows.append(
                _FactRow(
                    fact_id=fact.fact_id,
                    user_id=user_id,
                    source_entity_id=source_entity_id,
                    relation=fact.relation,
                    destination_entity_id=destination_entity_id,
                    valid_from=added_at,
                    valid_to=None,
                    world_valid_from=datetime.fromisoformat(fact.world_valid_from),
                    world_valid_to=_optional_datetime(fact.world_valid_to),
                    add_id=add_id,
                )
            )

        self._add_log[add_id] = _AddLogRow(
            add_id=add_id,
            user_id=user_id,
            session_id=context_result.metadata.session_id,
            messages=[dict(message) for message in context_result.messages],
            created_at=added_at,
        )
        self._entities.update(new_entity_rows)
        for invalidation in graph_update.invalidations:
            row = self._facts.get(invalidation.fact_id)
            if row is not None and row.valid_to is None:
                row.valid_to = added_at
                row.world_valid_to = row.world_valid_to or _optional_datetime(
                    invalidation.world_valid_to
                )
        for new_fact_row in new_fact_rows:
            self._facts[new_fact_row.fact_id] = new_fact_row

    def close(self) -> None:
        """Release the resources the store holds; an in-memory store holds none."""

    def _to_stored_fact(self, row: _FactRow) -> StoredFact:
        """Convert a fact row to a StoredFact, with the endpoint names resolved."""
        return StoredFact(
            fact_id=row.fact_id,
            user_id=row.user_id,
            source_entity_id=row.source_entity_id,
            source_entity=self._entities[row.source_entity_id].entity.name,
            relation=row.relation,
            destination_entity_id=row.destination_entity_id,
            destination_entity=self._entities[row.destination_entity_id].entity.name,
            valid_from=row.valid_from.isoformat(),
            valid_to=row.valid_to.isoformat() if row.valid_to else None,
            world_valid_from=row.world_valid_from.isoformat(),
            world_valid_to=(
                row.world_valid_to.isoformat() if row.world_valid_to else None
            ),
            add_id=row.add_id,
        )
