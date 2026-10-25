from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from aion_memory.graph_models import (
    AddLogRecord,
    GraphUpdate,
    StoredEntity,
    StoredFact,
)

if TYPE_CHECKING:
    from aion_memory.context_manager import ContextResult


class FactsStore(Protocol):
    """The storage the memory components depend on.

    Every method is scoped to one user and never returns another user's data.
    Implementations are internal: the library ships them, hosts do not write them.
    """

    def find_entity_by_name(self, user_id: str, name: str) -> StoredEntity | None:
        """Return the oldest entity of user_id with the given name, if any."""
        ...

    def find_similar_entities(
        self,
        user_id: str,
        embedding: list[float],
        minimum_similarity: float,
        limit: int,
    ) -> list[StoredEntity]:
        """Return entities by cosine similarity at or above the minimum, best first."""
        ...

    def get_valid_facts_for_entities(
        self,
        user_id: str,
        entity_ids: list[str],
        limit: int,
    ) -> list[StoredFact]:
        """Return system-valid facts touching any entity id, newest first."""
        ...

    def find_valid_fact_id(
        self,
        user_id: str,
        source_entity_id: str,
        relation: str,
        destination_entity_id: str,
    ) -> str | None:
        """Return the id of a valid fact with the same edge, if any."""
        ...

    def get_fact(self, user_id: str, fact_id: str) -> StoredFact:
        """Return a fact of user_id of any validity; raises KeyError if unknown or owned by another user."""
        ...

    def get_add_log_record(self, user_id: str, add_id: str) -> AddLogRecord:
        """Return the add log record for add_id of user_id; raises KeyError if unknown or owned by another user."""
        ...

    def get_most_recent_session_messages(
        self,
        user_id: str,
        session_id: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Return up to limit most recent messages of the session of user_id, oldest first."""
        ...

    def store_graph_update(
        self,
        graph_update: GraphUpdate,
        context_result: ContextResult,
    ) -> None:
        """Persist the add log, new entities, invalidations, and facts as one unit.

        An invalidation only closes a fact of the user the update belongs to.
        """
        ...

    def close(self) -> None:
        """Release the resources the store holds; it cannot be used afterwards."""
        ...
