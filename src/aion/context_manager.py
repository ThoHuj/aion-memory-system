from datetime import UTC, datetime

from pydantic import BaseModel, Field

from aion.facts_store import FactsStore


def format_messages(messages: list[dict[str, str]]) -> str:
    """Format messages as role-labeled lines; "(none)" when empty."""
    if not messages:
        return "(none)"
    blocks: list[str] = []
    for message in messages:
        role = message.get("role", "unknown").strip().lower()
        content = message.get("content", "").strip()
        blocks.append(f"{role}: {content}")
    return "\n".join(blocks)


class ContextMetadata(BaseModel):
    """Identifiers and the add time for one add call."""

    user_id: str
    session_id: str
    added_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ContextResult(BaseModel):
    """Context bundle passed into the graph extraction steps."""

    messages: list[dict[str, str]] = Field(default_factory=list)
    metadata: ContextMetadata
    new_messages_text: str
    recent_session_messages_text: str


class ContextManager:
    """Manage context for memory operations."""

    def __init__(
        self,
        facts_store: FactsStore,
        recent_session_messages_limit: int = 10,
    ) -> None:
        self._facts_store = facts_store
        self._recent_session_messages_limit = recent_session_messages_limit

    def produce_context_result(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: str,
    ) -> ContextResult:
        """Produce a context result; reads session messages before this add is logged."""
        recent_session_messages = self._facts_store.get_most_recent_session_messages(
            session_id,
            self._recent_session_messages_limit,
        )
        return ContextResult(
            messages=list(messages),
            metadata=ContextMetadata(user_id=user_id, session_id=session_id),
            new_messages_text=format_messages(messages),
            recent_session_messages_text=format_messages(recent_session_messages),
        )
