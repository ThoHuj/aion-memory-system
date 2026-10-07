from pydantic import BaseModel, Field


class ContextMetadata(BaseModel):
    """Identifiers for the user and session this context belongs to."""

    user_id: str
    session_id: str


class ContextLookup(BaseModel):
    """Retrieved and derived context used for fact extraction."""

    combined_messages: str
    combined_messages_embedding: list[float] = Field(default_factory=list)
    related_facts: list[str] = Field(default_factory=list)
    recent_session_messages: list[dict[str, str]] = Field(default_factory=list)


class ContextResult(BaseModel):
    """Full context bundle passed into fact extraction."""

    messages: list[dict[str, str]] = Field(default_factory=list)
    metadata: ContextMetadata
    context_lookup: ContextLookup


class ContextManager:
    """Manage context for memory operations."""

    def _parse_messages_to_string(self, messages: list[dict[str, str]]) -> str:
        """Parse a message list into a single string with lowercase role labels."""
        blocks: list[str] = []
        for message in messages:
            role = message.get("role", "unknown").strip().lower()
            content = message.get("content", "").strip()
            blocks.append(f"{role}: {content}")
        return "\n".join(blocks)

    def produce_context_result(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: str,
    ) -> ContextResult:
        """Produce a context result from messages and session metadata."""
        combined_messages = self._parse_messages_to_string(messages)
        return ContextResult(
            messages=list(messages),
            metadata=ContextMetadata(user_id=user_id, session_id=session_id),
            context_lookup=ContextLookup(combined_messages=combined_messages),
        )
