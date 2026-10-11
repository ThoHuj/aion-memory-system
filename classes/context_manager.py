from pydantic import BaseModel, Field

from classes.embedding_manager import EmbeddingManager
from classes.facts_store import (
    FactsStore,
    VectorSearchResult,
    order_retrieved_facts_by_recency,
)


class ContextMetadata(BaseModel):
    """Identifiers for the user and session this context belongs to."""

    user_id: str
    session_id: str


class ContextLookup(BaseModel):
    """Retrieved and derived context used for fact extraction."""

    combined_messages: str
    combined_messages_embedding: list[float] = Field(default_factory=list)
    vector_search_result: VectorSearchResult = Field(default_factory=VectorSearchResult)
    recent_session_messages: list[dict[str, str]] = Field(default_factory=list)


class ContextResult(BaseModel):
    """Full context bundle passed into fact extraction."""

    messages: list[dict[str, str]] = Field(default_factory=list)
    metadata: ContextMetadata
    context_lookup: ContextLookup


class ContextManager:
    """Manage context for memory operations."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        facts_store: FactsStore,
        nearest_facts_search_limit: int = 10,
        recent_session_messages_limit: int = 10,
    ) -> None:
        self._embedding_manager = embedding_manager
        self._facts_store = facts_store
        self._nearest_facts_search_limit = nearest_facts_search_limit
        self._recent_session_messages_limit = recent_session_messages_limit

    def _parse_messages_to_string(self, messages: list[dict[str, str]]) -> str:
        """Parse a message list into a single string with lowercase role labels."""
        blocks: list[str] = []
        for message in messages:
            role = message.get("role", "unknown").strip().lower()
            content = message.get("content", "").strip()
            blocks.append(f"{role}: {content}")
        return "\n".join(blocks)

    def _embed_parsed_messages_string(self, parsed_messages: str) -> list[float]:
        """Embed the parsed messages string for context lookup."""
        return self._embedding_manager.embed_string(parsed_messages)

    def produce_context_result(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        session_id: str,
    ) -> ContextResult:
        """Produce a context result from messages and session metadata."""
        combined_messages = self._parse_messages_to_string(messages)
        combined_messages_embedding = self._embed_parsed_messages_string(
            combined_messages
        )
        vector_search_result = order_retrieved_facts_by_recency(
            self._facts_store.search_nearest_facts(
                user_id,
                combined_messages,
                self._nearest_facts_search_limit,
            )
        )
        recent_session_messages = self._facts_store.get_most_recent_session_messages(
            session_id,
            self._recent_session_messages_limit,
        )
        return ContextResult(
            messages=list(messages),
            metadata=ContextMetadata(user_id=user_id, session_id=session_id),
            context_lookup=ContextLookup(
                combined_messages=combined_messages,
                combined_messages_embedding=combined_messages_embedding,
                vector_search_result=vector_search_result,
                recent_session_messages=recent_session_messages,
            ),
        )
