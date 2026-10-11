import uuid

from classes.facts_store import (
    FactsStore,
    VectorSearchResult,
    format_retrieved_fact_line,
    order_retrieved_facts_by_recency,
)
from classes.memory_system import MemorySystem
from classes.open_router_client import OpenRouterClient
from dev.session_logger import SessionLogger

_CHAT_RELEVANT_FACTS_LIMIT = 10


class MockAgent:
    """Mock external agent that uses the memory system."""

    def __init__(
        self,
        open_router_client: OpenRouterClient,
        memory_system: MemorySystem,
        facts_store: FactsStore,
        user_id: str = "Tjorven",
        session_id: str | None = None,
        session_logger: SessionLogger | None = None,
    ) -> None:
        self._open_router_client = open_router_client
        self._memory_system = memory_system
        self._facts_store = facts_store
        self._user_id = user_id
        self._session_id = session_id or str(uuid.uuid4())
        self._session_logger = session_logger
        self._messages: list[dict[str, str]] = []

    def import_conversation_history(self, messages: list[dict[str, str]]) -> None:
        """Replace the active conversation and run memory add for each assistant turn."""
        self._messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        if self._session_logger is not None:
            self._session_logger.log_conversation(self._messages)
        for index, message in enumerate(self._messages):
            if message["role"] != "assistant":
                continue
            turn_messages = self._turn_messages(index)
            if turn_messages:
                self._add_turn_to_memory(turn_messages)

    def handle_user_input(self, user_input: str) -> str:
        """Send user input to the model and store the latest turn in memory."""
        vector_search_result = order_retrieved_facts_by_recency(
            self._facts_store.search_nearest_facts(
                self._user_id,
                user_input,
                _CHAT_RELEVANT_FACTS_LIMIT,
            )
        )
        if self._session_logger is not None:
            self._session_logger.log_retrieved_facts_for_chat(
                user_input,
                [
                    format_retrieved_fact_line(retrieved_fact)
                    for retrieved_fact in vector_search_result.facts
                ],
            )
        relevant_facts_message = self._build_relevant_user_facts_message(
            vector_search_result
        )
        messages_for_completion = [
            *self._messages,
            relevant_facts_message,
            {"role": "user", "content": user_input},
        ]
        try:
            response = self._open_router_client.complete(messages_for_completion)
        except RuntimeError:
            raise

        if self._session_logger is not None:
            self._session_logger.log_chat_completion(messages_for_completion, response)

        self._messages.append({"role": "user", "content": user_input})
        self._messages.append({"role": "assistant", "content": response})
        if self._session_logger is not None:
            self._session_logger.log_conversation(self._messages)
        self._add_turn_to_memory(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response},
            ]
        )
        return response

    @staticmethod
    def _build_relevant_user_facts_message(
        vector_search_result: VectorSearchResult,
    ) -> dict[str, str]:
        """Build a separate chat message with semantically retrieved user facts."""
        if vector_search_result.facts:
            fact_lines = "\n".join(
                f"- {format_retrieved_fact_line(retrieved_fact)}"
                for retrieved_fact in vector_search_result.facts
            )
        else:
            fact_lines = "(none)"
        return {
            "role": "user",
            "content": (f"<relevant_user_facts>\n{fact_lines}\n</relevant_user_facts>"),
        }

    def _add_turn_to_memory(self, turn_messages: list[dict[str, str]]) -> None:
        """Run the memory add pipeline for one user/assistant turn."""
        add_result = self._memory_system.add(
            turn_messages,
            user_id=self._user_id,
            session_id=self._session_id,
        )
        if self._session_logger is not None:
            self._session_logger.log_memory_add(
                turn_messages,
                [stored_fact.model_dump() for stored_fact in add_result.stored_facts],
                add_result.extraction_context.model_dump(),
            )

    def _turn_messages(self, assistant_index: int) -> list[dict[str, str]]:
        """Return the user and assistant messages for the turn ending at assistant_index."""
        if assistant_index < 0 or assistant_index >= len(self._messages):
            return []

        assistant_message = self._messages[assistant_index]
        if assistant_message.get("role") != "assistant":
            return []

        assistant_content = assistant_message.get("content")
        if assistant_content is None or not assistant_content.strip():
            return []

        for index in range(assistant_index - 1, -1, -1):
            message = self._messages[index]
            if message.get("role") != "user":
                continue
            user_content = message.get("content")
            if user_content is None or not user_content.strip():
                return []
            return [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]

        return []
