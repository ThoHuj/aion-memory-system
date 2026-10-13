import json
import uuid
from collections.abc import Callable
from functools import partial

from classes.memory_system import FactData, MemorySystem
from classes.open_router_client import OpenRouterClient
from dev.agent_tools import GET_FACT_SOURCES_DEFINITION, get_fact_sources
from dev.session_logger import SessionLogger

_CHAT_RELEVANT_FACTS_LIMIT = 10
_MAX_TOOL_ROUNDS = 5


class MockAgent:
    """Mock external agent that uses the memory system."""

    def __init__(
        self,
        open_router_client: OpenRouterClient,
        memory_system: MemorySystem,
        user_id: str = "Tjorven",
        session_id: str | None = None,
        session_logger: SessionLogger | None = None,
    ) -> None:
        self._open_router_client = open_router_client
        self._memory_system = memory_system
        self._user_id = user_id
        self._session_id = session_id or str(uuid.uuid4())
        self._session_logger = session_logger
        self._messages: list[dict[str, object]] = []
        self._tool_definitions: list[dict[str, object]] = [GET_FACT_SOURCES_DEFINITION]
        self._tool_functions: dict[str, Callable[..., str]] = {
            "get_fact_sources": partial(get_fact_sources, memory_system),
        }

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
        retrieved_facts = sorted(
            self._memory_system.retrieve(
                user_input,
                self._user_id,
                _CHAT_RELEVANT_FACTS_LIMIT,
            ).retrieved_facts,
            key=lambda retrieved_fact: retrieved_fact.stored_at,
            reverse=True,
        )
        if self._session_logger is not None:
            self._session_logger.log_retrieved_facts_for_chat(
                user_input,
                [
                    self._format_fact_line(retrieved_fact)
                    for retrieved_fact in retrieved_facts
                ],
            )
        relevant_facts_message = self._build_relevant_user_facts_message(
            retrieved_facts
        )
        messages_for_completion = [
            *self._messages,
            relevant_facts_message,
            {"role": "user", "content": user_input},
        ]
        response, tool_exchange_messages = self._complete_with_tools(
            messages_for_completion
        )

        if self._session_logger is not None:
            self._session_logger.log_chat_completion(messages_for_completion, response)

        self._messages.append({"role": "user", "content": user_input})
        self._messages.extend(tool_exchange_messages)
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

    def _complete_with_tools(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, object]]]:
        """Run the model, executing requested tool calls until it returns text.

        Returns the final text and the assistant tool-call and tool result messages
        produced along the way, in order.
        """
        working_messages: list[dict[str, object]] = [*messages]
        tool_exchange_messages: list[dict[str, object]] = []
        for _ in range(_MAX_TOOL_ROUNDS):
            assistant_message = self._open_router_client.complete_with_tools(
                working_messages,
                self._tool_definitions,
            )
            tool_calls = assistant_message.get("tool_calls")
            if not tool_calls:
                return assistant_message.get("content") or "", tool_exchange_messages
            new_messages: list[dict[str, object]] = [assistant_message]
            for tool_call in tool_calls:
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": self._run_tool_call(tool_call),
                    }
                )
            working_messages.extend(new_messages)
            tool_exchange_messages.extend(new_messages)
        raise RuntimeError(f"Tool loop exceeded {_MAX_TOOL_ROUNDS} rounds")

    def _run_tool_call(self, tool_call: dict[str, object]) -> str:
        """Execute one tool call and return its result (or an error) as text."""
        function = tool_call["function"]
        tool_function = self._tool_functions.get(function["name"])
        if tool_function is None:
            return json.dumps({"error": f"unknown tool: {function['name']}"})
        try:
            arguments = json.loads(function["arguments"] or "{}")
            return tool_function(**arguments)
        except (json.JSONDecodeError, TypeError) as error:
            return json.dumps({"error": f"invalid tool arguments: {error}"})

    @staticmethod
    def _format_fact_line(retrieved_fact: FactData) -> str:
        """Format one retrieved fact with its storage timestamp for LLM context."""
        if retrieved_fact.stored_at:
            return f"[{retrieved_fact.stored_at}] {retrieved_fact.content}"
        return retrieved_fact.content

    @staticmethod
    def _build_relevant_user_facts_message(
        retrieved_facts: list[FactData],
    ) -> dict[str, str]:
        """Build a separate chat message with retrieved user facts."""
        if retrieved_facts:
            fact_lines = "\n".join(
                f"- [id: {retrieved_fact.fact_id}] "
                f"{MockAgent._format_fact_line(retrieved_fact)}"
                for retrieved_fact in retrieved_facts
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
