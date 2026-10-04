from classes.memory_system import MemorySystem
from classes.open_router_client import OpenRouterClient


class MockAgent:
    """Mock external agent that uses the memory system."""

    def __init__(
        self,
        open_router_client: OpenRouterClient,
        memory_system: MemorySystem,
    ) -> None:
        self._open_router_client = open_router_client
        self._memory_system = memory_system
        self._messages: list[dict[str, str]] = []

    def import_conversation_history(self, messages: list[dict[str, str]]) -> int:
        """Replace the active conversation and replay memory extraction per assistant turn."""
        self._messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        extraction_passes = 0
        for index, message in enumerate(self._messages):
            if message["role"] != "assistant":
                continue
            if self._extract_memories(index):
                extraction_passes += 1
        return extraction_passes

    def handle_user_input(self, user_input: str) -> str:
        """Send user input to the model and store the latest messages in memory."""
        self._messages.append({"role": "user", "content": user_input})
        try:
            response = self._open_router_client.complete(self._messages)
        except RuntimeError:
            self._messages.pop()
            raise

        self._messages.append({"role": "assistant", "content": response})
        self._extract_memories(len(self._messages) - 1)
        return response

    def _extract_memories(self, assistant_index: int) -> bool:
        """Run memory extraction on the user message and assistant reply for one turn."""
        turn_messages = self._turn_messages(assistant_index)
        if not turn_messages:
            return False
        self._memory_system.add(turn_messages)
        return True

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
