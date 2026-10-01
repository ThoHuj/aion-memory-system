from classes.memory_system import MemorySystem
from classes.open_router_client import OpenRouterClient

MESSAGES_FOR_MEMORY_AMOUNT = 5


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

    def import_conversation_history(
        self, messages: list[dict[str, str]]
    ) -> int:
        """Replace the active conversation and replay memory extraction per assistant turn."""
        self._messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        extraction_passes = 0
        for index, message in enumerate(self._messages):
            if message["role"] != "assistant":
                continue
            prefix = self._messages[: index + 1]
            if self._extract_memories_from_prefix(prefix):
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
        self._extract_memories_from_prefix(self._messages)
        return response

    def _extract_memories_from_prefix(
        self, messages: list[dict[str, str]]
    ) -> bool:
        """Run episodic extraction on the last few messages of a conversation prefix."""
        memory_extraction_messages = [
            message
            for message in messages[-MESSAGES_FOR_MEMORY_AMOUNT:]
            if (content := message.get("content")) is not None and content.strip()
        ]
        if not memory_extraction_messages:
            return False
        self._memory_system.add(memory_extraction_messages)
        return True
