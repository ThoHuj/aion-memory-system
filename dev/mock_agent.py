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

    def handle_user_input(self, user_input: str) -> str:
        """Send user input to the model and store the latest messages in memory."""
        self._messages.append({"role": "user", "content": user_input})
        try:
            response = self._open_router_client.complete(self._messages)
        except RuntimeError:
            self._messages.pop()
            raise

        self._messages.append({"role": "assistant", "content": response})

        # Trim empty messages before memory extraction
        memory_extraction_messages = [
            message
            for message in self._messages[-MESSAGES_FOR_MEMORY_AMOUNT:]
            if (content := message.get("content")) is not None and content.strip()
        ]
        # Extract memories
        self._memory_system.add(memory_extraction_messages)
        return response
