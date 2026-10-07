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

    def import_conversation_history(self, messages: list[dict[str, str]]) -> None:
        """Replace the active conversation with imported dialogue."""
        self._messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]

    def handle_user_input(self, user_input: str) -> str:
        """Send user input to the model and store the latest messages in memory."""
        self._messages.append({"role": "user", "content": user_input})
        try:
            response = self._open_router_client.complete(self._messages)
        except RuntimeError:
            self._messages.pop()
            raise

        self._messages.append({"role": "assistant", "content": response})
        return response
