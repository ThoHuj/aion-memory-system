from classes.open_router_client import OpenRouterClient


class Cli:
    """Command-line interface."""

    def __init__(self, open_router_client: OpenRouterClient) -> None:
        self._open_router_client = open_router_client
        self._messages: list[dict[str, str]] = []

    def run(self) -> None:
        """Run the interactive command-line loop."""
        while True:
            try:
                user_input = input("User: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input == "q":
                print("Quit")
                break
            self._handle_user_input(user_input)

    def _handle_user_input(self, user_input: str) -> None:
        """Send user input to the language model and print the response."""
        print()
        print("=" * 60)
        self._messages.append({"role": "user", "content": user_input})
        try:
            response = self._open_router_client.complete(self._messages)
        except RuntimeError as error:
            self._messages.pop()
            print(f"Error: {error}")
            return

        self._messages.append({"role": "assistant", "content": response})
        print("Assistant:")
        print(response)
        print()
        print("=" * 60)
