from dev.mock_agent import MockAgent


class Cli:
    """Command-line interface."""

    def __init__(self, mock_agent: MockAgent) -> None:
        self._mock_agent = mock_agent

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
        try:
            response = self._mock_agent.handle_user_input(user_input)
        except RuntimeError as error:
            print(f"Error: {error}")
            return

        print("Assistant:")
        print(response)
        print()
        print("=" * 60)
