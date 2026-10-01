from dev.mock_agent import MockAgent
from dev.slash_commands import SlashCommandError, execute_slash_command


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
            if user_input.strip().startswith("/"):
                self._handle_slash_command(user_input)
                continue
            self._handle_user_input(user_input)

    def _handle_slash_command(self, user_input: str) -> None:
        """Parse and run a slash command without calling the language model."""
        try:
            message = execute_slash_command(user_input, self._mock_agent)
        except SlashCommandError as error:
            print(f"Command error: {error}")
            return
        print(message)

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
