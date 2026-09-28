class Cli:
    """Command-line interface."""

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
        """Handle user input; LLM call and response printing will be added here."""
