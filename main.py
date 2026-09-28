import os
import sys

from dotenv import load_dotenv

from classes.cli import Cli
from classes.open_router_client import OpenRouterClient


def load_openrouter_config():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL")
    if not api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("OPENROUTER_MODEL is not set.", file=sys.stderr)
        sys.exit(1)
    return api_key, model


def main() -> None:
    api_key, model = load_openrouter_config()
    open_router_client = OpenRouterClient(api_key=api_key, model=model)

    cli = Cli(open_router_client=open_router_client)
    cli.run()


if __name__ == "__main__":
    main()
