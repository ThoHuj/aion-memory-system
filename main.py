import os
import sys

from dotenv import load_dotenv

from classes.facts_extractor import FactsExtractor
from classes.facts_store import FactsStore
from classes.memory_system import MemorySystem
from classes.open_router_client import OpenRouterClient
from dev.cli import Cli
from dev.mock_agent import MockAgent


def load_openrouter_config():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL")
    extraction_model = os.environ.get("OPENROUTER_EXTRACTION_MODEL")
    if not api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("OPENROUTER_MODEL is not set.", file=sys.stderr)
        sys.exit(1)
    if not extraction_model:
        print("OPENROUTER_EXTRACTION_MODEL is not set.", file=sys.stderr)
        sys.exit(1)
    return api_key, model, extraction_model


def main() -> None:
    api_key, model, extraction_model = load_openrouter_config()
    open_router_client = OpenRouterClient(api_key=api_key, model=model)
    extraction_client = OpenRouterClient(api_key=api_key, model=extraction_model)
    facts_store = FactsStore()
    facts_extractor = FactsExtractor(extraction_client=extraction_client)
    memory_system = MemorySystem(
        facts_store=facts_store,
        facts_extractor=facts_extractor,
    )
    mock_agent = MockAgent(
        open_router_client=open_router_client,
        memory_system=memory_system,
    )

    cli = Cli(mock_agent=mock_agent)
    cli.run()


if __name__ == "__main__":
    main()
