import os
import sys
import uuid

from dotenv import load_dotenv

from classes.conflict_resolver import ConflictResolver
from classes.context_manager import ContextManager
from classes.embedding_manager import EmbeddingManager
from classes.entity_extraction import EntityExtractor
from classes.facts_extraction import FactsExtractor
from classes.facts_store import FactsStore
from classes.memory_system import MemorySystem
from classes.open_router_client import OpenRouterClient
from dev.cli import Cli
from dev.mock_agent import MockAgent
from dev.session_logger import SessionLogger


def load_openrouter_config():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    chat_model = os.environ.get("OPENROUTER_CHAT_MODEL")
    extraction_model = os.environ.get("OPENROUTER_EXTRACTION_MODEL")
    embedding_model = os.environ.get("OPENROUTER_EMBEDDING_MODEL")
    if not api_key:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    if not chat_model:
        print("OPENROUTER_CHAT_MODEL is not set.", file=sys.stderr)
        sys.exit(1)
    if not extraction_model:
        print("OPENROUTER_EXTRACTION_MODEL is not set.", file=sys.stderr)
        sys.exit(1)
    if not embedding_model:
        print("OPENROUTER_EMBEDDING_MODEL is not set.", file=sys.stderr)
        sys.exit(1)
    return api_key, chat_model, extraction_model, embedding_model


def create_memory_system(
    api_key: str,
    extraction_model: str,
    embedding_model: str,
) -> tuple[MemorySystem, FactsStore]:
    """Wire OpenRouter clients and memory components into a ready-to-use MemorySystem."""
    extraction_client = OpenRouterClient(api_key=api_key, model=extraction_model)
    embedding_client = OpenRouterClient(api_key=api_key, model=embedding_model)
    embedding_manager = EmbeddingManager(embedding_client=embedding_client)
    facts_store = FactsStore(embedding_manager=embedding_manager)
    facts_extractor = FactsExtractor(extraction_client=extraction_client)
    entity_extractor = EntityExtractor(extraction_client=extraction_client)
    context_manager = ContextManager(
        embedding_manager=embedding_manager,
        facts_store=facts_store,
    )
    conflict_resolver = ConflictResolver()
    memory_system = MemorySystem(
        facts_store=facts_store,
        facts_extractor=facts_extractor,
        context_manager=context_manager,
        embedding_manager=embedding_manager,
        conflict_resolver=conflict_resolver,
        entity_extractor=entity_extractor,
    )
    return memory_system, facts_store


def main() -> None:
    api_key, chat_model, extraction_model, embedding_model = load_openrouter_config()
    memory_system, facts_store = create_memory_system(
        api_key,
        extraction_model,
        embedding_model,
    )
    open_router_client = OpenRouterClient(api_key=api_key, model=chat_model)
    session_id = str(uuid.uuid4())
    session_logger = SessionLogger(session_id=session_id, user_id="Tjorven")
    mock_agent = MockAgent(
        open_router_client=open_router_client,
        memory_system=memory_system,
        facts_store=facts_store,
        session_id=session_id,
        session_logger=session_logger,
    )
    print(f"Session log: {session_logger.log_path}", file=sys.stderr)

    cli = Cli(mock_agent=mock_agent)
    cli.run()


if __name__ == "__main__":
    main()
