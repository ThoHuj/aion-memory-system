import os
import sys
import uuid

from dotenv import load_dotenv

from aion_memory import create_memory_system
from aion_memory.open_router_client import OpenRouterClient
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


def load_database_url() -> str:
    """Read DATABASE_URL from the environment, exiting if it is missing."""
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    return database_url


def main() -> None:
    api_key, chat_model, extraction_model, embedding_model = load_openrouter_config()
    database_url = load_database_url()
    memory_system = create_memory_system(
        api_key,
        extraction_model,
        embedding_model,
        database_url,
    )
    open_router_client = OpenRouterClient(api_key=api_key, model=chat_model)
    session_id = str(uuid.uuid4())
    session_logger = SessionLogger(session_id=session_id, user_id="Tjorven")
    mock_agent = MockAgent(
        open_router_client=open_router_client,
        memory_system=memory_system,
        session_id=session_id,
        session_logger=session_logger,
    )
    print(f"Session log: {session_logger.log_path}", file=sys.stderr)

    cli = Cli(mock_agent=mock_agent)
    cli.run()


if __name__ == "__main__":
    main()
