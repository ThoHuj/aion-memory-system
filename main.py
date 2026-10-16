import os
import sys
import uuid

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from classes.conflict_resolution.conflict_resolver import ConflictResolver
from classes.context_manager import ContextManager
from classes.embedding_manager import EmbeddingManager
from classes.entity_extraction import EntityExtractor
from classes.entity_resolution.entity_resolver import EntityResolver
from classes.facts_store import FactsStore
from classes.graph_retriever import GraphRetriever
from classes.memory_system import MemorySystem
from classes.open_router_client import OpenRouterClient
from classes.relation_extraction.relation_extractor import RelationExtractor
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


def create_memory_system(
    api_key: str,
    extraction_model: str,
    embedding_model: str,
    database_url: str,
) -> MemorySystem:
    """Wire OpenRouter clients, Postgres, and memory components into a ready-to-use MemorySystem."""
    extraction_client = OpenRouterClient(api_key=api_key, model=extraction_model)
    embedding_client = OpenRouterClient(api_key=api_key, model=embedding_model)
    embedding_manager = EmbeddingManager(embedding_client=embedding_client)
    connection = psycopg.connect(database_url, autocommit=True, row_factory=dict_row)
    facts_store = FactsStore(connection)
    facts_store.create_schema()
    entity_extractor = EntityExtractor(extraction_client=extraction_client)
    relation_extractor = RelationExtractor(extraction_client=extraction_client)
    context_manager = ContextManager(facts_store=facts_store)
    entity_resolver = EntityResolver(
        extraction_client=extraction_client,
        facts_store=facts_store,
    )
    conflict_resolver = ConflictResolver(
        extraction_client=extraction_client,
        facts_store=facts_store,
    )
    graph_retriever = GraphRetriever(
        entity_extractor=entity_extractor,
        embedding_manager=embedding_manager,
        facts_store=facts_store,
    )
    return MemorySystem(
        facts_store=facts_store,
        context_manager=context_manager,
        entity_extractor=entity_extractor,
        relation_extractor=relation_extractor,
        embedding_manager=embedding_manager,
        entity_resolver=entity_resolver,
        conflict_resolver=conflict_resolver,
        graph_retriever=graph_retriever,
    )


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
