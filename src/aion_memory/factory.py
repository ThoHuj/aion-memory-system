from aion_memory.conflict_resolution.conflict_resolver import ConflictResolver
from aion_memory.context_manager import ContextManager
from aion_memory.embedding_manager import EmbeddingManager
from aion_memory.entity_extraction import EntityExtractor
from aion_memory.entity_resolution.entity_resolver import EntityResolver
from aion_memory.facts_store import PostgresFactsStore
from aion_memory.graph_retriever import GraphRetriever
from aion_memory.memory_system import MemorySystem
from aion_memory.open_router_client import OpenRouterClient
from aion_memory.relation_extraction.relation_extractor import RelationExtractor


def create_memory_system(
    api_key: str,
    extraction_model: str,
    embedding_model: str,
    database_url: str,
    create_schema: bool = True,
) -> MemorySystem:
    """Wire OpenRouter clients, Postgres, and memory components into a ready-to-use MemorySystem.

    The database must have pgvector. By default the pgvector extension and the tables are
    created if missing. With create_schema=False the database is left untouched, so the
    host sets it up itself with the SQL from schema_sql().
    """
    extraction_client = OpenRouterClient(api_key=api_key, model=extraction_model)
    embedding_client = OpenRouterClient(api_key=api_key, model=embedding_model)
    embedding_manager = EmbeddingManager(embedding_client=embedding_client)
    facts_store = PostgresFactsStore.connect(database_url)
    try:
        if create_schema:
            facts_store.create_schema()
        else:
            facts_store.require_pgvector()
    except BaseException:
        facts_store.close()
        raise
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
