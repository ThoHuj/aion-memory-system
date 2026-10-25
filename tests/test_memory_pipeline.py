"""The add and retrieve pipeline end to end, with scripted models instead of OpenRouter."""

import re
from typing import Any

from pydantic import BaseModel

from aion_memory.conflict_resolution.conflict_resolver import ConflictResolver
from aion_memory.context_manager import ContextManager
from aion_memory.embedding_manager import EmbeddingManager
from aion_memory.entity_extraction import EntityExtractor
from aion_memory.entity_resolution.entity_resolver import EntityResolver
from aion_memory.facts_store import FactsStore
from aion_memory.graph_retriever import GraphRetriever
from aion_memory.memory_system import MemorySystem
from aion_memory.relation_extraction.relation_extractor import RelationExtractor

UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
DOG_MESSAGES = [{"role": "user", "content": "I have a dog named Rex"}]


class ScriptedExtractionClient:
    """Stands in for the extraction model: answers each structured request from a script."""

    def __init__(self) -> None:
        self.entities: list[tuple[str, str]] = []
        self.relations: list[tuple[str, str, str]] = []
        self.invalidate_existing_facts = False
        self.requested_models: list[str] = []

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
    ) -> Any:
        """Return the scripted answer for the requested response model."""
        model_name = response_model.__name__
        self.requested_models.append(model_name)
        user_prompt = messages[-1]["content"]
        match model_name:
            case "_EntityExtractionPayload":
                data: dict[str, Any] = {
                    "entities": [
                        {"entity": name, "entity_type": entity_type}
                        for name, entity_type in self.entities
                    ]
                }
            case "_RelationExtractionPayload":
                data = {
                    "relations": [
                        {
                            "source": source,
                            "relationship": relation,
                            "destination": destination,
                            "world_valid_from": None,
                            "world_valid_to": None,
                        }
                        for source, relation, destination in self.relations
                    ]
                }
            case "_EntityResolutionPayload":
                data = {"matches": self._match_same_named_candidates(user_prompt)}
            case "_ConflictResolutionPayload":
                data = {"invalidations": self._invalidations(user_prompt)}
            case _:
                raise AssertionError(f"Unexpected response model {model_name}")
        return response_model.model_validate(data)

    def _match_same_named_candidates(self, prompt: str) -> list[dict[str, str | None]]:
        """Pick, for each new entity, the candidate with the same name."""
        matches: list[dict[str, str | None]] = []
        for block in prompt.split("new entity: ")[1:]:
            name = block.split(" (", 1)[0]
            candidate = re.search(
                rf"- id ({UUID_PATTERN}): {re.escape(name)} \(", block
            )
            matches.append(
                {
                    "entity_name": name,
                    "existing_entity_id": candidate.group(1) if candidate else None,
                }
            )
        return matches

    def _invalidations(self, prompt: str) -> list[dict[str, str | None]]:
        """Invalidate every listed existing fact when the script asks for it."""
        if not self.invalidate_existing_facts:
            return []
        fact_ids = re.findall(rf"({UUID_PATTERN}): ", prompt)
        return [{"fact_id": fact_id, "world_valid_to": None} for fact_id in fact_ids]


class DistinctEmbeddingClient:
    """Stands in for the embedding model: each distinct text gets its own direction."""

    def __init__(self, dimensions: int = 16) -> None:
        self._dimensions = dimensions
        self._vectors: dict[str, list[float]] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text; equal texts share one, different texts are orthogonal."""
        vectors: list[list[float]] = []
        for text in texts:
            if text not in self._vectors:
                position = len(self._vectors)
                assert position < self._dimensions, "Too many distinct texts"
                vector = [0.0] * self._dimensions
                vector[position] = 1.0
                self._vectors[text] = vector
            vectors.append(list(self._vectors[text]))
        return vectors


def build_memory_system(
    store: FactsStore,
    extraction_client: ScriptedExtractionClient,
) -> MemorySystem:
    """Wire the real components around a store and the scripted models."""
    embedding_manager = EmbeddingManager(embedding_client=DistinctEmbeddingClient())  # type: ignore[arg-type]
    entity_extractor = EntityExtractor(extraction_client=extraction_client)  # type: ignore[arg-type]
    return MemorySystem(
        facts_store=store,
        context_manager=ContextManager(facts_store=store),
        entity_extractor=entity_extractor,
        relation_extractor=RelationExtractor(extraction_client=extraction_client),  # type: ignore[arg-type]
        embedding_manager=embedding_manager,
        entity_resolver=EntityResolver(
            extraction_client=extraction_client,  # type: ignore[arg-type]
            facts_store=store,
        ),
        conflict_resolver=ConflictResolver(
            extraction_client=extraction_client,  # type: ignore[arg-type]
            facts_store=store,
        ),
        graph_retriever=GraphRetriever(
            entity_extractor=entity_extractor,
            embedding_manager=embedding_manager,
            facts_store=store,
        ),
    )


def retrieved_edges(
    memory_system: MemorySystem, user_id: str = "alice"
) -> list[tuple[str, str, str]]:
    """Retrieve for the user and return the facts as (source, relation, destination)."""
    result = memory_system.retrieve("what do you know about my dog?", user_id, 10)
    return [
        (data.fact.source_entity, data.fact.relation, data.fact.destination_entity)
        for data in result.retrieved_facts
    ]


def scripted_client() -> ScriptedExtractionClient:
    """A client that extracts alice and rex with the fact alice owns rex."""
    client = ScriptedExtractionClient()
    client.entities = [("alice", "person"), ("rex", "dog")]
    client.relations = [("alice", "owns", "rex")]
    return client


def test_added_fact_is_retrieved_with_its_source_for_that_user_only(
    facts_store: FactsStore,
) -> None:
    """A fact added for one user is retrieved with its origin, and not for another user."""
    memory_system = build_memory_system(facts_store, scripted_client())

    added = memory_system.add(DOG_MESSAGES, "alice", "chat-1")

    assert [fact.relation for fact in added.stored_facts] == ["owns"]
    result = memory_system.retrieve("what do you know about my dog?", "alice", 10)
    assert [data.fact.fact_id for data in result.retrieved_facts] == [
        added.stored_facts[0].fact_id
    ]
    assert result.retrieved_facts[0].origin.messages == DOG_MESSAGES
    assert retrieved_edges(memory_system, "bob") == []


def test_repeating_a_statement_does_not_store_the_fact_twice(
    facts_store: FactsStore,
) -> None:
    """The second add resolves rex to the stored entity and drops the duplicate fact."""
    client = scripted_client()
    memory_system = build_memory_system(facts_store, client)
    memory_system.add(DOG_MESSAGES, "alice", "chat-1")

    repeated = memory_system.add(DOG_MESSAGES, "alice", "chat-1")

    assert repeated.stored_facts == []
    assert "_EntityResolutionPayload" in client.requested_models
    assert retrieved_edges(memory_system) == [("alice", "owns", "rex")]


def test_a_contradicting_fact_invalidates_the_old_one(facts_store: FactsStore) -> None:
    """The new fact replaces the old one in retrieval, and the old one stays fetchable."""
    client = scripted_client()
    memory_system = build_memory_system(facts_store, client)
    first = memory_system.add(DOG_MESSAGES, "alice", "chat-1")
    old_fact_id = first.stored_facts[0].fact_id
    client.relations = [("alice", "gave_away", "rex")]
    client.invalidate_existing_facts = True

    second = memory_system.add(
        [{"role": "user", "content": "I gave Rex away"}], "alice", "chat-1"
    )

    assert [invalidation.fact_id for invalidation in second.invalidations] == [
        old_fact_id
    ]
    assert retrieved_edges(memory_system) == [("alice", "gave_away", "rex")]
    assert memory_system.get(old_fact_id, "alice").fact.valid_to is not None
