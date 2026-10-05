from enum import Enum

from pydantic import BaseModel, Field

from classes.memory_embedder import MemoryEmbedder
from classes.memory_models import (
    ExtractedEpisodicMemory,
    ExtractedSemanticFact,
    ExtractionResult,
)
from classes.memory_store import EpisodicMemoryRecord, MemoriesDocument, MemoryStore

DEFINITE_DUPLICATE_MIN_SIMILARITY = 0.90
LIKELY_DUPLICATE_MIN_SIMILARITY = 0.80
POSSIBLE_UPDATE_MIN_SIMILARITY = 0.70


class EpisodicSimilarityBand(str, Enum):
    """Similarity band between a candidate and a stored episodic memory."""

    DEFINITE_DUPLICATE = "definite_duplicate"
    LIKELY_DUPLICATE = "likely_duplicate"
    POSSIBLE_UPDATE = "possible_update"
    DISTINCT_MEMORY = "distinct_memory"


class EpisodicMemoryAction(str, Enum):
    """What to do with a candidate episodic memory after similarity analysis."""

    CREATE = "create"
    UPDATE = "update"
    MERGE = "merge"
    DISCARD = "discard"


class ClassifiedStoredEpisodicSimilarity(BaseModel):
    """One stored episodic memory and its similarity to a candidate."""

    stored_memory: EpisodicMemoryRecord
    similarity: float
    band: EpisodicSimilarityBand


class EpisodicCandidateSimilarityAnalysis(BaseModel):
    """Similarity analysis for one candidate against stored episodic memories."""

    overall_band: EpisodicSimilarityBand
    related_stored_memories: list[ClassifiedStoredEpisodicSimilarity] = Field(
        default_factory=list
    )


class ConflictResolutionResult(BaseModel):
    """Extraction output after conflict resolution; only pending creates are included."""

    episodic_memories_to_create: list[ExtractedEpisodicMemory] = Field(
        default_factory=list
    )
    semantic_facts_to_create: list[ExtractedSemanticFact] = Field(default_factory=list)


class ConflictResolver:
    """Reconcile newly extracted memories with what is already stored."""

    def __init__(
        self,
        memory_store: MemoryStore,
        memory_embedder: MemoryEmbedder,
    ) -> None:
        self._memory_store = memory_store
        self._memory_embedder = memory_embedder

    def resolve_extraction_conflicts(
        self,
        extraction_result: ExtractionResult,
    ) -> ConflictResolutionResult:
        """Embed candidates, resolve conflicts in place, return memories pending create."""
        stored_memories = self._memory_store.read_memories_document()
        embedded_episodic_memories = self._memory_embedder.embed_episodic_memory(
            extraction_result.episodic_memories,
        )

        episodic_memories_to_create: list[ExtractedEpisodicMemory] = []
        stored_memories_changed = False

        for candidate in embedded_episodic_memories:
            analysis = analyze_episodic_candidate_similarity(
                candidate,
                stored_memories,
            )
            action = decide_episodic_memory_action(analysis.overall_band)
            if action is EpisodicMemoryAction.CREATE:
                episodic_memories_to_create.append(candidate)
                continue

            stored_memories_changed |= self._apply_episodic_memory_action(
                candidate,
                analysis,
                action,
                stored_memories,
            )

        if stored_memories_changed:
            self._memory_store.write_memories_document(stored_memories)

        return ConflictResolutionResult(
            episodic_memories_to_create=episodic_memories_to_create,
            semantic_facts_to_create=list(extraction_result.semantic_facts),
        )

    def _apply_episodic_memory_action(
        self,
        candidate: ExtractedEpisodicMemory,
        analysis: EpisodicCandidateSimilarityAnalysis,
        action: EpisodicMemoryAction,
        stored_memories: MemoriesDocument,
    ) -> bool:
        """Mutate stored memories for update, merge, or discard; return True if document changed."""
        match action:
            case EpisodicMemoryAction.DISCARD:
                return False
            case EpisodicMemoryAction.UPDATE:
                target = primary_related_stored_memory(analysis)
                if target is None:
                    return False
                return self._memory_store.update_episodic_memory_in_document(
                    stored_memories,
                    memory_uuid=target.uuid,
                    content=candidate.content,
                    embedding=list(candidate.embedding),
                )
            case EpisodicMemoryAction.MERGE:
                return self._merge_episodic_memory(candidate, analysis, stored_memories)
            case EpisodicMemoryAction.CREATE:
                return False

    def _merge_episodic_memory(
        self,
        candidate: ExtractedEpisodicMemory,
        analysis: EpisodicCandidateSimilarityAnalysis,
        stored_memories: MemoriesDocument,
    ) -> bool:
        """Combine a candidate with an existing episodic memory."""
        _ = candidate, analysis, stored_memories
        return False


def classify_episodic_similarity(similarity: float) -> EpisodicSimilarityBand:
    """Map a cosine similarity score to an episodic similarity band."""
    if similarity >= DEFINITE_DUPLICATE_MIN_SIMILARITY:
        return EpisodicSimilarityBand.DEFINITE_DUPLICATE
    if similarity >= LIKELY_DUPLICATE_MIN_SIMILARITY:
        return EpisodicSimilarityBand.LIKELY_DUPLICATE
    if similarity >= POSSIBLE_UPDATE_MIN_SIMILARITY:
        return EpisodicSimilarityBand.POSSIBLE_UPDATE
    return EpisodicSimilarityBand.DISTINCT_MEMORY


def decide_episodic_memory_action(
    overall_band: EpisodicSimilarityBand,
) -> EpisodicMemoryAction:
    """Choose create, update, merge, or discard from the strongest similarity band."""
    if overall_band is EpisodicSimilarityBand.DISTINCT_MEMORY:
        return EpisodicMemoryAction.CREATE
    if overall_band is EpisodicSimilarityBand.POSSIBLE_UPDATE:
        return EpisodicMemoryAction.UPDATE
    if overall_band is EpisodicSimilarityBand.LIKELY_DUPLICATE:
        return EpisodicMemoryAction.MERGE
    return EpisodicMemoryAction.DISCARD


def analyze_episodic_candidate_similarity(
    candidate: ExtractedEpisodicMemory,
    stored_memories: MemoriesDocument,
) -> EpisodicCandidateSimilarityAnalysis:
    """Classify similarity between a candidate and all stored episodic memories."""
    if not candidate.embedding:
        return EpisodicCandidateSimilarityAnalysis(
            overall_band=EpisodicSimilarityBand.DISTINCT_MEMORY,
        )

    highest_similarity = 0.0
    related_stored_memories: list[ClassifiedStoredEpisodicSimilarity] = []
    for stored_memory in stored_memories.episodic_memories:
        if not stored_memory.embedding:
            continue
        similarity = cosine_similarity(candidate.embedding, stored_memory.embedding)
        if similarity > highest_similarity:
            highest_similarity = similarity
        band = classify_episodic_similarity(similarity)
        if band is EpisodicSimilarityBand.DISTINCT_MEMORY:
            continue
        related_stored_memories.append(
            ClassifiedStoredEpisodicSimilarity(
                stored_memory=stored_memory,
                similarity=similarity,
                band=band,
            )
        )

    return EpisodicCandidateSimilarityAnalysis(
        overall_band=classify_episodic_similarity(highest_similarity),
        related_stored_memories=related_stored_memories,
    )


def primary_related_stored_memory(
    analysis: EpisodicCandidateSimilarityAnalysis,
) -> EpisodicMemoryRecord | None:
    """Return the stored episodic memory with the highest similarity to the candidate."""
    if not analysis.related_stored_memories:
        return None
    return max(
        analysis.related_stored_memories,
        key=lambda related: related.similarity,
    ).stored_memory


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(
        left_value * right_value for left_value, right_value in zip(left, right)
    )
    left_norm = sum(left_value * left_value for left_value in left) ** 0.5
    right_norm = sum(right_value * right_value for right_value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
