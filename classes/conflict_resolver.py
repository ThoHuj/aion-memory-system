from enum import Enum

from pydantic import BaseModel, Field

from classes.memory_embedder import MemoryEmbedder
from classes.memory_models import ExtractedEpisodicMemory, ExtractionResult
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
    ) -> ExtractionResult:
        """Reconcile a new extraction result with memories already on disk."""
        stored_memories = self._memory_store.read_memories_document()
        embedded_episodic_memories = self._memory_embedder.embed_episodic_memory(
            extraction_result.episodic_memories,
        )

        episodic_memories_to_store: list[ExtractedEpisodicMemory] = []
        for candidate in embedded_episodic_memories:
            analysis = self._analyze_episodic_candidate_similarity(
                candidate,
                stored_memories,
            )
            if analysis.overall_band is EpisodicSimilarityBand.DISTINCT_MEMORY:
                episodic_memories_to_store.append(candidate)
                continue
            self._handle_episodic_memory_conflict(candidate, analysis)

        return extraction_result.model_copy(
            update={"episodic_memories": episodic_memories_to_store}
        )

    def _analyze_episodic_candidate_similarity(
        self,
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
            similarity = self._cosine_similarity(
                candidate.embedding,
                stored_memory.embedding,
            )
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

        overall_band = classify_episodic_similarity(highest_similarity)
        return EpisodicCandidateSimilarityAnalysis(
            overall_band=overall_band,
            related_stored_memories=related_stored_memories,
        )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        """Cosine similarity between two embedding vectors."""
        dot = sum(
            left_value * right_value for left_value, right_value in zip(left, right)
        )
        left_norm = sum(left_value * left_value for left_value in left) ** 0.5
        right_norm = sum(right_value * right_value for right_value in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _handle_episodic_memory_conflict(
        self,
        candidate: ExtractedEpisodicMemory,
        analysis: EpisodicCandidateSimilarityAnalysis,
    ) -> None:
        """Apply policy when a new episodic memory conflicts with existing ones."""
        _ = candidate, analysis


def classify_episodic_similarity(similarity: float) -> EpisodicSimilarityBand:
    """Map a cosine similarity score to an episodic similarity band."""
    if similarity >= DEFINITE_DUPLICATE_MIN_SIMILARITY:
        return EpisodicSimilarityBand.DEFINITE_DUPLICATE
    if similarity >= LIKELY_DUPLICATE_MIN_SIMILARITY:
        return EpisodicSimilarityBand.LIKELY_DUPLICATE
    if similarity >= POSSIBLE_UPDATE_MIN_SIMILARITY:
        return EpisodicSimilarityBand.POSSIBLE_UPDATE
    return EpisodicSimilarityBand.DISTINCT_MEMORY
