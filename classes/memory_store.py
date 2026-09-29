import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class MemoryStore:
    """MemoryStore manages storing and organizing memories extracted from conversations."""

    def __init__(
        self, episodic_memories_path: Path = Path("data/episodic_memories.json")
    ) -> None:
        self._episodic_memories_path = episodic_memories_path

    def load_episodic_memories(self) -> list[dict[str, object]]:
        """Load all episodic memories from storage."""
        if not self._episodic_memories_path.exists():
            return []

        file_contents = self._episodic_memories_path.read_text()
        elements = json.loads(file_contents)
        if not isinstance(elements, list):
            raise TypeError(
                f"Expected a JSON array in {self._episodic_memories_path}"
            )
        return elements

    def store_episodic_memory(
        self,
        description: str,
        keywords: list[str],
        extraction_reason: str,
        embedding: list[float],
    ) -> None:
        """Store an episodic memory."""
        episodic_memory = {
            "uuid": str(uuid.uuid4()),
            "description": description,
            "keywords": keywords,
            "extraction_reason": extraction_reason,
            "embedding": embedding,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._add_element_to_json(self._episodic_memories_path, episodic_memory)

    def _add_element_to_json(self, file_path: Path, element: dict[str, object]) -> None:
        """Add a new element to a JSON file."""
        if file_path.exists():
            file_contents = file_path.read_text()
            elements = json.loads(file_contents)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            elements = []

        if not isinstance(elements, list):
            raise TypeError(f"Expected a JSON array in {file_path}")

        elements.append(element)
        file_path.write_text(json.dumps(elements, indent=2))
