import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class MemoryStore:
    """Memory store."""

    def __init__(
        self, episodic_memories_path: Path = Path("data/episodic_memories.json")
    ) -> None:
        self._episodic_memories_path = episodic_memories_path

    def store_episodic_memory(
        self,
        description: str,
        keywords: list[str],
    ) -> None:
        """Store an episodic memory."""
        episodic_memory = {
            "uuid": str(uuid.uuid4()),
            "description": description,
            "keywords": keywords,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._add_element_to_json(self._episodic_memories_path, episodic_memory)

    def _add_element_to_json(
        self, file_path: Path, element: dict[str, object]
    ) -> None:
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
