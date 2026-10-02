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
            raise TypeError(f"Expected a JSON array in {self._episodic_memories_path}")
        return elements

    def load_episodic_memories_for_thread(
        self, thread_id: str
    ) -> list[dict[str, object]]:
        """Load episodic memories for one thread, oldest first by created_at."""
        thread_memories = [
            memory
            for memory in self.load_episodic_memories()
            if memory.get("thread_id") == thread_id
        ]
        thread_memories.sort(
            key=lambda memory: str(memory.get("created_at", "")),
        )
        return thread_memories

    def store_episodic_memory(
        self,
        thread_id: str,
        description: str,
        keywords: list[str],
        extraction_reason: str,
        embedding: list[float],
    ) -> None:
        """Store a new episodic memory."""
        now = datetime.now(UTC).isoformat()
        episodic_memory = {
            "uuid": str(uuid.uuid4()),
            "thread_id": thread_id,
            "description": description,
            "keywords": keywords,
            "extraction_reason": extraction_reason,
            "embedding": embedding,
            "created_at": now,
            "last_changed": now,
        }
        self._add_element_to_json(self._episodic_memories_path, episodic_memory)

    def update_episodic_memory(
        self,
        memory_uuid: str,
        description: str,
        keywords: list[str],
    ) -> None:
        """Update description and keywords; refresh last_changed and clear embedding."""
        elements = self.load_episodic_memories()
        for memory in elements:
            if memory.get("uuid") != memory_uuid:
                continue
            memory["description"] = description
            memory["keywords"] = keywords
            memory["last_changed"] = datetime.now(UTC).isoformat()
            memory["embedding"] = []
            self._write_episodic_memories(elements)
            return
        raise ValueError(f"Episodic memory not found: {memory_uuid}")

    def _write_episodic_memories(self, elements: list[dict[str, object]]) -> None:
        """Overwrite the episodic memories file."""
        self._episodic_memories_path.parent.mkdir(parents=True, exist_ok=True)
        self._episodic_memories_path.write_text(json.dumps(elements, indent=2))

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
