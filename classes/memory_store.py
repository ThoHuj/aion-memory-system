from pathlib import Path


class MemoryStore:
    """MemoryStore manages storing and organizing memories extracted from conversations."""

    def __init__(
        self, episodic_memories_path: Path = Path("data/episodic_memories.json")
    ) -> None:
        self._episodic_memories_path = episodic_memories_path
