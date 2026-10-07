from pathlib import Path


class MemoryStore:
    """Persists extracted memories as JSON."""

    def __init__(self, memories_path: Path = Path("data/memories.json")) -> None:
        self._memories_path = memories_path
