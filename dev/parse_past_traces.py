import json
from pathlib import Path

_DEV_DIRECTORY = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _DEV_DIRECTORY.parent
_DEFAULT_EPISODIC_MEMORIES_PATH = _REPOSITORY_ROOT / "data" / "episodic_memories.json"
_DEFAULT_OUTPUT_PATH = _DEV_DIRECTORY / "data" / "episodic_memories.json"


_STRIPPED_KEYS = ("embedding", "uuid")


def strip_embeddings_from_episodic_memories(
    episodic_memories_path: Path = _DEFAULT_EPISODIC_MEMORIES_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
) -> int:
    """
    Load episodic memories from JSON, remove ``embedding`` and ``uuid`` from each
    record, and write the result under ``dev/data/`` (default: ``episodic_memories.json``).

    The source file under ``data/`` is not modified.

    Returns:
        Number of memory records written to the output file.
    """
    if not episodic_memories_path.exists():
        raise FileNotFoundError(episodic_memories_path)

    elements = json.loads(episodic_memories_path.read_text(encoding="utf-8"))
    if not isinstance(elements, list):
        raise TypeError(f"Expected a JSON array in {episodic_memories_path}")

    received_count = len(elements)
    print(f"Received {received_count} episodic memories to parse.")

    for element in elements:
        if isinstance(element, dict):
            for key in _STRIPPED_KEYS:
                element.pop(key, None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(elements, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return received_count


if __name__ == "__main__":
    count = strip_embeddings_from_episodic_memories()
    print(
        f"Wrote {count} episodic memories (without embedding or uuid) to "
        f"{_DEFAULT_OUTPUT_PATH.relative_to(_REPOSITORY_ROOT)}."
    )
