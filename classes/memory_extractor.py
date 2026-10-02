import json

from classes.memory_store import MemoryStore
from classes.models import EpisodicMemoryExtractionResult
from classes.open_router_client import OpenRouterClient

EPISODIC_MEMORY_EXTRACTION_PROMPT = """You are a tool that is part of an agentic memory system.
Your purpose is to maintain a small, non-redundant set of episodic memories for an agent's long-term memory.
You write memories efficiently and concisely.

An episodic memory is a past event involving a decision, outcome, or development — not a passing update on the user's current state.
This is not a conversation log. Most exchanges yield no changes.

Brevity rules (apply to every description):
- Write at most one or two short sentences. Capture what mattered, not how the conversation unfolded step by step.
- Do not turn a memory into a summary of the whole thread, a timeline, or a checklist of sub-steps.
- Put specific names, identifiers, and retrieval hooks in keywords, not in long prose in description.
- On UPDATE, rewrite to a tighter summary that reflects the latest understanding. Drop redundant or superseded phrasing; do not append the entire new dialogue onto the old text.

Read the existing memories for this thread (JSON below). Then read the conversation messages.

Consolidation rules (follow strictly):
- Prefer UPDATE over CREATE. When new dialogue adds detail, clarification, correction, progress, or a final outcome related to something already stored, UPDATE the best-matching existing memory by uuid. Fold new facts into that single concise description; do not leave older partial versions as separate memories.
- CREATE only when the conversation clearly introduces a separate episodic beat that no existing memory covers (different decision, outcome, or development).
- If one existing memory already describes the same situation, subject, or ongoing arc as the new material, you must UPDATE it rather than CREATE a parallel memory.
- When unsure whether to UPDATE or CREATE, UPDATE the closest existing memory.
- Aim for the fewest memories that still capture distinct events. Do not fragment one story across multiple creates.
- If the messages add no new episodic information beyond what is already stored, return empty creates and empty updates.

For each CREATE:
- description: brief (see brevity rules)
- keywords: for retrieval
- extraction_reason: one word only

For each UPDATE:
- uuid: must match an existing memory uuid from the JSON below
- description: full replacement text, still brief (see brevity rules)
- keywords: full replacement list

Do not invent events not grounded in the conversation. Do not update uuids that are not listed below."""

_PROMPT_CONTEXT_KEYS = ("uuid", "description", "created_at", "last_changed")


def _build_episodic_extraction_system_prompt(
    thread_memories: list[dict[str, object]],
) -> str:
    """Build the episodic extraction system prompt with existing thread memories appended."""
    existing_for_prompt = [
        {key: memory[key] for key in _PROMPT_CONTEXT_KEYS if key in memory}
        for memory in thread_memories
    ]
    existing_json = json.dumps(existing_for_prompt, indent=2)
    return (
        f"{EPISODIC_MEMORY_EXTRACTION_PROMPT}\n\n"
        f"Existing memories for this thread (oldest first):\n{existing_json}"
    )


def _build_episodic_extraction_messages(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build the message list passed to the extraction model."""
    return [
        {"role": "system", "content": system_prompt},
        *messages,
    ]


class MemoryExtractor:
    """Extract episodic memory creates and updates from conversation via structured LLM output."""

    def __init__(
        self,
        open_router_client: OpenRouterClient,
        memory_store: MemoryStore,
    ) -> None:
        self._open_router_client = open_router_client
        self._memory_store = memory_store

    def extract_episodic_memories(
        self, thread_id: str, messages: list[dict[str, str]]
    ) -> EpisodicMemoryExtractionResult:
        """Return episodic memory creates and updates for the thread and messages."""
        thread_memories = self._memory_store.load_episodic_memories_for_thread(
            thread_id
        )
        system_prompt = _build_episodic_extraction_system_prompt(thread_memories)
        extraction_messages = _build_episodic_extraction_messages(
            system_prompt, messages
        )
        return self._open_router_client.complete_structured(
            extraction_messages,
            EpisodicMemoryExtractionResult,
        )
