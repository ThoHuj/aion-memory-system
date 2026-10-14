import json

from classes.memory_system import MemorySystem

GET_FACT_SOURCES_DEFINITION: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_fact_sources",
        "description": (
            "Get the original conversation messages that one or more remembered "
            "facts were extracted from, including when and in what context the "
            "user said it. Call this whenever the user asks for details, "
            "context, timing, origin, or proof about something you remember, "
            "or asks how or when you know something. Remembered facts are only "
            "short summaries; the source messages are the only way to give "
            "accurate details. Use the fact ids shown in <relevant_user_facts>. "
            "Always use the complete id exactly as shown (full UUID). Never "
            "shorten, abbreviate, or truncate ids, neither in tool calls nor "
            "in replies to the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Complete, unabbreviated fact ids to look up sources for."
                    ),
                },
            },
            "required": ["fact_ids"],
        },
    },
}


def get_fact_sources(memory_system: MemorySystem, fact_ids: list[str]) -> str:
    """Return a JSON string with the source messages for each given fact id.

    `memory_system` is bound by the calling code; the agent only supplies `fact_ids`.
    """
    results: list[dict[str, object]] = []
    for fact_id in fact_ids:
        try:
            fact_data = memory_system.get(fact_id)
        except KeyError:
            results.append({"fact_id": fact_id, "error": "fact not found"})
            continue
        source_messages = (
            fact_data.source.messages if fact_data.source is not None else []
        )
        results.append(
            {
                "fact_id": fact_id,
                "fact": fact_data.content,
                "source_messages": source_messages,
            }
        )
    return json.dumps(results, ensure_ascii=False)
