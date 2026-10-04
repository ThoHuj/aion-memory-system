from pydantic import BaseModel, Field

from classes.open_router_client import OpenRouterClient

_EXTRACTION_SYSTEM_PROMPT = """\
You extract agent memory from a short conversation window.

Produce:

1. episodic_memories — discrete episodes worth recalling later (what happened in \
this exchange, decisions, tasks, or ongoing situations the user described). \
Each episode is a third-person summary.

2. semantic_facts — stable distilled facts about the user or their world that \
should persist beyond this episode (preferences, constraints, relationships, \
standing obligations). One fact per item; concise standalone statements.

3. retrieval_keywords — lowercase hyphenated keywords summarizing the window for \
search and retrieval across episodic and semantic stores.

Rules:
- If the window is only greetings, small talk, or generic advice with no new \
user-specific information, return empty episodic_memories, semantic_facts, and \
retrieval_keywords.
- Do not invent information not supported by the conversation.
- Prefer one rich episodic memory over many thin duplicates.
- extraction_reason on an episodic memory is a short label for why it was stored \
(e.g. task, preference, problem, plan).
"""


class ExtractedEpisodicMemory(BaseModel):
    """Episodic memory extracted from a conversation window."""

    description: str = Field(
        description="Third-person summary of the episode to store."
    )
    extraction_reason: str = Field(
        description="Short label for why this episode was extracted."
    )


class ExtractedSemanticFact(BaseModel):
    """Semantic fact extracted from a conversation window."""

    content: str = Field(description="Standalone factual statement about the user.")


class ExtractionResult(BaseModel):
    """Structured output from memory and fact extraction."""

    episodic_memories: list[ExtractedEpisodicMemory] = Field(
        description="Episodic memories extracted from the window."
    )
    semantic_facts: list[ExtractedSemanticFact] = Field(
        description="Semantic facts extracted from the window."
    )
    retrieval_keywords: list[str] = Field(
        description="Keywords for retrieving content related to this window."
    )


def _format_conversation_for_prompt(messages: list[dict[str, str]]) -> str:
    """Render message history as plain text for the extraction user prompt."""
    blocks: list[str] = []
    for message in messages:
        role = message.get("role", "unknown").strip().capitalize()
        content = message.get("content", "").strip()
        blocks.append(f"{role}:\n{content}")
    return "\n\n".join(blocks)


def _build_extraction_messages(
    conversation_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build chat messages for the extraction model call."""
    conversation_text = _format_conversation_for_prompt(conversation_messages)
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract episodic memories, semantic facts, and retrieval keywords "
                f"from this conversation window:\n\n{conversation_text}"
            ),
        },
    ]


def extract_memories_and_facts_from_messages(
    extraction_client: OpenRouterClient,
    messages: list[dict[str, str]],
) -> ExtractionResult:
    """Build the extraction prompt, call the extraction model, and return structured results."""
    if not messages:
        return ExtractionResult(
            episodic_memories=[],
            semantic_facts=[],
            retrieval_keywords=[],
        )
    extraction_messages = _build_extraction_messages(messages)
    return extraction_client.complete_structured(
        extraction_messages,
        ExtractionResult,
    )
