# Field descriptions: source, relationship, destination are Mem0's RELATIONS_TOOL descriptions.
RELATION_EXTRACTION_RELATIONS_FIELD_DESCRIPTION = (
    "Relationships among the entities based on the provided text."
)
RELATION_EXTRACTION_SOURCE_FIELD_DESCRIPTION = "The source entity of the relationship."
RELATION_EXTRACTION_RELATIONSHIP_FIELD_DESCRIPTION = (
    "The relationship between the source and destination entities."
)
RELATION_EXTRACTION_DESTINATION_FIELD_DESCRIPTION = (
    "The destination entity of the relationship."
)
RELATION_EXTRACTION_WORLD_VALID_FROM_FIELD_DESCRIPTION = (
    "ISO 8601 date or datetime when the relationship became true, or null if unknown."
)
RELATION_EXTRACTION_WORLD_VALID_TO_FIELD_DESCRIPTION = (
    "ISO 8601 date or datetime when the relationship stopped being true, or null if "
    "unknown or still true."
)

# Mem0 v0.1.118 EXTRACT_RELATIONS_PROMPT, verbatim. USER_ID and CUSTOM_PROMPT are
# replaced at call time.
EXTRACT_RELATIONS_PROMPT = """

You are an advanced algorithm designed to extract structured information from text to construct knowledge graphs. Your goal is to capture comprehensive and accurate information. Follow these key principles:

1. Extract only explicitly stated information from the text.
2. Establish relationships among the entities provided.
3. Use "USER_ID" as the source entity for any self-references (e.g., "I," "me," "my," etc.) in user messages.
CUSTOM_PROMPT

Relationships:
    - Use consistent, general, and timeless relationship types.
    - Example: Prefer "professor" over "became_professor."
    - Relationships should only be established among the entities explicitly mentioned in the user message.

Entity Consistency:
    - Ensure that relationships are coherent and logically align with the context of the message.
    - Maintain consistent naming for entities across the extracted data.

Strive to construct a coherent and easily understandable knowledge graph by establishing all the relationships among the entities and adherence to the user’s context.

Adhere strictly to these guidelines to ensure high-quality knowledge graph extraction."""

# Replaces CUSTOM_PROMPT; {added_at} is the current time for relative dates.
RELATION_EXTRACTION_CUSTOM_PROMPT = """\
4. World time: world_valid_from is when a relationship became true and world_valid_to is when it stopped being true. When the text gives a relative or absolute time for a relationship or a change (e.g. "last month", "moved in 2020"), you MUST set it as an ISO 8601 date resolved against the current time {added_at}. Use null only when the text gives no time.
   Example: current time 2026-10-15, "My mother moved to Berlin last month" -> mother -- lives_in -- berlin with world_valid_from "2026-09-15".
5. Facts about the user come only from what the user states or confirms. A question carries no information, not even an opposite version of what it asks.
6. Assistant messages are a source only about the assistant itself and information it explicitly provides, never about the user or their world.
7. Never extract relationships that appear only in the recent session messages. When the text refers to an entity by role or pronoun (e.g. "my mother"), use the entity named in the list of entities."""

# User message: Mem0's format plus a recent-session background section.
RELATION_EXTRACTION_USER_PROMPT = """\
List of entities: {entity_names}. 

Text: {new_messages}

The messages below are background only, for resolving references; do not extract \
relationships that appear only there.

<recent_session_messages>
{recent_session_messages}
</recent_session_messages>\
"""


def format_relation_extraction_system_prompt(user_id: str, added_at: str) -> str:
    """Fill the Mem0 relations prompt with the raw user id and the world-time rule."""
    custom_prompt = RELATION_EXTRACTION_CUSTOM_PROMPT.format(added_at=added_at)
    return EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_id).replace(
        "CUSTOM_PROMPT",
        custom_prompt,
    )


def format_relation_extraction_user_prompt(
    entity_names: list[str],
    new_messages: str,
    recent_session_messages: str,
) -> str:
    """Fill the user prompt template with entities, new messages and background."""
    return RELATION_EXTRACTION_USER_PROMPT.format(
        entity_names=entity_names,
        new_messages=new_messages,
        recent_session_messages=recent_session_messages,
    )
