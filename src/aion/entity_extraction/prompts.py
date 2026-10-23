# Field descriptions below are Mem0's EXTRACT_ENTITIES_TOOL descriptions.
ENTITY_EXTRACTION_ENTITIES_FIELD_DESCRIPTION = "An array of entities with their types."
ENTITY_EXTRACTION_ENTITY_FIELD_DESCRIPTION = "The name or identifier of the entity."
ENTITY_EXTRACTION_ENTITY_TYPE_FIELD_DESCRIPTION = "The type or category of the entity."

# System message (Mem0 verbatim) with {user_id} filled in at call time.
ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "You are a smart assistant who understands entities and their types in a given text. "
    "If user message contains self reference such as 'I', 'me', 'my' etc. then use "
    "{user_id} as the source entity. Extract all the entities from the text. "
    "***DO NOT*** answer the question itself if the given text is a question. "
    "Extra rules: entities about the user come only from what the user states or "
    "confirms. A question carries no information, so extract nothing from it beyond "
    "the entities it names. Assistant messages are a source only about the assistant "
    "itself and information it explicitly provides. Never extract entities that "
    "appear only in the recent session messages, but when the text refers to one of "
    "them by role or pronoun (e.g. 'my mother'), use its name from there."
)

# User message. The recent session messages are background only.
ENTITY_EXTRACTION_USER_PROMPT = """\
Extract entities from the text in <text> below. The messages in \
<recent_session_messages> are background only, for resolving references; do not extract \
entities that appear only there.

<recent_session_messages>
{recent_session_messages}
</recent_session_messages>

<text>
{text}
</text>\
"""


def format_entity_extraction_system_prompt(user_id: str) -> str:
    """Fill the system prompt template with the raw user id."""
    return ENTITY_EXTRACTION_SYSTEM_PROMPT.format(user_id=user_id)


def format_entity_extraction_user_prompt(
    text: str,
    recent_session_messages: str,
) -> str:
    """Fill the user prompt template with the text and session background."""
    return ENTITY_EXTRACTION_USER_PROMPT.format(
        text=text,
        recent_session_messages=recent_session_messages,
    )
