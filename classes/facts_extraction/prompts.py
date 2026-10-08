# JSON schema field description for the `facts` array in structured extraction output.
FACTS_EXTRACTION_FACTS_FIELD_DESCRIPTION = (
    "Semantic facts extracted from the new messages."
)

# System message for the semantic fact-extraction chat completion (one turn per call).
FACTS_EXTRACTION_SYSTEM_PROMPT = """\
You extract stable semantic facts about the user and their world from conversation \
context. Each fact should be a concise standalone statement. Do not invent \
information that is not supported by the conversation.\
"""

# User message for fact extraction; placeholders are preformatted context sections.
FACTS_EXTRACTION_USER_PROMPT = """\
Extract semantic facts for this turn. Ground new facts only in <new_messages>; \
treat <semantic_search> and <recent_session_messages> as background.

<semantic_search>
{semantic_search}
</semantic_search>

<recent_session_messages>
{recent_session_messages}
</recent_session_messages>

<new_messages>
{new_messages}
</new_messages>\
"""


def format_facts_extraction_user_prompt(
    semantic_search: str,
    recent_session_messages: str,
    new_messages: str,
) -> str:
    """Fill the user prompt template with context section text."""
    return FACTS_EXTRACTION_USER_PROMPT.format(
        semantic_search=semantic_search,
        recent_session_messages=recent_session_messages,
        new_messages=new_messages,
    )
