# JSON schema field description for the `facts` array in structured extraction output.
FACTS_EXTRACTION_FACTS_FIELD_DESCRIPTION = """\
Facts extracted from the new messages. \
Empty if the new messages contain nothing new of lasting value. \
Each fact is a self-contained statement of 15 to 80 words that keeps names, numbers, qualifiers \
and dates as written.
"""

# System message for the semantic fact-extraction chat completion (one turn per call).
FACTS_EXTRACTION_SYSTEM_PROMPT = """\
You extract facts from the provided messages for a long-term memory store.

A fact belongs in long-term memory only if it would still be useful in a future conversation: \
information about the user, their world, their preferences, decisions, plans, goals or events, \
and substantive information the assistant provides. Extract every such item from the new messages, \
from either speaker.

Extract information, not conversation. Do not record that something was said, asked or \
acknowledged; record only the information it carries. Content with no lasting informational value \
is not a fact, regardless of who said it or how it is phrased. If nothing in the new messages \
qualifies, return an empty list.

Write each fact as a self-contained statement that keeps proper nouns, numbers, qualifiers and dates. \
Split detail-rich content into several facts rather than compressing it. \
Use the recent session messages to resolve pronouns and references.

Memory already holds what the semantic search shows, so only extract what it does not already \
contain. Judge by meaning, not wording: a fact is already known if the semantic search or the \
recent session messages convey the same information, however it is phrased. If a fact only \
partly overlaps with something known, extract only the new or changed part. Never return two facts \
that convey the same information. When unsure whether something is new, leave it out.

- **Do not** extract facts from the semantic search or recent session sections; \
they are for avoiding repeats and resolving references only.
- **Do not** infer gender, age or ethnicity from names.
- **Do not** invent information the messages do not support.\
"""

# User message for fact extraction. Example of what the model sees:
#
# <semantic_search>
# [2026-10-11T18:00:00+00:00] User prefers Python for backend services.
# [2026-10-10T12:30:00+00:00] User lives in Berlin.
# </semantic_search>
#
# <recent_session_messages>
# user: Can we use FastAPI for the API?
# assistant: FastAPI fits well with your Python stack.
# </recent_session_messages>
#
# <new_messages>
# user: I'm moving the API to async next week.
# assistant: I'll remember the async migration timeline.
# </new_messages>
FACTS_EXTRACTION_USER_PROMPT = """\
Extract facts for this turn. Ground new facts only in <new_messages>. Treat <semantic_search> \
and <recent_session_messages> as background: use them to resolve references and to skip any \
fact whose meaning is already present, even if worded differently. If nothing new of lasting value remains, return no facts.

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
