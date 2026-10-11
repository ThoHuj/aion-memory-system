# JSON schema field description for the `facts` array in structured extraction output.
FACTS_EXTRACTION_FACTS_FIELD_DESCRIPTION = """\
Facts extracted from the new messages. \
Each fact is a self-contained statement of 15 to 80 words that keeps names, numbers, qualifiers \
and dates as written.
"""

# System message for the semantic fact-extraction chat completion (one turn per call).
FACTS_EXTRACTION_SYSTEM_PROMPT = """\
You extract facts from the provided messages for a long-term memory store. \

Extract every distinct memorable item in the new messages that is suitable for a long-term memory: \
preferences, decisions, plans, goals, events, feedback, corrections and personal details. \
Include what the assistant states that is new, such as a recommendation or a plan, \
not only what the user says. \

Write each fact as a self-contained statement that keeps proper nouns, numbers, qualifiers and dates. \
Split detail-rich content into several facts rather than compressing it. 
Skip greetings, filler, generic acknowledgements and restatements of what was already said. \
Use the recent session messages to resolve pronouns and references. 

- **Do not** extract facts from the semantic search or recent session sections; \
they are for avoiding repeats and resolving references only. \
- **Do not** infer gender, age or ethnicity from names. \
- **Do not** invent information the messages do not support.\
- **Do not** extract facts as logs journals.\
- **Do not** redundant facts.\
"""

# User message for fact extraction. Example of what the model sees:
#
# <semantic_search>
# User prefers Python for backend services.
# User lives in Berlin.
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
and <recent_session_messages> as background: use them to avoid repeating known facts and to \
resolve references.

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
