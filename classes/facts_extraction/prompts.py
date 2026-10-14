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
and information about the assistant's own work. Extract every such item from the new messages.

Judge each item by who is its source. Facts about the user and their world come only from what \
the user states or confirms. The assistant's messages are not evidence about the user or their \
world: its questions, guesses, assumptions and elaborations are not facts. A question carries no \
information, not even a negated or opposite version of what it asks: what was asked about but not \
answered is unknown, and unknown is never recorded. The assistant is a source only about itself and its own work, such as what it \
did, discovered, learned or decided, and for information it explicitly provides, such as an \
answer or a recommendation. Before writing a fact about the user or their world, check that the \
user's own words support it; if the only support is something the assistant said, drop it.

Extract information, not conversation. Do not record that something was said, asked or \
acknowledged; record only the information it carries. Content with no lasting informational value \
is not a fact, regardless of who said it or how it is phrased. If nothing in the new messages \
qualifies, return an empty list.

Write each fact as a self-contained statement that keeps proper nouns, numbers, qualifiers and dates. \
Split detail-rich content into several facts rather than compressing it. \
Use the recent session messages to resolve pronouns and references.

Memory already holds the facts listed under already known facts, so only extract what they do not \
contain. Judge by meaning, not wording: information is already known if the already known facts or \
the recent session messages convey it, however it is phrased, and the new messages repeating or \
recalling it does not make it new. If a fact only \
partly overlaps with something known, extract only the new or changed part. Never return two facts \
that convey the same information. When unsure whether something is new, leave it out.

- **Do not** extract facts from the already known facts or recent session messages; \
they are for avoiding repeats and resolving references only.
- **Do not** infer gender, age or ethnicity from names.
- **Do not** invent information the messages do not support.\
"""

# User message for fact extraction. Example of what the model sees:
#
# <new_messages>
# user: I'm moving the API to async next week.
# assistant: I'll remember the async migration timeline.
# </new_messages>
#
# <recent_session_messages>
# user: Can we use FastAPI for the API?
# assistant: FastAPI fits well with your Python stack.
# </recent_session_messages>
#
# <already_known_facts>
# [2026-10-11T18:00:00+00:00] User prefers Python for backend services.
# [2026-10-10T12:30:00+00:00] User lives in Berlin.
# </already_known_facts>
FACTS_EXTRACTION_USER_PROMPT = """\
Extract facts from <new_messages>. <recent_session_messages> and <already_known_facts> are \
background only: use them to resolve references and to skip anything already known, even if \
worded differently. Return only information that is new in <new_messages>; if there is none, \
return no facts.

<new_messages>
{new_messages}
</new_messages>

<recent_session_messages>
{recent_session_messages}
</recent_session_messages>

<already_known_facts>
{semantic_search}
</already_known_facts>\
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
