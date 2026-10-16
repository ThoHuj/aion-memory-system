# JSON schema field description for the `matches` array in structured resolution output.
ENTITY_RESOLUTION_MATCHES_FIELD_DESCRIPTION = """\
One match per new entity that has candidates. Use the exact new entity name as given.\
"""

# JSON schema field description for `existing_entity_id` in each match.
ENTITY_RESOLUTION_EXISTING_ENTITY_ID_FIELD_DESCRIPTION = """\
The id of the candidate that is the same real-world thing as the new entity, or null \
when it is not clearly the same as any candidate.\
"""

# System message for the entity-resolution chat completion.
ENTITY_RESOLUTION_SYSTEM_PROMPT = """\
You are a graph memory manager that decides whether entities mentioned in new messages \
are already stored in the memory graph.

You receive the new messages, recent session messages as background, and a list of new \
entities. Each new entity comes with candidate stored entities. Each candidate lists its \
stored facts as `source -- relationship -- destination` lines.

Guidelines:
1. Return a candidate id only when the messages and the candidate's facts make it clear \
that the new entity and the candidate are the same real-world thing.
2. The same name can be different things. For example, the user's mother Maria and the \
user's coworker Maria are different entities.
3. Similar names alone are not enough. Use the candidate's facts and the context.
4. The same normalized name together with a consistent role or consistent stored facts is strong evidence of the same entity. Return that candidate unless the messages or facts explicitly contradict it (e.g. a different relationship to the user, or a different person mentioned in the same context). A new fact about the entity, such as a new city or job, is not a contradiction.
5. When the name matches but the evidence is genuinely ambiguous, return null.
6. Only return ids that appear in the candidate list of that entity.\
"""

# User message for entity resolution. Example of what the model sees:
#
# <new_messages>
# user: My mother Maria moved to Berlin.
# </new_messages>
#
# <recent_session_messages>
# (none)
# </recent_session_messages>
#
# <new_entities>
# new entity: maria (person)
# - id 1234: maria (person)
#   mother_maria -- lives_in -- hamburg
# </new_entities>
ENTITY_RESOLUTION_USER_PROMPT = """\
Decide for each new entity in <new_entities> whether it is one of its candidates.

<new_messages>
{new_messages}
</new_messages>

<recent_session_messages>
{recent_session_messages}
</recent_session_messages>

<new_entities>
{entity_blocks}
</new_entities>\
"""


def format_entity_resolution_user_prompt(
    new_messages: str,
    recent_session_messages: str,
    entity_blocks: str,
) -> str:
    """Fill the user prompt template with messages and entity candidate blocks."""
    return ENTITY_RESOLUTION_USER_PROMPT.format(
        new_messages=new_messages,
        recent_session_messages=recent_session_messages,
        entity_blocks=entity_blocks,
    )
