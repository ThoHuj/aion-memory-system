# JSON schema field description for the `entities` array in structured extraction output.
ENTITY_EXTRACTION_ENTITIES_FIELD_DESCRIPTION = """\
Entities in the fact, each with its exact text as written in the fact. \
"""

# System message for the entity-extraction chat completion (one fact per call).
ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
You extract entities from a single semantic fact for a memory index. For each entity, \
return its text exactly as it appears in the fact. Only return text that appears in the fact. \

Do not return dates, times, numbers, amounts or pronouns. Do not invent entities the fact \
does not support.\
"""

# User message for entity extraction. Example of what the model sees:
#
# <fact>
# User is moving the API to async implementations next week.
# </fact>
ENTITY_EXTRACTION_USER_PROMPT = """\
Extract entities from the fact in <fact> below.

<fact>
{fact}
</fact>\
"""


def format_entity_extraction_user_prompt(fact: str) -> str:
    """Fill the user prompt template with a fact string."""
    return ENTITY_EXTRACTION_USER_PROMPT.format(fact=fact)
