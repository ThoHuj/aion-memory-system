# JSON schema field description for the `entities` array in structured extraction output.
ENTITY_EXTRACTION_ENTITIES_FIELD_DESCRIPTION = (
    "Named entities mentioned in the fact (people, places, organizations, etc.)."
)

# System message for the entity-extraction chat completion (one fact per call).
ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
You extract named entities from a single semantic fact. Return people, organizations, \
places, products, and other distinct proper nouns or stable identifiers explicitly \
mentioned or clearly implied. Do not invent entities not supported by the fact.\
"""

# User message used for entity extraction; `{fact}` is the semantic fact text.
ENTITY_EXTRACTION_USER_PROMPT = """\
Extract entities from the fact in <fact> below.

<fact>
{fact}
</fact>\
"""


def format_entity_extraction_user_prompt(fact: str) -> str:
    """Fill the user prompt template with a fact string."""
    return ENTITY_EXTRACTION_USER_PROMPT.format(fact=fact)
