# JSON schema field description for the `invalidations` array in structured output.
CONFLICT_RESOLUTION_INVALIDATIONS_FIELD_DESCRIPTION = """\
Existing relationships to invalidate, each identified by its id from the existing \
memories list.\
"""

# JSON schema field description for `world_valid_to` in each invalidation.
CONFLICT_RESOLUTION_WORLD_VALID_TO_FIELD_DESCRIPTION = """\
ISO 8601 date or datetime when the relationship stopped being true, only when the new \
information states or clearly implies it. Otherwise null.\
"""

# System message for the invalidation chat completion. Adapted from Mem0
# v0.1.118 DELETE_RELATIONS_SYSTEM_PROMPT (delete changed to invalidate).
CONFLICT_RESOLUTION_SYSTEM_PROMPT = """\
You are a graph memory manager specializing in identifying, managing, and optimizing relationships within graph-based memories. Your primary task is to analyze a list of existing relationships and determine which ones should be invalidated based on the new information provided.
Input:
1. Existing Graph Memories: A list of current graph memories, each containing an id, source, relationship, and destination information.
2. New Text: The new information to be integrated into the existing graph structure.
3. Use "{user_id}" as node for any self-references (e.g., "I," "me," "my," etc.) in user messages.

Guidelines:
1. Identification: Use the new information to evaluate existing relationships in the memory graph.
2. Invalidation Criteria: Invalidate a relationship only if it meets at least one of these conditions:
   - Outdated or Inaccurate: The new information is more recent or accurate.
   - Contradictory: The new information conflicts with or negates the existing information.
3. DO NOT INVALIDATE if their is a possibility of same type of relationship but different destination nodes.
4. Comprehensive Analysis:
   - Thoroughly examine each existing relationship against the new information and invalidate as necessary.
   - Multiple invalidations may be required based on the new information.
5. Semantic Integrity:
   - Ensure that invalidations maintain or improve the overall semantic structure of the graph.
   - Avoid invalidating relationships that are NOT contradictory/outdated to the new information.
6. Temporal Awareness: Prioritize recency when timestamps are available.
7. Necessity Principle: Only INVALIDATE relationships that must be invalidated and are contradictory/outdated to the new information to maintain an accurate and coherent memory graph.

Note: DO NOT INVALIDATE if their is a possibility of same type of relationship but different destination nodes. 

For example: 
Existing Memory: alice -- loves_to_eat -- pizza
New Information: Alice also loves to eat burger.

Do not invalidate in the above example because there is a possibility that Alice loves to eat both pizza and burger.

Memory Format:
id: source -- relationship -- destination

Provide a list of invalidation instructions, each specifying the id of the relationship to be invalidated. For each invalidated relationship, when the new information gives a relative or absolute time for the change (e.g. "last month", "moved in 2020"), you MUST set world_valid_to to that date as ISO 8601, resolved against the current time {added_at}. Use null only when the new information gives no time.
Example: current time 2026-10-15, existing "5: maria -- lives_in -- hamburg", new information "My mother moved to Berlin last month" -> invalidate id 5 with world_valid_to "2026-09-15".

Recent session messages may be provided as background for resolving references only. Base invalidations on the new information.\
"""

# User message for invalidation. Mem0's get_delete_messages format plus background.
CONFLICT_RESOLUTION_USER_PROMPT = """\
Here are the existing memories: {existing_memories} 

 New Information: {new_messages}

Recent session messages (background only):
{recent_session_messages}\
"""


def format_conflict_resolution_system_prompt(user_id: str, added_at: str) -> str:
    """Fill the system prompt with the raw user id and the add time."""
    return CONFLICT_RESOLUTION_SYSTEM_PROMPT.format(
        user_id=user_id,
        added_at=added_at,
    )


def format_conflict_resolution_user_prompt(
    existing_memories: str,
    new_messages: str,
    recent_session_messages: str,
) -> str:
    """Fill the user prompt template with existing memories and new messages."""
    return CONFLICT_RESOLUTION_USER_PROMPT.format(
        existing_memories=existing_memories,
        new_messages=new_messages,
        recent_session_messages=recent_session_messages,
    )
