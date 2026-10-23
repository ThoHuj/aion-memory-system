CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS add_log (
    add_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    messages JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_entity_id TEXT NOT NULL REFERENCES entities (entity_id),
    relation TEXT NOT NULL,
    destination_entity_id TEXT NOT NULL REFERENCES entities (entity_id),
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    world_valid_from TIMESTAMPTZ NOT NULL,
    world_valid_to TIMESTAMPTZ,
    add_id TEXT NOT NULL REFERENCES add_log (add_id)
);

CREATE INDEX IF NOT EXISTS entities_user_id_index ON entities (user_id);
CREATE INDEX IF NOT EXISTS facts_source_entity_id_index ON facts (source_entity_id);
CREATE INDEX IF NOT EXISTS facts_destination_entity_id_index ON facts (destination_entity_id);
CREATE INDEX IF NOT EXISTS add_log_session_id_index ON add_log (session_id, created_at);
