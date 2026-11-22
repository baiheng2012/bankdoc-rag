-- Enable the pgvector extension (adds the VECTOR column type and distance operators)
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per source document (e.g. one PDF file).
-- access_level supports permission-based filtering: a user can only
-- retrieve chunks whose access_level is <= their own clearance.
--   0 = public, 1 = internal, 2 = restricted
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    source_uri    TEXT NOT NULL UNIQUE,      -- file path or URL
    title         TEXT NOT NULL,
    access_level  SMALLINT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per text chunk. Each chunk carries its own embedding vector.
-- VECTOR(384) matches the output size of the bge-small embedding model.
CREATE TABLE IF NOT EXISTS chunks (
    id            SERIAL PRIMARY KEY,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,          -- position of the chunk within the document
    page_number   INTEGER,                   -- source page, used for citations
    content       TEXT NOT NULL,
    access_level  SMALLINT NOT NULL DEFAULT 0,  -- copied from the document for fast filtering
    embedding     VECTOR(384) NOT NULL
);

-- HNSW index for fast approximate nearest-neighbour search using cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Index to speed up the permission filter.
CREATE INDEX IF NOT EXISTS chunks_access_idx ON chunks (access_level);

-- Audit log: every question is recorded for governance and traceability.
-- Regulated environments (like a central bank) require this.
CREATE TABLE IF NOT EXISTS audit_log (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id       TEXT NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT,
    cited_chunks  INTEGER[],                 -- ids of the chunks used as evidence
    blocked       BOOLEAN NOT NULL DEFAULT FALSE,  -- true if a guardrail stopped the request
    block_reason  TEXT
);

-- API users. Only a SHA-256 hash of each key is stored, never the key itself.
-- clearance is the user's access level (0 public, 1 internal, 2 restricted);
-- it is looked up on the server for every request.
CREATE TABLE IF NOT EXISTS api_users (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL UNIQUE,
    api_key_hash  TEXT NOT NULL UNIQUE,
    clearance     SMALLINT NOT NULL DEFAULT 0 CHECK (clearance BETWEEN 0 AND 2),
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,   -- set to FALSE to revoke access
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
