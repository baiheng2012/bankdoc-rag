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
