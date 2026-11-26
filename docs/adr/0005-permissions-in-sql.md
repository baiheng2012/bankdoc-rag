# ADR 0005: Enforce permissions in SQL, with clearance decided by the server

Status: accepted

## Context
Documents carry an access level (0 public, 1 internal, 2 restricted). Filtering after generation, or asking the model to hide content, would let restricted text reach the prompt.

## Decision
- The retrieval query contains `WHERE access_level <= clearance`, applied before ranking, so a user's restricted chunks never become candidates.
- Clearance is looked up from the authenticated API key on the server. The request schema forbids extra fields, so a client that sends its own `clearance` gets HTTP 422.
- API keys are stored as SHA-256 hashes; a key can be revoked with `is_active = false`.
- The behaviour is tested against a real PostgreSQL with pgvector (three integration tests, run in CI): level 0 sees only public chunks, level 1 sees public and internal, level 2 sees all.

## Consequences
- Verified manually: a revoked key returns 401 while another user's key keeps working. Blocked requests are written to the audit table.
- Access levels here come from folder names and are a demonstration, not a classification system.
- API keys are a simplification. A production deployment would use OIDC (for example Microsoft Entra ID) and map groups to clearance; `get_current_user` is the single place to change.
- Duplicate versions of a document at different levels crowd out top_k; avoid ingesting two versions of the same file.
