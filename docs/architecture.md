# Architecture

## Request flow

See the diagram in the README. Guardrails run at the entrance (script check,
PII redaction, injection detection), in the middle (retrieved chunks are scanned
for injected instructions) and at the exit (citation check, prompt-leak check,
numeric grounding, output PII redaction).

## Module responsibilities

Dependencies point one way: `api -> generation -> retrieval -> core`.

| Package | Responsibility |
|---|---|
| `core/` | Settings, database pool, local embedding model, audit writer |
| `ingestion/` | PDF -> clean text -> sentence-aware chunks (never crossing pages) -> embeddings -> one transaction per document. Idempotent |
| `retrieval/` | pgvector similarity search with the permission filter inside the SQL; publication-date parsing and newest-first ordering |
| `generation/` | Prompt, LLM provider abstraction (Ollama / OpenAI / Anthropic), citation parsing, orchestration in `answerer.py` |
| `guardrails/` | Pure functions: PII redaction, injection rules, script check, numeric grounding, prompt-leak check |
| `api/` | FastAPI app, API-key authentication, request/response schemas. No business logic |
| `evaluation/` | Case loader, deterministic checks, runner, report. The checks have their own unit tests |

## Data model

- `documents(source_uri UNIQUE, title, access_level)` and `chunks(document_id, page_number, content, access_level, embedding VECTOR(384))`.
  `access_level` is copied onto each chunk so the permission filter needs no JOIN.
- `api_users(user_id, api_key_hash, clearance, is_active)`: only a SHA-256 hash of each key is stored.
- `audit_log(user_id, question, answer, cited_chunks, blocked, block_reason)`: the question is stored after PII redaction.
