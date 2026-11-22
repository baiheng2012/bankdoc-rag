"""Audit logging.

Every question that reaches the system is recorded: who asked, what they
asked, what we answered, which chunks were cited, and whether the answer
was withheld. Regulated organisations need this for traceability: for any
answer you must be able to reconstruct what evidence was used.

This module takes plain values (not AnswerResult) so that core/ does not
depend on generation/. Dependencies should point one way: api -> generation
-> retrieval -> core.
"""

from bankdoc.core.db import get_connection


def write_audit_log(
    user_id: str,
    question: str,
    answer: str | None,
    cited_chunk_ids: list[int],
    blocked: bool,
    block_reason: str | None,
) -> None:
    """Insert one row into audit_log.

    Args:
        user_id: Authenticated caller.
        question: The question as received.
        answer: The text returned to the user (or the refusal sentence).
        cited_chunk_ids: Database ids of the chunks cited as evidence.
        blocked: True if no answer was delivered (refusal or guardrail stop).
        block_reason: Machine-readable reason, e.g. "no_relevant_sources".

    Raises:
        Any database error. The caller must NOT swallow it: if the audit
        record cannot be written, the request should fail ("fail closed").
    """
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_log "
            "(user_id, question, answer, cited_chunks, blocked, block_reason) "
            "VALUES (%s, %s, %s, %s::int[], %s, %s)",
            (user_id, question, answer, cited_chunk_ids, blocked, block_reason),
        )
