"""Integration tests: permission filtering against a REAL PostgreSQL + pgvector.

Unit tests prove the Python logic; these prove the SQL. They insert three tiny
documents (public / internal / restricted) that share one hand-made vector,
replace the embedding model with a function returning that same vector (no
model needed), and check which chunks retrieve() returns for each clearance.

The test chunks have similarity 1.0 to the query, so they always rank first,
even if the database also holds real documents.

Run locally (the database container must be up):
    RUN_DB_TESTS=1 pytest -q -m integration
In CI the same variable is set and a pgvector service container is used.
"""

import os

import numpy as np
import pytest

from bankdoc.core.db import close_pool, get_connection
from bankdoc.retrieval import retriever

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_DB_TESTS") != "1", reason="set RUN_DB_TESTS=1 to run database tests"
    ),
]

EMBEDDING_DIM = 384
TEST_URI_PREFIX = "zz-test/"  # every test document uses this prefix, so cleanup is safe


def _delete_test_documents() -> None:
    """Remove test documents. Chunks go with them (ON DELETE CASCADE)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE source_uri LIKE %s", (TEST_URI_PREFIX + "%",))


@pytest.fixture
def seeded(monkeypatch):
    """Insert one document and one chunk per access level; return {level: chunk_id}."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = 1.0  # a unit vector: identical for all test chunks and for the query

    _delete_test_documents()  # in case an earlier aborted run left rows behind
    chunk_ids: dict[int, int] = {}
    with get_connection() as conn:
        for level in (0, 1, 2):
            document_id = conn.execute(
                "INSERT INTO documents (source_uri, title, access_level) "
                "VALUES (%s, %s, %s) RETURNING id",
                (f"{TEST_URI_PREFIX}level{level}.pdf", f"Test level {level}", level),
            ).fetchone()[0]
            chunk_ids[level] = conn.execute(
                "INSERT INTO chunks "
                "(document_id, chunk_index, page_number, content, access_level, embedding) "
                "VALUES (%s, 0, 1, %s, %s, %s) RETURNING id",
                (document_id, f"test content level {level}", level, vector),
            ).fetchone()[0]

    # retriever.py did "from ... import embed_query", so patch the name THERE.
    monkeypatch.setattr(retriever, "embed_query", lambda question: vector)
    yield chunk_ids
    _delete_test_documents()
    close_pool()


def returned_ids(clearance: int) -> set[int]:
    return {c.chunk_id for c in retriever.retrieve("anything", clearance=clearance, top_k=10)}


def test_public_user_sees_only_public_chunks(seeded):
    got = returned_ids(0)
    assert seeded[0] in got
    assert seeded[1] not in got and seeded[2] not in got


def test_internal_user_sees_public_and_internal_but_not_restricted(seeded):
    got = returned_ids(1)
    assert seeded[0] in got and seeded[1] in got
    assert seeded[2] not in got


def test_restricted_user_sees_everything(seeded):
    got = returned_ids(2)
    assert {seeded[0], seeded[1], seeded[2]} <= got
