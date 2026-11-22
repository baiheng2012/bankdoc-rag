"""Smoke test for Step 1.

Checks that (1) the database is reachable and has the pgvector extension,
and (2) the embedding model loads and produces vectors that behave
sensibly: related sentences should score higher than unrelated ones.

Run from the project root:
    PYTHONPATH=src python scripts/smoke_test.py
"""

from bankdoc.core.config import get_settings
from bankdoc.core.db import close_pool, get_connection
from bankdoc.core.embedder import embed_documents, embed_query


def check_database() -> None:
    """Verify the DB connection and that the pgvector extension is installed."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert row is not None, "pgvector extension is not installed"
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
        print("[OK] Database reachable. Tables:", [t[0] for t in tables])


def check_embeddings() -> None:
    """Verify the embedding model and that semantic similarity makes sense."""
    settings = get_settings()
    docs = [
        "The Bank of Canada raised its policy interest rate to curb inflation.",
        "A recipe for chocolate chip cookies requires flour, sugar and butter.",
    ]
    doc_vectors = embed_documents(docs)
    query_vector = embed_query("What did the central bank do about inflation?")

    assert doc_vectors.shape == (2, settings.embedding_dim), "unexpected vector shape"

    # Vectors are normalised, so the dot product is the cosine similarity.
    score_related = float(doc_vectors[0] @ query_vector)
    score_unrelated = float(doc_vectors[1] @ query_vector)
    print(f"[OK] Embedding dim = {doc_vectors.shape[1]}")
    print(f"     related score   = {score_related:.3f}")
    print(f"     unrelated score = {score_unrelated:.3f}")
    assert score_related > score_unrelated, "related text should score higher"


if __name__ == "__main__":
    try:
        check_database()
        check_embeddings()
        print("\nAll smoke checks passed.")
    finally:
        close_pool()
