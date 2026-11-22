"""Database writes for ingestion.

All functions take an open connection so the caller decides the
transaction boundary. The pipeline wraps one whole document in a single
transaction: either everything is stored, or nothing is.
"""

import numpy as np
import psycopg

from bankdoc.ingestion.chunker import Chunk


def find_document_id(conn: psycopg.Connection, source_uri: str) -> int | None:
    """Return the id of an already-ingested document, or None."""
    row = conn.execute(
        "SELECT id FROM documents WHERE source_uri = %s", (source_uri,)
    ).fetchone()
    return row[0] if row else None


def delete_document(conn: psycopg.Connection, document_id: int) -> None:
    """Delete a document. Its chunks are removed by ON DELETE CASCADE."""
    conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def insert_document(
    conn: psycopg.Connection, source_uri: str, title: str, access_level: int
) -> int:
    """Insert a document row and return its new id."""
    row = conn.execute(
        "INSERT INTO documents (source_uri, title, access_level) "
        "VALUES (%s, %s, %s) RETURNING id",
        (source_uri, title, access_level),
    ).fetchone()
    return row[0]


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    access_level: int,
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> None:
    """Insert all chunks of one document in a single batch.

    Args:
        conn: Open connection (inside a transaction).
        document_id: Id returned by insert_document().
        access_level: Copied onto every chunk so retrieval can filter on
            the chunks table alone, without a JOIN.
        chunks: The chunks, in document order.
        embeddings: Array of shape (len(chunks), dim), same order as chunks.
    """
    rows = [
        (document_id, c.chunk_index, c.page_number, c.content, access_level, embeddings[i])
        for i, c in enumerate(chunks)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks "
            "(document_id, chunk_index, page_number, content, access_level, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            rows,
        )
