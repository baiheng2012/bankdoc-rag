"""Vector retrieval with permission filtering.

Responsibility: given a question and a user clearance level, return the
most relevant chunks together with everything needed to cite them.
"""

from dataclasses import dataclass

from bankdoc.core.config import get_settings
from bankdoc.core.db import get_connection
from bankdoc.core.embedder import embed_query

# The permission filter (access_level <= clearance) is part of the SQL query,
# so chunks the user may not see never enter the candidate set at all.
# "<=>" is pgvector's cosine DISTANCE operator; similarity = 1 - distance.
SEARCH_SQL = """
SELECT c.id, d.title, d.source_uri, c.page_number, c.content,
       1 - (c.embedding <=> %(vec)s) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.access_level <= %(clearance)s
ORDER BY c.embedding <=> %(vec)s
LIMIT %(k)s
"""


@dataclass
class RetrievedChunk:
    """One search hit with its citation metadata."""

    chunk_id: int
    title: str
    source_uri: str  # file name, contains the report date, used in citations
    page_number: int
    content: str
    score: float  # cosine similarity, higher is more relevant


def retrieve(question: str, clearance: int = 0, top_k: int | None = None) -> list[RetrievedChunk]:
    """Find the chunks most similar to the question that the user may see.

    Args:
        question: The user's question in natural language.
        clearance: The user's access level (0 public, 1 internal, 2 restricted).
        top_k: Number of chunks to return. Defaults to the TOP_K setting.

    Returns:
        Chunks ordered from most to least relevant.
    """
    k = top_k if top_k is not None else get_settings().top_k
    vector = embed_query(question)
    with get_connection() as conn:
        rows = conn.execute(
            SEARCH_SQL, {"vec": vector, "clearance": clearance, "k": k}
        ).fetchall()
    return [
        RetrievedChunk(
            chunk_id=r[0], title=r[1], source_uri=r[2],
            page_number=r[3], content=r[4], score=float(r[5]),
        )
        for r in rows
    ]
