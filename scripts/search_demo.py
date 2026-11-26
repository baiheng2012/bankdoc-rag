"""Quick manual check of retrieval quality (a preview of Step 3).

Usage:
    python scripts/search_demo.py "What is the outlook for inflation?"
    python scripts/search_demo.py "your question" --clearance 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankdoc.core.config import get_settings
from bankdoc.core.db import close_pool, get_connection
from bankdoc.core.embedder import embed_query

SEARCH_SQL = """
SELECT d.title, c.page_number, c.access_level, c.content,
       1 - (c.embedding <=> %(vec)s) AS score   -- cosine similarity
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.access_level <= %(clearance)s            -- permission filter, applied IN the query
ORDER BY c.embedding <=> %(vec)s                 -- nearest neighbours first
LIMIT %(k)s
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--clearance", type=int, default=0, help="0 public, 1 internal, 2 restricted")
    args = parser.parse_args()

    try:
        vec = embed_query(args.question)
        with get_connection() as conn:
            rows = conn.execute(
                SEARCH_SQL,
                {"vec": vec, "clearance": args.clearance, "k": get_settings().top_k},
            ).fetchall()
        for title, page, level, content, score in rows:
            print(f"\n[{score:.3f}] {title} (page {page}, level {level})")
            print(content[:400] + ("..." if len(content) > 400 else ""))
    finally:
        close_pool()


if __name__ == "__main__":
    main()
