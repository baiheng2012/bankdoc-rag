"""Show exactly which numbered sources the LLM receives for a question, and
which of them contain the given figures. Retrieval only: no LLM call.

It mirrors the retrieval steps of answer_question() (redact, strip
placeholders, retrieve, threshold, newest-first, drop injected chunks).
If answerer.py changes, update this script too.

Usage:
    python scripts/inspect_context.py "your question" 0.47 0.26 0.66
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankdoc.core.config import get_settings  # noqa: E402
from bankdoc.core.db import close_pool  # noqa: E402
from bankdoc.guardrails.injection import detect_injection  # noqa: E402
from bankdoc.guardrails.pii import redact_pii, strip_placeholders  # noqa: E402
from bankdoc.retrieval.recency import sort_newest_first  # noqa: E402
from bankdoc.retrieval.retriever import retrieve  # noqa: E402


def main() -> None:
    question, figures = sys.argv[1], sys.argv[2:]
    settings = get_settings()
    try:
        query = strip_placeholders(redact_pii(question).text)
        retrieved = retrieve(query, 0)
    finally:
        close_pool()

    candidates = sort_newest_first([c for c in retrieved if c.score >= settings.min_score])
    relevant = [c for c in candidates if not detect_injection(c.content)]

    print(f"min_score={settings.min_score}, top_k={settings.top_k}: "
          f"{len(relevant)} of {len(retrieved)} retrieved chunks reach the LLM\n")
    for number, chunk in enumerate(relevant, start=1):
        # Plain substring test: "0.26" would also match "10.26". Good enough for a manual check.
        found = [f for f in figures if f in chunk.content]
        print(f"[{number}] {chunk.source_uri} p{chunk.page_number} "
              f"chunk_id={chunk.chunk_id} score={chunk.score:.3f} contains={found}")

    below = [c for c in retrieved if c.score < settings.min_score]
    if below:
        print(f"\n{len(below)} retrieved chunk(s) fell below min_score and were NOT sent.")


if __name__ == "__main__":
    main()
