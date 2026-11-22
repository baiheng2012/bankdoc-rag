"""Ask a question from the command line.

Usage:
    python scripts/ask.py "What is the outlook for inflation?"
    python scripts/ask.py "your question" --clearance 1 --debug
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankdoc.core.db import close_pool  # noqa: E402
from bankdoc.generation.answerer import answer_question  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--clearance", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Also print all retrieved chunks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        result = answer_question(args.question, args.clearance)
    finally:
        close_pool()

    print("\n=== ANSWER ===")
    print(result.answer)
    if result.pii_findings:
        print(f"\n(PII redacted from your question: {result.pii_findings})")
        print(f"(Question as processed: {result.redacted_question})")

    if result.refused:
        print(f"\n(refused, reason: {result.refusal_reason})")
    else:
        print("\n=== SOURCES CITED ===")
        for c in result.citations:
            print(f"- {c.source_uri}, page {c.page_number}  (score {c.score:.3f})")
        if result.invalid_citations:
            print(f"\nWARNING: model cited non-existent sources: {result.invalid_citations}")

    if args.debug:
        print("\n=== ALL RETRIEVED (debug) ===")
        for c in result.retrieved:
            print(f"[{c.score:.3f}] {c.source_uri} p{c.page_number}")


if __name__ == "__main__":
    main()
