"""Recency helpers.

Central bank forecasts are superseded by every new report. If an old
projection and a new one are mixed in one answer, the answer is misleading.
We therefore extract the publication date from the file name and let the
answer step put the newest sources first.
"""

import re
from datetime import date

# File names such as "mpr-2026-10-28.pdf" contain the publication date.
_DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def publication_date(source_uri: str) -> date | None:
    """Extract the publication date from a file name.

    Args:
        source_uri: The stored file name, e.g. "mpr-2026-10-28.pdf".

    Returns:
        The date, or None if the name has no valid YYYY-MM-DD part
        (e.g. Staff Working Papers like "swp2026-39.pdf").
    """
    match = _DATE_PATTERN.search(source_uri)
    if not match:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:  # e.g. month 13
        return None


def sort_newest_first(chunks: list) -> list:
    """Order chunks from newest to oldest publication date.

    Chunks without a date go last. Python's sort is stable, so chunks with
    the same date keep their relevance order from the retriever.

    Args:
        chunks: Objects with a source_uri attribute (RetrievedChunk).
    """

    def sort_key(chunk) -> int:
        d = publication_date(chunk.source_uri)
        return d.toordinal() if d else 0  # undated -> 0 -> sorted last

    return sorted(chunks, key=sort_key, reverse=True)
